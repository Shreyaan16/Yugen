import json
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from src.constants import (EVAL_CF_METRICS_FILE, EVAL_HYBRID_METRICS_FILE, EVAL_SUMMARY_FILE)
from src.entity.config_entity import (CFRecommenderConfig, EvaluationConfig, HybridRecommenderConfig,)
from src.entity.artifact_entity import EvaluationArtifact
from src.components.recommender import (ContentBasedRecommender, UserBasedRecommender, CFRecommender, HybridRecommender)


class Evaluation:
    def __init__(self, config: EvaluationConfig | None = None):
        self.config = config or EvaluationConfig()
        os.makedirs(self.config.evaluation_dir, exist_ok=True)

        self.content: ContentBasedRecommender | None = None
        self.user_based: UserBasedRecommender | None = None
        self.cf: CFRecommender | None = None
        self.hybrid: HybridRecommender | None = None
        self.anime_df: pd.DataFrame | None = None
        self.user_anime_ratings_df: pd.DataFrame | None = None
        self.ratings_df: pd.DataFrame | None = None

        self._cf_cfg = CFRecommenderConfig()
        self._hy_cfg = HybridRecommenderConfig()

    def fit(self, content: ContentBasedRecommender, user_based: UserBasedRecommender, cf: CFRecommender,
        hybrid: HybridRecommender, anime_df: pd.DataFrame, user_anime_ratings_df: pd.DataFrame,
        ratings_df: pd.DataFrame,) -> "Evaluation":
        self.content = content
        self.user_based = user_based
        self.cf = cf
        self.hybrid = hybrid
        self.anime_df = anime_df
        self.user_anime_ratings_df = user_anime_ratings_df
        self.ratings_df = ratings_df
        return self

    def _build_user_vec(self, train_ratings: pd.DataFrame) -> np.ndarray | None:
        """Rebuild a user's TF-IDF-weighted vector from a subset of their ratings."""
        anime_id_to_idx = self.content.anime_id_to_idx
        tfidf_matrix = self.content.tfidf_matrix

        in_idx = train_ratings[train_ratings["anime_id"].isin(anime_id_to_idx)]
        if len(in_idx) == 0:
            return None

        mean_r = in_idx["rating"].mean()
        centered = (in_idx["rating"] - mean_r).astype(np.float32).values
        anime_idxs = in_idx["anime_id"].map(anime_id_to_idx).values

        n_anime = tfidf_matrix.shape[0]
        row = csr_matrix(
            (centered, (np.zeros(len(centered), dtype=np.int64), anime_idxs)),
            shape=(1, n_anime),
        )
        user_vec_sparse = row @ tfidf_matrix
        user_vec_sparse = normalize(user_vec_sparse, norm="l2", axis=1)
        return user_vec_sparse.toarray().astype(np.float32)

    def _cf_top_k(self, user_id: int, user_vec: np.ndarray, seen: set, k: int) -> list[int]:
        ub = self.user_based
        cfg = self._cf_cfg
        self_idx = ub.user_id_to_idx.get(int(user_id))

        sims, nbr_idxs = ub.index.search(user_vec, cfg.k_neighbors + 5)
        sims, nbr_idxs = sims[0], nbr_idxs[0]

        neighbor_ids, neighbor_sims = [], []
        for s, ni in zip(sims, nbr_idxs):
            if ni == -1 or ni == ub.avg_idx or ni == self_idx:
                continue
            nid = ub.idx_to_user_id.get(int(ni))
            if nid is None:
                continue
            neighbor_ids.append(nid)
            neighbor_sims.append(float(s))
            if len(neighbor_ids) == cfg.k_neighbors:
                break
        if not neighbor_ids:
            return []

        sim_map = dict(zip(neighbor_ids, neighbor_sims))
        uar = self.user_anime_ratings_df
        nbr = uar[uar["user_id"].isin(sim_map) & ~uar["anime_id"].isin(seen)].copy()
        if len(nbr) == 0:
            return []
        nbr["sim"] = nbr["user_id"].map(sim_map)

        # Center each neighbor's rating against their own mean (over all their ratings),
        # so the score reflects "lift above this user's personal baseline" rather than
        # raw rating level. Avoids the score collapsing onto globally-popular anime.
        neighbor_means = uar[uar["user_id"].isin(sim_map)].groupby("user_id")["rating"].mean()
        nbr["centered"] = nbr["rating"] - nbr["user_id"].map(neighbor_means)
        nbr["w_centered"] = nbr["centered"] * nbr["sim"]

        agg = nbr.groupby("anime_id").agg(
            num_neighbors=("user_id", "size"),
            weighted_sum=("w_centered", "sum"),
            sim_sum=("sim", "sum"),
        )
        agg = agg[agg["num_neighbors"] >= cfg.min_support]
        if len(agg) == 0:
            return []
        agg["score"] = agg["weighted_sum"] / agg["sim_sum"]

        if cfg.min_num_ratings > 0 and "num_ratings" in self.ratings_df.columns:
            reliable = set(
                self.ratings_df.loc[self.ratings_df["num_ratings"] >= cfg.min_num_ratings, "anime_id"]
            )
            agg = agg[agg.index.isin(reliable)]

        return agg["score"].sort_values(ascending=False).head(k).index.tolist()

    def _hybrid_top_k(self, user_id: int, user_vec: np.ndarray, seen: set, k: int) -> list[int]:
        cb = self.content
        ub = self.user_based
        cfg = self._hy_cfg
        self_idx = ub.user_id_to_idx.get(int(user_id))

        sims, nbr_idxs = ub.index.search(user_vec, cfg.k_neighbors + 5)
        sims, nbr_idxs = sims[0], nbr_idxs[0]
        neighbor_ids, neighbor_sims = [], []
        for s, ni in zip(sims, nbr_idxs):
            if ni == -1 or ni == ub.avg_idx or ni == self_idx:
                continue
            nid = ub.idx_to_user_id.get(int(ni))
            if nid is None:
                continue
            neighbor_ids.append(nid)
            neighbor_sims.append(float(s))
            if len(neighbor_ids) == cfg.k_neighbors:
                break

        cf_scores = pd.Series(dtype="float32", name="cf_score")
        if neighbor_ids:
            sim_map = dict(zip(neighbor_ids, neighbor_sims))
            uar = self.user_anime_ratings_df
            nbr = uar[uar["user_id"].isin(sim_map) & ~uar["anime_id"].isin(seen)].copy()
            if len(nbr) > 0:
                nbr["sim"] = nbr["user_id"].map(sim_map)
                neighbor_means = uar[uar["user_id"].isin(sim_map)].groupby("user_id")["rating"].mean()
                nbr["centered"] = nbr["rating"] - nbr["user_id"].map(neighbor_means)
                nbr["w_centered"] = nbr["centered"] * nbr["sim"]
                agg = nbr.groupby("anime_id").agg(
                    num_neighbors=("user_id", "size"),
                    weighted_sum=("w_centered", "sum"),
                    sim_sum=("sim", "sum"),
                )
                agg = agg[agg["num_neighbors"] >= cfg.min_support]
                cf_scores = (agg["weighted_sum"] / agg["sim_sum"]).rename("cf_score")

        c_scores_raw, c_idxs = cb.index.search(user_vec, cfg.content_pool)
        c_scores_raw, c_idxs = c_scores_raw[0], c_idxs[0]
        rows = []
        for s, ai in zip(c_scores_raw, c_idxs):
            if ai == -1:
                continue
            aid = cb.idx_to_anime_id[int(ai)]
            if aid in seen:
                continue
            rows.append((aid, float(s)))
        content_scores = pd.Series(dict(rows), name="content_score", dtype="float32")

        candidates = pd.concat([cf_scores, content_scores], axis=1)
        if cfg.min_num_ratings > 0 and "num_ratings" in self.ratings_df.columns:
            reliable = set(
                self.ratings_df.loc[self.ratings_df["num_ratings"] >= cfg.min_num_ratings, "anime_id"]
            )
            candidates = candidates[candidates.index.isin(reliable)]
        if len(candidates) == 0:
            return []

        def minmax(s):
            s = s.astype("float32")
            lo, hi = s.min(skipna=True), s.max(skipna=True)
            if pd.isna(lo) or hi == lo:
                return s.fillna(0.0) * 0.0
            return ((s - lo) / (hi - lo)).fillna(0.0)

        candidates["cf_n"] = minmax(candidates["cf_score"])
        candidates["content_n"] = minmax(candidates["content_score"])
        candidates["final_score"] = (
            cfg.alpha * candidates["cf_n"] + (1 - cfg.alpha) * candidates["content_n"]
        )
        return candidates["final_score"].sort_values(ascending=False).head(k).index.tolist()

    @staticmethod
    def _metrics(recs: list[int], relevant: set, k: int) -> dict:
        top = recs[:k]
        hits = [1 if aid in relevant else 0 for aid in top]
        precision = sum(hits) / k if k > 0 else 0.0
        recall = sum(hits) / len(relevant) if relevant else 0.0
        dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return {"precision@k": precision, "recall@k": recall, "ndcg@k": ndcg}

    def _evaluate_user(self, user_id: int, rng: np.random.Generator) -> dict | None:
        cfg = self.config
        uar = self.user_anime_ratings_df
        user_rats = uar[uar["user_id"] == user_id]
        if len(user_rats) < cfg.min_user_ratings:
            return None

        n_hold = max(1, int(round(len(user_rats) * cfg.holdout_ratio)))
        hold_pos = rng.choice(len(user_rats), size=n_hold, replace=False)
        mask = np.zeros(len(user_rats), dtype=bool)
        mask[hold_pos] = True
        holdout = user_rats.iloc[mask]
        train = user_rats.iloc[~mask]

        relevant = set(holdout.loc[holdout["rating"] >= cfg.relevant_threshold, "anime_id"].tolist())
        if not relevant:
            return None

        user_vec = self._build_user_vec(train)
        if user_vec is None or not np.isfinite(user_vec).all() or np.linalg.norm(user_vec) == 0:
            return None

        seen = set(train["anime_id"].tolist())
        cf_recs = self._cf_top_k(user_id, user_vec, seen, cfg.top_k)
        hybrid_recs = self._hybrid_top_k(user_id, user_vec, seen, cfg.top_k)

        return {
            "user_id": int(user_id),
            "n_relevant": len(relevant),
            "cf": self._metrics(cf_recs, relevant, cfg.top_k),
            "hybrid": self._metrics(hybrid_recs, relevant, cfg.top_k),
        }

    def _aggregate(self, per_user: list[dict], key: str) -> dict:
        if not per_user:
            return {"users_evaluated": 0, "precision@k": 0.0, "recall@k": 0.0, "ndcg@k": 0.0}
        df = pd.DataFrame([r[key] for r in per_user])
        return {
            "users_evaluated": len(per_user),
            "precision@k": float(df["precision@k"].mean()),
            "recall@k": float(df["recall@k"].mean()),
            "ndcg@k": float(df["ndcg@k"].mean()),
        }

    def run(self) -> EvaluationArtifact:
        cfg = self.config
        rng = np.random.default_rng(cfg.random_state)

        uar = self.user_anime_ratings_df
        counts = uar.groupby("user_id").size()
        eligible = counts[counts >= cfg.min_user_ratings].index.to_numpy()
        if len(eligible) == 0:
            raise ValueError("no users meet min_user_ratings threshold")

        n_sample = min(cfg.n_users_sample, len(eligible))
        sampled = rng.choice(eligible, size=n_sample, replace=False)

        per_user: list[dict] = []
        for i, uid in enumerate(sampled, 1):
            result = self._evaluate_user(int(uid), rng)
            if result is not None:
                per_user.append(result)
            if i % 50 == 0:
                print(f"  evaluated {i}/{n_sample} (kept {len(per_user)})")

        cf_metrics = self._aggregate(per_user, "cf")
        hybrid_metrics = self._aggregate(per_user, "hybrid")
        summary = {
            "config": {
                "top_k": cfg.top_k,
                "holdout_ratio": cfg.holdout_ratio,
                "relevant_threshold": cfg.relevant_threshold,
                "n_users_sample": cfg.n_users_sample,
                "min_user_ratings": cfg.min_user_ratings,
                "random_state": cfg.random_state,
            },
            "cf": cf_metrics,
            "hybrid": hybrid_metrics,
        }

        out_dir = cfg.evaluation_dir
        cf_path = os.path.join(out_dir, EVAL_CF_METRICS_FILE)
        hybrid_path = os.path.join(out_dir, EVAL_HYBRID_METRICS_FILE)
        summary_path = os.path.join(out_dir, EVAL_SUMMARY_FILE)

        with open(cf_path, "w", encoding="utf-8") as f:
            json.dump(cf_metrics, f, indent=2)
        with open(hybrid_path, "w", encoding="utf-8") as f:
            json.dump(hybrid_metrics, f, indent=2)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\nCF      : {cf_metrics}")
        print(f"Hybrid  : {hybrid_metrics}")

        return EvaluationArtifact(
            evaluation_dir=out_dir,
            cf_metrics_path=cf_path,
            hybrid_metrics_path=hybrid_path,
            summary_path=summary_path,
        )
