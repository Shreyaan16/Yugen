import json
import os
import mlflow
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from src.constants import (
    EVAL_METRICS_FILE,
    EVAL_SUMMARY_FILE,
    PARAMS_PATH,
    EXPT_NAME,
)
from src.entity.config_entity import EvaluationConfig, HybridRecommenderConfig
from src.entity.artifact_entity import EvaluationArtifact
from src.components.recommender import (
    ContentBasedRecommender,
    UserBasedRecommender,
    HybridRecommender,
)
from src.utils import read_yaml


class Evaluation:
    def __init__(self, config: EvaluationConfig | None = None):
        self.config = config or EvaluationConfig()
        os.makedirs(self.config.evaluation_dir, exist_ok=True)
        os.makedirs(self.config.content_eval_dir, exist_ok=True)
        os.makedirs(self.config.user_eval_dir, exist_ok=True)
        os.makedirs(self.config.hybrid_eval_dir, exist_ok=True)

        self.content: ContentBasedRecommender | None = None
        self.user_based: UserBasedRecommender | None = None
        self.hybrid: HybridRecommender | None = None
        self.anime_df: pd.DataFrame | None = None
        self.user_anime_ratings_df: pd.DataFrame | None = None
        self.ratings_df: pd.DataFrame | None = None

        self._hy_cfg = HybridRecommenderConfig()

    def fit(self, content: ContentBasedRecommender, user_based: UserBasedRecommender,
            hybrid: HybridRecommender, anime_df: pd.DataFrame,
            user_anime_ratings_df: pd.DataFrame, ratings_df: pd.DataFrame) -> "Evaluation":
        self.content = content
        self.user_based = user_based
        self.hybrid = hybrid
        self.anime_df = anime_df
        self.user_anime_ratings_df = user_anime_ratings_df
        self.ratings_df = ratings_df
        return self

    # ------------------------------------------------------------------
    # Content-based: genre overlap @ K
    # For each sampled anime, recommend top-K and check the fraction of recs
    # sharing >=1 genre with the seed. Sanity check that content sim is sane.
    # ------------------------------------------------------------------
    def evaluate_content(self) -> dict:
        cfg = self.config
        cb = self.content
        anime_df = self.anime_df

        rng = np.random.default_rng(cfg.random_state)
        eligible = anime_df[anime_df["genres"].apply(lambda g: isinstance(g, list) and len(g) > 0)]
        n = min(cfg.content_sample_anime, len(eligible))
        sample = eligible.sample(n=n, random_state=cfg.random_state)

        genres_by_aid = dict(zip(anime_df["anime_id"], anime_df["genres"]))

        per_anime_overlap = []
        for aid in sample["anime_id"]:
            try:
                recs = cb.recommend(int(aid), k=cfg.top_k)
            except ValueError:
                continue
            seed_genres = set(genres_by_aid.get(int(aid), []) or [])
            if not seed_genres or len(recs) == 0:
                continue
            hits = 0
            for rec_aid in recs["anime_id"]:
                rec_genres = set(genres_by_aid.get(int(rec_aid), []) or [])
                if seed_genres & rec_genres:
                    hits += 1
            per_anime_overlap.append(hits / len(recs))

        return {
            "anime_evaluated": len(per_anime_overlap),
            "top_k": cfg.top_k,
            f"genre_overlap@{cfg.top_k}": float(np.mean(per_anime_overlap)) if per_anime_overlap else 0.0,
        }

    # ------------------------------------------------------------------
    # UserBased: Jaccard overlap of watched anime
    # For each sampled user, look at top-K neighbors. Jaccard(target_anime,
    # neighbor_anime) measures whether neighbors actually share taste.
    # ------------------------------------------------------------------
    def evaluate_user_based(self) -> dict:
        cfg = self.config
        ub = self.user_based
        uar = self.user_anime_ratings_df

        anime_by_user = uar.groupby("user_id")["anime_id"].agg(set).to_dict()
        warm_users = [uid for uid, idx in ub.user_id_to_idx.items() if idx != ub.avg_idx]
        if not warm_users:
            return {"users_evaluated": 0, "k_neighbors": cfg.user_k_neighbors, "mean_jaccard": 0.0}

        rng = np.random.default_rng(cfg.random_state)
        n = min(cfg.user_sample_users, len(warm_users))
        sampled = rng.choice(warm_users, size=n, replace=False)

        per_user_jaccard = []
        for uid in sampled:
            target_set = anime_by_user.get(int(uid), set())
            if not target_set:
                continue
            try:
                nbrs = ub.similar_users(int(uid), k=cfg.user_k_neighbors)
            except ValueError:
                continue
            if len(nbrs) == 0:
                continue
            neighbor_jaccards = []
            for nid in nbrs["user_id"]:
                nbr_set = anime_by_user.get(int(nid), set())
                if not nbr_set:
                    continue
                union = target_set | nbr_set
                if not union:
                    continue
                neighbor_jaccards.append(len(target_set & nbr_set) / len(union))
            if neighbor_jaccards:
                per_user_jaccard.append(float(np.mean(neighbor_jaccards)))

        return {
            "users_evaluated": len(per_user_jaccard),
            "k_neighbors": cfg.user_k_neighbors,
            "mean_jaccard": float(np.mean(per_user_jaccard)) if per_user_jaccard else 0.0,
        }

    # ------------------------------------------------------------------
    # Hybrid: Hit@K via per-user random holdout
    # Hide 20% of each user's ratings; check whether any high-rated holdout
    # item appears in the recommender's top-K. Hit@K = fraction of users
    # with at least one such hit.
    # ------------------------------------------------------------------
    def _build_user_vec(self, train_ratings: pd.DataFrame) -> np.ndarray | None:
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

    def _hybrid_top_k(self, user_id: int, user_vec: np.ndarray, seen: set, k: int) -> list[int]:
        cb = self.content
        ub = self.user_based
        cfg = self._hy_cfg
        self_idx = ub.user_id_to_idx.get(int(user_id))

        sims, nbr_idxs = ub.index.search(user_vec, cfg.k_neighbors + 5)
        sims, nbr_idxs = sims[0], nbr_idxs[0]
        neighbor_ids: list[int] = []
        for s, ni in zip(sims, nbr_idxs):
            if ni == -1 or ni == ub.avg_idx or ni == self_idx:
                continue
            nid = ub.idx_to_user_id.get(int(ni))
            if nid is None:
                continue
            neighbor_ids.append(nid)
            if len(neighbor_ids) == cfg.k_neighbors:
                break
        if not neighbor_ids:
            return []

        uar = self.user_anime_ratings_df
        pool_rows = uar[
            uar["user_id"].isin(neighbor_ids)
            & (uar["rating"] >= cfg.neighbor_rating_threshold)
            & ~uar["anime_id"].isin(seen)
        ]
        pool = pool_rows["anime_id"].unique().tolist()

        if cfg.min_num_ratings > 0 and "num_ratings" in self.ratings_df.columns:
            reliable = set(
                self.ratings_df.loc[self.ratings_df["num_ratings"] >= cfg.min_num_ratings, "anime_id"]
            )
            pool = [aid for aid in pool if aid in reliable]
        pool = [aid for aid in pool if aid in cb.anime_id_to_idx]
        if not pool:
            return []

        pool_idxs = np.array([cb.anime_id_to_idx[aid] for aid in pool], dtype=np.int64)
        pool_vecs = cb.vectors[pool_idxs]
        scores = pool_vecs @ user_vec.ravel().astype(np.float32)
        order = np.argsort(-scores)[:k]
        return [pool[i] for i in order]

    def _hit_for_user(self, user_id: int, rng: np.random.Generator) -> int | None:
        cfg = self.config
        uar = self.user_anime_ratings_df
        user_rats = uar[uar["user_id"] == user_id]
        if len(user_rats) < cfg.hybrid_min_user_ratings:
            return None

        n_hold = max(1, int(round(len(user_rats) * cfg.hybrid_holdout_ratio)))
        hold_pos = rng.choice(len(user_rats), size=n_hold, replace=False)
        mask = np.zeros(len(user_rats), dtype=bool)
        mask[hold_pos] = True
        holdout = user_rats.iloc[mask]
        train = user_rats.iloc[~mask]

        relevant = set(holdout.loc[holdout["rating"] >= cfg.hybrid_relevant_threshold, "anime_id"])
        if not relevant:
            return None

        user_vec = self._build_user_vec(train)
        if user_vec is None or not np.isfinite(user_vec).all() or np.linalg.norm(user_vec) == 0:
            return None

        seen = set(train["anime_id"].tolist())
        recs = self._hybrid_top_k(int(user_id), user_vec, seen, cfg.top_k)
        return int(any(aid in relevant for aid in recs))

    def evaluate_hybrid(self) -> dict:
        cfg = self.config
        uar = self.user_anime_ratings_df
        rng = np.random.default_rng(cfg.random_state)

        counts = uar.groupby("user_id").size()
        eligible = counts[counts >= cfg.hybrid_min_user_ratings].index.to_numpy()
        if len(eligible) == 0:
            return {"users_evaluated": 0, "top_k": cfg.top_k, f"hit@{cfg.top_k}": 0.0}

        n = min(cfg.hybrid_n_users_sample, len(eligible))
        sampled = rng.choice(eligible, size=n, replace=False)

        hits, total = 0, 0
        for i, uid in enumerate(sampled, 1):
            res = self._hit_for_user(int(uid), rng)
            if res is None:
                continue
            hits += res
            total += 1
            if i % 50 == 0:
                print(f"  hybrid eval {i}/{n} (kept {total}, hits {hits})")

        return {
            "users_evaluated": total,
            "top_k": cfg.top_k,
            f"hit@{cfg.top_k}": (hits / total) if total else 0.0,
        }

    # ------------------------------------------------------------------
    # MLflow logging
    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_params(params: dict, prefix: str = "") -> dict:
        flat = {}
        for key, value in params.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                flat.update(Evaluation._flatten_params(value, full_key))
            elif isinstance(value, (list, tuple)):
                flat[full_key] = ",".join(str(v) for v in value)
            else:
                flat[full_key] = value
        return flat

    def _log_mlflow(self, params: dict, content_m: dict, user_m: dict, hybrid_m: dict) -> None:
        mlflow.set_experiment(EXPT_NAME)
        flat_params = self._flatten_params(params)
        with mlflow.start_run(run_name="evaluation"):
            for key, value in flat_params.items():
                mlflow.log_param(key, value)
            for prefix, metrics in [("content", content_m), ("user", user_m), ("hybrid", hybrid_m)]:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"{prefix}_{key.replace('@', '_at_')}", value)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(self) -> EvaluationArtifact:
        cfg = self.config

        print("Evaluating content-based ...")
        content_m = self.evaluate_content()
        print(f"  {content_m}")

        print("Evaluating user-based ...")
        user_m = self.evaluate_user_based()
        print(f"  {user_m}")

        print("Evaluating hybrid ...")
        hybrid_m = self.evaluate_hybrid()
        print(f"  {hybrid_m}")

        content_path = os.path.join(cfg.content_eval_dir, EVAL_METRICS_FILE)
        user_path = os.path.join(cfg.user_eval_dir, EVAL_METRICS_FILE)
        hybrid_path = os.path.join(cfg.hybrid_eval_dir, EVAL_METRICS_FILE)
        summary_path = os.path.join(cfg.evaluation_dir, EVAL_SUMMARY_FILE)

        with open(content_path, "w", encoding="utf-8") as f:
            json.dump(content_m, f, indent=2)
        with open(user_path, "w", encoding="utf-8") as f:
            json.dump(user_m, f, indent=2)
        with open(hybrid_path, "w", encoding="utf-8") as f:
            json.dump(hybrid_m, f, indent=2)

        summary = {
            "config": {
                "top_k": cfg.top_k,
                "random_state": cfg.random_state,
                "content_sample_anime": cfg.content_sample_anime,
                "user_sample_users": cfg.user_sample_users,
                "user_k_neighbors": cfg.user_k_neighbors,
                "hybrid_holdout_ratio": cfg.hybrid_holdout_ratio,
                "hybrid_relevant_threshold": cfg.hybrid_relevant_threshold,
                "hybrid_n_users_sample": cfg.hybrid_n_users_sample,
                "hybrid_min_user_ratings": cfg.hybrid_min_user_ratings,
            },
            "content": content_m,
            "user_based": user_m,
            "hybrid": hybrid_m,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        params = read_yaml(PARAMS_PATH) or {}
        self._log_mlflow(params, content_m, user_m, hybrid_m)

        return EvaluationArtifact(
            evaluation_dir=cfg.evaluation_dir,
            content_metrics_path=content_path,
            user_metrics_path=user_path,
            hybrid_metrics_path=hybrid_path,
            summary_path=summary_path,
        )
