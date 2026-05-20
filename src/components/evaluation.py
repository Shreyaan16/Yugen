import json
import os
import mlflow
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from src.constants import (
    EVAL_HYBRID_METRICS_FILE,
    EVAL_SUMMARY_FILE,
    PARAMS_PATH,
    EXPT_NAME,
)
from src.entity.config_entity import (EvaluationConfig, HybridRecommenderConfig,)
from src.entity.artifact_entity import EvaluationArtifact
from src.components.recommender import (ContentBasedRecommender, UserBasedRecommender, HybridRecommender)
from src.utils import read_yaml


class Evaluation:
    def __init__(self, config: EvaluationConfig | None = None):
        self.config = config or EvaluationConfig()
        os.makedirs(self.config.evaluation_dir, exist_ok=True)

        self.content: ContentBasedRecommender | None = None
        self.user_based: UserBasedRecommender | None = None
        self.hybrid: HybridRecommender | None = None
        self.anime_df: pd.DataFrame | None = None
        self.user_anime_ratings_df: pd.DataFrame | None = None
        self.ratings_df: pd.DataFrame | None = None

        self._hy_cfg = HybridRecommenderConfig()

    def fit(self, content: ContentBasedRecommender, user_based: UserBasedRecommender,
        hybrid: HybridRecommender, anime_df: pd.DataFrame, user_anime_ratings_df: pd.DataFrame,
        ratings_df: pd.DataFrame,) -> "Evaluation":
        self.content = content
        self.user_based = user_based
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

    def _hybrid_top_k(self, user_id: int, user_vec: np.ndarray, seen: set, k: int) -> list[int]:
        """New hybrid: CF neighbors define candidate pool, content sim ranks it."""
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
        hybrid_recs = self._hybrid_top_k(user_id, user_vec, seen, cfg.top_k)

        return {
            "user_id": int(user_id),
            "n_relevant": len(relevant),
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

    def _log_mlflow(self, params: dict, hybrid_metrics: dict) -> None:
        mlflow.set_experiment(EXPT_NAME)

        flat_params = self._flatten_params(params)
        with mlflow.start_run(run_name="evaluation"):
            for key, value in flat_params.items():
                mlflow.log_param(key, value)

            for key, value in hybrid_metrics.items():
                mlflow.log_metric(f"hybrid_{key.replace('@', '_at_')}", value)

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
            "hybrid": hybrid_metrics,
        }

        out_dir = cfg.evaluation_dir
        hybrid_path = os.path.join(out_dir, EVAL_HYBRID_METRICS_FILE)
        summary_path = os.path.join(out_dir, EVAL_SUMMARY_FILE)

        with open(hybrid_path, "w", encoding="utf-8") as f:
            json.dump(hybrid_metrics, f, indent=2)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\nHybrid  : {hybrid_metrics}")

        params = read_yaml(PARAMS_PATH) or {}
        self._log_mlflow(params, hybrid_metrics)

        return EvaluationArtifact(
            evaluation_dir=out_dir,
            hybrid_metrics_path=hybrid_path,
            summary_path=summary_path,
        )
