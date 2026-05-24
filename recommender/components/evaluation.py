import ast
import json
import os
import mlflow
import numpy as np
import pandas as pd
from recommender.constants import *
from recommender.entity.config_entity import EvaluationConfig, ContentBasedRecommenderConfig
from recommender.entity.artifact_entity import EvaluationArtifact
from recommender.components.content_based import ContentBasedRecommender
from recommender.utils import read_yaml

_params = read_yaml(PARAMS_PATH)
_ev = _params["evaluation"]
_hy = _params["hybrid"]

TOP_K: int                   = _ev["top_k"]
RANDOM_STATE: int            = _ev["random_state"]
CONTENT_SAMPLE_ANIME: int    = _ev["content"]["sample_anime"]
HYBRID_N_USERS_SAMPLE: int   = _ev["hybrid"]["n_users_sample"]
HYBRID_MIN_USER_RATINGS: int = _ev["hybrid"]["min_user_ratings"]
HYBRID_LIKED_THRESHOLD: float = _ev["hybrid"]["liked_threshold"]
HY_MIN_NUM_RATINGS: int      = _hy["min_num_ratings"]


def _parse_genres(val) -> set:
    if isinstance(val, list):
        return set(val)
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            return set(parsed) if isinstance(parsed, list) else set()
        except (ValueError, SyntaxError):
            return set()
    return set()


class Evaluation:
    def __init__(self, config: EvaluationConfig | None = None):
        self.config = config or EvaluationConfig()
        os.makedirs(self.config.evaluation_dir, exist_ok=True)
        os.makedirs(self.config.content_eval_dir, exist_ok=True)
        os.makedirs(self.config.hybrid_eval_dir, exist_ok=True)
        self._preprocessed_dir = os.path.join(ARTIFACT_DIR, DATA_PREPROCESSING_DIR_NAME)
        self.content: ContentBasedRecommender | None = None
        self.user_anime_ratings_df: pd.DataFrame | None = None
        self.ratings_df: pd.DataFrame | None = None

    def _load_artifacts(self) -> None:
        if self.content is not None:
            return
        print("[Evaluation] Loading ContentBasedRecommender artifacts …")
        cb = ContentBasedRecommender(config=ContentBasedRecommenderConfig())
        cb._load_if_needed()
        self.content = cb
        print(f"   Content index loaded ({cb.index.ntotal:,} anime vectors)")

        self.user_anime_ratings_df = pd.read_csv(os.path.join(self._preprocessed_dir, USER_ANIME_RATINGS_FILE_NAME))
        ratings_path = os.path.join(self._preprocessed_dir, RATINGS_FILE_NAME)
        self.ratings_df = (pd.read_csv(ratings_path) if os.path.exists(ratings_path) else pd.DataFrame())
        print(f"   user_anime_ratings: {len(self.user_anime_ratings_df):,} rows")

    def _genres_by_anime_id(self) -> dict[int, set]:
        return {
            int(aid): _parse_genres(g)
            for aid, g in zip(self.content.anime_df["anime_id"], self.content.anime_df["genres"])
        }

    # ------------------------------------------------------------------
    # Shared: genre_overlap@K
    #   Given a set of seed genres and a list of recommended anime_ids,
    #   return the fraction of recs that share at least one genre.
    # ------------------------------------------------------------------
    @staticmethod
    def _genre_overlap(seed_genres: set, rec_ids: list[int], genres_map: dict[int, set]) -> float:
        if not seed_genres or not rec_ids:
            return 0.0
        hits = sum(1 for aid in rec_ids if seed_genres & genres_map.get(aid, set()))
        return hits / len(rec_ids)

    # ------------------------------------------------------------------
    # 1. Content-based: genre_overlap@K
    #    Seed = one anime; recs should share genres with it.
    # ------------------------------------------------------------------
    def evaluate_content(self) -> dict:
        self._load_artifacts()
        cb = self.content
        genres_map = self._genres_by_anime_id()
        eligible = cb.anime_df[cb.anime_df["genres"].apply(lambda g: len(_parse_genres(g)) > 0)]
        sample = eligible.sample(n=min(CONTENT_SAMPLE_ANIME, len(eligible)), random_state=RANDOM_STATE)
        overlaps = []
        for aid in sample["anime_id"]:
            try:
                recs = cb.recommend(int(aid), k=TOP_K)
            except ValueError:
                continue
            seed_genres = genres_map.get(int(aid), set())
            overlap = self._genre_overlap(seed_genres, recs["anime_id"].tolist(), genres_map)
            if seed_genres:
                overlaps.append(overlap)
        return {
            "anime_evaluated": len(overlaps),
            "top_k": TOP_K,
            f"genre_overlap@{TOP_K}": float(np.mean(overlaps)) if overlaps else 0.0,
        }

    # ------------------------------------------------------------------
    # 2. Hybrid: genre_overlap@K
    #    Seed = user's liked genres (rating >= liked_threshold);
    #    recs should share genres with those.
    # ------------------------------------------------------------------
    def evaluate_hybrid(self) -> dict:
        self._load_artifacts()
        cb = self.content
        uar = self.user_anime_ratings_df
        genres_map = self._genres_by_anime_id()
        # Reliability filter (optional)
        reliable: set | None = None
        if HY_MIN_NUM_RATINGS > 0 and not self.ratings_df.empty and "num_ratings" in self.ratings_df.columns:
            reliable = set(self.ratings_df.loc[self.ratings_df["num_ratings"] >= HY_MIN_NUM_RATINGS, "anime_id"])
        # Sample users with enough ratings
        counts = uar.groupby("user_id").size()
        eligible = counts[counts >= HYBRID_MIN_USER_RATINGS].index.to_numpy()
        if len(eligible) == 0:
            return {"users_evaluated": 0, "top_k": TOP_K, f"genre_overlap@{TOP_K}": 0.0}
        rng = np.random.default_rng(RANDOM_STATE)
        sampled = rng.choice(eligible, size=min(HYBRID_N_USERS_SAMPLE, len(eligible)), replace=False)
        overlaps = []
        for uid in sampled:
            user_rats = uar[uar["user_id"] == uid]
            # Liked genres = union of genres from highly-rated anime
            liked_ids = user_rats.loc[user_rats["rating"] >= HYBRID_LIKED_THRESHOLD, "anime_id"]
            seed_genres = set().union(*[genres_map.get(int(a), set()) for a in liked_ids])
            if not seed_genres:
                continue
            # Build user taste vector: sum of (centered_rating × anime_vec)
            in_cb = user_rats[user_rats["anime_id"].isin(cb.anime_id_to_idx)]
            if in_cb.empty:
                continue
            ratings_arr = in_cb["rating"].astype(np.float32).values
            centered = ratings_arr - ratings_arr.mean()
            idxs = np.array([cb.anime_id_to_idx[int(a)] for a in in_cb["anime_id"]], dtype=np.int64)
            user_vec = (centered[:, None] * cb.vectors[idxs]).sum(axis=0).astype(np.float32)
            norm = np.linalg.norm(user_vec)
            if norm < 1e-12:
                continue
            user_vec = (user_vec / norm).reshape(1, -1)
            # Search CB FAISS, filter seen + unreliable
            seen = set(user_rats["anime_id"].tolist())
            k_fetch = min(TOP_K + len(seen) + 50, cb.index.ntotal)
            _, faiss_idxs = cb.index.search(user_vec, k_fetch)
            recs: list[int] = []
            for ai in faiss_idxs[0]:
                if ai == -1:
                    continue
                aid = cb.idx_to_anime_id[int(ai)]
                if aid in seen:
                    continue
                if reliable is not None and aid not in reliable:
                    continue
                recs.append(aid)
                if len(recs) == TOP_K:
                    break
            if recs:
                overlaps.append(self._genre_overlap(seed_genres, recs, genres_map))
        return {
            "users_evaluated": len(overlaps),
            "top_k": TOP_K,
            f"genre_overlap@{TOP_K}": float(np.mean(overlaps)) if overlaps else 0.0,
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

    def _log_mlflow(self, content_m: dict, hybrid_m: dict) -> None:
        flat_params = self._flatten_params(read_yaml(PARAMS_PATH) or {})
        mlflow.set_experiment(EXPT_NAME)
        with mlflow.start_run(run_name="evaluation"):
            for key, value in flat_params.items():
                mlflow.log_param(key, value)
            for prefix, metrics in [("content", content_m), ("hybrid", hybrid_m)]:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"{prefix}_{key.replace('@', '_at_')}", value)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(self) -> EvaluationArtifact:
        cfg = self.config

        print("\n[Evaluation] ── Content-based ──────────────────────────────")
        content_m = self.evaluate_content()
        print(f"  {content_m}")

        print("\n[Evaluation] ── Hybrid ─────────────────────────────────────")
        hybrid_m = self.evaluate_hybrid()
        print(f"  {hybrid_m}")

        content_path = os.path.join(cfg.content_eval_dir, EVAL_METRICS_FILE)
        hybrid_path  = os.path.join(cfg.hybrid_eval_dir,  EVAL_METRICS_FILE)
        summary_path = os.path.join(cfg.evaluation_dir,   EVAL_SUMMARY_FILE)

        for path, data in [(content_path, content_m), (hybrid_path, hybrid_m)]:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"content": content_m, "hybrid": hybrid_m}, f, indent=2)

        print("\n[Evaluation] Logging to MLflow …")
        self._log_mlflow(content_m, hybrid_m)
        print("[Evaluation]  Done.\n")

        return EvaluationArtifact(
            evaluation_dir=cfg.evaluation_dir,
            content_metrics_path=content_path,
            hybrid_metrics_path=hybrid_path,
            summary_path=summary_path,
        )
