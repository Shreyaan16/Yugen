import ast
import json
import os
import mlflow
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from recommender.constants import *
from recommender.entity.config_entity import (EvaluationConfig, ContentBasedRecommenderConfig, UserBasedRecommenderConfig)
from recommender.entity.artifact_entity import EvaluationArtifact
from recommender.components.content_based import ContentBasedRecommender
from recommender.components.user_based import UserBasedRecommender
from recommender.utils import read_yaml

_params = read_yaml(PARAMS_PATH)
_ev = _params["evaluation"]
_hy = _params["hybrid"]

TOP_K: int = _ev["top_k"]
RANDOM_STATE: int = _ev["random_state"]
CONTENT_SAMPLE_ANIME: int = _ev["content"]["sample_anime"]
USER_SAMPLE_USERS: int = _ev["user_based"]["sample_users"]
USER_K_NEIGHBORS: int = _ev["user_based"]["k_neighbors"]
HYBRID_HOLDOUT_RATIO: float = _ev["hybrid"]["holdout_ratio"]
HYBRID_RELEVANT_THRESHOLD: float = _ev["hybrid"]["relevant_threshold"]
HYBRID_N_USERS_SAMPLE: int = _ev["hybrid"]["n_users_sample"]
HYBRID_MIN_USER_RATINGS: int = _ev["hybrid"]["min_user_ratings"]
HY_K_NEIGHBORS: int = _hy["k_neighbors"]
HY_NEIGHBOR_RATING_THRESHOLD: float = _hy["neighbor_rating_threshold"]
HY_MIN_NUM_RATINGS: int = _hy["min_num_ratings"]

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
        os.makedirs(self.config.user_eval_dir, exist_ok=True)
        os.makedirs(self.config.hybrid_eval_dir, exist_ok=True)
        self._preprocessed_dir = os.path.join(ARTIFACT_DIR, DATA_PREPROCESSING_DIR_NAME)
        self.content: ContentBasedRecommender | None = None
        self.user_based: UserBasedRecommender | None = None
        self.anime_df: pd.DataFrame | None = None
        self.user_anime_ratings_df: pd.DataFrame | None = None
        self.ratings_df: pd.DataFrame | None = None

    def _load_artifacts(self) -> None:
        if self.content is not None:
            return  # already loaded
        print("[Evaluation] Loading ContentBasedRecommender artifacts …")
        anime_df = pd.read_csv(os.path.join(self._preprocessed_dir, ANIME_FILE_NAME))
        cb = ContentBasedRecommender(config=ContentBasedRecommenderConfig())
        cb.load(anime_df)
        self.content = cb
        self.anime_df = anime_df
        print(f"   Content index loaded ({cb.index.ntotal:,} anime vectors)")

        print("[Evaluation] Loading UserBasedRecommender artifacts …")
        ub = UserBasedRecommender(config=UserBasedRecommenderConfig())
        ub._load_if_needed()
        self.user_based = ub
        print(f"   User index loaded   ({ub.index.ntotal:,} user vectors, avg_idx={ub.avg_idx})")

        print("[Evaluation] Loading preprocessed data …")
        self.user_anime_ratings_df = pd.read_csv(
            os.path.join(self._preprocessed_dir, USER_ANIME_RATINGS_FILE_NAME)
        )
        ratings_path = os.path.join(self._preprocessed_dir, RATINGS_FILE_NAME)
        self.ratings_df = (
            pd.read_csv(ratings_path) if os.path.exists(ratings_path) else pd.DataFrame()
        )
        print(f"   user_anime_ratings: {len(self.user_anime_ratings_df):,} rows")

    # ------------------------------------------------------------------
    # 1. Content-based: genre_overlap@K
    # ------------------------------------------------------------------
    def evaluate_content(self) -> dict:
        self._load_artifacts()
        cb = self.content
        anime_df = self.anime_df

        genres_by_aid = {
            int(aid): _parse_genres(g)
            for aid, g in zip(anime_df["anime_id"], anime_df["genres"])
        }

        eligible = anime_df[anime_df["genres"].apply(lambda g: len(_parse_genres(g)) > 0)]
        n = min(CONTENT_SAMPLE_ANIME, len(eligible))
        sample = eligible.sample(n=n, random_state=RANDOM_STATE)

        per_anime_overlap = []
        for aid in sample["anime_id"]:
            try:
                recs = cb.recommend(int(aid), k=TOP_K)
            except ValueError:
                continue
            seed_genres = genres_by_aid.get(int(aid), set())
            if not seed_genres or len(recs) == 0:
                continue
            hits = sum(
                1 for rec_aid in recs["anime_id"]
                if seed_genres & genres_by_aid.get(int(rec_aid), set())
            )
            per_anime_overlap.append(hits / len(recs))

        return {
            "anime_evaluated": len(per_anime_overlap),
            "top_k": TOP_K,
            f"genre_overlap@{TOP_K}": float(np.mean(per_anime_overlap)) if per_anime_overlap else 0.0,
        }

    # ------------------------------------------------------------------
    # 2. User-based: mean_jaccard
    # ------------------------------------------------------------------
    def evaluate_user_based(self) -> dict:
        self._load_artifacts()
        ub = self.user_based
        uar = self.user_anime_ratings_df

        anime_by_user = uar.groupby("user_id")["anime_id"].agg(set).to_dict()
        warm_users = [uid for uid, idx in ub.user_id_to_idx.items() if idx != ub.avg_idx]
        if not warm_users:
            return {"users_evaluated": 0, "k_neighbors": USER_K_NEIGHBORS, "mean_jaccard": 0.0}

        rng = np.random.default_rng(RANDOM_STATE)
        n = min(USER_SAMPLE_USERS, len(warm_users))
        sampled = rng.choice(warm_users, size=n, replace=False)

        per_user_jaccard = []
        for uid in sampled:
            target_set = anime_by_user.get(int(uid), set())
            if not target_set:
                continue
            try:
                nbrs = ub.similar_users(int(uid), k=USER_K_NEIGHBORS)
            except ValueError:
                continue
            if len(nbrs) == 0:
                continue
            neighbor_jaccards = []
            for nid in nbrs["user_id"]:
                nbr_set = anime_by_user.get(int(nid), set())
                union = target_set | nbr_set
                if not union:
                    continue
                neighbor_jaccards.append(len(target_set & nbr_set) / len(union))
            if neighbor_jaccards:
                per_user_jaccard.append(float(np.mean(neighbor_jaccards)))

        return {
            "users_evaluated": len(per_user_jaccard),
            "k_neighbors": USER_K_NEIGHBORS,
            "mean_jaccard": float(np.mean(per_user_jaccard)) if per_user_jaccard else 0.0,
        }

    # ------------------------------------------------------------------
    # 3. Hybrid: hit@K via per-user random holdout
    # ------------------------------------------------------------------
    def _build_user_vec(self, train_ratings: pd.DataFrame) -> np.ndarray | None:
        """Rebuild a user taste vector from held-in ratings only (no leakage)."""
        cb = self.content
        in_idx = train_ratings[train_ratings["anime_id"].isin(cb.anime_id_to_idx)]
        if len(in_idx) == 0:
            return None

        mean_r = in_idx["rating"].mean()
        centered = (in_idx["rating"] - mean_r).astype(np.float32).values
        anime_idxs = in_idx["anime_id"].map(cb.anime_id_to_idx).values

        n_anime = cb.tfidf_matrix.shape[0]
        row = csr_matrix(
            (centered, (np.zeros(len(centered), dtype=np.int64), anime_idxs)),
            shape=(1, n_anime),
        )
        user_vec_sparse = normalize(row @ cb.tfidf_matrix, norm="l2", axis=1)
        return user_vec_sparse.toarray().astype(np.float32)

    def _hybrid_top_k(self, user_id: int, user_vec: np.ndarray, seen: set, k: int) -> list[int]:
        """Mirrors HybridRecommender._score_pool for warm users."""
        cb = self.content
        ub = self.user_based
        self_idx = ub.user_id_to_idx.get(int(user_id))

        # --- CF: find neighbors ---
        sims, nbr_idxs = ub.index.search(user_vec, HY_K_NEIGHBORS + 5)
        neighbor_ids: list[int] = []
        for s, ni in zip(sims[0], nbr_idxs[0]):
            if ni == -1 or ni == ub.avg_idx or ni == self_idx:
                continue
            nid = ub.idx_to_user_id.get(int(ni))
            if nid is None:
                continue
            neighbor_ids.append(nid)
            if len(neighbor_ids) == HY_K_NEIGHBORS:
                break
        if not neighbor_ids:
            return []

        # --- Build candidate pool from neighbors ---
        uar = self.user_anime_ratings_df
        pool_rows = uar[
            uar["user_id"].isin(neighbor_ids)
            & (uar["rating"] >= HY_NEIGHBOR_RATING_THRESHOLD)
            & ~uar["anime_id"].isin(seen)
        ]
        pool = pool_rows["anime_id"].unique().tolist()

        # Reliability filter (if ratings.csv with num_ratings is available)
        if HY_MIN_NUM_RATINGS > 0 and "num_ratings" in self.ratings_df.columns:
            reliable = set(
                self.ratings_df.loc[
                    self.ratings_df["num_ratings"] >= HY_MIN_NUM_RATINGS, "anime_id"
                ]
            )
            pool = [aid for aid in pool if aid in reliable]

        pool = [aid for aid in pool if aid in cb.anime_id_to_idx]
        if not pool:
            return []

        # --- Content ranking ---
        pool_idxs = np.array([cb.anime_id_to_idx[aid] for aid in pool], dtype=np.int64)
        pool_vecs = (
            cb.vectors[pool_idxs]
            if cb.vectors is not None
            else np.vstack([cb.index.reconstruct(int(i)) for i in pool_idxs])
        )
        scores = pool_vecs @ user_vec.ravel().astype(np.float32)
        order = np.argsort(-scores)[:k]
        return [pool[i] for i in order]

    def _hit_for_user(self, user_id: int, rng: np.random.Generator) -> int | None:
        uar = self.user_anime_ratings_df
        user_rats = uar[uar["user_id"] == user_id]
        if len(user_rats) < HYBRID_MIN_USER_RATINGS:
            return None

        # Random holdout split
        n_hold = max(1, int(round(len(user_rats) * HYBRID_HOLDOUT_RATIO)))
        hold_pos = rng.choice(len(user_rats), size=n_hold, replace=False)
        mask = np.zeros(len(user_rats), dtype=bool)
        mask[hold_pos] = True
        holdout = user_rats.iloc[mask]
        train = user_rats.iloc[~mask]

        # Relevant = held-out items with rating >= threshold
        relevant = set(
            holdout.loc[holdout["rating"] >= HYBRID_RELEVANT_THRESHOLD, "anime_id"]
        )
        if not relevant:
            return None

        # Rebuild user vec from held-in only (no leakage)
        user_vec = self._build_user_vec(train)
        if user_vec is None or not np.isfinite(user_vec).all() or np.linalg.norm(user_vec) == 0:
            return None

        seen = set(train["anime_id"].tolist())
        recs = self._hybrid_top_k(int(user_id), user_vec, seen, TOP_K)
        return int(any(aid in relevant for aid in recs))

    def evaluate_hybrid(self) -> dict:
        self._load_artifacts()
        uar = self.user_anime_ratings_df
        rng = np.random.default_rng(RANDOM_STATE)

        counts = uar.groupby("user_id").size()
        eligible = counts[counts >= HYBRID_MIN_USER_RATINGS].index.to_numpy()
        if len(eligible) == 0:
            return {"users_evaluated": 0, "top_k": TOP_K, f"hit@{TOP_K}": 0.0}

        n = min(HYBRID_N_USERS_SAMPLE, len(eligible))
        sampled = rng.choice(eligible, size=n, replace=False)

        hits, total = 0, 0
        for i, uid in enumerate(sampled, 1):
            res = self._hit_for_user(int(uid), rng)
            if res is None:
                continue
            hits += res
            total += 1
            if i % 50 == 0:
                print(f"  hybrid eval {i}/{n}  (evaluated {total}, hits {hits})")

        return {
            "users_evaluated": total,
            "top_k": TOP_K,
            f"hit@{TOP_K}": (hits / total) if total else 0.0,
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

    def _log_mlflow(self, content_m: dict, user_m: dict, hybrid_m: dict) -> None:
        all_params = read_yaml(PARAMS_PATH) or {}
        flat_params = self._flatten_params(all_params)
        mlflow.set_experiment(EXPT_NAME)
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

        print("\n[Evaluation] ── Content-based ──────────────────────────────")
        content_m = self.evaluate_content()
        print(f"  {content_m}")

        print("\n[Evaluation] ── User-based ─────────────────────────────────")
        user_m = self.evaluate_user_based()
        print(f"  {user_m}")

        print("\n[Evaluation] ── Hybrid ─────────────────────────────────────")
        hybrid_m = self.evaluate_hybrid()
        print(f"  {hybrid_m}")

        # --- Persist metrics ---
        content_path = os.path.join(cfg.content_eval_dir, EVAL_METRICS_FILE)
        user_path = os.path.join(cfg.user_eval_dir, EVAL_METRICS_FILE)
        hybrid_path = os.path.join(cfg.hybrid_eval_dir, EVAL_METRICS_FILE)
        summary_path = os.path.join(cfg.evaluation_dir, EVAL_SUMMARY_FILE)

        for path, data in [
            (content_path, content_m),
            (user_path, user_m),
            (hybrid_path, hybrid_m),
        ]:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        summary = {
            "params": {
                "top_k": TOP_K,
                "random_state": RANDOM_STATE,
                "content": {"sample_anime": CONTENT_SAMPLE_ANIME},
                "user_based": {"sample_users": USER_SAMPLE_USERS, "k_neighbors": USER_K_NEIGHBORS},
                "hybrid": {
                    "holdout_ratio": HYBRID_HOLDOUT_RATIO,
                    "relevant_threshold": HYBRID_RELEVANT_THRESHOLD,
                    "n_users_sample": HYBRID_N_USERS_SAMPLE,
                    "min_user_ratings": HYBRID_MIN_USER_RATINGS,
                    "k_neighbors": HY_K_NEIGHBORS,
                    "neighbor_rating_threshold": HY_NEIGHBOR_RATING_THRESHOLD,
                    "min_num_ratings": HY_MIN_NUM_RATINGS,
                },
            },
            "content": content_m,
            "user_based": user_m,
            "hybrid": hybrid_m,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\n[Evaluation] Logging to MLflow …")
        self._log_mlflow(content_m, user_m, hybrid_m)

        print("[Evaluation]  Done.\n")
        return EvaluationArtifact(
            evaluation_dir=cfg.evaluation_dir,
            content_metrics_path=content_path,
            user_metrics_path=user_path,
            hybrid_metrics_path=hybrid_path,
            summary_path=summary_path,
        )
