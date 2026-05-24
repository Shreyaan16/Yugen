import os
from collections import defaultdict
import numpy as np
import pandas as pd
from recommender.constants import PARAMS_PATH
from recommender.entity.config_entity import (HybridRecommenderConfig, ContentBasedRecommenderConfig)
from recommender.entity.artifact_entity import HybridRecommenderArtifact
from recommender.components.content_based import ContentBasedRecommender
from recommender.utils import read_yaml

_params = read_yaml(PARAMS_PATH)
_hy     = _params["hybrid"]


class HybridRecommender:
    def __init__(self, config: HybridRecommenderConfig | None = None):
        self.config    = config or HybridRecommenderConfig()
        self.top_n: int            = _hy["top_n"]
        self.min_num_ratings: int  = _hy["min_num_ratings"]
        uar = pd.read_csv(os.path.join(self.config.preprocessed_dir, USER_ANIME_RATINGS_FILE_NAME))
        self.anime_df: pd.DataFrame = pd.read_csv(os.path.join(self.config.preprocessed_dir, ANIME_FILE_NAME))
        ratings_path = os.path.join(self.config.preprocessed_dir, RATINGS_FILE_NAME)
        self._ratings_df: pd.DataFrame = (pd.read_csv(ratings_path) if os.path.exists(ratings_path) else pd.DataFrame())
        _uid  = uar["user_id"].values
        _aid  = uar["anime_id"].values
        _rat  = uar["rating"].values.astype(np.float32)
        _map: dict[int, dict[int, float]] = defaultdict(dict)
        for uid, aid, r in zip(_uid, _aid, _rat):
            _map[int(uid)][int(aid)] = float(r)
        self._user_ratings: dict[int, dict[int, float]] = dict(_map)
        self._content: ContentBasedRecommender | None = None

    def _load_if_needed(self) -> None:
        """Load CB FAISS + mappings. User FAISS is never loaded for serving."""
        if self._content is not None:
            return
        content_based_dir = os.path.dirname(self.config.anime_id_to_idx_path)
        cb_cfg = ContentBasedRecommenderConfig(preprocessed_dir=self.config.preprocessed_dir,content_based_dir=content_based_dir)
        cb = ContentBasedRecommender(config=cb_cfg)
        cb._load_if_needed()
        self._content = cb

    def _build_user_vec(self, user_id: int) -> np.ndarray:
        """
        Compute a taste vector for the user from their rating history.
        Method:
          v = sum_i [ (rating_i - mean_rating) * anime_vec_i ]   (normalised)
        This lives in the same space as CB anime vectors, so we can search
        the CB FAISS index directly — no user FAISS needed.
        Cold users (no ratings) get the mean anime vector.
        """
        cb = self._content
        user_ratings = self._user_ratings.get(int(user_id), {})
        if not user_ratings:
            return self._cold_vec()
        anime_ids    = list(user_ratings.keys())
        ratings_arr  = np.array(list(user_ratings.values()), dtype=np.float32)
        centered     = ratings_arr - ratings_arr.mean()
        # Map to CB index positions, skip anime not in CB
        valid_cb_idxs: list[int]   = []
        valid_weights: list[float] = []
        for aid, w in zip(anime_ids, centered):
            ci = cb.anime_id_to_idx.get(int(aid))
            if ci is not None:
                valid_cb_idxs.append(ci)
                valid_weights.append(float(w))
        if not valid_cb_idxs:
            return self._cold_vec()
        idxs    = np.array(valid_cb_idxs, dtype=np.int64)
        weights = np.array(valid_weights,  dtype=np.float32)
        # Weighted sum of anime content vectors (cb.vectors is dense float32)
        user_vec = (weights[:, None] * cb.vectors[idxs]).sum(axis=0)

        norm = np.linalg.norm(user_vec)
        if norm > 1e-12:
            user_vec /= norm

        return user_vec.reshape(1, -1).astype(np.float32)

    def _cold_vec(self) -> np.ndarray:
        """Mean of all anime vectors — sensible starting point for new users."""
        v = self._content.vectors.mean(axis=0).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-12)
        return v.reshape(1, -1)

    # ── Reliability filter ────────────────────────────────────────────────────
    def _reliable_anime_ids(self) -> set | None:
        if self.min_num_ratings > 0 and "num_ratings" in self._ratings_df.columns:
            return set(
                self._ratings_df.loc[
                    self._ratings_df["num_ratings"] >= self.min_num_ratings, "anime_id"
                ]
            )
        return None

    # ── Main API ──────────────────────────────────────────────────────────────
    def recommend(self, user_id: int) -> pd.DataFrame:
        """
        Return top-N anime for user_id.
        """
        self._load_if_needed()
        cb = self._content

        seen     = set(self._user_ratings.get(int(user_id), {}).keys())
        user_vec = self._build_user_vec(user_id)
        reliable = self._reliable_anime_ids()

        # Fetch extra candidates to account for filtering
        k_fetch = self.top_n + len(seen) + 100
        k_fetch = min(k_fetch, cb.index.ntotal)

        scores_raw, faiss_idxs = cb.index.search(user_vec, k_fetch)

        results: list[tuple[int, float]] = []
        for score, ai in zip(scores_raw[0], faiss_idxs[0]):
            if ai == -1:
                continue
            aid = cb.idx_to_anime_id[int(ai)]
            if aid in seen:
                continue
            if reliable is not None and aid not in reliable:
                continue
            results.append((aid, float(score)))
            if len(results) == self.top_n:
                break

        if not results:
            return pd.DataFrame(columns=["anime_id", "title", "genres", "era", "score"])

        recs = pd.DataFrame(results, columns=["anime_id", "score"])
        return recs.merge(
            self.anime_df[["anime_id", "title", "genres", "era"]],
            on="anime_id",
            how="left",
        )[["anime_id", "title", "genres", "era", "score"]]

    def save(self) -> HybridRecommenderArtifact:
        return HybridRecommenderArtifact(status="success")

    def run(self) -> HybridRecommenderArtifact:
        return self.save()
