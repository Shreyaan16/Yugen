import os
import numpy as np
import pandas as pd
from recommender.constants import *
from recommender.entity.config_entity import (HybridRecommenderConfig, ContentBasedRecommenderConfig, UserBasedRecommenderConfig,)
from recommender.entity.artifact_entity import HybridRecommenderArtifact
from recommender.components.content_based import ContentBasedRecommender
from recommender.components.user_based import UserBasedRecommender
from recommender.utils import read_yaml

_params = read_yaml(PARAMS_PATH)
_hy = _params["hybrid"]


class HybridRecommender:
    def __init__(self, config: HybridRecommenderConfig | None = None):
        self.config = config or HybridRecommenderConfig()
        self.anime_df: pd.DataFrame = pd.read_csv(os.path.join(self.config.preprocessed_dir, ANIME_FILE_NAME))
        self.user_anime_ratings_df: pd.DataFrame = pd.read_csv(os.path.join(self.config.preprocessed_dir, USER_ANIME_RATINGS_FILE_NAME))
        ratings_path = os.path.join(self.config.preprocessed_dir, RATINGS_FILE_NAME)
        self.ratings_df: pd.DataFrame = (pd.read_csv(ratings_path) if os.path.exists(ratings_path) else pd.DataFrame())
        self.k_neighbors: int = _hy["k_neighbors"]
        self.top_n: int = _hy["top_n"]
        self.neighbor_rating_threshold: float = _hy["neighbor_rating_threshold"]
        self.min_num_ratings: int = _hy["min_num_ratings"]
        self._content = None
        self._user_based = None

    def _load_if_needed(self) -> None:
        if self._content is not None and self._user_based is not None:
            return
        content_based_dir = os.path.dirname(self.config.anime_id_to_idx_path)
        user_based_dir = os.path.dirname(self.config.user_id_to_idx_path)

        cb_cfg = ContentBasedRecommenderConfig(preprocessed_dir=self.config.preprocessed_dir, content_based_dir=content_based_dir)
        cb = ContentBasedRecommender(config=cb_cfg)
        cb._load_if_needed()
        self._content = cb

        ub_cfg = UserBasedRecommenderConfig(
            preprocessed_dir=self.config.preprocessed_dir,
            user_based_dir=user_based_dir,
            anime_id_to_idx_path=self.config.anime_id_to_idx_path,
            tfidf_vectorizer_path=self.config.tfidf_vectorizer_path,
            tfidf_matrix_path=os.path.join(content_based_dir, TFIDF_MATRIX_FILE),
        )
        ub = UserBasedRecommender(config=ub_cfg)
        ub._load_if_needed()
        self._user_based = ub

    def save(self) -> HybridRecommenderArtifact:
        return HybridRecommenderArtifact(status="success")

    def run(self) -> HybridRecommenderArtifact:
        return self.save()

    def _reliable_anime_ids(self) -> set | None:
        if self.min_num_ratings > 0 and "num_ratings" in self.ratings_df.columns:
            return set(
                self.ratings_df.loc[
                    self.ratings_df["num_ratings"] >= self.min_num_ratings, "anime_id"
                ]
            )
        return None

    def _score_pool(
        self,
        user_vec: np.ndarray,
        self_idx: int | None,
        seen: set,
        reliable: set | None,
        is_cold: bool,
    ) -> tuple[list[int], np.ndarray]:
        ub = self._user_based
        cb = self._content

        if is_cold:
            k_fetch = self.top_n + len(seen) + 50
            scores_raw, idxs = cb.index.search(user_vec, k_fetch)
            pool_aids, pool_scores = [], []
            for s, ai in zip(scores_raw[0], idxs[0]):
                if ai == -1:
                    continue
                aid = cb.idx_to_anime_id[int(ai)]
                if aid in seen:
                    continue
                if reliable is not None and aid not in reliable:
                    continue
                pool_aids.append(aid)
                pool_scores.append(float(s))
            return pool_aids, np.array(pool_scores, dtype=np.float32)

        sims, nbr_idxs = ub.index.search(user_vec, self.k_neighbors + 5)
        sims, nbr_idxs = sims[0], nbr_idxs[0]
        neighbor_ids: list[int] = []
        for s, ni in zip(sims, nbr_idxs):
            if ni == -1 or ni == self_idx or ni == ub.avg_idx:
                continue
            nid = ub.idx_to_user_id.get(int(ni))
            if nid is None:
                continue
            neighbor_ids.append(nid)
            if len(neighbor_ids) == self.k_neighbors:
                break
        if not neighbor_ids:
            return [], np.array([], dtype=np.float32)

        uar = self.user_anime_ratings_df
        pool_rows = uar[
            uar["user_id"].isin(neighbor_ids)
            & (uar["rating"] >= self.neighbor_rating_threshold)
            & ~uar["anime_id"].isin(seen)
        ]
        pool = pool_rows["anime_id"].unique().tolist()
        if reliable is not None:
            pool = [aid for aid in pool if aid in reliable]
        pool = [aid for aid in pool if aid in cb.anime_id_to_idx]
        if not pool:
            return [], np.array([], dtype=np.float32)

        pool_idxs = np.array([cb.anime_id_to_idx[aid] for aid in pool], dtype=np.int64)
        pool_vecs = (
            cb.vectors[pool_idxs]
            if cb.vectors is not None
            else np.vstack([cb.index.reconstruct(int(i)) for i in pool_idxs])
        )
        scores = pool_vecs @ user_vec.ravel().astype(np.float32)
        return pool, scores.astype(np.float32)

    def recommend(self, user_id: int) -> pd.DataFrame:
        self._load_if_needed()
        ub = self._user_based
        cb = self._content

        u_idx = ub.user_id_to_idx.get(int(user_id))
        if u_idx is None:
            raise ValueError(f"user_id {user_id} not found")

        is_cold = u_idx == ub.avg_idx
        user_vec = ub.get_user_vec(ub.avg_idx if is_cold else u_idx)

        uar = self.user_anime_ratings_df
        seen = set(uar.loc[uar["user_id"] == user_id, "anime_id"].tolist())
        reliable = self._reliable_anime_ids()

        pool_aids, scores = self._score_pool(
            user_vec=user_vec,
            self_idx=u_idx,
            seen=seen,
            reliable=reliable,
            is_cold=is_cold,
        )

        if not pool_aids:
            return pd.DataFrame(columns=["anime_id", "title", "genres", "era", "score"])

        order = np.argsort(-scores)[: self.top_n]
        recs = pd.DataFrame({
            "anime_id": [pool_aids[i] for i in order],
            "score": [float(scores[i]) for i in order],
        })
        return recs.merge(
            self.anime_df[["anime_id", "title", "genres", "era"]],
            on="anime_id",
            how="left",
        )[["anime_id", "title", "genres", "era", "score"]]

