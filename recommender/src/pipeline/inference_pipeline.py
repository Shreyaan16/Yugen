from __future__ import annotations

import os
import pandas as pd

from recommender.src.components.recommender import (
    ContentBasedRecommender,
    UserBasedRecommender,
    HybridRecommender,
)
from recommender.src.constants import (
    ANIME_FILE_NAME,
    USERS_FILE_NAME,
    USER_ANIME_RATINGS_FILE_NAME,
    RATINGS_FILE_NAME,
)
from recommender.src.entity.config_entity import (
    DataPreprocessingConfig,
    ContentBasedRecommenderConfig,
    UserBasedRecommenderConfig,
    HybridRecommenderConfig,
)


def _parse_list_cell(value):
    if isinstance(value, str) and value.startswith("["):
        return eval(value)
    return []


class InferencePipeline:
    def __init__(self):
        self.preproc_cfg = DataPreprocessingConfig()
        self.anime_df: pd.DataFrame | None = None
        self.users_df: pd.DataFrame | None = None
        self.uar_df: pd.DataFrame | None = None
        self.ratings_df: pd.DataFrame | None = None
        self.content: ContentBasedRecommender | None = None
        self.user_based: UserBasedRecommender | None = None
        self.hybrid: HybridRecommender | None = None

    def _load_data(self) -> None:
        preproc_dir = self.preproc_cfg.data_preprocessing_dir
        raw_dir = self.preproc_cfg.raw_data_dir

        self.anime_df = pd.read_csv(os.path.join(preproc_dir, ANIME_FILE_NAME))
        for col in ["genres", "producers", "studios"]:
            self.anime_df[col] = self.anime_df[col].apply(_parse_list_cell)
        self.users_df = pd.read_csv(os.path.join(preproc_dir, USERS_FILE_NAME))
        self.uar_df = pd.read_csv(os.path.join(preproc_dir, USER_ANIME_RATINGS_FILE_NAME))
        self.ratings_df = pd.read_csv(os.path.join(raw_dir, RATINGS_FILE_NAME))

    def load(self) -> "InferencePipeline":
        self._load_data()

        self.content = ContentBasedRecommender(ContentBasedRecommenderConfig()).load(self.anime_df)

        self.user_based = UserBasedRecommender(UserBasedRecommenderConfig()).load()
        # user_vectors_sparse is not restored; downstream lookups go through
        # ub.get_user_vec, which falls back to index.reconstruct on demand.

        self.hybrid = HybridRecommender(HybridRecommenderConfig()).fit(
            self.content, self.user_based, self.anime_df, self.uar_df, self.ratings_df
        )
        return self

    def _pick_sample_user(self) -> int:
        ub = self.user_based
        return next(uid for uid, i in ub.user_id_to_idx.items() if i != ub.avg_idx)

    def run(self) -> None:
        if self.content is None:
            self.load()

        sample_anime_id = int(self.anime_df["anime_id"].iloc[0])
        sample_uid = self._pick_sample_user()

        print(f"\nContent recs for anime {sample_anime_id}:")
        print(self.content.recommend(sample_anime_id, k=5))

        print(f"\nSimilar users for user {sample_uid}:")
        print(self.user_based.similar_users(sample_uid, k=5))

        print(f"\nHybrid recs for user {sample_uid}:")
        print(self.hybrid.recommend(sample_uid))
