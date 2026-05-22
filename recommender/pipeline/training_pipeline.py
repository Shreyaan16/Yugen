import os
import pandas as pd
from recommender.constants import *
from recommender.components.content_based import ContentBasedRecommender
from recommender.components.user_based import UserBasedRecommender
from recommender.entity.artifact_entity import (ContentBasedRecommenderArtifact, UserBasedRecommenderArtifact)
from recommender.entity.config_entity import (ContentBasedRecommenderConfig, UserBasedRecommenderConfig)


class TrainingPipeline:
    def __init__(self, cb_config: ContentBasedRecommenderConfig | None = None, ub_config: UserBasedRecommenderConfig | None = None):
        self.cb_config = cb_config or ContentBasedRecommenderConfig()
        self.ub_config = ub_config or UserBasedRecommenderConfig()
        self._preprocessed_dir = self.cb_config.preprocessed_dir

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print(f"[TrainingPipeline] Loading preprocessed data from: {self._preprocessed_dir}")
        anime_df = pd.read_csv(os.path.join(self._preprocessed_dir, ANIME_FILE_NAME))
        users_df = pd.read_csv(os.path.join(self._preprocessed_dir, USERS_FILE_NAME))
        ratings_df = pd.read_csv(os.path.join(self._preprocessed_dir, USER_ANIME_RATINGS_FILE_NAME))
        print(
            f"  anime: {len(anime_df):,} rows | "
            f"users: {len(users_df):,} rows | "
            f"ratings: {len(ratings_df):,} rows"
        )
        return anime_df, users_df, ratings_df

    def _run_content_based(self, anime_df: pd.DataFrame) -> tuple[ContentBasedRecommender, ContentBasedRecommenderArtifact]:
        print("\n[TrainingPipeline] Stage 1 — Content-Based Recommender")
        cb = ContentBasedRecommender(config=self.cb_config)
        artifact = cb.run(anime_df)
        print(f"   FAISS index saved   → {artifact.index_path}")
        print(f"   anime_id→idx map    → {artifact.anime_id_to_idx_path}")
        print(f"   TF-IDF vectorizer   → {artifact.tfidf_vectorizer_path}")
        print(f"   TF-IDF matrix       → {artifact.tfidf_matrix_path}")
        return cb, artifact

    def _run_user_based(self, ratings_df: pd.DataFrame, users_df: pd.DataFrame,) -> tuple[UserBasedRecommender, UserBasedRecommenderArtifact]:
        print("\n[TrainingPipeline] Stage 2 — User-Based Recommender")
        ub = UserBasedRecommender(config=self.ub_config)
        artifact = ub.run(ratings_df, users_df)
        print(f"   User FAISS index    → {artifact.index_path}")
        print(f"   user_id→idx map     → {artifact.user_id_to_idx_path}")
        return ub, artifact

    def run(self) -> dict:
        anime_df, users_df, ratings_df = self._load_data()
        cb, cb_artifact = self._run_content_based(anime_df)
        ub, ub_artifact = self._run_user_based(ratings_df, users_df)
        print("\n[TrainingPipeline]   Training complete.\n")
        return {"content_based": cb_artifact, "user_based": ub_artifact}

if __name__ == "__main__":
    TrainingPipeline().run()
