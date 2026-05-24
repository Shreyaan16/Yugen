import os
import pandas as pd
from recommender.constants import *
from recommender.components.content_based import ContentBasedRecommender
from recommender.entity.artifact_entity import ContentBasedRecommenderArtifact
from recommender.entity.config_entity import ContentBasedRecommenderConfig


class TrainingPipeline:
    def __init__(self, cb_config: ContentBasedRecommenderConfig | None = None):
        self.cb_config = cb_config or ContentBasedRecommenderConfig()
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

    def run(self) -> dict:
        anime_df, users_df, ratings_df = self._load_data()
        cb, cb_artifact = self._run_content_based(anime_df)
        print("\n[TrainingPipeline]   Training complete.\n")
        return {"content_based": cb_artifact}

if __name__ == "__main__":
    TrainingPipeline().run()
