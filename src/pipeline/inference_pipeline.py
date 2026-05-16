import os
import pandas as pd

from src.components.recommender import ContentBasedRecommender, HybridRecommender
from src.constants import (
    SPLIT_FILE_NAME,
    ANIME_FILE_NAME,
    CONTENT_INDEX_FILE,
    CONTENT_MAL_TO_FAISS_FILE,
    CONTENT_FAISS_TO_MAL_FILE,
    USER_INDEX_FILE,
    USER_ID_TO_FAISS_FILE,
    USER_ID_TO_VECTOR_FILE,
    HYBRID_RATINGS_MATRIX_FILE,
)
from src.entity.config_entity import (
    ContentBasedRecommenderConfig,
    UserBasedRecommenderConfig,
    HybridRecommenderConfig,
    DataPreprocessingConfig,
)
from src.entity.artifact_entity import (
    ContentBasedRecommenderArtifact,
    UserBasedRecommenderArtifact,
    HybridRecommenderArtifact,
)


class InferencePipeline:
    """Loads pre-trained artifacts from disk and serves recommendations."""

    def __init__(self):
        self.pre_cfg = DataPreprocessingConfig()
        self.content_cfg = ContentBasedRecommenderConfig()
        self.user_cfg = UserBasedRecommenderConfig()
        self.hybrid_cfg = HybridRecommenderConfig()

        self.content_artifact = ContentBasedRecommenderArtifact(
            content_based_dir=self.content_cfg.content_based_dir,
            index_path=os.path.join(self.content_cfg.content_based_dir, CONTENT_INDEX_FILE),
            mal_to_faiss_path=os.path.join(self.content_cfg.content_based_dir, CONTENT_MAL_TO_FAISS_FILE),
            faiss_to_mal_path=os.path.join(self.content_cfg.content_based_dir, CONTENT_FAISS_TO_MAL_FILE),
        )
        self.user_artifact = UserBasedRecommenderArtifact(
            user_based_dir=self.user_cfg.user_based_dir,
            index_path=os.path.join(self.user_cfg.user_based_dir, USER_INDEX_FILE),
            user_id_to_faiss_path=os.path.join(self.user_cfg.user_based_dir, USER_ID_TO_FAISS_FILE),
            user_id_to_vector_path=os.path.join(self.user_cfg.user_based_dir, USER_ID_TO_VECTOR_FILE),
        )
        self.hybrid_artifact = HybridRecommenderArtifact(
            hybrid_dir=self.hybrid_cfg.hybrid_dir,
            ratings_matrix_path=os.path.join(self.hybrid_cfg.hybrid_dir, HYBRID_RATINGS_MATRIX_FILE),
        )

        train_df = pd.read_csv(os.path.join(self.pre_cfg.train_data_dir, SPLIT_FILE_NAME))
        anime_df = pd.read_csv(
            os.path.join(self.pre_cfg.raw_data_dir, ANIME_FILE_NAME),
            usecols=["MAL_ID", "Name"],
        )
        anime_titles: dict[int, str] = {
            int(mal_id): str(name) for mal_id, name in zip(anime_df["MAL_ID"], anime_df["Name"])
        }

        self.content_recommender = ContentBasedRecommender(self.content_cfg).load(train_df)
        self.hybrid_recommender = HybridRecommender(self.hybrid_cfg).load(
            self.content_artifact, self.user_artifact, self.hybrid_artifact, anime_titles
        )

    def recommend_for_user(self, user_id: int, top_k: int = 10) -> pd.DataFrame:
        return self.hybrid_recommender.recommend(user_id=user_id, top_k=top_k)

    def recommend_similar_to(self, title: str, n: int = 10) -> pd.DataFrame | None:
        return self.content_recommender.recommend(title=title, n=n)


if __name__ == "__main__":
    pipeline = InferencePipeline()
    print(pipeline.recommend_for_user(user_id=0, top_k=10))
    print(pipeline.recommend_similar_to(title="Mobile Suit Gundam", n=5))
