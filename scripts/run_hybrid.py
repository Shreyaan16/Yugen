import os
import pandas as pd

from src.components.recommender import HybridRecommender
from src.constants import (
    RATINGS_SPLIT_FILE_NAME,
    ANIME_FILE_NAME,
    CONTENT_INDEX_FILE,
    CONTENT_MAL_TO_FAISS_FILE,
    CONTENT_FAISS_TO_MAL_FILE,
    USER_INDEX_FILE,
    USER_ID_TO_FAISS_FILE,
    USER_ID_TO_VECTOR_FILE,
)
from src.entity.config_entity import (
    HybridRecommenderConfig,
    ContentBasedRecommenderConfig,
    UserBasedRecommenderConfig,
    DataPreprocessingConfig,
)
from src.entity.artifact_entity import (
    ContentBasedRecommenderArtifact,
    UserBasedRecommenderArtifact,
)

if __name__ == "__main__":
    pre_cfg = DataPreprocessingConfig()
    content_cfg = ContentBasedRecommenderConfig()
    user_cfg = UserBasedRecommenderConfig()

    content_artifact = ContentBasedRecommenderArtifact(
        content_based_dir=content_cfg.content_based_dir,
        index_path=os.path.join(content_cfg.content_based_dir, CONTENT_INDEX_FILE),
        mal_to_faiss_path=os.path.join(content_cfg.content_based_dir, CONTENT_MAL_TO_FAISS_FILE),
        faiss_to_mal_path=os.path.join(content_cfg.content_based_dir, CONTENT_FAISS_TO_MAL_FILE),
    )
    user_artifact = UserBasedRecommenderArtifact(
        user_based_dir=user_cfg.user_based_dir,
        index_path=os.path.join(user_cfg.user_based_dir, USER_INDEX_FILE),
        user_id_to_faiss_path=os.path.join(user_cfg.user_based_dir, USER_ID_TO_FAISS_FILE),
        user_id_to_vector_path=os.path.join(user_cfg.user_based_dir, USER_ID_TO_VECTOR_FILE),
    )

    ratings_df = pd.read_csv(os.path.join(pre_cfg.train_data_dir, RATINGS_SPLIT_FILE_NAME))
    anime_df = pd.read_csv(
        os.path.join(pre_cfg.raw_data_dir, ANIME_FILE_NAME), usecols=["MAL_ID", "Name"]
    )
    anime_titles = {
        int(mal_id): str(name) for mal_id, name in zip(anime_df["MAL_ID"], anime_df["Name"])
    }
    artifact = HybridRecommender(HybridRecommenderConfig()).run(
        ratings_df, content_artifact, user_artifact, anime_titles
    )
    print(artifact)
