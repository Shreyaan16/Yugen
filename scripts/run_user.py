import os
import pandas as pd

from src.components.recommender import UserBasedRecommender
from src.constants import (
    RATINGS_SPLIT_FILE_NAME,
    CONTENT_INDEX_FILE,
    CONTENT_MAL_TO_FAISS_FILE,
    CONTENT_FAISS_TO_MAL_FILE,
)
from src.entity.config_entity import (
    UserBasedRecommenderConfig,
    ContentBasedRecommenderConfig,
    DataPreprocessingConfig,
)
from src.entity.artifact_entity import ContentBasedRecommenderArtifact

if __name__ == "__main__":
    pre_cfg = DataPreprocessingConfig()
    content_cfg = ContentBasedRecommenderConfig()

    content_artifact = ContentBasedRecommenderArtifact(
        content_based_dir=content_cfg.content_based_dir,
        index_path=os.path.join(content_cfg.content_based_dir, CONTENT_INDEX_FILE),
        mal_to_faiss_path=os.path.join(content_cfg.content_based_dir, CONTENT_MAL_TO_FAISS_FILE),
        faiss_to_mal_path=os.path.join(content_cfg.content_based_dir, CONTENT_FAISS_TO_MAL_FILE),
    )

    ratings_df = pd.read_csv(os.path.join(pre_cfg.train_data_dir, RATINGS_SPLIT_FILE_NAME))
    artifact = UserBasedRecommender(UserBasedRecommenderConfig()).run(ratings_df, content_artifact)
    print(artifact)
