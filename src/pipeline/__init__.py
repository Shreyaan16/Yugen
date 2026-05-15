import os
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.recommender import ContentBasedRecommender, UserBasedRecommender
from src.constants import SPLIT_FILE_NAME, RATINGS_SPLIT_FILE_NAME
from src.entity.config_entity import (
    DataIngestionConfig,
    DataPreprocessingConfig,
    ContentBasedRecommenderConfig,
    UserBasedRecommenderConfig,
)


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_preprocessing_config = DataPreprocessingConfig()
        self.content_based_config = ContentBasedRecommenderConfig()
        self.user_based_config = UserBasedRecommenderConfig()

    def run(self):
        data_ingestion = DataIngestion(self.data_ingestion_config)
        # data_ingestion.run()

        data_preprocessing = DataPreprocessing(self.data_preprocessing_config)
        # data_preprocessing_artifact = data_preprocessing.run()

        train_df = pd.read_csv(os.path.join(self.data_preprocessing_config.train_data_dir, SPLIT_FILE_NAME))
        content_recommender = ContentBasedRecommender(self.content_based_config)
        content_artifact = content_recommender.run(train_df)

        ratings_df = pd.read_csv(os.path.join(self.data_preprocessing_config.train_data_dir, RATINGS_SPLIT_FILE_NAME))
        user_recommender = UserBasedRecommender(self.user_based_config)
        user_artifact = user_recommender.run(ratings_df, content_artifact)

        return content_artifact, user_artifact


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
