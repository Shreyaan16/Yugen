from __future__ import annotations

import os
import pandas as pd

from recommender.src.components.loading import Loading
from recommender.src.components.data_ingestion import DataIngestion
from recommender.src.components.data_preprocessing import DataPreprocessing
from recommender.src.components.recommender import (
    ContentBasedRecommender,
    UserBasedRecommender,
    HybridRecommender,
)
from recommender.src.components.evaluation import Evaluation
from recommender.src.constants import (
    ANIME_FILE_NAME,
    USERS_FILE_NAME,
    USER_ANIME_RATINGS_FILE_NAME,
    RATINGS_FILE_NAME,
)
from recommender.src.entity.config_entity import (
    LoadingConfig,
    DataIngestionConfig,
    DataPreprocessingConfig,
    ContentBasedRecommenderConfig,
    UserBasedRecommenderConfig,
    HybridRecommenderConfig,
    EvaluationConfig,
)


def _parse_list_cell(value):
    if isinstance(value, str) and value.startswith("["):
        return eval(value)
    return []


class TrainingPipeline:
    def __init__(self, schema: str = "public"):
        self.schema = schema

    def run_loading(self) -> list[str]:
        with Loading(config=LoadingConfig(schema=self.schema)) as loader:
            return loader.run().uploaded_blobs

    def run_data_ingestion(self) -> str:
        artifact = DataIngestion(DataIngestionConfig()).run()
        return artifact.data_ingestion_dir

    def run_data_preprocessing(self) -> str:
        artifact = DataPreprocessing(DataPreprocessingConfig()).run()
        return artifact.data_preprocessing_dir

    def _load_preprocessed(self, preproc_dir: str, raw_dir: str):
        anime_df = pd.read_csv(os.path.join(preproc_dir, ANIME_FILE_NAME))
        for col in ["genres", "producers", "studios"]:
            anime_df[col] = anime_df[col].apply(_parse_list_cell)
        users_df = pd.read_csv(os.path.join(preproc_dir, USERS_FILE_NAME))
        uar_df = pd.read_csv(os.path.join(preproc_dir, USER_ANIME_RATINGS_FILE_NAME))
        ratings_df = pd.read_csv(os.path.join(raw_dir, RATINGS_FILE_NAME))
        return anime_df, users_df, uar_df, ratings_df

    def run_recommenders(self, preproc_dir: str, raw_dir: str):
        anime_df, users_df, uar_df, ratings_df = self._load_preprocessed(preproc_dir, raw_dir)

        content = ContentBasedRecommender(ContentBasedRecommenderConfig())
        content.run(anime_df)

        user_based = UserBasedRecommender(UserBasedRecommenderConfig())
        user_based.run(uar_df, users_df, content)

        hybrid = HybridRecommender(HybridRecommenderConfig())
        hybrid.fit(content, user_based, anime_df, uar_df, ratings_df)
        hybrid.save()

        return content, user_based, hybrid, anime_df, users_df, uar_df, ratings_df

    def run_evaluation(self, content, user_based, hybrid, anime_df, uar_df, ratings_df) -> str:
        evaluator = Evaluation(EvaluationConfig())
        evaluator.fit(content, user_based, hybrid, anime_df, uar_df, ratings_df)
        artifact = evaluator.run()
        return artifact.evaluation_dir

    def run(self) -> None:
        uploaded = self.run_loading()
        print(f"Loaded {len(uploaded)} tables")

        ingest_dir = self.run_data_ingestion()
        print(f"Data ingested to {ingest_dir}")

        preproc_cfg = DataPreprocessingConfig()
        preproc_dir = self.run_data_preprocessing()
        print(f"Data preprocessed to {preproc_dir}")

        content, user_based, hybrid, anime_df, _, uar_df, ratings_df = self.run_recommenders(
            preproc_cfg.data_preprocessing_dir, preproc_cfg.raw_data_dir
        )
        print("Recommenders trained")

        eval_dir = self.run_evaluation(content, user_based, hybrid, anime_df, uar_df, ratings_df)
        print(f"Evaluation metrics saved to {eval_dir}")
