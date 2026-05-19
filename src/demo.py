from __future__ import annotations

import os
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.loading import Loading
from src.components.recommender import (
	CFRecommender,
	ContentBasedRecommender,
	HybridRecommender,
	UserBasedRecommender,
)
from src.constants import (
	ANIME_FILE_NAME,
	CONTAINER_NAME,
	RATINGS_FILE_NAME,
	USERS_FILE_NAME,
	USER_ANIME_RATINGS_FILE_NAME,
)
from src.entity.config_entity import (
	CFRecommenderConfig,
	ContentBasedRecommenderConfig,
	DataIngestionConfig,
	DataPreprocessingConfig,
	HybridRecommenderConfig,
	LoadingConfig,
	UserBasedRecommenderConfig,
)


def run_loading(schema: str = "public") -> list[str]:
	config = LoadingConfig(schema=schema)
	with Loading(config=config) as loader:
		artifact = loader.run()
		return artifact.uploaded_blobs


def run_data_ingestion() -> str:
	config = DataIngestionConfig()
	ingestor = DataIngestion(data_ingestion_config=config)
	artifact = ingestor.run()
	return artifact.data_ingestion_dir


def run_data_preprocessing() -> str:
	config = DataPreprocessingConfig()
	preprocessor = DataPreprocessing(data_preprocessing_config=config)
	artifact = preprocessor.run()
	return artifact.data_preprocessing_dir


def _load_preprocessed(preproc_dir: str, raw_dir: str):
	anime_df = pd.read_csv(os.path.join(preproc_dir, ANIME_FILE_NAME))
	for col in ["genres", "producers", "studios"]:
		anime_df[col] = anime_df[col].apply(
			lambda s: eval(s) if isinstance(s, str) and s.startswith("[") else []
		)
	users_df = pd.read_csv(os.path.join(preproc_dir, USERS_FILE_NAME))
	user_anime_ratings_df = pd.read_csv(os.path.join(preproc_dir, USER_ANIME_RATINGS_FILE_NAME))
	ratings_df = pd.read_csv(os.path.join(raw_dir, RATINGS_FILE_NAME))
	return anime_df, users_df, user_anime_ratings_df, ratings_df


def run_recommenders(preproc_dir: str, raw_dir: str):
	anime_df, users_df, user_anime_ratings_df, ratings_df = _load_preprocessed(preproc_dir, raw_dir)

	content = ContentBasedRecommender(ContentBasedRecommenderConfig())
	content_artifact = content.run(anime_df)
	print(f"Content-based saved to {content_artifact.content_based_dir}")

	user_based = UserBasedRecommender(UserBasedRecommenderConfig())
	user_artifact = user_based.run(user_anime_ratings_df, users_df, content)
	print(f"User-based saved to {user_artifact.user_based_dir}")

	cf = CFRecommender(CFRecommenderConfig())
	cf.fit(user_based, anime_df, user_anime_ratings_df, ratings_df)
	cf.save()

	hybrid = HybridRecommender(HybridRecommenderConfig())
	hybrid.fit(content, user_based, anime_df, user_anime_ratings_df, ratings_df)
	hybrid.save()

	sample_anime_id = int(anime_df["anime_id"].iloc[0])
	print(f"\nContent recs for anime {sample_anime_id}:")
	print(content.recommend(sample_anime_id, k=5))

	sample_uid = next(uid for uid, i in user_based.user_id_to_idx.items() if i != user_based.avg_idx)
	print(f"\nSimilar users for user {sample_uid}:")
	print(user_based.similar_users(sample_uid, k=5))

	print(f"\nCF recs for user {sample_uid}:")
	print(cf.recommend(sample_uid))

	print(f"\nHybrid recs for user {sample_uid}:")
	print(hybrid.recommend(sample_uid))


if __name__ == "__main__":
	# uploaded_tables = run_loading()
	# print(f"Uploaded {len(uploaded_tables)} tables to {CONTAINER_NAME}")
	# data_ingestion_dir = run_data_ingestion()
	# print(f"Data ingested to {data_ingestion_dir}")
	# data_preprocessing_dir = run_data_preprocessing()
	# print(f"Data preprocessed to {data_preprocessing_dir}")

	preproc_cfg = DataPreprocessingConfig()
	run_recommenders(preproc_cfg.data_preprocessing_dir, preproc_cfg.raw_data_dir)
