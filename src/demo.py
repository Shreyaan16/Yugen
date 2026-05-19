from __future__ import annotations

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.loading import Loading
from src.constants import CONTAINER_NAME
from src.entity.config_entity import (
	DataIngestionConfig,
	DataPreprocessingConfig,
	LoadingConfig,
)

def run_loading(schema: str = "public",) -> list[str]:
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

if __name__ == "__main__":
	# uploaded_tables = run_loading()
	# print(f"Uploaded {len(uploaded_tables)} tables to {CONTAINER_NAME}")
	# data_ingestion_dir = run_data_ingestion()
	# print(f"Data ingested to {data_ingestion_dir}")
	data_preprocessing_dir = run_data_preprocessing()
	print(f"Data preprocessed to {data_preprocessing_dir}")