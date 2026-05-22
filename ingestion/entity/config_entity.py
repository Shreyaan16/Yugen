import os
from dataclasses import dataclass
from ingestion.constants import *

@dataclass
class LoadingConfig:
    pg_connection_string: str = DATABASE_URL
    azure_connection_string: str = CONNECTION_STRING
    azure_container_name: str = CONTAINER_NAME
    schema: str = SCHEMA_NAME

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR_NAME)

@dataclass
class DataPreprocessingConfig:
    data_preprocessing_dir: str = os.path.join(ARTIFACT_DIR, DATA_PREPROCESSING_DIR_NAME)
    data_ingestion_dir: str = os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR_NAME)
