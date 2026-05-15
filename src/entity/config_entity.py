import os
from dataclasses import dataclass, field
from src.constants import *


@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = ARTIFACT_DIR


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)


@dataclass
class DataPreprocessingConfig:
    data_preprocessing_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_PREPROCESSING_DIR_NAME)
    raw_data_dir: str = os.path.join(training_pipeline_config.artifact_dir, RAW_DATA_DIR_NAME)
    train_data_dir: str = os.path.join(data_preprocessing_dir, TRAIN_DATA_DIR_NAME)
    test_data_dir: str = os.path.join(data_preprocessing_dir, TEST_DATA_DIR_NAME)
    val_data_dir: str = os.path.join(data_preprocessing_dir, VAL_DATA_DIR_NAME)


@dataclass
class ContentBasedRecommenderConfig:
    content_based_dir: str = os.path.join(training_pipeline_config.artifact_dir, RECOMMENDER_DIR_NAME, CONTENT_BASED_DIR_NAME)
    max_features: int = CONTENT_MAX_FEATURES
    n_components: int = CONTENT_N_COMPONENTS
    hnsw_m: int = CONTENT_HNSW_M
    ef_construction: int = CONTENT_EF_CONSTRUCTION
    random_state: int = RANDOM_STATE


@dataclass
class UserBasedRecommenderConfig:
    user_based_dir: str = os.path.join(training_pipeline_config.artifact_dir, RECOMMENDER_DIR_NAME, USER_BASED_DIR_NAME)
    hnsw_m: int = USER_HNSW_M
    ef_construction: int = USER_EF_CONSTRUCTION
