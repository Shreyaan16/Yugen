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
