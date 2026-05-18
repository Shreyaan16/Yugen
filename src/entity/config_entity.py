import os
from dataclasses import dataclass
from src.constants import *
from src.utils import read_yaml

_params = read_yaml(PARAMS_PATH)
_pp = _params["data_preprocessing"]
_cb = _params["content_based"]
_ub = _params["user_based"]
_hy = _params["hybrid"]
_ev = _params["evaluation"]


@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = ARTIFACT_DIR


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

_data_preprocessing_dir = os.path.join(training_pipeline_config.artifact_dir, DATA_PREPROCESSING_DIR_NAME)


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)


@dataclass
class LoadingConfig:
    pg_connection_string: str = DATABASE_URL
    azure_connection_string: str = CONNECTION_STRING
    azure_container_name: str = CONTAINER_NAME
    schema: str = "public"


@dataclass
class DataPreprocessingConfig:
    data_preprocessing_dir: str = _data_preprocessing_dir
    raw_data_dir: str = os.path.join(training_pipeline_config.artifact_dir, RAW_DATA_DIR_NAME)
    train_data_dir: str = os.path.join(_data_preprocessing_dir, TRAIN_DATA_DIR_NAME)
    test_data_dir: str = os.path.join(_data_preprocessing_dir, TEST_DATA_DIR_NAME)
    val_data_dir: str = os.path.join(_data_preprocessing_dir, VAL_DATA_DIR_NAME)
    train_ratio: float = _pp["train_ratio"]
    val_ratio: float = _pp["val_ratio"]
    test_ratio: float = _pp["test_ratio"]
    random_state: int = _pp["random_state"]


@dataclass
class ContentBasedRecommenderConfig:
    content_based_dir: str = os.path.join(training_pipeline_config.artifact_dir, RECOMMENDER_DIR_NAME, CONTENT_BASED_DIR_NAME)
    max_features: int = _cb["max_features"]
    n_components: int = _cb["n_components"]
    hnsw_m: int = _cb["hnsw_m"]
    ef_construction: int = _cb["ef_construction"]
    random_state: int = _cb["random_state"]


@dataclass
class UserBasedRecommenderConfig:
    user_based_dir: str = os.path.join(training_pipeline_config.artifact_dir, RECOMMENDER_DIR_NAME, USER_BASED_DIR_NAME)
    hnsw_m: int = _ub["hnsw_m"]
    ef_construction: int = _ub["ef_construction"]


@dataclass
class HybridRecommenderConfig:
    hybrid_dir: str = os.path.join(training_pipeline_config.artifact_dir, RECOMMENDER_DIR_NAME, HYBRID_DIR_NAME)
    similar_users_k: int = _hy["similar_users_k"]
    top_k: int = _hy["top_k"]


@dataclass
class EvaluationConfig:
    evaluation_dir: str = os.path.join(training_pipeline_config.artifact_dir, EVALUATION_DIR_NAME)
    top_k: int = _ev["top_k"]
    similar_users_k: int = _ev["similar_users_k"]
    relevant_threshold: float = _ev["relevant_threshold"]
    content_neighbors_k: int = _ev["content_neighbors_k"]
    content_sample_size: int = _ev["content_sample_size"]
    random_state: int = _ev["random_state"]
