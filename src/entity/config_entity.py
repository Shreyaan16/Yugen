import os
from dataclasses import dataclass, field
from src.constants import *
from src.utils import read_yaml

_params = read_yaml(PARAMS_PATH)
_cb = _params["content_based"]
_ub = _params["user_based"]
_cf = _params["cf"]
_hy = _params["hybrid"]
_ev = _params["evaluation"]


@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = ARTIFACT_DIR


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

_data_preprocessing_dir = os.path.join(training_pipeline_config.artifact_dir, DATA_PREPROCESSING_DIR_NAME)
_recommender_dir = os.path.join(training_pipeline_config.artifact_dir, RECOMMENDER_DIR_NAME)


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


@dataclass
class ContentBasedRecommenderConfig:
    content_based_dir: str = os.path.join(_recommender_dir, CONTENT_BASED_DIR_NAME)
    max_features: int = _cb["max_features"]
    ngram_range: tuple = field(default_factory=lambda: tuple(_cb["ngram_range"]))
    min_df: int = _cb["min_df"]
    random_state: int = _cb["random_state"]


@dataclass
class UserBasedRecommenderConfig:
    user_based_dir: str = os.path.join(_recommender_dir, USER_BASED_DIR_NAME)
    min_ratings: int = _ub["min_ratings"]
    chunk_size: int = _ub["chunk_size"]


@dataclass
class CFRecommenderConfig:
    cf_dir: str = os.path.join(_recommender_dir, CF_DIR_NAME)
    k_neighbors: int = _cf["k_neighbors"]
    top_n: int = _cf["top_n"]
    min_support: int = _cf["min_support"]
    min_num_ratings: int = _cf["min_num_ratings"]


@dataclass
class HybridRecommenderConfig:
    hybrid_dir: str = os.path.join(_recommender_dir, HYBRID_DIR_NAME)
    k_neighbors: int = _hy["k_neighbors"]
    top_n: int = _hy["top_n"]
    alpha: float = _hy["alpha"]
    min_support: int = _hy["min_support"]
    min_num_ratings: int = _hy["min_num_ratings"]
    content_pool: int = _hy["content_pool"]


@dataclass
class EvaluationConfig:
    evaluation_dir: str = os.path.join(training_pipeline_config.artifact_dir, EVALUATION_DIR_NAME)
    top_k: int = _ev["top_k"]
    holdout_ratio: float = _ev["holdout_ratio"]
    relevant_threshold: float = _ev["relevant_threshold"]
    n_users_sample: int = _ev["n_users_sample"]
    min_user_ratings: int = _ev["min_user_ratings"]
    random_state: int = _ev["random_state"]
