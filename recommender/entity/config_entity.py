import os
from dataclasses import dataclass, field
from recommender.constants import *

@dataclass
class ContentBasedRecommenderConfig:
    preprocessed_dir: str = field(default_factory=lambda: os.path.join(ARTIFACT_DIR, DATA_PREPROCESSING_DIR_NAME))
    content_based_dir: str = field(default_factory=lambda: os.path.join(ARTIFACT_DIR, RECOMMENDER_DIR_NAME, CONTENT_BASED_DIR_NAME))

@dataclass
class UserBasedRecommenderConfig:
    preprocessed_dir: str = field(default_factory=lambda: os.path.join(ARTIFACT_DIR, DATA_PREPROCESSING_DIR_NAME))
    user_based_dir: str = field(default_factory=lambda: os.path.join(ARTIFACT_DIR, RECOMMENDER_DIR_NAME, USER_BASED_DIR_NAME))
    anime_id_to_idx_path: str = field(default_factory=lambda: os.path.join(ARTIFACT_DIR, RECOMMENDER_DIR_NAME, CONTENT_BASED_DIR_NAME, ANIME_ID_TO_IDX_FILE))
    tfidf_vectorizer_path: str = field(default_factory=lambda: os.path.join(ARTIFACT_DIR, RECOMMENDER_DIR_NAME, CONTENT_BASED_DIR_NAME, TFIDF_VECTORIZER_FILE))
    tfidf_matrix_path: str = field(default_factory=lambda: os.path.join(ARTIFACT_DIR, RECOMMENDER_DIR_NAME, CONTENT_BASED_DIR_NAME, TFIDF_MATRIX_FILE))

@dataclass
class HybridRecommenderConfig:
    preprocessed_dir: str = field(default_factory=lambda: os.path.join(ARTIFACT_DIR, DATA_PREPROCESSING_DIR_NAME))
    anime_id_to_idx_path: str = field(default_factory=lambda: os.path.join(ARTIFACT_DIR, RECOMMENDER_DIR_NAME, CONTENT_BASED_DIR_NAME, ANIME_ID_TO_IDX_FILE))
    tfidf_vectorizer_path: str = field(default_factory=lambda: os.path.join(ARTIFACT_DIR, RECOMMENDER_DIR_NAME, CONTENT_BASED_DIR_NAME, TFIDF_VECTORIZER_FILE))
    user_id_to_idx_path: str = field(default_factory=lambda: os.path.join(ARTIFACT_DIR, RECOMMENDER_DIR_NAME, USER_BASED_DIR_NAME, USER_ID_TO_IDX_FILE))

@dataclass
class EvaluationConfig:
    evaluation_dir: str = field(default_factory=lambda: os.path.join(ARTIFACT_DIR, EVALUATION_DIR_NAME))
    content_eval_dir: str = field(default_factory=lambda: os.path.join(ARTIFACT_DIR, EVALUATION_DIR_NAME, EVAL_CONTENT_DIR_NAME))
    user_eval_dir: str = field(default_factory=lambda: os.path.join(ARTIFACT_DIR, EVALUATION_DIR_NAME, EVAL_USER_DIR_NAME))
    hybrid_eval_dir: str = field(default_factory=lambda: os.path.join(ARTIFACT_DIR, EVALUATION_DIR_NAME, EVAL_HYBRID_DIR_NAME))