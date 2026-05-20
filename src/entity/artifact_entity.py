from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    data_ingestion_dir: str


@dataclass
class LoadingArtifact:
    uploaded_blobs: list[str]


@dataclass
class DataPreprocessingArtifact:
    data_preprocessing_dir: str
    raw_data_dir: str


@dataclass
class ContentBasedRecommenderArtifact:
    content_based_dir: str
    index_path: str
    anime_id_to_idx_path: str
    tfidf_vectorizer_path: str


@dataclass
class UserBasedRecommenderArtifact:
    user_based_dir: str
    index_path: str
    user_id_to_idx_path: str


@dataclass
class HybridRecommenderArtifact:
    hybrid_dir: str


@dataclass
class EvaluationArtifact:
    evaluation_dir: str
    hybrid_metrics_path: str
    summary_path: str
