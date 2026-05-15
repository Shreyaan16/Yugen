from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    data_ingestion_dir: str

@dataclass
class DataPreprocessingArtifact:
    data_preprocessing_dir: str
    raw_data_dir: str
    train_data_dir: str
    test_data_dir: str
    val_data_dir: str

@dataclass
class ContentBasedRecommenderArtifact:
    content_based_dir: str
    index_path: str
    mal_to_faiss_path: str
    faiss_to_mal_path: str

@dataclass
class UserBasedRecommenderArtifact:
    user_based_dir: str
    index_path: str
    user_id_to_faiss_path: str
    user_id_to_vector_path: str

@dataclass
class HybridRecommenderArtifact:
    hybrid_dir: str
    ratings_matrix_path: str
