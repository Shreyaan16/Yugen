from dataclasses import dataclass

@dataclass
class ContentBasedRecommenderArtifact:
    content_based_dir: str
    index_path: str
    anime_id_to_idx_path: str
    tfidf_vectorizer_path: str
    tfidf_matrix_path: str

@dataclass
class HybridRecommenderArtifact:
    status: str

@dataclass
class EvaluationArtifact:
    evaluation_dir: str
    content_metrics_path: str
    hybrid_metrics_path: str
    summary_path: str
