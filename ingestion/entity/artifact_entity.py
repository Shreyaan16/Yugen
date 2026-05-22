from dataclasses import dataclass

@dataclass
class LoadingArtifact:
    uploaded_blobs: list[str]

@dataclass
class DataIngestionArtifact:
    data_ingestion_dir: str

@dataclass
class DataPreprocessingArtifact:
    data_preprocessing_dir: str
