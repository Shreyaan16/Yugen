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
