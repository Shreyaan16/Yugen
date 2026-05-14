from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.entity.config_entity import DataIngestionConfig, DataPreprocessingConfig


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_preprocessing_config = DataPreprocessingConfig()

    def run(self):
        data_ingestion = DataIngestion(self.data_ingestion_config)
        data_ingestion_artifact = data_ingestion.run()

        data_preprocessing = DataPreprocessing(self.data_preprocessing_config)
        data_preprocessing_artifact = data_preprocessing.run()

        return data_preprocessing_artifact


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
