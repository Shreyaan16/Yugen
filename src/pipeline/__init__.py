from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def run(self):
        data_ingestion = DataIngestion(self.data_ingestion_config)
        data_ingestion_artifact = data_ingestion.run()
        return data_ingestion_artifact


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
