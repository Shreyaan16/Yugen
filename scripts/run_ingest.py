from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig

if __name__ == "__main__":
    DataIngestion(DataIngestionConfig()).run()
