import os
from azure.storage.blob import BlobServiceClient
from src.constants import CONNECTION_STRING, CONTAINER_NAME
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        os.makedirs(self.data_ingestion_config.data_ingestion_dir, exist_ok=True)

    def download_csv_from_azure(self) -> None:
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)

        for blob in container_client.list_blobs():
            file_path = os.path.join(self.data_ingestion_config.data_ingestion_dir, blob.name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            blob_client = container_client.get_blob_client(blob.name)
            with open(file_path, "wb") as f:
                f.write(blob_client.download_blob().readall())

    def run(self) -> DataIngestionArtifact:
        self.download_csv_from_azure()
        return DataIngestionArtifact(data_ingestion_dir=self.data_ingestion_config.data_ingestion_dir)
