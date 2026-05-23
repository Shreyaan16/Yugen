from ingestion.components.data_ingestion import DataIngestion
from ingestion.entity.config_entity import DataIngestionConfig


class IngestionPipeline:
    def __init__(self, config: DataIngestionConfig | None = None):
        self.config = config or DataIngestionConfig()

    def run(self) -> None:
        print("[IngestionPipeline] Downloading CSVs from Azure ...")
        artifact = DataIngestion(self.config).run()
        print(f"  Data saved to: {artifact.data_ingestion_dir}")
        print("[IngestionPipeline] Done.\n")


if __name__ == "__main__":
    IngestionPipeline().run()
