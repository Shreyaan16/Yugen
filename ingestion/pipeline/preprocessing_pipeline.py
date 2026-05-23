from ingestion.components.data_preprocessing import DataPreprocessing
from ingestion.entity.config_entity import DataIngestionConfig, DataPreprocessingConfig


class PreprocessingPipeline:
    def __init__(self, config: DataPreprocessingConfig | None = None):
        self.config = config or DataPreprocessingConfig(
            data_ingestion_dir=DataIngestionConfig().data_ingestion_dir
        )

    def run(self) -> None:
        print("[PreprocessingPipeline] Cleaning and aggregating data ...")
        artifact = DataPreprocessing(self.config).run()
        print(f"  Preprocessed data saved to: {artifact.data_preprocessing_dir}")
        print("[PreprocessingPipeline] Done.\n")


if __name__ == "__main__":
    PreprocessingPipeline().run()
