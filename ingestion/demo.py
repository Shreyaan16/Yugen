"""
demo.py — Run the full ingestion + preprocessing pipeline.

Run from the project root:
    python -m ingestion.demo
"""

from ingestion.components.loading import Loading
from ingestion.components.data_ingestion import DataIngestion
from ingestion.components.data_preprocessing import DataPreprocessing
from ingestion.entity.config_entity import LoadingConfig, DataIngestionConfig, DataPreprocessingConfig


def run_pipeline() -> None:
    # ── Step 1: Export tables from PostgreSQL → Azure Blob Storage ──
    print("Step 1/3  Loading: exporting tables from PostgreSQL to Azure …")
    loading_artifact = Loading(LoadingConfig()).run()
    print(f"  Uploaded {len(loading_artifact.uploaded_blobs)} blobs: {loading_artifact.uploaded_blobs}")

    # ── Step 2: Download CSVs from Azure → local artifacts ──────────
    print("\nStep 2/3  DataIngestion: downloading CSVs from Azure …")
    ingestion_artifact = DataIngestion(DataIngestionConfig()).run()
    print(f"  Data saved to: {ingestion_artifact.data_ingestion_dir}")

    # ── Step 3: Preprocess raw CSVs → cleaned artifacts ─────────────
    print("\nStep 3/3  DataPreprocessing: cleaning and aggregating data …")
    preprocessing_artifact = DataPreprocessing(
        DataPreprocessingConfig(data_ingestion_dir=ingestion_artifact.data_ingestion_dir)
    ).run()
    print(f"  Preprocessed data saved to: {preprocessing_artifact.data_preprocessing_dir}")

    print("\nPipeline complete.")


if __name__ == "__main__":
    run_pipeline()
