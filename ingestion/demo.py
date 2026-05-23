"""
demo.py — Run the full ingestion + preprocessing pipeline.

Run from the project root:
    python -m ingestion.demo
"""

from ingestion.pipeline.loading_pipeline import LoadingPipeline
from ingestion.pipeline.ingestion_pipeline import IngestionPipeline
from ingestion.pipeline.preprocessing_pipeline import PreprocessingPipeline


def run_pipeline() -> None:
    LoadingPipeline().run()
    IngestionPipeline().run()
    PreprocessingPipeline().run()
    print("Pipeline complete.")


if __name__ == "__main__":
    run_pipeline()
