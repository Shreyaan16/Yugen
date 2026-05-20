from __future__ import annotations

from src.constants import CONTAINER_NAME
from src.entity.config_entity import DataPreprocessingConfig
from src.pipeline import *


if __name__ == "__main__":
	uploaded_tables = run_loading()
	print(f"Uploaded {len(uploaded_tables)} tables to {CONTAINER_NAME}")
	data_ingestion_dir = run_data_ingestion()
	print(f"Data ingested to {data_ingestion_dir}")
	data_preprocessing_dir = run_data_preprocessing()
	print(f"Data preprocessed to {data_preprocessing_dir}")
	preproc_cfg = DataPreprocessingConfig()
	run_recommenders_demo(preproc_cfg.data_preprocessing_dir, preproc_cfg.raw_data_dir)
	eval_dir = run_evaluation(preproc_cfg.data_preprocessing_dir, preproc_cfg.raw_data_dir)
	print(f"\nEvaluation metrics saved to {eval_dir}")
