from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

ROOT_DIR: Path = Path(__file__).resolve().parents[2]

PIPELINE_NAME: str = "YuGen"
ARTIFACT_DIR: str = os.path.join(ROOT_DIR, "artifacts")

COLS_TO_DROP_CONFIG: str = os.path.join(ROOT_DIR, "src", "config", "cols_to_drop.yaml")
FEATURE_BINS_CONFIG: str = os.path.join(ROOT_DIR, "src", "config", "feature_bins.yaml")

RANDOM_STATE: int = 42
TRAIN_RATIO: float = 0.8
VAL_RATIO: float = 0.1
TEST_RATIO: float = 0.1
SPLIT_FILE_NAME: str = "data.csv"
RATINGS_SPLIT_FILE_NAME: str = "ratings.csv"
ANIME_FILE_NAME: str = "anime.csv"
ANIME_SYNOPSIS_FILE_NAME: str = "anime_with_synopsis.csv"
RATINGS_FILE_NAME: str = "rating_complete.csv"

###################### AZURE ENV ############################
CONNECTION_STRING: str = os.environ["CONNECTION_STRING"]
CONTAINER_NAME: str = os.environ["CONTAINER_NAME"]
REGION: str = os.environ["REGION"]
STORAGE_ACCOUNT_NAME: str = os.environ["STORAGE_ACCOUNT_NAME"]

###################### DATA INGESTION ############################
DATA_INGESTION_DIR_NAME: str = "data_ingestion"

###################### DATA PREPROCESSING ############################
RAW_DATA_DIR_NAME: str = "data_ingestion"
DATA_PREPROCESSING_DIR_NAME: str = "data_preprocessing"
TRAIN_DATA_DIR_NAME: str = "train_data"
TEST_DATA_DIR_NAME: str = "test_data"
VAL_DATA_DIR_NAME: str = "val_data"
