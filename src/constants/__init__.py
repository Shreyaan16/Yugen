from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

ROOT_DIR: Path = Path(__file__).resolve().parents[2]

PIPELINE_NAME: str = "YuGen"
ARTIFACT_DIR: str = os.path.join(ROOT_DIR, "artifacts")

PARAMS_PATH: str = os.path.join(ROOT_DIR, "params.yaml")
COLS_TO_DROP_CONFIG: str = os.path.join(ROOT_DIR, "src", "config", "cols_to_drop.yaml")
FEATURE_BINS_CONFIG: str = os.path.join(ROOT_DIR, "src", "config", "feature_bins.yaml")

###################### AZURE ENV ############################
CONNECTION_STRING: str = os.environ["CONNECTION_STRING"]
CONTAINER_NAME: str = os.environ["V2_CONTAINER_NAME"]
REGION: str = os.environ["REGION"]
STORAGE_ACCOUNT_NAME: str = os.environ["STORAGE_ACCOUNT_NAME"]

###################### POSTGRES ENV #########################
DATABASE_URL: str = os.environ["DATABASE_URL"]

###################### DIR NAMES ############################
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
RAW_DATA_DIR_NAME: str = "data_ingestion"
DATA_PREPROCESSING_DIR_NAME: str = "data_preprocessing"
TRAIN_DATA_DIR_NAME: str = "train_data"
TEST_DATA_DIR_NAME: str = "test_data"
VAL_DATA_DIR_NAME: str = "val_data"

RECOMMENDER_DIR_NAME: str = "recommender"
CONTENT_BASED_DIR_NAME: str = "content_based"
USER_BASED_DIR_NAME: str = "user_based"
HYBRID_DIR_NAME: str = "hybrid"
EVALUATION_DIR_NAME: str = "evaluation"

###################### FILE NAMES ############################
SPLIT_FILE_NAME: str = "data.csv"
RATINGS_SPLIT_FILE_NAME: str = "ratings.csv"
ANIME_FILE_NAME: str = "anime.csv"
ANIME_SYNOPSIS_FILE_NAME: str = "anime_with_synopsis.csv"
RATINGS_FILE_NAME: str = "rating_complete.csv"

CONTENT_INDEX_FILE: str = "content_faiss_index.bin"
CONTENT_MAL_TO_FAISS_FILE: str = "mal_id_to_faiss_id.json"
CONTENT_FAISS_TO_MAL_FILE: str = "faiss_id_to_mal_id.json"

USER_INDEX_FILE: str = "user_faiss_index.bin"
USER_ID_TO_FAISS_FILE: str = "user_id_to_faiss_id.json"
USER_ID_TO_VECTOR_FILE: str = "user_id_to_vector.json"

HYBRID_RATINGS_MATRIX_FILE: str = "ratings_csr.npz"

EVAL_HYBRID_METRICS_FILE: str = "hybrid_metrics.json"
EVAL_CONTENT_METRICS_FILE: str = "content_metrics.json"
EVAL_USER_METRICS_FILE: str = "user_metrics.json"
EVAL_SUMMARY_FILE: str = "summary.json"
