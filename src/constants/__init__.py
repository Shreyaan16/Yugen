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

RECOMMENDER_DIR_NAME: str = "recommender"
CONTENT_BASED_DIR_NAME: str = "content_based"
USER_BASED_DIR_NAME: str = "user_based"
CF_DIR_NAME: str = "cf"
HYBRID_DIR_NAME: str = "hybrid"
EVALUATION_DIR_NAME: str = "evaluation"

###################### FILE NAMES ############################
ANIME_FILE_NAME: str = "anime.csv"
USERS_FILE_NAME: str = "users.csv"
USER_ANIME_RATINGS_FILE_NAME: str = "user_anime_ratings.csv"
RATINGS_FILE_NAME: str = "ratings.csv"

CONTENT_INDEX_FILE: str = "anime.index"
ANIME_ID_TO_IDX_FILE: str = "anime_id_to_idx.json"
TFIDF_VECTORIZER_FILE: str = "tfidf_vectorizer.pkl"

USER_INDEX_FILE: str = "users.index"
USER_ID_TO_IDX_FILE: str = "user_id_to_idx.json"

EVAL_CF_METRICS_FILE: str = "cf_metrics.json"
EVAL_HYBRID_METRICS_FILE: str = "hybrid_metrics.json"
EVAL_SUMMARY_FILE: str = "summary.json"

######################## MLFLOW CONSTANTS ######################
EXPT_NAME: str = os.environ["MLFLOW_EXPT_NAME"]