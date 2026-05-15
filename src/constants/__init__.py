from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

ROOT_DIR: Path = Path(__file__).resolve().parents[2]

PIPELINE_NAME: str = "YuGen"
ARTIFACT_DIR: str = os.path.join(ROOT_DIR, "artifacts")

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

RANDOM_STATE: int = 42
TRAIN_RATIO: float = 0.8
VAL_RATIO: float = 0.1
TEST_RATIO: float = 0.1

COLS_TO_DROP_CONFIG: str = os.path.join(ROOT_DIR, "src", "config", "cols_to_drop.yaml")
FEATURE_BINS_CONFIG: str = os.path.join(ROOT_DIR, "src", "config", "feature_bins.yaml")

###################### RECOMMENDER ############################
RECOMMENDER_DIR_NAME: str = "recommender"
CONTENT_BASED_DIR_NAME: str = "content_based"

CONTENT_MAX_FEATURES: int = 15000
CONTENT_N_COMPONENTS: int = 256
CONTENT_HNSW_M: int = 32
CONTENT_EF_CONSTRUCTION: int = 200

CONTENT_INDEX_FILE: str = "content_faiss_index.bin"
CONTENT_MAL_TO_FAISS_FILE: str = "mal_id_to_faiss_id.json"
CONTENT_FAISS_TO_MAL_FILE: str = "faiss_id_to_mal_id.json"

USER_BASED_DIR_NAME: str = "user_based"
USER_HNSW_M: int = 32
USER_EF_CONSTRUCTION: int = 200
USER_INDEX_FILE: str = "user_faiss_index.bin"
USER_ID_TO_FAISS_FILE: str = "user_id_to_faiss_id.json"
USER_ID_TO_VECTOR_FILE: str = "user_id_to_vector.json"

HYBRID_DIR_NAME: str = "hybrid"
HYBRID_SIMILAR_USERS_K: int = 50
HYBRID_TOP_K: int = 10
HYBRID_RATINGS_MATRIX_FILE: str = "ratings_csr.npz"
SPLIT_FILE_NAME: str = "data.csv"
RATINGS_SPLIT_FILE_NAME: str = "ratings.csv"
ANIME_FILE_NAME: str = "anime.csv"
ANIME_SYNOPSIS_FILE_NAME: str = "anime_with_synopsis.csv"
RATINGS_FILE_NAME: str = "rating_complete.csv"