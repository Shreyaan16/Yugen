from dotenv import load_dotenv
import os
from pathlib import Path
load_dotenv()

ROOT_DIR: Path = Path(__file__).resolve().parents[2]
ARTIFACT_DIR: str = os.path.join(ROOT_DIR, "artifacts")
PARAMS_PATH: str = os.path.join(ROOT_DIR, "params.yaml")

###################### DIR NAMES ############################
DATA_PREPROCESSING_DIR_NAME: str = "data_preprocessing"
RECOMMENDER_DIR_NAME: str = "recommender"
CONTENT_BASED_DIR_NAME: str = "content_based"
USER_BASED_DIR_NAME: str = "user_based"
EVALUATION_DIR_NAME: str = "evaluation"

######################## COLUMN NAMES ########################
COLUMN_NAMES: list[str] = ["title", "synopsis", "rating", "ep_bin", "dur_bin", "era", "source", "genres", "producers", "studios"]
LIST_COLUMNS: set[str] = {"genres", "producers", "studios"}

###################### FILE NAMES ############################
ANIME_FILE_NAME: str = "anime.csv"
USERS_FILE_NAME: str = "users.csv"
USER_ANIME_RATINGS_FILE_NAME: str = "user_anime_ratings.csv"
RATINGS_FILE_NAME: str = "ratings.csv"

CONTENT_INDEX_FILE: str = "anime.index"
ANIME_ID_TO_IDX_FILE: str = "anime_id_to_idx.json"
TFIDF_VECTORIZER_FILE: str = "tfidf_vectorizer.pkl"
TFIDF_MATRIX_FILE: str = "tfidf_matrix.npz"

USER_INDEX_FILE: str = "users.index"
USER_ID_TO_IDX_FILE: str = "user_id_to_idx.json"

EVAL_CONTENT_DIR_NAME: str = "content_based"
EVAL_USER_DIR_NAME: str = "user_based"
EVAL_HYBRID_DIR_NAME: str = "hybrid"

EVAL_METRICS_FILE: str = "metrics.json"
EVAL_SUMMARY_FILE: str = "summary.json"

######################## MLFLOW CONSTANTS ######################
EXPT_NAME: str = os.environ["MLFLOW_EXPT_NAME"]