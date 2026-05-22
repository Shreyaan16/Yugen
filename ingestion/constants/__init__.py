from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

ROOT_DIR: Path = Path(__file__).resolve().parents[2]

ARTIFACT_DIR: str = os.path.join(ROOT_DIR, "artifacts")

PARAMS_PATH: str = os.path.join(ROOT_DIR, "params.yaml")
COLS_TO_DROP_CONFIG: str = os.path.join(ROOT_DIR, "ingestion", "config", "cols_to_drop.yaml")
FEATURE_BINS_CONFIG: str = os.path.join(ROOT_DIR, "ingestion", "config", "feature_bins.yaml")

###################### AZURE ENV ############################
CONNECTION_STRING: str = os.environ["CONNECTION_STRING"]
CONTAINER_NAME: str = os.environ["V2_CONTAINER_NAME"]

###################### POSTGRES ENV #########################
DATABASE_URL: str = os.environ["DATABASE_URL"]
SCHEMA_NAME: str = "public"

###################### DIR NAMES ############################
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_PREPROCESSING_DIR_NAME: str = "data_preprocessing"


######################## TABLE NAME #########################
NAMES = ["anime", "anime_genres", "anime_producers", "anime_studios", "genres", "producers", "studios", "sources", "ratings", "users", "user_anime_ratings"]


###################### FILE NAMES ###########################
ANIME_FILE_NAME: str = "anime.csv"
USERS_FILE_NAME: str = "users.csv"
USER_ANIME_RATINGS_FILE_NAME: str = "user_anime_ratings.csv"
RATINGS_FILE_NAME: str = "ratings.csv"
