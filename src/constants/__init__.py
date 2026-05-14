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
