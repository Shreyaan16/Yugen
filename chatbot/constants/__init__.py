from pathlib import Path
import os
import yaml
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
_PROJECT_ROOT = Path(__file__).parent.parent.parent

with (_PROJECT_ROOT / "params.yaml").open("r", encoding="utf-8") as _f:
    _params = yaml.safe_load(_f)

_chatbot = _params["chatbot"]

MODEL            = _chatbot["model"]
FUZZY_SEARCH_CFG = _chatbot["fuzzy_search"]
SIMILARITY_CFG   = _chatbot["similarity_search"]
WEB_SEARCH_CFG   = _chatbot["web_search"]
MEMORY_CFG       = _chatbot["memory"]

_ARTIFACT_DIR = _PROJECT_ROOT / "artifacts"
ANIME_CSV    = str(_ARTIFACT_DIR / "data_preprocessing" / "anime.csv")
VECTORIZER   = str(_ARTIFACT_DIR / "recommender" / "content_based" / "tfidf_vectorizer.pkl")
FAISS_INDEX  = str(_ARTIFACT_DIR / "recommender" / "content_based" / "anime.index")
ID_MAP       = str(_ARTIFACT_DIR / "recommender" / "content_based" / "anime_id_to_idx.json")
FEATURE_BINS = str(_PROJECT_ROOT / "ingestion" / "config" / "feature_bins.yaml")

ARTIFACT_PATHS = {"anime_csv": ANIME_CSV,  "vectorizer": VECTORIZER, "faiss_index": FAISS_INDEX, "id_map": ID_MAP, "feature_bins": FEATURE_BINS}

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT"))
REDIS_DB = int(os.getenv("REDIS_DB"))