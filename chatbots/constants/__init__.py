from pathlib import Path
import os
import yaml
from dotenv import load_dotenv
 
load_dotenv()
 
# ── API Keys ──────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY")

 
# ── Params ────────────────────────────────────────────────────
_PARAMS_PATH = Path(__file__).parent.parent / "params.yaml"
 
with _PARAMS_PATH.open("r", encoding="utf-8") as _f:
    PARAMS: dict = yaml.safe_load(_f)
 
# ── Shortcuts into PARAMS (avoids repeated .get() chains) ─────
MODEL: str                   = PARAMS["model"]
ARTIFACT_PATHS: dict         = PARAMS["artifact_paths"]
FUZZY_SEARCH_CFG: dict       = PARAMS["fuzzy_search"]
SIMILARITY_SEARCH_CFG: dict  = PARAMS["similarity_search"]
WEB_SEARCH_CFG: dict         = PARAMS["web_search"]
MEMORY_CFG: dict             = PARAMS["memory"]