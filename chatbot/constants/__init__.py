from pathlib import Path
import os
import yaml
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not set in environment / .env")
if not TAVILY_API_KEY:
    raise EnvironmentError("TAVILY_API_KEY not set in environment / .env")

# ── Load params.yaml from project root ────────────────────────
# __file__ → chatbots/constants/__init__.py
# .parent  → chatbots/constants/
# .parent  → chatbots/
# .parent  → project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_PARAMS_PATH  = _PROJECT_ROOT / "params.yaml"

if not _PARAMS_PATH.exists():
    raise FileNotFoundError(f"params.yaml not found at: {_PARAMS_PATH}")

with _PARAMS_PATH.open("r", encoding="utf-8") as _f:
    _ALL: dict = yaml.safe_load(_f)

# ── Chatbot config block ──────────────────────────────────────
CFG: dict = _ALL["chatbot"]

MODEL:                str  = CFG["model"]
ARTIFACT_PATHS:       dict = CFG["artifact_paths"]
FUZZY_SEARCH_CFG:     dict = CFG["fuzzy_search"]
SIMILARITY_CFG:       dict = CFG["similarity_search"]
WEB_SEARCH_CFG:       dict = CFG["web_search"]
MEMORY_CFG:           dict = CFG["memory"]

# Resolve artifact paths relative to project root
ARTIFACT_PATHS = {k: str(_PROJECT_ROOT / v) for k, v in ARTIFACT_PATHS.items()}