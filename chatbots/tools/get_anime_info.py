import ast
from pathlib import Path
import pandas as pd
import yaml
from langchain_core.tools import tool
from chatbots.constants import ARTIFACT_PATHS


# ── Helpers ───────────────────────────────────────────────────
def _load_feature_bins(config_path: str | Path = ARTIFACT_PATHS["feature_bins"]) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"feature_bins config not found at: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_na(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def _build_range_labels(bins: list) -> list[str]:
    ranges = []
    for start, end in zip(bins[:-1], bins[1:]):
        s = int(start) if float(start).is_integer() else start
        if end == float("inf"):
            ranges.append(f"{s}+")
        else:
            e = int(end) if float(end).is_integer() else end
            ranges.append(f"{s}-{e}")
    return ranges


def _map_bin_value(value, cfg: dict) -> str:
    if _is_na(value):
        return ""
    labels      = cfg.get("labels", [])
    bins        = cfg.get("bins", [])
    text_labels = cfg.get("text_labels")

    if not labels or not bins:
        return value

    range_labels  = _build_range_labels(bins)
    label_to_range = dict(zip(labels, range_labels))
    text_to_range  = dict(zip(text_labels, range_labels)) if text_labels else {}

    return label_to_range.get(value) or text_to_range.get(value) or value


# ── Core logic ────────────────────────────────────────────────
_FIELDS = ["anime_id", "title", "rating", "synopsis",
           "ep_bin", "dur_bin", "era", "source",
           "genres", "producers", "studios"]

_LIST_COLS = ["genres", "producers", "studios"]


def fetch_anime_info(
    anime_df: pd.DataFrame,
    anime_id: int,
    config: dict | None = None,
) -> dict:
    if "anime_id" not in anime_df.columns:
        raise KeyError("anime_df must include an 'anime_id' column")

    if config is None:
        config = _load_feature_bins()

    row = anime_df.loc[anime_df["anime_id"] == int(anime_id)]
    if row.empty:
        return {"error": "Anime not found", "anime_id": anime_id}

    record = row.iloc[0][_FIELDS].to_dict()

    for col in _LIST_COLS:
        val = record.get(col)
        if isinstance(val, str):
            try:
                record[col] = ast.literal_eval(val)
            except (SyntaxError, ValueError):
                record[col] = []
        elif _is_na(val):
            record[col] = []

    if _is_na(record.get("synopsis")):
        record["synopsis"] = ""

    record["ep_bin"]  = _map_bin_value(record.get("ep_bin"),  config.get("ep_bin",  {}))
    record["dur_bin"] = _map_bin_value(record.get("dur_bin"), config.get("dur_bin", {}))
    record["era"]     = _map_bin_value(record.get("era"),     config.get("era",     {}))

    return record


# ── LangChain tool factory ────────────────────────────────────
def make_get_anime_info_tool(anime_df: pd.DataFrame):
    """Return a LangChain @tool bound to the given DataFrame."""

    @tool("get_anime_info")
    def get_anime_info(anime_id: int) -> dict:
        """Get detailed information about an anime from the local dataset.

        Returns title, synopsis, rating, episode count, duration, era,
        source, genres, producers, and studios.

        Args:
            anime_id: The integer anime ID (obtain via fuzzy_find_anime first).
        """
        return fetch_anime_info(anime_df, anime_id)

    return get_anime_info