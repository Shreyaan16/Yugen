import ast
from pathlib import Path
import pandas as pd
import yaml
from langchain_core.tools import tool


_FIELDS   = ["anime_id", "title", "rating", "synopsis",
             "ep_bin", "dur_bin", "era", "source",
             "genres", "producers", "studios"]
_LIST_COLS = ["genres", "producers", "studios"]


class GetAnimeInfoTool:
    """
    Fetches a single anime record from the local DataFrame and decodes
    binned columns (ep_bin, dur_bin, era) into human-readable ranges.
    """

    def __init__(self, anime_df: pd.DataFrame, feature_bins_path: str) -> None:
        self._df     = anime_df
        self._config = self._load_bins(feature_bins_path)

    # ── Helpers ───────────────────────────────────────────────
    @staticmethod
    def _load_bins(path: str) -> dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"feature_bins config not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _is_na(value) -> bool:
        return value is None or (isinstance(value, float) and pd.isna(value))

    @staticmethod
    def _range_labels(bins: list) -> list[str]:
        out = []
        for start, end in zip(bins[:-1], bins[1:]):
            s = int(start) if float(start).is_integer() else start
            out.append(f"{s}+" if end == float("inf")
                       else f"{s}-{int(end) if float(end).is_integer() else end}")
        return out

    def _decode_bin(self, value, cfg: dict) -> str:
        if self._is_na(value):
            return ""
        labels = cfg.get("labels", [])
        bins   = cfg.get("bins",   [])
        if not labels or not bins:
            return value
        ranges       = self._range_labels(bins)
        label_map    = dict(zip(labels, ranges))
        text_map     = dict(zip(cfg.get("text_labels", []), ranges))
        return label_map.get(value) or text_map.get(value) or value

    # ── Core logic ────────────────────────────────────────────
    def run(self, anime_id: int) -> dict:
        row = self._df.loc[self._df["anime_id"] == int(anime_id)]
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
            elif self._is_na(val):
                record[col] = []

        if self._is_na(record.get("synopsis")):
            record["synopsis"] = ""

        record["ep_bin"]  = self._decode_bin(record.get("ep_bin"),  self._config.get("ep_bin",  {}))
        record["dur_bin"] = self._decode_bin(record.get("dur_bin"), self._config.get("dur_bin", {}))
        record["era"]     = self._decode_bin(record.get("era"),     self._config.get("era",     {}))

        return record

    # ── LangChain tool ────────────────────────────────────────
    def as_tool(self):
        instance = self

        @tool("get_anime_info")
        def get_anime_info(anime_id: int) -> dict:
            """Get detailed information about an anime from the local dataset.

            Returns title, synopsis, rating, episode count, duration, era,
            source, genres, producers, and studios.

            Args:
                anime_id: The integer anime ID (obtain via fuzzy_find_anime first).
            """
            return instance.run(anime_id)

        return get_anime_info