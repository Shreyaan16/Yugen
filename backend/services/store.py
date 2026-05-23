"""
store.py — Application-level singleton.

All heavy I/O (CSVs, FAISS indexes, AnimeAgent) happens once at import time
so every request reuses the same in-memory objects.
"""

import ast
import os
import pandas as pd

from recommender.constants import ARTIFACT_DIR, DATA_PREPROCESSING_DIR_NAME
from recommender.components.content_based import ContentBasedRecommender
from recommender.components.hybrid import HybridRecommender

_PREPROCESSED_DIR = os.path.join(ARTIFACT_DIR, DATA_PREPROCESSING_DIR_NAME)
_INGESTION_DIR    = os.path.join(ARTIFACT_DIR, "data_ingestion")


def _parse_list(val) -> list:
    """Convert a CSV-serialised list string back to a Python list."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    return []


class _AppStore:
    def __init__(self) -> None:
        # ── Preprocessed anime CSV ─────────────────────────────────────────
        self.anime_df: pd.DataFrame = pd.read_csv(
            os.path.join(_PREPROCESSED_DIR, "anime.csv")
        )
        for col in ("genres", "producers", "studios"):
            if col in self.anime_df.columns:
                self.anime_df[col] = self.anime_df[col].apply(_parse_list)

        # ── Ingestion anime CSV (extras: favorites, watching, completed) ────
        _ing = pd.read_csv(
            os.path.join(_INGESTION_DIR, "anime.csv"),
            usecols=["id", "favorites", "watching", "completed"],
        ).rename(columns={"id": "anime_id"})

        # ── Merged detail view ────────────────────────────────────────────
        self._detail_df: pd.DataFrame = self.anime_df.merge(
            _ing, on="anime_id", how="left"
        )

        # ── Content-based recommender (lazy-loads FAISS on first use) ─────
        self.cb: ContentBasedRecommender = ContentBasedRecommender()

        # ── Hybrid recommender (lazy-loads indexes on first use) ──────────
        self.hybrid: HybridRecommender = HybridRecommender()

        # ── Chatbot agent (loads FAISS + vectorizer at startup) ───────────
        # Wrapped so the API starts even if recommender hasn't been trained.
        try:
            from chatbot.agent import AnimeAgent
            self.agent = AnimeAgent()
        except Exception as exc:
            print(f"[store] AnimeAgent init failed (chatbot unavailable): {exc}")
            self.agent = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_detail(self, anime_id: int) -> dict | None:
        row = self._detail_df.loc[self._detail_df["anime_id"] == anime_id]
        if row.empty:
            return None
        return row.iloc[0].to_dict()


# Module-level singleton — created once when the module is first imported.
store = _AppStore()
