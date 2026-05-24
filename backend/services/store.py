"""
store.py — Application-level singleton.

All heavy I/O (CSVs, FAISS indexes, AnimeAgent) happens once at import time
so every request reuses the same in-memory objects.
"""

import ast
import os
import threading
import pandas as pd
from recommender.constants import ARTIFACT_DIR, DATA_PREPROCESSING_DIR_NAME, ANIME_FILE_NAME, USER_ANIME_RATINGS_FILE_NAME
from recommender.components.content_based import ContentBasedRecommender
from recommender.components.hybrid import HybridRecommender
from chatbot.agent import AnimeAgent

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
        self.anime_df: pd.DataFrame = pd.read_csv(os.path.join(_PREPROCESSED_DIR, ANIME_FILE_NAME))
        for col in ("genres", "producers", "studios"):
            if col in self.anime_df.columns:
                self.anime_df[col] = self.anime_df[col].apply(_parse_list)
        # ── User-anime ratings CSV ─────────────────────────────────────────
        self.user_ratings_df: pd.DataFrame = pd.read_csv(os.path.join(_PREPROCESSED_DIR, USER_ANIME_RATINGS_FILE_NAME))
        # ── User details CSV (from ingestion — has user_id, username, email) ─
        self.user_df: pd.DataFrame = pd.read_csv(os.path.join(_INGESTION_DIR, "users.csv"))

        # ── Ingestion anime CSV — only grab columns not already in preprocessed ──
        # (ingestion has title/synopsis/etc. too — loading them would cause _x/_y duplicates)
        _ing = (
            pd.read_csv(os.path.join(_INGESTION_DIR, "anime.csv"),
                        usecols=["id", "source_id", "favorites", "watching", "completed"])
            .rename(columns={"id": "anime_id"})
        )
        # ── Merged detail view — preprocessed cols + ingestion-only cols ──────
        # Drop 'source' from preprocessed (ingestion has the cleaner 'source_id')
        _prep_cols = [c for c in self.anime_df.columns if c != "source"]
        self._detail_df: pd.DataFrame = self.anime_df[_prep_cols].merge(_ing, on="anime_id", how="left")
        # ── Content-based recommender ──────────────────────────────────────
        self.cb: ContentBasedRecommender = ContentBasedRecommender()
        # ── Hybrid recommender ────────────────────────────────────────────
        self.hybrid: HybridRecommender = HybridRecommender()
        try:
            self.agent = AnimeAgent()
        except Exception as exc:
            print(f"[store] AnimeAgent init failed (chatbot unavailable): {exc}")
            self.agent = None

        # ── Pre-warm recommenders in background so first request is instant ─
        threading.Thread(target=self._prewarm, daemon=True).start()

    # ── Background pre-warm ───────────────────────────────────────────────────
    def _prewarm(self) -> None:
        try:
            print("[store] Pre-warming content-based recommender…")
            self.cb._load_if_needed()      # load CB FAISS + reconstruct dense vectors
            self.hybrid._content = self.cb # inject into hybrid — no user FAISS needed
            print("[store] Recommenders ready. (User FAISS not loaded — not needed for serving)")
        except Exception as exc:
            print(f"[store] Pre-warm failed: {exc}")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def get_detail(self, anime_id: int) -> dict | None:
        row = self._detail_df.loc[self._detail_df["anime_id"] == anime_id]
        if row.empty:
            return None
        deduped = row.loc[:, ~row.columns.duplicated()]
        return deduped.iloc[0].to_dict()

# Module-level singleton — created once when the module is first imported.
store = _AppStore()
