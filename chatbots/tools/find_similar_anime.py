import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from sklearn.preprocessing import normalize
from langchain_core.tools import tool


class FindSimilarAnimeTool:
    """
    Content-based anime recommender backed by a TF-IDF + FAISS index.
    Accepts a free-text description and a list of genres, returns the top-k
    most similar anime from the index.
    """

    def __init__(
        self,
        anime_df: pd.DataFrame,
        cfg: dict,
        vectorizer_path: str,
        index_path: str,
        id_map_path: str,
    ) -> None:
        self._k        = cfg["k"]
        self._title_map= self._build_title_map(anime_df)

        # Load artifacts once at construction time
        self._vectorizer = self._load_pickle(vectorizer_path)
        self._index      = faiss.read_index(index_path)
        self._idx_to_id  = self._load_id_map(id_map_path)

    # ── Helpers ───────────────────────────────────────────────
    @staticmethod
    def _load_pickle(path: str):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Vectorizer not found: {p}")
        with p.open("rb") as f:
            return pickle.load(f)

    @staticmethod
    def _load_id_map(path: str) -> dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"ID map not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        # stored as anime_id → idx; we need idx → anime_id
        return {int(v): int(k) for k, v in raw.items()}

    @staticmethod
    def _build_title_map(anime_df: pd.DataFrame) -> dict:
        if anime_df is not None and {"anime_id", "title"} <= set(anime_df.columns):
            return dict(zip(anime_df["anime_id"].astype(int), anime_df["title"]))
        return {}

    @staticmethod
    def _join_genres(genres) -> str:
        if isinstance(genres, list):
            return " ".join(str(g) for g in genres)
        return str(genres) if genres is not None else ""

    # ── Core logic ────────────────────────────────────────────
    def run(self, description: str, genres: list[str]) -> list[dict]:
        soup      = f"{description or ''} {self._join_genres(genres)}".strip().lower()
        query     = self._vectorizer.transform([soup])
        query_vec = normalize(query, norm="l2", axis=1).astype(np.float32).toarray()

        scores, neighbors = self._index.search(query_vec, int(self._k))

        results = []
        for score, idx in zip(scores[0], neighbors[0]):
            if idx == -1:
                continue
            anime_id = self._idx_to_id.get(int(idx))
            if anime_id is None:
                continue
            results.append({
                "anime_id":   int(anime_id),
                "title":      self._title_map.get(int(anime_id), ""),
                "similarity": float(score),
            })
        return results

    # ── LangChain tool ────────────────────────────────────────
    def as_tool(self):
        instance = self

        @tool("find_similar_anime")
        def find_similar_anime(description: str, genres: list[str]) -> list[dict]:
            """Find anime similar to a given description and list of genres.

            Use when the user asks for recommendations or 'something like X'.
            Returns top-k anime with anime_id, title, and similarity score.

            Args:
                description: Free-text description of what the user wants to watch.
                genres:      List of genre strings, e.g. ["Action", "Fantasy"].
            """
            return instance.run(description, genres)

        return find_similar_anime