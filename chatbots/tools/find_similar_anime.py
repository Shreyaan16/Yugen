import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sklearn.preprocessing import normalize
from langchain_core.tools import tool
from chatbots.constants import ARTIFACT_PATHS, SIMILARITY_SEARCH_CFG


# ── Core logic ────────────────────────────────────────────────
def _join_genres(genres) -> str:
    if isinstance(genres, list):
        return " ".join(str(g) for g in genres)
    return str(genres) if genres is not None else ""


def fetch_similar_anime(
    description: str,
    genres: list[str],
    anime_df: pd.DataFrame | None = None,
    k: int = SIMILARITY_SEARCH_CFG["k"],
    vectorizer_path: str | Path = ARTIFACT_PATHS["vectorizer"],
    index_path: str | Path = ARTIFACT_PATHS["faiss_index"],
    id_map_path: str | Path = ARTIFACT_PATHS["id_map"],
) -> list[dict]:
    vectorizer_path = Path(vectorizer_path)
    index_path      = Path(index_path)
    id_map_path     = Path(id_map_path)

    missing = [str(p) for p in (vectorizer_path, index_path, id_map_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifact files: {missing}")

    with vectorizer_path.open("rb") as f:
        vectorizer = pickle.load(f)

    with id_map_path.open("r", encoding="utf-8") as f:
        anime_id_to_idx = {int(k): int(v) for k, v in json.load(f).items()}

    idx_to_anime_id = {v: k for k, v in anime_id_to_idx.items()}
    index = faiss.read_index(str(index_path))

    soup      = f"{description or ''} {_join_genres(genres)}".strip().lower()
    query     = vectorizer.transform([soup])
    query_vec = normalize(query, norm="l2", axis=1).astype(np.float32).toarray()

    title_by_id: dict = {}
    if anime_df is not None and "anime_id" in anime_df.columns and "title" in anime_df.columns:
        title_by_id = dict(zip(anime_df["anime_id"].astype(int), anime_df["title"]))

    scores, neighbors = index.search(query_vec, int(k))

    results = []
    for score, idx in zip(scores[0], neighbors[0]):
        if idx == -1:
            continue
        anime_id = idx_to_anime_id.get(int(idx))
        if anime_id is None:
            continue
        results.append({
            "anime_id":   int(anime_id),
            "title":      title_by_id.get(int(anime_id), ""),
            "similarity": float(score),
        })

    return results


# ── LangChain tool factory ────────────────────────────────────
def make_find_similar_tool(anime_df: pd.DataFrame):
    """Return a LangChain @tool bound to the given DataFrame."""

    @tool("find_similar_anime")
    def find_similar_anime(description: str, genres: list[str]) -> list[dict]:
        """Find anime similar to a given description and list of genres.

        Use when the user asks for recommendations or 'something like X'.
        Returns top 10 similar anime with anime_id, title, and similarity score.

        Args:
            description: Free-text description of what the user wants to watch.
            genres:      List of genre strings, e.g. ["Action", "Fantasy"].
        """
        return fetch_similar_anime(description, genres, anime_df=anime_df)

    return find_similar_anime