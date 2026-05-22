import re
import pandas as pd
from langchain_core.tools import tool
from chatbots.constants import FUZZY_SEARCH_CFG

# ── Title normalisation ───────────────────────────────────────
_STOPWORDS = {
    "the", "a", "an", "and", "of", "to", "in", "on", "for",
    "with", "season", "movie", "film", "ova", "special",
    "part", "episode", "series",
}


def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(t for t in text.split() if t not in _STOPWORDS)


# ── Core logic (pure function, testable without LangChain) ────
def fuzzy_find_anime_titles(
    query: str,
    anime_df: pd.DataFrame,
    top_n: int = FUZZY_SEARCH_CFG["top_n"],
    min_score: int = FUZZY_SEARCH_CFG["min_score"],
    ambiguous_delta: int = FUZZY_SEARCH_CFG["ambiguous_delta"],
) -> dict:
    try:
        from rapidfuzz import fuzz, process
    except ImportError as exc:
        raise ImportError("rapidfuzz is required: pip install rapidfuzz") from exc

    if "title" not in anime_df.columns:
        raise KeyError("anime_df must include a 'title' column")

    titles = anime_df["title"].fillna("").astype(str).tolist()
    norm_titles = [_normalize(t) for t in titles]
    norm_query = _normalize(query)

    if not norm_query:
        return {"query": query, "matches": [], "needs_clarification": True}

    matches = process.extract(
        norm_query, norm_titles,
        scorer=fuzz.token_set_ratio,
        limit=top_n,
    )

    results = []
    for _, score, idx in matches:
        if score < min_score:
            continue
        row = anime_df.iloc[idx]
        results.append({
            "anime_id": int(row["anime_id"]) if "anime_id" in row else None,
            "title":    row.get("title"),
            "score":    int(score),
        })

    needs_clarification = (
        len(results) > 1
        and abs(results[0]["score"] - results[1]["score"]) <= ambiguous_delta
    )

    return {"query": query, "matches": results, "needs_clarification": needs_clarification}


# ── LangChain tool factory ─────────────────────────────────────
def make_fuzzy_find_tool(anime_df: pd.DataFrame):
    """Return a LangChain @tool bound to the given DataFrame."""

    @tool("fuzzy_find_anime")
    def fuzzy_find_anime(query: str) -> dict:
        """Search for anime by name in the local dataset.

        Always call this first when a user mentions an anime by name.
        Returns up to 5 matches with anime_id, title, and match score.
        If needs_clarification is True, ask the user to pick one.

        Args:
            query: The anime title or partial name to search for.
        """
        return fuzzy_find_anime_titles(query, anime_df)

    return fuzzy_find_anime