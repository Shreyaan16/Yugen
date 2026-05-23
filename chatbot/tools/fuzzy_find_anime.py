import re
import pandas as pd
from langchain_core.tools import tool


_STOPWORDS = {
    "the", "a", "an", "and", "of", "to", "in", "on", "for",
    "with", "season", "movie", "film", "ova", "special",
    "part", "episode", "series",
}


class FuzzyFindAnimeTool:
    """
    Fuzzy title search over the local anime DataFrame.

    Attributes
    ----------
    top_n           : max candidates to return
    min_score       : minimum rapidfuzz score to include a result
    ambiguous_delta : score gap below which clarification is requested
    """

    def __init__(self, anime_df: pd.DataFrame, cfg: dict) -> None:
        self._df             = anime_df
        self._top_n          = cfg["top_n"]
        self._min_score      = cfg["min_score"]
        self._ambiguous_delta= cfg["ambiguous_delta"]

    # ── Helpers ───────────────────────────────────────────────
    @staticmethod
    def _normalize(text: str) -> str:
        text = (text or "").lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+",         " ", text).strip()
        return " ".join(t for t in text.split() if t not in _STOPWORDS)

    # ── Core logic ────────────────────────────────────────────
    def run(self, query: str) -> dict:
        try:
            from rapidfuzz import fuzz, process
        except ImportError as exc:
            raise ImportError("rapidfuzz is required: pip install rapidfuzz") from exc

        titles      = self._df["title"].fillna("").astype(str).tolist()
        norm_titles = [self._normalize(t) for t in titles]
        norm_query  = self._normalize(query)

        if not norm_query:
            return {"query": query, "matches": [], "needs_clarification": True}

        raw = process.extract(
            norm_query, norm_titles,
            scorer=fuzz.token_set_ratio,
            limit=self._top_n,
        )

        results = []
        for _, score, idx in raw:
            if score < self._min_score:
                continue
            row = self._df.iloc[idx]
            results.append({
                "anime_id": int(row["anime_id"]) if "anime_id" in row else None,
                "title":    row.get("title"),
                "score":    int(score),
            })

        needs_clarification = (
            len(results) > 1
            and abs(results[0]["score"] - results[1]["score"]) <= self._ambiguous_delta
        )
        return {"query": query, "matches": results, "needs_clarification": needs_clarification}

    # ── LangChain tool ────────────────────────────────────────
    def as_tool(self):
        instance = self

        @tool("fuzzy_find_anime")
        def fuzzy_find_anime(query: str) -> dict:
            """Search for anime by name in the local dataset.

            Always call this first when a user mentions an anime by name.
            Returns up to 5 matches with anime_id, title, and match score.
            If needs_clarification is True, ask the user to pick one.

            Args:
                query: The anime title or partial name to search for.
            """
            return instance.run(query)

        return fuzzy_find_anime