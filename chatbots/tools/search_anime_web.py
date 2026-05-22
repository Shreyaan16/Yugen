import asyncio
from typing import Optional
import httpx
from langchain_core.tools import tool

TAVILY_URL = "https://api.tavily.com/search"


class SearchAnimeWebTool:
    """
    Searches the web via Tavily for anime information not in the local dataset
    (characters, voice actors, sequels, release dates, news, etc.).
    """

    def __init__(self, tavily_api_key: str, cfg: dict) -> None:
        self._api_key        = tavily_api_key
        self._max_results    = cfg["max_results"]
        self._search_depth   = cfg["search_depth"]
        self._include_domains= cfg["include_domains"]

    # ── Async core ────────────────────────────────────────────
    async def _call_tavily(
        self,
        user_query: str,
        anime_title: str,
        anime_id: Optional[int],
    ) -> dict:
        parts        = [anime_title]
        if anime_id:
            parts.append(f"anime ID {anime_id}")
        parts       += ["anime", user_query]
        search_query = " ".join(parts)

        payload = {
            "api_key":             self._api_key,
            "query":               search_query,
            "search_depth":        self._search_depth,
            "max_results":         self._max_results,
            "include_answer":      True,
            "include_raw_content": False,
            "include_domains":     self._include_domains,
        }

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(TAVILY_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()

        return {
            "query_used": search_query,
            "answer":     data.get("answer", ""),
            "results": [
                {
                    "title":   r.get("title", ""),
                    "url":     r.get("url", ""),
                    "snippet": r.get("content", ""),
                    "score":   round(r.get("score", 0), 4),
                }
                for r in data.get("results", [])
            ],
        }

    # ── Sync wrapper ──────────────────────────────────────────
    def run(
        self,
        user_query: str,
        anime_title: str,
        anime_id: Optional[int] = None,
    ) -> dict:
        coro = self._call_tavily(user_query, anime_title, anime_id)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        # Inside an existing loop (Jupyter / FastAPI) → need nest_asyncio
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError as exc:
            raise RuntimeError(
                "nest_asyncio required in async contexts: pip install nest_asyncio"
            ) from exc
        return loop.run_until_complete(coro)

    # ── LangChain tool ────────────────────────────────────────
    def as_tool(self):
        instance = self

        @tool("search_anime_web")
        def search_anime_web(
            user_query: str,
            anime_title: str,
            anime_id: int = None,
        ) -> dict:
            """Search the web for anime info not available in the local dataset.

            Use when the user asks about characters, voice actors, sequels,
            release schedule, news, or anything not in get_anime_info.

            Args:
                user_query:  The specific question or topic to search for.
                anime_title: The anime title (required).
                anime_id:    The anime ID (optional, improves accuracy).
            """
            return instance.run(user_query, anime_title, anime_id)

        return search_anime_web