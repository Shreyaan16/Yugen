import asyncio
from typing import Optional
import httpx
from langchain_core.tools import tool
from chatbots.constants import TAVILY_API_KEY, WEB_SEARCH_CFG

TAVILY_URL = "https://api.tavily.com/search"


# ── Core async logic ──────────────────────────────────────────
async def _search_tavily(
    user_query: str,
    anime_title: str,
    anime_id: Optional[int] = None,
    max_results: int = WEB_SEARCH_CFG["max_results"],
    search_depth: str = WEB_SEARCH_CFG["search_depth"],
    include_domains: list[str] = WEB_SEARCH_CFG["include_domains"],
) -> dict:
    parts = [anime_title]
    if anime_id:
        parts.append(f"anime ID {anime_id}")
    parts += ["anime", user_query]
    search_query = " ".join(parts)

    payload = {
        "api_key":           TAVILY_API_KEY,
        "query":             search_query,
        "search_depth":      search_depth,
        "max_results":       max_results,
        "include_answer":    True,
        "include_raw_content": False,
        "include_domains":   include_domains,
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


# ── Sync wrapper (handles both notebook and server loops) ─────
def search_anime_info(
    user_query: str,
    anime_title: str,
    anime_id: Optional[int] = None,
) -> dict:
    coro = _search_tavily(user_query, anime_title, anime_id)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Running inside an existing loop (Jupyter / FastAPI) → need nest_asyncio
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError as exc:
        raise RuntimeError(
            "nest_asyncio is required in async contexts. "
            "Install with: pip install nest_asyncio"
        ) from exc
    return loop.run_until_complete(coro)


# ── LangChain tool factory ────────────────────────────────────
def make_search_anime_web_tool():
    """Return the web-search LangChain @tool (no DataFrame needed)."""

    @tool("search_anime_web")
    def search_anime_web(
        user_query: str,
        anime_title: str,
        anime_id: int = None,
    ) -> dict:
        """Search the web for anime info not available in the local dataset.

        Use when the user asks about details not in get_anime_info
        (characters, voice actors, sequels, release schedule, news, etc.).

        Args:
            user_query:  The specific question or topic to search for.
            anime_title: The anime title (required).
            anime_id:    The anime ID (optional, improves accuracy).
        """
        return search_anime_info(user_query, anime_title, anime_id)

    return search_anime_web