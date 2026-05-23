from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from backend.constants import ALL_ANIME_DEFAULT_LIMIT
from backend.controllers.animeControllers import (
    get_all_anime,
    get_anime_detail,
    get_similar_anime,
    chat,
)

router = APIRouter()


class ChatRequest(BaseModel):
    message:   str
    thread_id: str | None = None


# ── Home page ─────────────────────────────────────────────────────────────────

@router.get("/anime")
def list_anime(
    page:   int = Query(1, ge=1, description="Page number"),
    limit:  int = Query(ALL_ANIME_DEFAULT_LIMIT, ge=1, description="Results per page"),
    search: str = Query("", description="Partial title search"),
    genre:  str = Query("", description="Filter by exact genre name"),
):
    """All anime — paginated cards (title + genres). Auth optional."""
    return get_all_anime(page=page, limit=limit, search=search, genre=genre)


# ── Anime detail page ─────────────────────────────────────────────────────────

@router.get("/anime/{anime_id}")
def anime_detail(anime_id: int):
    """Full anime info + favorites/watching/completed. Auth optional."""
    return get_anime_detail(anime_id)


# ── Similar anime tab ─────────────────────────────────────────────────────────

@router.get("/anime/{anime_id}/similar")
def similar_anime(anime_id: int, request: Request):
    """
    Logged-in  → hybrid personalised recommendations.
    Guest      → content-based anime similarity.
    Returns cards (title + genres).
    """
    return get_similar_anime(anime_id, request)


# ── Chatbot ───────────────────────────────────────────────────────────────────

@router.post("/chat")
def chatbot(body: ChatRequest, request: Request):
    """
    Send a message to the anime chatbot.
    Omit thread_id on the first call — a new session is created and returned.
    Pass the returned thread_id on all subsequent calls to preserve context.
    """
    return chat(body.message, body.thread_id, request)
