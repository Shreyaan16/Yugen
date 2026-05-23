from fastapi import APIRouter, Depends, Query
from backend.models.animeModels import *
from backend.constants import ALL_ANIME_DEFAULT_LIMIT
from backend.utils import get_optional_user
from backend.controllers.animeControllers import *

router = APIRouter()

@router.get("/anime", response_model=AnimeListResponse)
def list_anime(
    page:   int = Query(1, ge=1),
    limit:  int = Query(ALL_ANIME_DEFAULT_LIMIT, ge=1),
    search: str = Query(""),
    genre:  str = Query(""),
):
    """All anime — paginated cards (title + genres). Auth optional."""
    return get_all_anime(page=page, limit=limit, search=search, genre=genre)


@router.get("/anime/{anime_id}", response_model=AnimeDetailResponse)
def anime_detail(anime_id: int):
    """Full anime info + favorites / watching / completed. Auth optional."""
    return get_anime_detail(anime_id)


@router.get("/anime/{anime_id}/similar", response_model=list[AnimeCardResponse])
def similar_anime(anime_id: int, user_id: int | None = Depends(get_optional_user)):
    """
    Logged-in  → hybrid personalised recommendations.
    Guest      → content-based anime similarity.
    Returns cards (title + genres).
    """
    return get_similar_anime(anime_id, user_id)


@router.post("/chat", response_model=ChatResponse)
def chatbot(
    body: ChatScheme,
    user_id: int | None = Depends(get_optional_user),
):
    """
    Send a message to the anime chatbot.
    Omit thread_id on the first call — a new session is created and returned.
    Pass the returned thread_id on subsequent calls to keep context.
    """
    return chat(body.message, body.thread_id, user_id)


@router.get("/chat/{thread_id}/history", response_model=list[ChatMessageResponse])
def chat_history(thread_id: str):
    """
    Returns all messages from the current chat session in order.
    Each entry has role ('human' | 'ai' | 'tool') and content.
    """
    return get_chat_history(thread_id)
