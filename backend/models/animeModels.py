from __future__ import annotations
from typing import Any
from pydantic import BaseModel

class ChatScheme(BaseModel):
    message:   str
    thread_id: str | None = None

class AnimeCardResponse(BaseModel):
    """Card shown in list / similar-anime views."""
    anime_id: int
    title:    str
    genres:   list[str]

class AnimeListResponse(BaseModel):
    """Paginated home-page response."""
    total: int
    page:  int
    limit: int
    anime: list[AnimeCardResponse]

class AnimeDetailResponse(BaseModel):
    """
    Full detail for a single anime.
    Fields sourced from the merged view of:
      • artifacts/data_preprocessing/anime.csv  (core metadata)
      • artifacts/data_ingestion/anime.csv       (engagement stats)
    All fields except anime_id are optional because not every anime row will
    have every column populated across both CSVs.
    """
    anime_id:  int
    title:     str | None = None
    # preprocessed / derived fields
    rating:    float | None = None
    synopsis:  str   | None = None
    ep_bin:    str   | None = None
    dur_bin:   str   | None = None
    era:       str   | None = None
    source_id: Any          = None
    genres:    list[str]    = []
    producers: list[str]    = []
    studios:   list[str]    = []
    # engagement stats from data_ingestion
    favorites: int   | None = None
    watching:  int   | None = None
    completed: int   | None = None


class ChatResponse(BaseModel):
    """Response returned by POST /chat."""
    reply:     str
    thread_id: str


class ChatMessageResponse(BaseModel):
    """A single turn in the chat history (GET /chat/{thread_id}/history)."""
    role:    str 
    content: str
