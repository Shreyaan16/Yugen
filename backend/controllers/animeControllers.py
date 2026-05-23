import ast

import pandas as pd
from fastapi import HTTPException, Request
from jose import jwt, JWTError

from backend.constants import (
    SECRET_KEY, ALGORITHM,
    ALL_ANIME_DEFAULT_LIMIT, ALL_ANIME_MAX_LIMIT, TOP_K_SIMILAR,
)
from backend.services.store import store


# ── Utilities ─────────────────────────────────────────────────────────────────

def _parse_list(val) -> list:
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    return []


def _get_optional_user(request: Request) -> int | None:
    """Decode Bearer token; returns user_id or None if absent/invalid."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return int(payload["sub"])
    except (JWTError, ValueError, TypeError, KeyError):
        return None


def _to_card(row) -> dict:
    """Serialise a DataFrame row / dict to an anime card (title + genres)."""
    genres = row.get("genres", []) if isinstance(row, dict) else getattr(row, "genres", [])
    return {
        "anime_id": int(row["anime_id"]),
        "title":    str(row.get("title", "")),
        "genres":   _parse_list(genres),
    }


def _clean(value):
    """Replace float NaN with None for JSON serialisation."""
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


# ── Controllers ────────────────────────────────────────────────────────────────

def get_all_anime(
    page:   int = 1,
    limit:  int = ALL_ANIME_DEFAULT_LIMIT,
    search: str = "",
    genre:  str = "",
) -> dict:
    limit = min(limit, ALL_ANIME_MAX_LIMIT)
    df = store.anime_df

    if search:
        df = df[df["title"].str.contains(search, case=False, na=False)]

    if genre:
        gl = genre.lower()
        df = df[df["genres"].apply(lambda gs: any(gl == g.lower() for g in gs))]

    total = len(df)
    start = (page - 1) * limit
    page_df = df.iloc[start : start + limit]

    return {
        "total": total,
        "page":  page,
        "limit": limit,
        "anime": [_to_card(row) for _, row in page_df.iterrows()],
    }


def get_anime_detail(anime_id: int) -> dict:
    detail = store.get_detail(anime_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Anime not found")
    return {k: _clean(v) for k, v in detail.items()}


def get_similar_anime(anime_id: int, request: Request) -> list[dict]:
    user_id = _get_optional_user(request)

    if user_id is not None:
        # Logged-in → hybrid personalised recommendations
        try:
            df = store.hybrid.recommend(user_id)
            return [_to_card(row) for _, row in df.iterrows()]
        except Exception:
            # user not in index yet (registered after last train) → fall through
            pass

    # Guest or hybrid fallback → content-based similarity to this anime
    try:
        df = store.cb.recommend(anime_id, k=TOP_K_SIMILAR)
        return [_to_card(row) for _, row in df.iterrows()]
    except ValueError:
        raise HTTPException(status_code=404, detail="Anime not found in recommender index")


def chat(message: str, thread_id: str | None, request: Request) -> dict:
    if store.agent is None:
        raise HTTPException(
            status_code=503,
            detail="Chatbot unavailable: recommender artifacts not yet trained",
        )

    if thread_id is None:
        user_id = _get_optional_user(request)
        label   = str(user_id) if user_id else "anon"
        thread_id = store.agent.new_session(label)

    reply = store.agent.chat(message, thread_id=thread_id)
    return {"reply": reply, "thread_id": thread_id}
