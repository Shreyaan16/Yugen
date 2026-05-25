import time
import logging
from fastapi import HTTPException
from backend.constants import ALL_ANIME_DEFAULT_LIMIT, ALL_ANIME_MAX_LIMIT, TOP_K_SIMILAR
from backend.services.store import store
from backend.services import chat_history
from backend.utils import to_card, clean

log = logging.getLogger(__name__)

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
        df = df[df["genres"].apply(lambda gs: any(gl == str(g).lower() for g in gs) if isinstance(gs, list) else False)]
    total = len(df)
    start = (page - 1) * limit
    page_df = df.iloc[start : start + limit]
    return {
        "total": total,
        "page":  page,
        "limit": limit,
        "anime": [to_card(row) for _, row in page_df.iterrows()],
    }


def get_anime_detail(anime_id: int) -> dict:
    detail = store.get_detail(anime_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Anime not found")
    return {k: clean(v) for k, v in detail.items()}


def get_similar_anime(anime_id: int, user_id: int | None) -> list[dict]:
    if user_id is not None:
        # Logged-in → hybrid personalised recommendations
        try:
            t0 = time.perf_counter()
            df = store.hybrid.recommend(user_id)
            log.info("hybrid.recommend(%s) took %.3fs, returned %d rows", user_id, time.perf_counter() - t0, len(df))
            return [to_card(row) for _, row in df.iterrows()]
        except Exception as exc:
            log.warning("hybrid.recommend failed (%s), falling back to CB", exc)
    # Guest (or hybrid fallback) → content-based similarity
    try:
        t0 = time.perf_counter()
        df = store.cb.recommend(anime_id, k=TOP_K_SIMILAR)
        log.info("cb.recommend(%s) took %.3fs", anime_id, time.perf_counter() - t0)
        return [to_card(row) for _, row in df.iterrows()]
    except ValueError:
        raise HTTPException(status_code=404, detail="Anime not found in recommender index")


def chat(message: str, thread_id: str | None, user_id: int | None) -> dict:
    if store.agent is None:
        raise HTTPException(status_code=503, detail="Chatbot unavailable: recommender artifacts not yet trained")
    if thread_id is None:
        label = str(user_id) if user_id else "anon"
        thread_id = store.agent.new_session(label)
    reply = store.agent.chat(message, thread_id=thread_id)
    chat_history.append_message(thread_id, "human", message)
    chat_history.append_message(thread_id, "ai", reply)
    return {"reply": reply, "thread_id": thread_id}


def get_chat_history(thread_id: str) -> list[dict]:
    if store.agent is None:
        raise HTTPException(status_code=503, detail="Chatbot unavailable: recommender artifacts not yet trained")
    history = chat_history.get_history(thread_id)
    return history if history else store.agent.get_history(thread_id)
