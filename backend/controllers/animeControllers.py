import os
import time
import logging
import threading
import pandas as pd
from fastapi import HTTPException
from backend.constants import ALL_ANIME_DEFAULT_LIMIT, ALL_ANIME_MAX_LIMIT, TOP_K_SIMILAR
from backend.services.store import store
from sqlalchemy.orm import Session
from backend.services import chat_history
from backend.utils import to_card, clean
from backend.models.authModels import UserRating, AnimeRating
from recommender.constants import (
    ARTIFACT_DIR, DATA_PREPROCESSING_DIR_NAME,
    USER_ANIME_RATINGS_FILE_NAME, RATINGS_FILE_NAME,
)
from sqlalchemy import func as sa_func, text

_ratings_lock = threading.Lock()
_UAR_CSV  = os.path.join(ARTIFACT_DIR, DATA_PREPROCESSING_DIR_NAME, USER_ANIME_RATINGS_FILE_NAME)
_STAT_CSV = os.path.join(ARTIFACT_DIR, DATA_PREPROCESSING_DIR_NAME, RATINGS_FILE_NAME)

# Bayesian-average damping constant (reverse-engineered from dataset, m ≈ 50)
_BAYES_M = 50

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

def rate_anime(anime_id: int, rating: int, user_id: int, db: Session) -> dict:
    # 1. Confirm the anime exists in our catalogue
    if store.anime_df[store.anime_df["anime_id"] == anime_id].empty:
        raise HTTPException(status_code=404, detail="Anime not found")

    # 2. Upsert into user_anime_ratings.
    #    The table was imported from CSV so it likely has no unique constraint on
    #    (user_id, anime_id) — ON CONFLICT would fail.  DELETE + INSERT is safe
    #    in all cases and leaves exactly one row per user-anime pair.
    db.execute(
        text("DELETE FROM user_anime_ratings WHERE user_id = :u AND anime_id = :a"),
        {"u": user_id, "a": anime_id},
    )
    db.execute(
        text("INSERT INTO user_anime_ratings (user_id, anime_id, rating) VALUES (:u, :a, :r)"),
        {"u": user_id, "a": anime_id, "r": rating},
    )
    db.commit()

    # 3. Recalculate aggregate stats from user_anime_ratings
    agg = db.execute(
        text("""
            SELECT COUNT(*) AS num_ratings, AVG(rating::float) AS mean_rating
            FROM user_anime_ratings
            WHERE anime_id = :anime_id
        """),
        {"anime_id": anime_id},
    ).one()
    new_num  = int(agg.num_ratings)
    new_mean = float(agg.mean_rating)

    # Bayesian average: score = (v*R + m*C) / (v+m)
    # C = weighted global mean = Σ(v*R) / Σv  across all anime
    stat_df = store.hybrid._ratings_df
    if not stat_df.empty and "num_ratings" in stat_df.columns:
        total_votes = stat_df["num_ratings"].sum()
        C = float((stat_df["num_ratings"] * stat_df["mean_rating"]).sum() / total_votes) if total_votes else new_mean
    else:
        C = new_mean
    new_score = (new_num * new_mean + _BAYES_M * C) / (new_num + _BAYES_M)

    # 4. Upsert ratings aggregate table.
    #    Table has no unique constraint (bulk-loaded from CSV), so ON CONFLICT
    #    won't work.  UPDATE first; INSERT only when no row existed yet.
    result = db.execute(
        text("""
            UPDATE ratings
            SET num_ratings      = :n,
                mean_rating      = :m,
                popularity_score = :s
            WHERE anime_id = :a
        """),
        {"a": anime_id, "n": new_num, "m": new_mean, "s": new_score},
    )
    if result.rowcount == 0:
        db.execute(
            text("""
                INSERT INTO ratings (anime_id, num_ratings, mean_rating, popularity_score)
                VALUES (:a, :n, :m, :s)
            """),
            {"a": anime_id, "n": new_num, "m": new_mean, "s": new_score},
        )
    db.commit()

    # 5. Mirror all changes in-memory + write both CSVs under a single lock
    with _ratings_lock:
        # --- user_anime_ratings ---
        uar  = store.user_ratings_df
        mask = (uar["user_id"] == user_id) & (uar["anime_id"] == anime_id)
        if mask.any():
            store.user_ratings_df.loc[mask, "rating"] = rating
        else:
            store.user_ratings_df = pd.concat(
                [uar, pd.DataFrame([{"user_id": user_id, "anime_id": anime_id, "rating": rating}])],
                ignore_index=True,
            )
        store.user_ratings_df.to_csv(_UAR_CSV, index=False)

        # --- ratings (aggregate stats) ---
        sdf   = store.hybrid._ratings_df
        smask = (sdf["anime_id"] == anime_id) if not sdf.empty else pd.Series([], dtype=bool)
        if not sdf.empty and smask.any():
            store.hybrid._ratings_df.loc[smask, "num_ratings"]      = new_num
            store.hybrid._ratings_df.loc[smask, "mean_rating"]      = new_mean
            store.hybrid._ratings_df.loc[smask, "popularity_score"] = new_score
        else:
            store.hybrid._ratings_df = pd.concat(
                [sdf, pd.DataFrame([{
                    "anime_id":         anime_id,
                    "num_ratings":      new_num,
                    "mean_rating":      new_mean,
                    "popularity_score": new_score,
                }])],
                ignore_index=True,
            )
        store.hybrid._ratings_df.to_csv(_STAT_CSV, index=False)

    return {"message": "Rating saved successfully", "anime_id": anime_id, "rating": rating}