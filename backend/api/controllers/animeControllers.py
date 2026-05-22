from collections import defaultdict
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from backend.api.constants import *
from backend.api.controllers.authControllers import get_db
from backend.api.models.authModels import RateSchema, User, UserRating

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=True)


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    creds_exc = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        sub = payload.get("sub")
        if sub is None:
            raise creds_exc
        user_id = int(sub)
    except (JWTError, ValueError):
        raise creds_exc

    user = db.query(User).filter(User.user_id == user_id).first()
    if user is None:
        raise creds_exc
    return user


def list_genres(db: Session) -> dict[str, Any]:
    rows = db.execute(text("SELECT id, genre_name FROM genres WHERE genre_name <> 'Unknown' ORDER BY genre_name")).mappings().all()
    return {"items": [dict(r) for r in rows]}

def list_all_anime(db: Session, limit: int, offset: int) -> dict[str, Any]:
    limit = max(1, min(limit, ALL_ANIME_MAX_LIMIT))
    offset = max(0, offset)

    total = db.execute(text("SELECT COUNT(*) FROM anime")).scalar_one()

    anime_rows = db.execute(
        text(
            """
            SELECT a.id, a.title, a.era, a.rating,
                   r.num_ratings, r.mean_rating, r.popularity_score
            FROM anime a
            LEFT JOIN ratings r ON r.anime_id = a.id
            ORDER BY COALESCE(r.popularity_score, 0) DESC, a.id ASC
            LIMIT :limit OFFSET :offset
            """
        ),
        {"limit": limit, "offset": offset},
    ).mappings().all()

    ids = [row["id"] for row in anime_rows]
    genres_by_anime: dict[int, list[str]] = defaultdict(list)
    if ids:
        genre_stmt = text(
            """
            SELECT ag.anime_id, g.genre_name
            FROM anime_genres ag
            JOIN genres g ON g.id = ag.genre_id
            WHERE ag.anime_id IN :ids
            ORDER BY g.genre_name
            """
        ).bindparams(bindparam("ids", expanding=True))
        for r in db.execute(genre_stmt, {"ids": ids}).fetchall():
            genres_by_anime[r[0]].append(r[1])

    items = [
        {**dict(row), "genres": genres_by_anime.get(row["id"], [])}
        for row in anime_rows
    ]
    return {"total": total, "limit": limit, "offset": offset, "items": items}


def get_anime_detail(anime_id: int, db: Session) -> dict[str, Any]:
    anime = db.execute(
        text(
            """
            SELECT a.id, a.title, a.synopsis, a.era, a.rating, a.ep_bin, a.dur_bin,
                   s.source_name AS source,
                   r.num_ratings, r.mean_rating, r.popularity_score
            FROM anime a
            LEFT JOIN sources s ON s.id = a.source_id
            LEFT JOIN ratings r ON r.anime_id = a.id
            WHERE a.id = :id
            """
        ),
        {"id": anime_id},
    ).mappings().first()
    if anime is None:
        raise HTTPException(status_code=404, detail="Anime not found")

    genres = [
        r[0]
        for r in db.execute(
            text(
                """
                SELECT g.genre_name FROM anime_genres ag
                JOIN genres g ON g.id = ag.genre_id
                WHERE ag.anime_id = :id ORDER BY g.genre_name
                """
            ),
            {"id": anime_id},
        ).fetchall()
    ]

    studios = [
        r[0]
        for r in db.execute(
            text(
                """
                SELECT st.studio_name FROM anime_studios ast
                JOIN studios st ON st.id = ast.studio_id
                WHERE ast.anime_id = :id ORDER BY st.studio_name
                """
            ),
            {"id": anime_id},
        ).fetchall()
    ]

    producers = [
        r[0]
        for r in db.execute(
            text(
                """
                SELECT p.producer_name FROM anime_producers ap
                JOIN producers p ON p.id = ap.producer_id
                WHERE ap.anime_id = :id ORDER BY p.producer_name
                """
            ),
            {"id": anime_id},
        ).fetchall()
    ]

    return {**dict(anime), "genres": genres, "studios": studios, "producers": producers}


def rate_anime(payload: RateSchema, current_user: User, db: Session) -> dict[str, Any]:
    exists = db.execute(
        text("SELECT 1 FROM anime WHERE id = :id"),
        {"id": payload.anime_id},
    ).first()
    if exists is None:
        raise HTTPException(status_code=404, detail="Anime not found")

    existing = (
        db.query(UserRating)
        .filter(
            UserRating.user_id == current_user.user_id,
            UserRating.anime_id == payload.anime_id,
        )
        .first()
    )

    if existing is None:
        existing = UserRating(
            user_id=current_user.user_id,
            anime_id=payload.anime_id,
            rating=payload.rating,
        )
        db.add(existing)
        action = "created"
    else:
        existing.rating = payload.rating
        action = "updated"

    db.commit()
    
    db.refresh(existing)
    return {
        "message": f"Rating {action}",
        "user_id": existing.user_id,
        "anime_id": existing.anime_id,
        "rating": existing.rating,
    }


def get_similar_anime(anime_id: int, db: Session, recommender, k: int = TOP_K_SIMILAR) -> dict[str, Any]:
    k = max(1, min(k, TOP_K_SIMILAR))

    target_exists = db.execute(
        text("SELECT 1 FROM anime WHERE id = :id"),
        {"id": anime_id},
    ).first()
    if target_exists is None:
        raise HTTPException(status_code=404, detail="Anime not found")

    pairs = recommender.recommend_by_mal_id(anime_id, n=k)
    if not pairs:
        return {"anime_id": anime_id, "items": []}

    sim_by_id = {mid: sim for mid, sim in pairs}
    ids = list(sim_by_id.keys())

    stmt = text(
        """
        SELECT a.id, a.title, a.era, a.rating,
               r.num_ratings, r.mean_rating, r.popularity_score
        FROM anime a
        LEFT JOIN ratings r ON r.anime_id = a.id
        WHERE a.id IN :ids
        """
    ).bindparams(bindparam("ids", expanding=True))
    rows = {r["id"]: dict(r) for r in db.execute(stmt, {"ids": ids}).mappings().all()}

    items = []
    for mid in ids:
        row = rows.get(mid)
        if row is None:
            continue
        row["similarity"] = round(sim_by_id[mid], 4)
        items.append(row)

    return {"anime_id": anime_id, "items": items}


def recommend_for_user(current_user: User, db: Session, hybrid_recommender) -> dict[str, Any]:
    user_id = current_user.user_id

    if user_id in hybrid_recommender.user_id_to_uidx:
        try:
            df = hybrid_recommender.recommend(user_id=user_id, top_k=TOP_K)
        except KeyError:
            df = None

        if df is not None and len(df) > 0:
            mal_ids = [int(x) for x in df["mal_id"].tolist()]
            score_by_id = {int(mid): float(s) for mid, s in zip(df["mal_id"], df["score"])}

            stmt = text(
                """
                SELECT a.id, a.title, a.era, a.rating,
                       r.num_ratings, r.mean_rating, r.popularity_score
                FROM anime a
                LEFT JOIN ratings r ON r.anime_id = a.id
                WHERE a.id IN :ids
                """
            ).bindparams(bindparam("ids", expanding=True))
            rows = {r["id"]: dict(r) for r in db.execute(stmt, {"ids": mal_ids}).mappings().all()}

            items = []
            for mid in mal_ids:
                row = rows.get(mid)
                if row is None:
                    continue
                row["hybrid_score"] = round(score_by_id[mid], 4)
                items.append(row)
            return {"strategy": "hybrid", "user_id": user_id, "items": items}

    # cold-start fallback: popularity-ranked, excluding anime the user has already rated
    fallback_stmt = text(
        """
        SELECT a.id, a.title, a.era, a.rating,
               r.num_ratings, r.mean_rating, r.popularity_score
        FROM anime a
        JOIN ratings r ON r.anime_id = a.id
        WHERE a.id NOT IN (SELECT anime_id FROM user_ratings WHERE user_id = :uid)
        ORDER BY r.popularity_score DESC, r.num_ratings DESC
        LIMIT :k
        """
    )
    rows = db.execute(fallback_stmt, {"uid": user_id, "k": TOP_K}).mappings().all()
    return {"strategy": "popularity", "user_id": user_id, "items": [dict(r) for r in rows]}


__all__ = [
    "ALL_ANIME_DEFAULT_LIMIT",
    "get_current_user",
    "get_anime_detail",
    "get_similar_anime",
    "list_all_anime",
    "list_genres",
    "rate_anime",
    "recommend_for_user",
]
