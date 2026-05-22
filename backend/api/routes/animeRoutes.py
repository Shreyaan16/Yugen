from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

from backend.api.constants import ALL_ANIME_DEFAULT_LIMIT, ALL_ANIME_MAX_LIMIT, TOP_K_SIMILAR
from backend.api.controllers.animeControllers import (
    get_anime_detail,
    get_current_user,
    get_similar_anime,
    list_all_anime,
    list_genres,
    rate_anime,
    recommend_for_user,
)
from backend.api.controllers.authControllers import get_db
from backend.api.models.authModels import RateSchema, User

router = APIRouter()


@router.get("/genres")
def genres(db: Session = Depends(get_db)):
    return list_genres(db)


@router.get("/all_anime")
def all_anime(
    limit: int = Query(ALL_ANIME_DEFAULT_LIMIT, ge=1, le=ALL_ANIME_MAX_LIMIT),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    return list_all_anime(db, limit=limit, offset=offset)


@router.get("/anime/{anime_id}")
def anime_detail(anime_id: int, db: Session = Depends(get_db)):
    return get_anime_detail(anime_id, db)


@router.post("/rate")
def rate(
    payload: RateSchema,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return rate_anime(payload, current_user, db)


@router.get("/similar_anime/{anime_id}")
def similar_anime(
    anime_id: int,
    request: Request,
    k: int = Query(TOP_K_SIMILAR, ge=1, le=TOP_K_SIMILAR),
    db: Session = Depends(get_db),
):
    return get_similar_anime(anime_id, db, request.app.state.content_recommender, k=k)


@router.get("/recommend_for_me")
def recommend_for_me(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return recommend_for_user(current_user, db, request.app.state.hybrid_recommender)
