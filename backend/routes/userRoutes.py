from fastapi import APIRouter, Depends
from backend.utils import get_current_user
from backend.controllers.userControllers import get_user_profile, get_user_ratings

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me")
def my_profile(user_id: int = Depends(get_current_user)):
    """Returns the logged-in user's username and email."""
    return get_user_profile(user_id)


@router.get("/me/ratings")
def my_ratings(user_id: int = Depends(get_current_user)):
    """Returns all anime the user has rated with title, genres, and their score."""
    return get_user_ratings(user_id)
