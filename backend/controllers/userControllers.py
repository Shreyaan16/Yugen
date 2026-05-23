from fastapi import HTTPException
from sqlalchemy.orm import Session
from backend.models.authModels import User
from backend.models.userModels import UserProfileResponse, RatedAnimeResponse
from backend.services.store import store
from backend.utils import _parse_list


def get_user_profile(user_id: int) -> UserProfileResponse:
    user = store.user_df[store.user_df["user_id"] == user_id]
    if user.empty:
        raise HTTPException(status_code=404, detail="User not found")
    user = user.iloc[0]
    return UserProfileResponse(user_id=user["user_id"], username=user["username"], email=user["email"])


def get_user_ratings(user_id: int) -> list[RatedAnimeResponse]:
    # Filter ratings CSV for this user
    user_rows = store.user_ratings_df[store.user_ratings_df["user_id"] == user_id]

    if user_rows.empty:
        return []

    # Merge with anime data to get title + genres in one vectorised step
    merged = user_rows.merge(
        store.anime_df[["anime_id", "title", "genres"]],
        on="anime_id",
        how="left",
    )

    return [
        RatedAnimeResponse(
            anime_id=int(row["anime_id"]),
            title=str(row["title"]) if row["title"] == row["title"] else "",
            genres=_parse_list(row.get("genres", [])),
            rating=int(row["rating"]),
        )
        for _, row in merged.iterrows()
    ]
