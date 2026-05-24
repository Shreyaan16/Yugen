from pydantic import BaseModel

class UserProfileResponse(BaseModel):
    user_id:  int
    username: str
    email:    str

class RatedAnimeResponse(BaseModel):
    anime_id: int
    title:    str
    genres:   list[str]
    rating:   int
