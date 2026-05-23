from sqlalchemy import Column, Integer, String, SmallInteger, ForeignKey, DateTime, func
from pydantic import BaseModel, EmailStr, conint
from backend.database import Base

class User(Base):
    __tablename__ = "users"
    user_id  = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    email    = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)

class UserRating(Base):
    __tablename__ = "user_ratings"
    user_id    = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True)
    anime_id   = Column(Integer, primary_key=True)
    rating     = Column(SmallInteger, nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class RegisterSchema(BaseModel):
    username: str
    email:    EmailStr
    password: str

class LoginSchema(BaseModel):
    email:    EmailStr
    password: str

class TokenSchema(BaseModel):
    access_token: str
    token_type:   str

class RateSchema(BaseModel):
    anime_id: int
    rating:  conint(ge=1, le=10)

class LogoutScheme(BaseModel):
    thread_id: str | None = None