from sqlalchemy import Column, Integer, String
from pydantic import BaseModel, EmailStr
from api.controllers.authControllers import Base

class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)

class RegisterSchema(BaseModel):
    username: str
    email: EmailStr
    password: str
    favourite_genres: list[int]

class LoginSchema(BaseModel):
    email: EmailStr
    password: str

class TokenSchema(BaseModel):
    access_token: str
    token_type: str
