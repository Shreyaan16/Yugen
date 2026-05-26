from fastapi import APIRouter, Depends, Security
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.utils import required_bearer
from backend.controllers.authControllers import register_user, login_user, logout_user
from backend.models.authModels import RegisterSchema, LoginSchema, LogoutScheme

router = APIRouter()

@router.post("/register")
def register(user: RegisterSchema, db: Session = Depends(get_db)):
    return register_user(user, db)


@router.post("/login")
def login(user: LoginSchema, db: Session = Depends(get_db)):
    return login_user(user, db)


@router.post("/logout")
def logout(body: LogoutScheme, credentials: HTTPAuthorizationCredentials = Security(required_bearer),):
    """Revokes the Bearer token. Pass chatbot_session_id as thread_id to also clear chat history."""
    return logout_user(credentials, body.thread_id)
