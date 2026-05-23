from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database import Base, engine, get_db
from backend.controllers.authControllers import register_user, login_user
from backend.models.authModels import RegisterSchema, LoginSchema

router = APIRouter()

Base.metadata.create_all(bind=engine)


@router.post("/register")
def register(user: RegisterSchema, db: Session = Depends(get_db)):
    return register_user(user, db)


@router.post("/login")
def login(user: LoginSchema, db: Session = Depends(get_db)):
    return login_user(user, db)
