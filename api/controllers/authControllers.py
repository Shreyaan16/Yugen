from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from api.constants import (
    DATABASE_URL,
    TOP_K,
    SECRET_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    UNKNOWN_GENRE_ID,
)
from jose import jwt
import hashlib
from datetime import datetime, timedelta
from fastapi import HTTPException

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

from api.models.authModels import User, RegisterSchema, LoginSchema  # noqa: E402  (Base must exist first)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(password: str):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def verify_password(plain_password, hashed_password):
    return hash_password(plain_password) == hashed_password

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def register_user(user: RegisterSchema, db: Session):
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    if not user.favourite_genres:
        raise HTTPException(status_code=400, detail="Please select at least one favourite genre")

    genre_ids = list({gid for gid in user.favourite_genres if gid != UNKNOWN_GENRE_ID})
    if not genre_ids:
        raise HTTPException(status_code=400, detail="Please select at least one valid genre")

    valid_stmt = text(
        "SELECT id FROM genres WHERE id IN :ids AND id <> :unknown"
        ).bindparams(bindparam("ids", expanding=True))
    valid_rows = db.execute(
        valid_stmt,
        {"ids": genre_ids, "unknown": UNKNOWN_GENRE_ID},
    ).fetchall()
    valid_ids = {row[0] for row in valid_rows}
    invalid = set(genre_ids) - valid_ids
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid genre ids: {sorted(invalid)}")

    top_anime_stmt = text(
        """
        SELECT a.id, a.title, a.synopsis, a.era, a.rating
        FROM anime a
        WHERE a.id IN (
            SELECT DISTINCT anime_id FROM anime_genres WHERE genre_id IN :ids
        )
        ORDER BY random()
        LIMIT :k
        """
    ).bindparams(bindparam("ids", expanding=True))
    top_anime = db.execute(
        top_anime_stmt,
        {"ids": list(valid_ids), "k": TOP_K},
    ).mappings().all()

    new_user = User(username=user.username, email=user.email, password=hash_password(user.password))

    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {
        "message": "User registered successfully",
        "user_id": new_user.user_id,
        "recommended_anime": [dict(row) for row in top_anime],
    }


def login_user(user: LoginSchema, db: Session):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(data={"sub": str(db_user.user_id)})
    return {"access_token": token, "token_type": "bearer"}