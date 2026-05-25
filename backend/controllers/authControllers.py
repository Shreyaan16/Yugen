import uuid
import hashlib
from datetime import datetime, timedelta
from sqlalchemy import text
from sqlalchemy.orm import Session
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jose import jwt, JWTError
from backend.services import token_blacklist
from backend.constants import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES, TOP_K
from backend.services.store import store
from backend.services import chat_history
from backend.models.authModels import User, RegisterSchema, LoginSchema

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def verify_password(plain: str, hashed: str) -> bool:
    return hash_password(plain) == hashed

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    to_encode["exp"] = datetime.now() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def register_user(user: RegisterSchema, db: Session) -> dict:
    # 1. Email uniqueness check
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    # 2. Global top-K popular anime as the initial recommendation
    top_anime = db.execute(
        text(
            """
            SELECT a.id        AS anime_id,
                   a.title,
                   a.synopsis,
                   a.era,
                   a.rating,
                   r.num_ratings,
                   r.mean_rating,
                   r.popularity_score
            FROM   anime   a
            JOIN   ratings r ON r.anime_id = a.id
            ORDER  BY r.popularity_score DESC, r.num_ratings DESC
            LIMIT  :k
            """
        ),
        {"k": TOP_K},
    ).mappings().all()
    # 3. Persist user
    new_user = User(
        username=user.username,
        email=user.email,
        password=hash_password(user.password),
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {
        "message": "User registered successfully",
        "user_id": new_user.user_id,
        "recommended_anime": [dict(row) for row in top_anime],
    }


def login_user(user: LoginSchema, db: Session) -> dict:
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token     = create_access_token(data={"sub": str(db_user.user_id)})
    thread_id = f"{db_user.user_id}:{uuid.uuid4().hex[:8]}"  # fresh chatbot session per login
    return {"access_token": token, "token_type": "bearer", "chatbot_session_id": thread_id}


def logout_user(credentials: HTTPAuthorizationCredentials, thread_id: str | None) -> dict:
    raw_token = credentials.credentials
    # Validate the token is genuine before blacklisting it
    try:
        jwt.decode(raw_token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    token_blacklist.revoke(raw_token)
    # Clear the chatbot session if the client provided a thread_id
    if thread_id:
        try:
            if store.agent:
                store.agent.clear_session(thread_id)
            chat_history.clear_history(thread_id)
        except Exception:
            pass  # non-fatal
    return {"message": "Logged out successfully"}
