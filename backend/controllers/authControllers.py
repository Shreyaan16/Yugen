import os
import json
import uuid
import hashlib
from datetime import datetime, timedelta

from sqlalchemy import text
from sqlalchemy.orm import Session
from fastapi import HTTPException
from jose import jwt

from backend.constants import (
    SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES, TOP_K,
)
from backend.database import engine, Base, get_db  # noqa: F401  (re-exported for routes)
from backend.models.authModels import User, RegisterSchema, LoginSchema

from recommender.constants import (
    ARTIFACT_DIR, RECOMMENDER_DIR_NAME,
    USER_BASED_DIR_NAME, USER_INDEX_FILE, USER_ID_TO_IDX_FILE,
)

_USER_DIR = os.path.join(ARTIFACT_DIR, RECOMMENDER_DIR_NAME, USER_BASED_DIR_NAME)


# ── Auth helpers ───────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(plain: str, hashed: str) -> bool:
    return hash_password(plain) == hashed


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    to_encode["exp"] = datetime.now() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# ── Cold-start helper ──────────────────────────────────────────────────────────

def _allocate_cold_start(user_id: int) -> None:
    """
    Point the new user at the average vector (avg_idx) in user_id_to_idx.json.

    The avg vector is always the last entry added to the FAISS index during
    training (index.ntotal - 1 after load). We only touch the JSON map —
    the FAISS index itself is never modified here.

    No-ops silently if the recommender hasn't been trained yet; the user will
    be picked up as a cold-start user on the next training run.
    """
    index_path  = os.path.join(_USER_DIR, USER_INDEX_FILE)
    id_map_path = os.path.join(_USER_DIR, USER_ID_TO_IDX_FILE)

    if not os.path.exists(index_path) or not os.path.exists(id_map_path):
        return

    import faiss
    avg_idx = faiss.read_index(index_path).ntotal - 1  # avg vec is always last

    with open(id_map_path, encoding="utf-8") as f:
        user_id_to_idx: dict = json.load(f)

    user_id_to_idx[str(user_id)] = avg_idx

    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(user_id_to_idx, f)


# ── Controllers ────────────────────────────────────────────────────────────────

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

    # 4. Map user to the avg (cold-start) vector in the user FAISS index
    try:
        _allocate_cold_start(new_user.user_id)
    except Exception as exc:
        print(f"[register_user] Cold-start allocation failed (non-fatal): {exc}")

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
