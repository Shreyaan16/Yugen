"""
Shared helpers used across backend controllers.
"""
import ast
import pandas as pd
from jose import jwt, JWTError
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from backend.constants import SECRET_KEY, ALGORITHM
from backend.services import token_blacklist

optional_bearer = HTTPBearer(auto_error=False)   # public endpoints (guest ok)
required_bearer = HTTPBearer(auto_error=True)    # protected endpoints (must be logged in)

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(required_bearer)) -> int:
    """Raises 401 if token is missing, revoked, or invalid. Returns user_id."""
    token = credentials.credentials
    if token_blacklist.is_revoked(token):
        raise HTTPException(status_code=401, detail="Token has been revoked — please log in again")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return int(payload["sub"])
    except (JWTError, ValueError, TypeError, KeyError):
        raise HTTPException(status_code=401, detail="Invalid or expired token")

def get_optional_user(credentials: HTTPAuthorizationCredentials | None = Security(optional_bearer)) -> int | None:
    """Returns user_id if valid non-revoked token present, else None (guest)."""
    if credentials is None:
        return None
    token = credentials.credentials
    if token_blacklist.is_revoked(token):
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return int(payload["sub"])
    except (JWTError, ValueError, TypeError, KeyError):
        return None

def _parse_list(val) -> list:
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    return []

def to_card(row) -> dict:
    """Serialise a DataFrame row (Series or dict) to an anime card (anime_id, title, genres)."""
    return {
        "anime_id": int(row["anime_id"]),
        "title":    str(row.get("title") or ""),
        "genres":   _parse_list(row.get("genres", [])),
    }

def clean(value):
    """Replace scalar NaN/NaT with None for JSON serialisation. Leaves lists/dicts untouched."""
    if isinstance(value, (list, dict)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
