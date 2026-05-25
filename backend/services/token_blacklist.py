"""
Redis-backed JWT blacklist.
Tokens are held until their JWT exp timestamp.
"""
import time
from jose import jwt
from backend.constants import SECRET_KEY, ALGORITHM
from backend.services.redis_client import get_redis_client

def _revoked_key(token: str) -> str:
    return f"revoked:{token}"

def revoke(token: str) -> None:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_exp": False})
        exp_ts = int(payload.get("exp", 0))
        ttl = max(exp_ts - int(time.time()), 1)
        client = get_redis_client()
        client.set(_revoked_key(token), "1", ex=ttl)
    except Exception as exc:
        raise RuntimeError("Failed to revoke token in Redis") from exc

def is_revoked(token: str) -> bool:
    try:
        client = get_redis_client()
        return bool(client.exists(_revoked_key(token)))
    except Exception as exc:
        raise RuntimeError("Failed to check token revocation in Redis") from exc
