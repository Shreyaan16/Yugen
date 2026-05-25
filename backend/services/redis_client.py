from backend.constants import REDIS_HOST, REDIS_PORT, REDIS_DB
import redis

_redis_client = None

def get_redis_client() -> redis.Redis:
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    host = REDIS_HOST
    port = REDIS_PORT
    db = REDIS_DB
    _redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
    try:
        _redis_client.ping()
    except Exception as exc:
        raise RuntimeError("Failed to connect to Redis") from exc
    return _redis_client
