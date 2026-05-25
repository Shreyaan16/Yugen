import json
import logging
from backend.services.redis_client import get_redis_client
log = logging.getLogger(__name__)

def _history_key(thread_id: str) -> str:
    return f"chat:history:{thread_id}"

def append_message(thread_id: str, role: str, content: str) -> None:
    try:
        payload = json.dumps({"role": role, "content": content})
        client = get_redis_client()
        client.rpush(_history_key(thread_id), payload)
    except Exception as exc:
        log.warning("Failed to append chat history: %s", exc)


def get_history(thread_id: str) -> list[dict]:
    try:
        client = get_redis_client()
        items = client.lrange(_history_key(thread_id), 0, -1)
        return [json.loads(item) for item in items]
    except Exception as exc:
        log.warning("Failed to fetch chat history: %s", exc)
        return []

def clear_history(thread_id: str) -> None:
    try:
        client = get_redis_client()
        client.delete(_history_key(thread_id))
    except Exception as exc:
        log.warning("Failed to clear chat history: %s", exc)
