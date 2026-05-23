"""
In-memory JWT blacklist.
Tokens are held until the process restarts
"""
_blacklist: set[str] = set()

def revoke(token: str) -> None:
    _blacklist.add(token)
    
def is_revoked(token: str) -> bool:
    return token in _blacklist
