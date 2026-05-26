"""
database/redis_client.py
─────────────────────────
Redis connection pool and all key-based helpers used across the system.

Why a connection pool?
  Instead of opening a new TCP connection on every request, a pool keeps N
  connections alive and hands them out to concurrent requests — much faster.

Usage in FastAPI routes:
    from database.redis_client import get_redis
    r: redis.Redis = Depends(get_redis)
"""

import secrets
import redis
from config.settings import settings

# ── Connection Pool ───────────────────────────────────────────────────────────
# Created once at module import time (process-wide singleton).
# max_connections=20 means at most 20 simultaneous Redis connections.
# decode_responses=True means Redis returns str instead of bytes.
_pool = redis.ConnectionPool.from_url(
    settings.redis_url,
    max_connections=20,
    decode_responses=True,
)


def get_redis() -> redis.Redis:
    """
    FastAPI dependency — yields a Redis client backed by the shared pool.

    Usage:
        @router.get("/example")
        async def example(r: redis.Redis = Depends(get_redis)):
            r.set("key", "value")
    """
    return redis.Redis(connection_pool=_pool)


# =============================================================================
# Key name helpers
# =============================================================================
# Centralising key names here means if you ever rename a key prefix you change
# it in exactly one place, and nothing else breaks.

def jwt_blacklist_key(jti: str) -> str:
    """Redis key for a blacklisted JWT token id."""
    return f"blacklist:{jti}"


def invite_token_key(token: str) -> str:
    """Redis key for a pending invite token."""
    return f"invite:{token}"


def rate_limit_key(ip: str, action: str) -> str:
    """Redis key for a rate-limit counter (per IP, per action)."""
    return f"ratelimit:{action}:{ip}"


# =============================================================================
# Token blacklisting (logout / token revocation)
# =============================================================================

def blacklist_token(jti: str, expires_in: int, r: redis.Redis) -> None:
    """
    Mark a JWT as blacklisted until its natural expiry time.

    Parameters
    ----------
    jti        : the unique token id (jti claim from JWT payload)
    expires_in : seconds until the token naturally expires
                 (set this to remaining TTL so the key auto-deletes)
    r          : Redis client
    """
    r.setex(jwt_blacklist_key(jti), expires_in, "1")


def is_token_blacklisted(jti: str, r: redis.Redis) -> bool:
    """Return True if the given jti has been blacklisted (i.e. logged out)."""
    return r.exists(jwt_blacklist_key(jti)) == 1


# =============================================================================
# Invite tokens (email-based admin onboarding)
# =============================================================================

def create_invite(email: str, role: str, r: redis.Redis) -> str:
    """
    Generate a one-time invite token and store it in Redis for 24 hours.

    The stored value is "email|role" so we can recover both on redemption.

    Returns
    -------
    str : the URL-safe token to send to the invitee
    """
    token = secrets.token_urlsafe(32)
    r.setex(invite_token_key(token), 86400, f"{email}|{role}")  # 24 hours
    return token


def consume_invite(token: str, r: redis.Redis) -> dict | None:
    """
    Redeem an invite token (atomic get-and-delete via GETDEL).

    Returns
    -------
    dict with keys 'email' and 'role', or None if token is invalid/expired.
    """
    value = r.getdel(invite_token_key(token))
    if not value:
        return None
    email, role = value.split("|", 1)
    return {"email": email, "role": role}


# =============================================================================
# Rate limiting (sliding window counter)
# =============================================================================

def check_rate_limit(
    ip: str,
    action: str,
    limit: int,
    window: int,
    r: redis.Redis,
) -> bool:
    """
    Increment a counter and check whether the caller is within the rate limit.

    Uses a fixed-window counter pattern:
      - INCR the key (creates it at 0 if it doesn't exist, then returns 1)
      - On first increment (count == 1) set the TTL = window seconds
      - Return True if count <= limit (request is allowed)

    Parameters
    ----------
    ip     : client IP address
    action : logical action name (e.g. "face_login")
    limit  : max requests allowed in the window
    window : window duration in seconds
    r      : Redis client

    Returns
    -------
    bool : True = request is within limit, False = rate limit exceeded
    """
    key = rate_limit_key(ip, action)
    count = r.incr(key)
    if count == 1:
        r.expire(key, window)
    return count <= limit
