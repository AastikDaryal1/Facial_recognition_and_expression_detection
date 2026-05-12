"""
api/auth/service.py
───────────────────
Password hashing, JWT creation/decoding, Redis token blacklist.
"""

import uuid
from datetime import datetime, timedelta, timezone

import redis.asyncio as aioredis
from jose import JWTError, jwt
from passlib.context import CryptContext

from config.settings import (
    JWT_SECRET_KEY, JWT_ALGORITHM,
    JWT_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS,
    REDIS_URL,
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── Password ──────────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ── JWT ───────────────────────────────────────────────────────────────────────

def _make_token(data: dict, expires_delta: timedelta, token_type: str) -> str:
    payload = data.copy()
    now = datetime.now(timezone.utc)
    payload.update({
        "iat"  : now,
        "exp"  : now + expires_delta,
        "jti"  : str(uuid.uuid4()),   # unique ID — used for blacklisting
        "type" : token_type,
    })
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def create_access_token(user_id: str, role: str, org_id: str | None) -> str:
    return _make_token(
        {"sub": user_id, "role": role, "org_id": org_id},
        timedelta(minutes=JWT_EXPIRE_MINUTES),
        "access",
    )


def create_refresh_token(user_id: str) -> str:
    return _make_token(
        {"sub": user_id},
        timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
        "refresh",
    )


def decode_token(token: str) -> dict:
    """Decode and return the payload. Raises JWTError on failure."""
    return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])


# ── Redis blacklist ───────────────────────────────────────────────────────────

async def get_redis() -> aioredis.Redis:
    return await aioredis.from_url(REDIS_URL, decode_responses=True)


async def blacklist_token(jti: str, ttl_seconds: int) -> None:
    """Add jti to Redis with the token's remaining TTL."""
    r = await get_redis()
    await r.setex(f"bl:{jti}", ttl_seconds, "1")
    await r.aclose()


async def is_blacklisted(jti: str) -> bool:
    r = await get_redis()
    result = await r.exists(f"bl:{jti}")
    await r.aclose()
    return bool(result)