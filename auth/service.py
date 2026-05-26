"""
auth/service.py
────────────────
Business logic for authentication — password hashing, JWT creation,
user login, and invite-based signup.

This module contains ONLY pure logic — no FastAPI Request/Response objects.
The router calls these functions and handles HTTP details.
"""

import uuid
from datetime import datetime, timedelta, timezone

from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from database.models import User, UserSession
from database.redis_client import consume_invite

# ── Password hashing ─────────────────────────────────────────────────────────
# bcrypt with cost factor 12 — slow enough to resist brute-force attacks,
# fast enough that a single hash takes ~0.3s on a modern CPU.
_pwd_context = CryptContext(schemes=["bcrypt"], bcrypt__rounds=12, deprecated="auto")

ALGORITHM = "HS256"


def hash_password(plain: str) -> str:
    """Hash a plaintext password using bcrypt (cost=12)."""
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if plain matches the stored bcrypt hash."""
    return _pwd_context.verify(plain, hashed)


# ── JWT creation ─────────────────────────────────────────────────────────────

def create_access_token(user_id: uuid.UUID, role: str) -> str:
    """
    Create a short-lived JWT access token.

    Payload claims:
      sub  — subject (user UUID as string)
      role — user role (super_admin / authority_admin / user_admin)
      jti  — unique token ID used for blacklisting on logout
      exp  — expiry timestamp
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "role": role,
        "jti": str(uuid.uuid4()),   # unique ID per token — needed for blacklisting
        "exp": now + timedelta(minutes=settings.jwt_expire_minutes),
        "iat": now,
        "type": "access",
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def create_refresh_token(user_id: uuid.UUID) -> str:
    """
    Create a long-lived JWT refresh token.

    Refresh tokens only carry sub + jti + exp + type.
    They cannot be used as access tokens (the router checks type == "refresh").
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "jti": str(uuid.uuid4()),
        "exp": now + timedelta(days=settings.refresh_token_expire_days),
        "iat": now,
        "type": "refresh",
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """
    Decode and verify a JWT.

    Raises
    ------
    jose.JWTError : if the token is invalid, expired, or tampered with.
    """
    return jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])


# ── Login ─────────────────────────────────────────────────────────────────────

async def login_user(
    email: str,
    password: str,
    db: AsyncSession,
) -> User | None:
    """
    Verify credentials and return the User if correct, else None.

    Also updates `last_login` timestamp on success.
    """
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(password, user.password_hash):
        return None

    if not user.is_active:
        return None

    # Update last login timestamp
    user.last_login = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(user)
    return user


# ── Invite-based Signup ───────────────────────────────────────────────────────

async def signup_user(
    token: str,
    full_name: str,
    password: str,
    db: AsyncSession,
    r,                  # redis.Redis
) -> User | None:
    """
    Complete registration using a pre-issued invite token.

    Flow:
      1. Consume the invite token from Redis (atomic get-and-delete).
      2. If token is invalid or expired → return None.
      3. Create the User with hashed password and the role from the invite.
      4. Commit and return the new user.
    """
    invite = consume_invite(token, r)
    if invite is None:
        return None   # invalid or expired token

    email = invite["email"]
    role  = invite["role"]

    # Check for duplicate email (race condition guard)
    existing = await db.execute(select(User).where(User.email == email))
    if existing.scalar_one_or_none():
        return None

    new_user = User(
        email=email,
        full_name=full_name,
        password_hash=hash_password(password),
        role=role,
        is_active=True,
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return new_user
