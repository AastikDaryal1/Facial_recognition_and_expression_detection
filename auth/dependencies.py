"""
auth/dependencies.py
─────────────────────
FastAPI dependency functions for authentication and role-based access control.

These are injected into routes via Depends():

    @router.get("/protected")
    async def route(user: User = Depends(get_current_user)):
        ...

    @router.get("/admin-only")
    async def route(user: User = Depends(require_role(["super_admin"]))):
        ...
"""

import uuid
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import redis

from auth.service import decode_token
from database.models import User
from database.session import get_db
from database.redis_client import get_redis, is_token_blacklisted

# ── Bearer token extractor ────────────────────────────────────────────────────
# HTTPBearer reads the "Authorization: Bearer <token>" header automatically.
bearer_scheme = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
    r: redis.Redis = Depends(get_redis),
) -> User:
    """
    FastAPI dependency — decode the Bearer JWT and return the authenticated User.

    Checks (in order):
      1. Token is a valid, non-expired JWT signed with our secret.
      2. Token type is "access" (not a refresh token).
      3. Token has not been blacklisted (i.e. user has not logged out).
      4. User exists in the database.
      5. User account is active (not disabled).

    Raises
    ------
    HTTP 401 on any failure — deliberately vague to avoid info leakage.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_token(credentials.credentials)
    except JWTError:
        raise credentials_exception

    # Reject refresh tokens used as access tokens
    if payload.get("type") != "access":
        raise credentials_exception

    jti: str = payload.get("jti")
    sub: str = payload.get("sub")
    if not jti or not sub:
        raise credentials_exception

    # Check blacklist (logout revocation)
    if is_token_blacklisted(jti, r):
        raise credentials_exception

    # Load user from DB
    try:
        user_uuid = uuid.UUID(sub)
    except ValueError:
        raise credentials_exception

    result = await db.execute(select(User).where(User.id == user_uuid))
    user = result.scalar_one_or_none()

    if user is None or not user.is_active:
        raise credentials_exception

    return user


def require_role(allowed_roles: list[str]):
    """
    Dependency factory for role-based access control.

    Usage:
        @router.post("/invite")
        async def invite(user: User = Depends(require_role(["super_admin"]))):
            ...

    Raises
    ------
    HTTP 403 if the authenticated user's role is not in allowed_roles.
    """
    async def _check_role(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role.value not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role: {allowed_roles}. Your role: {current_user.role.value}",
            )
        return current_user

    return _check_role
