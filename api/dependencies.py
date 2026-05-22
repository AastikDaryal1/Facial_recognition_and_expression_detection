"""
api/dependencies.py
───────────────────
get_current_user  — decodes JWT, checks blacklist, returns User
require_role      — role guard factory for route protection
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.service import decode_token, is_blacklisted
from api.models import User
from db.base import get_db

bearer_scheme = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    token = credentials.credentials
    try:
        payload = decode_token(token)
    except JWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token.")

    if payload.get("type") != "access":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not an access token.")

    if await is_blacklisted(payload["jti"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token has been revoked.")

    import uuid
    try:
        user_id = uuid.UUID(payload["sub"])
    except ValueError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid user ID in token.")

    user = await db.get(User, user_id)
    if not user or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found or deactivated.")

    # Attach payload to user so logout can read jti/exp
    user._token_payload = payload
    return user


def require_role(allowed_roles: list[str]):
    """
    Usage:  Depends(require_role(["super_admin", "org_admin"]))
    Returns a FastAPI dependency that enforces role.
    """
    async def _guard(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role.value not in allowed_roles:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Requires one of: {allowed_roles}. Your role: {current_user.role.value}",
            )
        return current_user
    return _guard