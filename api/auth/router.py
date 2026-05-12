"""
api/auth/router.py
──────────────────
POST /auth/signup    — create first super_admin account (seeds an org)
POST /auth/login     — returns access + refresh tokens
POST /auth/refresh   — rotate access token
POST /auth/logout    — blacklists current token
POST /auth/invite    — org_admin / super_admin issues an invite token
POST /auth/signup-invite — user registers using an invite token
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.schemas import (
    InviteRequest, InviteSignupRequest,
    LoginRequest, SignupRequest, TokenResponse, UserOut,
)
from api.auth.service import (
    blacklist_token, create_access_token, create_refresh_token,
    decode_token, hash_password, is_blacklisted, verify_password,
)
from api.dependencies import get_current_user, require_role
from api.models import Organisation, User, UserRole
from db.base import get_db

router = APIRouter()


# ── Signup (seeds super_admin + org) ─────────────────────────────────────────

@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(payload: SignupRequest, db: AsyncSession = Depends(get_db)):
    """
    First-time setup only. Creates a super_admin + their organisation.
    After the first super_admin exists, use /auth/invite to add more users.
    """
    existing = await db.scalar(select(User).where(User.role == UserRole.super_admin))
    if existing:
        raise HTTPException(400, "Super admin already exists. Use invite flow instead.")

    existing_email = await db.scalar(select(User).where(User.email == payload.email))
    if existing_email:
        raise HTTPException(400, "Email already registered.")

    org = Organisation(name=payload.org_name)
    db.add(org)
    await db.flush()   # get org.id before committing

    user = User(
        email         = payload.email,
        password_hash = hash_password(payload.password),
        role          = UserRole.super_admin,
        org_id        = org.id,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return TokenResponse(
        access_token  = create_access_token(str(user.id), user.role.value, str(user.org_id)),
        refresh_token = create_refresh_token(str(user.id)),
    )


# ── Login ─────────────────────────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = await db.scalar(select(User).where(User.email == payload.email))
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials.")
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Account is deactivated.")

    user.last_login_at = datetime.now(timezone.utc)
    await db.commit()

    return TokenResponse(
        access_token  = create_access_token(str(user.id), user.role.value, str(user.org_id)),
        refresh_token = create_refresh_token(str(user.id)),
    )


# ── Refresh ───────────────────────────────────────────────────────────────────

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: str, db: AsyncSession = Depends(get_db)):
    try:
        payload = decode_token(refresh_token)
    except Exception:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid refresh token.")

    if payload.get("type") != "refresh":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not a refresh token.")

    if await is_blacklisted(payload["jti"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token has been revoked.")

    user = await db.get(User, payload["sub"])
    if not user or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found.")

    return TokenResponse(
        access_token  = create_access_token(str(user.id), user.role.value, str(user.org_id)),
        refresh_token = create_refresh_token(str(user.id)),
    )


# ── Logout ────────────────────────────────────────────────────────────────────

@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(current_user: User = Depends(get_current_user)):
    # The jti and exp are stored on the request's token — we grab them from state
    # This works because get_current_user attaches token_payload to the user object
    payload  = current_user._token_payload
    jti      = payload["jti"]
    exp      = payload["exp"]
    now_ts   = int(__import__("time").time())
    ttl      = max(exp - now_ts, 0)
    await blacklist_token(jti, ttl)


# ── Invite ────────────────────────────────────────────────────────────────────

@router.post("/invite", response_model=dict)
async def invite_user(
    payload      : InviteRequest,
    current_user : User = Depends(require_role(["super_admin", "org_admin"])),
    db           : AsyncSession = Depends(get_db),
):
    """
    Generates a signed invite token. Share this with the new user —
    they call POST /auth/signup-invite to complete registration.
    """
    from api.auth.service import _make_token
    from datetime import timedelta

    # org_admin can only invite user or org_admin (not super_admin)
    if current_user.role == UserRole.org_admin and payload.role == UserRole.super_admin:
        raise HTTPException(403, "org_admin cannot invite super_admin.")

    token = _make_token(
        {
            "sub"    : str(current_user.id),
            "invite_email": payload.email,
            "invite_role" : payload.role.value,
            "org_id" : str(current_user.org_id),
        },
        timedelta(hours=48),
        "invite",
    )
    return {"invite_token": token, "expires_in": "48 hours"}


# ── Signup via invite ─────────────────────────────────────────────────────────

@router.post("/signup-invite", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup_invite(payload: InviteSignupRequest, db: AsyncSession = Depends(get_db)):
    try:
        invite = decode_token(payload.invite_token)
    except Exception:
        raise HTTPException(400, "Invalid or expired invite token.")

    if invite.get("type") != "invite":
        raise HTTPException(400, "Not an invite token.")

    if invite.get("invite_email") != payload.email:
        raise HTTPException(400, "Email does not match invite.")

    existing = await db.scalar(select(User).where(User.email == payload.email))
    if existing:
        raise HTTPException(400, "Email already registered.")

    user = User(
        email         = payload.email,
        password_hash = hash_password(payload.password),
        role          = UserRole(invite["invite_role"]),
        org_id        = invite["org_id"],
        invited_by    = invite["sub"],
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return TokenResponse(
        access_token  = create_access_token(str(user.id), user.role.value, str(user.org_id)),
        refresh_token = create_refresh_token(str(user.id)),
    )