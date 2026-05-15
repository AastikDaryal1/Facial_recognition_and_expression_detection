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

from api.email_service import send_invite_email
import uuid
from datetime import datetime, timezone, timedelta
import traceback

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.schemas import (
    InviteRequest, InviteSignupRequest,
    LoginRequest, SignupRequest, TokenResponse, UserOut,
)
from api.auth.service import (
    blacklist_token, create_access_token, create_refresh_token,
    decode_token, hash_password, is_blacklisted, verify_password,
    _make_token,
)
from api.dependencies import get_current_user, require_role
from api.models import Organisation, User, UserRole
from api.routers.audit import write_audit_log
from db.base import get_db

router = APIRouter()

@router.get("/check-setup")
async def check_setup(db: AsyncSession = Depends(get_db)):
    """
    Returns whether a super admin already exists.
    Used by the frontend to show first-time setup vs normal login.
    """
    existing = await db.scalar(select(User).where(User.role == UserRole.super_admin))
    return {"setup_complete": existing is not None}


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

    try:
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
        await write_audit_log(
            db=db, actor_id=user.id, org_id=org.id,
            action="auth.signup",
            target_type="user", target_id=str(user.id),
            detail={"email": user.email, "role": user.role.value},
        )
        await db.commit()
        await db.refresh(user)

        return TokenResponse(
            access_token  = create_access_token(str(user.id), user.role.value, str(user.org_id)),
            refresh_token = create_refresh_token(str(user.id)),
        )
    except Exception as e:
        print("SIGNUP ERROR TRACEBACK:")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Signup failed: {str(e)}")


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

class RefreshRequest(BaseModel):
    refresh_token: str

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(payload_body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    refresh_token = payload_body.refresh_token
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
    if current_user.role == UserRole.org_admin and payload.role == UserRole.super_admin:
        raise HTTPException(403, "org_admin cannot invite super_admin.")

    # Determine the target org:
    # - super_admin must supply org_id in the payload (they have no org of their own)
    # - org_admin always uses their own org (payload.org_id is ignored)
    if current_user.role == UserRole.super_admin:
        if not payload.org_id:
            raise HTTPException(400, "super_admin must specify an org_id when inviting.")
        try:
            effective_org_id = uuid.UUID(payload.org_id)   # string → UUID object for db.get()
        except ValueError:
            raise HTTPException(400, "Invalid org_id format.")
    else:
        effective_org_id = current_user.org_id             # already a UUID object

    # Generate invite token
    token = _make_token(
        {
            "sub"          : str(current_user.id),
            "invite_email" : payload.email,
            "invite_role"  : payload.role.value,
            "org_id"       : str(effective_org_id),
        },
        timedelta(hours=48),
        "invite",
    )

    # Get org name for email (db.get() needs UUID object — guaranteed above)
    from api.models import Organisation
    org = await db.get(Organisation, effective_org_id)
    org_name = org.name if org else "VisionX"

    # Send email (non-blocking — don't fail if email fails)
    from api.email_service import send_invite_email
    email_sent = await send_invite_email(
        to_email     = payload.email,
        invite_token = token,
        invited_by   = current_user.email,
        role         = payload.role.value,
        org_name     = org_name,
    )

    await write_audit_log(
        db=db, actor_id=current_user.id, org_id=effective_org_id,
        action="auth.invite",
        target_type="user", target_id=None,
        detail={"invited_email": payload.email, "role": payload.role.value, "org_id": str(effective_org_id)},
    )
    await db.commit()

    return {
        "invite_token" : token,
        "expires_in"   : "48 hours",
        "email_sent"   : email_sent,
    }


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

    try:
        user = User(
            email         = payload.email,
            password_hash = hash_password(payload.password),
            role          = UserRole(invite["invite_role"]),
            org_id        = uuid.UUID(invite["org_id"]) if invite.get("org_id") else None,
            invited_by    = uuid.UUID(invite["sub"]) if invite.get("sub") else None,
        )
        db.add(user)
        await write_audit_log(
            db=db, actor_id=user.id, org_id=user.org_id,
            action="auth.invite_accept",
            target_type="user", target_id=str(user.id),
            detail={"email": user.email, "role": user.role.value, "invited_by": invite["sub"]},
        )
        await db.commit()
        await db.refresh(user)

        return TokenResponse(
            access_token  = create_access_token(str(user.id), user.role.value, str(user.org_id)),
            refresh_token = create_refresh_token(str(user.id)),
        )
    except Exception as e:
        print("SIGNUP-INVITE ERROR TRACEBACK:")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Signup-invite failed: {str(e)}")