"""
auth/router.py
───────────────
FastAPI router for all authentication endpoints.

Endpoints
---------
POST /auth/login         → email+password → JWT access + refresh tokens
POST /auth/signup        → invite token + name + password → create admin account
POST /auth/invite        → super_admin only → issue one-time invite token
POST /auth/logout        → blacklist current access token
POST /auth/face-login    → base64 image → JWT if face recognized
"""

import base64
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import redis
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user, require_role
from auth.schemas import (
    FaceLoginRequest,
    InviteRequest,
    LoginRequest,
    SignupRequest,
    TokenResponse,
    UserResponse,
)
from auth.service import (
    create_access_token,
    create_refresh_token,
    decode_token,
    login_user,
    signup_user,
)
from audit.service import write_audit_log
from config.settings import settings
from database.models import AuditAction, User
from database.redis_client import (
    blacklist_token,
    check_rate_limit,
    create_invite,
    get_redis,
)
from database.session import get_db
from jose import JWTError
from utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["Auth"])


# ── POST /auth/login ──────────────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    r: redis.Redis = Depends(get_redis),
):
    """
    Authenticate with email + password.
    Returns a short-lived access token and a long-lived refresh token.
    """
    user = await login_user(body.email, body.password, db)

    if user is None:
        await write_audit_log(
            db,
            action=AuditAction.LOGIN_FAIL,
            ip_address=request.client.host,
            detail={"email": body.email},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    access_token  = create_access_token(user.id, user.role.value)
    refresh_token = create_refresh_token(user.id)

    await write_audit_log(
        db,
        action=AuditAction.LOGIN_OK,
        actor_id=user.id,
        ip_address=request.client.host,
    )

    log.info("Login OK: %s (%s)", user.email, user.role.value)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        role=user.role.value,
    )


# ── POST /auth/signup ─────────────────────────────────────────────────────────

@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    body: SignupRequest,
    db: AsyncSession = Depends(get_db),
    r: redis.Redis = Depends(get_redis),
):
    """
    Complete registration using a one-time invite token.
    The invite token must have been issued by a super_admin via POST /auth/invite.
    """
    user = await signup_user(body.token, body.full_name, body.password, db, r)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired invite token",
        )

    log.info("New admin registered: %s (%s)", user.email, user.role.value)
    return UserResponse.model_validate(user)


# ── POST /auth/invite ─────────────────────────────────────────────────────────

@router.post("/invite")
async def invite(
    body: InviteRequest,
    request: Request,
    current_user: User = Depends(require_role(["super_admin"])),
    db: AsyncSession = Depends(get_db),
    r: redis.Redis = Depends(get_redis),
):
    """
    Issue a one-time invite token (super_admin only).
    The token expires in 24 hours.
    Send this token to the invitee out-of-band (email, Slack, etc.).
    """
    token = create_invite(body.email, body.role.value, r)

    await write_audit_log(
        db,
        action=AuditAction.INVITE_SENT,
        actor_id=current_user.id,
        ip_address=request.client.host,
        detail={"invited_email": body.email, "assigned_role": body.role.value},
    )

    log.info("Invite issued by %s → %s (%s)", current_user.email, body.email, body.role.value)

    return {
        "invite_token": token,
        "email": body.email,
        "role": body.role.value,
        "expires_in": "24 hours",
    }


# ── POST /auth/logout ─────────────────────────────────────────────────────────

@router.post("/logout")
async def logout(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    r: redis.Redis = Depends(get_redis),
):
    """
    Invalidate the current access token by adding its jti to the blacklist.
    After this call, the same token will be rejected by get_current_user().
    """
    # Re-decode to get jti and exp for TTL calculation
    auth_header = request.headers.get("authorization", "")
    raw_token = auth_header.removeprefix("Bearer ").strip()

    try:
        payload = decode_token(raw_token)
        jti = payload["jti"]
        exp = payload["exp"]
        # Time remaining until natural expiry — we blacklist only that long
        remaining_ttl = max(1, int(exp - time.time()))
        blacklist_token(jti, remaining_ttl, r)
    except (JWTError, KeyError):
        pass  # Already invalid, nothing to blacklist

    await write_audit_log(
        db,
        action="LOGOUT",
        actor_id=current_user.id,
        ip_address=request.client.host,
    )

    log.info("Logout: %s", current_user.email)
    return {"message": "Logged out successfully"}


# ── POST /auth/face-login ─────────────────────────────────────────────────────

@router.post("/face-login", response_model=TokenResponse)
async def face_login(
    body: FaceLoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    r: redis.Redis = Depends(get_redis),
):
    """
    Authenticate via face recognition.

    Flow:
      1. Rate limit: max 5 attempts per IP per 60 seconds.
      2. Decode base64 image → numpy BGR array.
      3. Run RetinaFace + FaceNet → embedding.
      4. Search pgvector for nearest face embedding.
      5. If confidence >= threshold → return tokens.
    """
    client_ip = request.client.host

    # Step 1: Rate limit check
    allowed = check_rate_limit(client_ip, "face_login", limit=5, window=60, r=r)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many face login attempts. Try again in 60 seconds.",
        )

    # Step 2: Decode base64 image
    try:
        image_bytes = base64.b64decode(body.image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Image decode failed")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    # Step 3 & 4: Run inference pipeline
    try:
        from pipelines.inference import run_face_identification
        result = await run_face_identification(img_bgr, db)
    except Exception as exc:
        log.error("Face login inference error: %s", exc, exc_info=True)
        await write_audit_log(
            db, AuditAction.FACE_LOGIN_FAIL,
            ip_address=client_ip,
            detail={"error": str(exc)},
        )
        raise HTTPException(status_code=500, detail="Face recognition error")

    # Step 5: Check result
    if result is None or result.get("status") != "confirmed":
        await write_audit_log(
            db, AuditAction.FACE_LOGIN_FAIL,
            ip_address=client_ip,
            detail={"status": result.get("status") if result else "no_face"},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Face not recognized",
        )

    # Look up the person and their enrolled_by user
    person_id = result.get("person_id")
    from sqlalchemy import select
    from database.models import Person
    person_row = await db.execute(select(Person).where(Person.id == person_id))
    person = person_row.scalar_one_or_none()

    if person is None or person.enrolled_by is None:
        raise HTTPException(status_code=401, detail="Person not linked to an admin account")

    from database.models import User as UserModel
    user_row = await db.execute(select(UserModel).where(UserModel.id == person.enrolled_by))
    user = user_row.scalar_one_or_none()
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="Account inactive")

    access_token  = create_access_token(user.id, user.role.value)
    refresh_token = create_refresh_token(user.id)

    await write_audit_log(
        db, AuditAction.FACE_LOGIN_OK,
        actor_id=user.id,
        ip_address=client_ip,
        detail={"confidence": result.get("confidence"), "person_id": str(person_id)},
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        role=user.role.value,
    )
