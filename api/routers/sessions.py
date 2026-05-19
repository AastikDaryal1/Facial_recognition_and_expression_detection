"""
api/routers/sessions.py
────────────────────────
Session history endpoints.

Who can do what
---------------
user        → GET /sessions           (own sessions only, view only)
user        → GET /sessions/{id}      (own session only)
org_admin   → GET /sessions           (all sessions in their org)
org_admin   → GET /sessions/{id}      (any session in their org)
org_admin   → PATCH /sessions/{id}    (add/edit a note on a session)
org_admin   → DELETE /sessions/{id}   (delete a session in their org)
super_admin → all of the above, across all orgs
super_admin → DELETE /sessions/{id}   (any session)

Endpoints
---------
GET    /sessions           → list sessions (scoped by role)
GET    /sessions/{id}      → get one session
PATCH  /sessions/{id}/note → add/update a note (org_admin + super_admin)
DELETE /sessions/{id}      → delete a session (org_admin for own org, super_admin for all)
"""

from uuid import UUID
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_current_user
from api.models import Session, User, UserRole
from db.base import get_db

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class SessionOut(BaseModel):
    id              : str
    user_id         : str
    org_id          : Optional[str]
    n_faces         : int
    n_identified    : int
    elapsed_s       : float
    results_json    : Optional[dict | list]
    annotated_image : Optional[str]
    note            : Optional[str]
    created_at      : str

    model_config = {"from_attributes": True}


class NoteRequest(BaseModel):
    note: str


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _session_out(s: Session) -> dict:
    return {
        "id"             : str(s.id),
        "user_id"        : str(s.user_id),
        "org_id"         : str(s.org_id) if s.org_id else None,
        "n_faces"        : s.n_faces,
        "n_identified"   : s.n_identified,
        "elapsed_s"      : s.elapsed_s,
        "results_json"   : s.results_json,
        "annotated_image": s.annotated_image,
        "note"           : s.note,
        "created_at"     : s.created_at.isoformat() if s.created_at else None,
    }


async def _get_session_or_404(session_id: str, db: AsyncSession) -> Session:
    try:
        sid = UUID(session_id)
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid session ID format.")
    session = await db.get(Session, sid)
    if not session:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found.")
    return session


def _check_session_access(current_user: User, session: Session) -> None:
    """
    - user      → can only see their own sessions
    - org_admin → can see all sessions within their org
    - super_admin → can see everything
    """
    if current_user.role == UserRole.super_admin:
        return
    if current_user.role == UserRole.org_admin:
        if session.org_id != current_user.org_id:
            raise HTTPException(status.HTTP_403_FORBIDDEN,
                                "You can only access sessions within your own organisation.")
        return
    # regular user — own sessions only
    if session.user_id != current_user.id:
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            "You can only access your own sessions.")


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("", response_model=list[SessionOut])
async def list_sessions(
    page         : int  = Query(1, ge=1),
    page_size    : int  = Query(20, ge=1, le=100),
    current_user : User = Depends(get_current_user),
    db           : AsyncSession = Depends(get_db),
):
    """
    List sessions, most recent first.
    - user        → own sessions only
    - org_admin   → all sessions in their org
    - super_admin → all sessions everywhere
    """
    try:
        offset = (page - 1) * page_size
        stmt = select(Session).order_by(desc(Session.created_at)).offset(offset).limit(page_size)

        if current_user.role == UserRole.super_admin:
            pass  # no filter — sees all
        elif current_user.role == UserRole.org_admin:
            stmt = stmt.where(Session.org_id == current_user.org_id)
        else:
            stmt = stmt.where(Session.user_id == current_user.id)

        result = await db.execute(stmt)
        sessions = result.scalars().all()
        return [_session_out(s) for s in sessions]
    except Exception as e:
        print("LIST SESSIONS ERROR TRACEBACK:")
        import traceback
        traceback.print_exc()
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Failed to list sessions: {str(e)}")


@router.get("/{session_id}", response_model=SessionOut)
async def get_session(
    session_id   : str,
    current_user : User = Depends(get_current_user),
    db           : AsyncSession = Depends(get_db),
):
    """Get a single session by ID."""
    session = await _get_session_or_404(session_id, db)
    _check_session_access(current_user, session)
    return _session_out(session)


@router.patch("/{session_id}/note", response_model=SessionOut)
async def update_session_note(
    session_id   : str,
    payload      : NoteRequest,
    current_user : User = Depends(get_current_user),
    db           : AsyncSession = Depends(get_db),
):
    """
    Add or update a note on a session.
    - org_admin   → can annotate any session in their org
    - super_admin → can annotate any session
    - user        → cannot annotate (403)
    """
    if current_user.role == UserRole.user:
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            "Users cannot annotate sessions.")

    session = await _get_session_or_404(session_id, db)
    _check_session_access(current_user, session)

    session.note = payload.note
    await db.commit()
    await db.refresh(session)
    return _session_out(session)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id   : str,
    current_user : User = Depends(get_current_user),
    db           : AsyncSession = Depends(get_db),
):
    """
    Delete a session.
    - org_admin   → can delete sessions within their own org
    - super_admin → can delete any session
    - user        → cannot delete (403)
    """
    if current_user.role == UserRole.user:
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            "Users cannot delete sessions.")

    session = await _get_session_or_404(session_id, db)
    _check_session_access(current_user, session)

    await db.delete(session)
    await db.commit()