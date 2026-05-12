"""
api/routers/audit.py
─────────────────────
Audit log endpoints + shared write_audit_log() helper.

The write_audit_log() function should be imported and called by other
routers whenever a mutating action happens (invite, deactivate, delete, etc.).

Who can view audit logs
-----------------------
org_admin   → sees only logs where org_id matches their own org
super_admin → sees ALL audit logs across all orgs
user        → no access (403)

Endpoints
---------
GET /audit        → paginated list, filterable by action and date
GET /audit/{id}   → single log entry detail
"""

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select, desc, and_
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import require_role
from api.models import AuditLog, User, UserRole
from db.base import get_db

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper — import this in any router to record an action
# ─────────────────────────────────────────────────────────────────────────────

async def write_audit_log(
    db          : AsyncSession,
    actor_id    : Optional[uuid.UUID],
    org_id      : Optional[uuid.UUID],
    action      : str,
    target_type : Optional[str] = None,
    target_id   : Optional[str] = None,
    detail      : Optional[dict] = None,
) -> None:
    """
    Record an audit log entry. Call this inside any router after a
    mutating action, before committing the main transaction.

    Usage example (inside any router):
        from api.routers.audit import write_audit_log
        await write_audit_log(
            db          = db,
            actor_id    = current_user.id,
            org_id      = current_user.org_id,
            action      = "user.invite",
            target_type = "user",
            target_id   = str(invited_user.id),
            detail      = {"email": invited_user.email, "role": invited_user.role.value},
        )

    Standard action strings (use these for consistency):
        auth.signup           auth.login            auth.logout
        auth.invite           auth.invite_accept
        user.role_change      user.deactivate        user.activate      user.delete
        org.create            org.update             org.deactivate     org.activate    org.delete
        person.create         person.update          person.delete      person.enrolled
        session.delete        session.note_update
    """
    log_entry = AuditLog(
        actor_id    = actor_id,
        org_id      = org_id,
        action      = action,
        target_type = target_type,
        target_id   = target_id,
        detail      = detail or {},
    )
    db.add(log_entry)
    # The caller must commit (db.commit()) after calling this.


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class AuditLogOut(BaseModel):
    id          : str
    actor_id    : Optional[str]
    org_id      : Optional[str]
    action      : str
    target_type : Optional[str]
    target_id   : Optional[str]
    detail      : Optional[dict]
    created_at  : str

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _audit_out(log: AuditLog) -> dict:
    return {
        "id"         : str(log.id),
        "actor_id"   : str(log.actor_id) if log.actor_id else None,
        "org_id"     : str(log.org_id)   if log.org_id   else None,
        "action"     : log.action,
        "target_type": log.target_type,
        "target_id"  : log.target_id,
        "detail"     : log.detail,
        "created_at" : log.created_at.isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("", response_model=list[AuditLogOut])
async def list_audit_logs(
    page         : int           = Query(1, ge=1),
    page_size    : int           = Query(50, ge=1, le=200),
    action       : Optional[str] = Query(None, description="Filter by action, e.g. 'user.invite'"),
    date_from    : Optional[str] = Query(None, description="ISO date string, e.g. '2026-01-01'"),
    date_to      : Optional[str] = Query(None, description="ISO date string, e.g. '2026-12-31'"),
    current_user : User          = Depends(require_role(["super_admin", "org_admin"])),
    db           : AsyncSession  = Depends(get_db),
):
    """
    List audit logs, most recent first.
    - super_admin → sees all logs across all orgs
    - org_admin   → sees only logs for their own org

    Optional filters:
    - action    → filter by action string (e.g. 'user.invite')
    - date_from → logs from this date onward
    - date_to   → logs up to this date
    """
    offset = (page - 1) * page_size
    filters = []

    # Scope by org for org_admin
    if current_user.role == UserRole.org_admin:
        filters.append(AuditLog.org_id == current_user.org_id)

    # Optional filters
    if action:
        filters.append(AuditLog.action == action)

    if date_from:
        try:
            dt_from = datetime.fromisoformat(date_from)
            filters.append(AuditLog.created_at >= dt_from)
        except ValueError:
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                "Invalid date_from format. Use ISO format: 2026-01-01")

    if date_to:
        try:
            dt_to = datetime.fromisoformat(date_to)
            filters.append(AuditLog.created_at <= dt_to)
        except ValueError:
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                "Invalid date_to format. Use ISO format: 2026-12-31")

    stmt = (
        select(AuditLog)
        .where(and_(*filters) if filters else True)
        .order_by(desc(AuditLog.created_at))
        .offset(offset)
        .limit(page_size)
    )

    result = await db.execute(stmt)
    logs = result.scalars().all()
    return [_audit_out(log) for log in logs]


@router.get("/{log_id}", response_model=AuditLogOut)
async def get_audit_log(
    log_id       : str,
    current_user : User = Depends(require_role(["super_admin", "org_admin"])),
    db           : AsyncSession = Depends(get_db),
):
    """Get a single audit log entry by ID."""
    try:
        lid = uuid.UUID(log_id)
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid log ID format.")

    log = await db.get(AuditLog, lid)
    if not log:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Audit log entry not found.")

    # org_admin can only see logs from their own org
    if current_user.role == UserRole.org_admin and log.org_id != current_user.org_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            "You can only access audit logs for your own organisation.")

    return _audit_out(log)