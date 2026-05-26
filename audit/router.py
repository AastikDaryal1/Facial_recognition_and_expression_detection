"""
audit/router.py
────────────────
Router for viewing audit logs.
"""

from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import require_role
from database.models import AuditLog, User
from database.session import get_db

router = APIRouter(tags=["Audit"])

@router.get("")
async def list_audit_logs(
    actor_id: str | None = None,
    action: str | None = None,
    from_date: datetime | None = Query(None, alias="from"),
    to_date: datetime | None = Query(None, alias="to"),
    page: int = 1,
    page_size: int = 50,
    current_user: User = Depends(require_role(["authority_admin", "super_admin"])),
    db: AsyncSession = Depends(get_db),
):
    """
    Get audit logs with optional filtering and pagination.
    """
    stmt = select(AuditLog)
    
    if actor_id:
        stmt = stmt.where(AuditLog.actor_id == actor_id)
    if action:
        stmt = stmt.where(AuditLog.action == action)
    if from_date:
        stmt = stmt.where(AuditLog.timestamp >= from_date)
    if to_date:
        stmt = stmt.where(AuditLog.timestamp <= to_date)
        
    stmt = stmt.order_by(AuditLog.timestamp.desc())
    stmt = stmt.offset((page - 1) * page_size).limit(page_size)
    
    result = await db.execute(stmt)
    logs = result.scalars().all()
    
    return logs
