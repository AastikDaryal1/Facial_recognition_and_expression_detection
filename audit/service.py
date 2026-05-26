"""
audit/service.py
─────────────────
Service for writing audit logs.
"""

from typing import Optional
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from database.models import AuditLog

async def write_audit_log(
    db: AsyncSession,
    action: str,
    actor_id: Optional[uuid.UUID] = None,
    target_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    detail: Optional[dict] = None
):
    """
    Writes an audit log entry to the database.
    """
    log_entry = AuditLog(
        actor_id=actor_id,
        action=action,
        target_id=str(target_id) if target_id else None,
        ip_address=ip_address,
        detail=detail or {}
    )
    db.add(log_entry)
    await db.commit()
