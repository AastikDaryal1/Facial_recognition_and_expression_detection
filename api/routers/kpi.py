"""
api/routers/kpi.py
───────────────────
KPI Analytics endpoints — aggregate data from sessions & audit_logs.

All queries run against existing tables, no extra KPI table needed.

Who can view
────────────
super_admin → platform-wide KPIs
org_admin   → KPIs scoped to their own org

Endpoints
─────────
GET /kpi/summary             → top-level stat cards
GET /kpi/sessions-over-time  → session counts grouped by day/week
GET /kpi/emotion-distribution→ emotion breakdown across all results
GET /kpi/top-identified      → most frequently recognised persons
GET /kpi/latency-trend       → avg latency per day
GET /kpi/user-activity       → sessions per user
GET /kpi/hourly-heatmap      → session counts by hour of day
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func, cast, Integer, Float, text
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import require_role
from api.models import Session, User, UserRole, AuditLog
from db.base import get_db

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _date_filter(days: int) -> datetime:
    """Return a datetime N days ago (UTC)."""
    return datetime.utcnow() - timedelta(days=days)


def _scope_query(stmt, current_user: User):
    """Narrow a sessions query by org if the caller is an org_admin."""
    if current_user.role == UserRole.org_admin:
        stmt = stmt.where(Session.org_id == current_user.org_id)
    return stmt


# ─────────────────────────────────────────────────────────────────────────────
# 1. Summary — top-level KPI cards
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/summary")
async def kpi_summary(
    days: int = Query(30, ge=1, le=365, description="Look-back window in days"),
    current_user: User = Depends(require_role(["super_admin", "org_admin"])),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns:
      total_sessions, total_faces, total_identified,
      identification_rate, avg_latency_s, active_users,
      total_audit_events
    """
    since = _date_filter(days)

    # ── Sessions aggregates ──────────────────────────────────────────────
    base = select(
        func.count(Session.id).label("total_sessions"),
        func.coalesce(func.sum(Session.n_faces), 0).label("total_faces"),
        func.coalesce(func.sum(Session.n_identified), 0).label("total_identified"),
        func.coalesce(func.avg(Session.elapsed_s), 0).label("avg_latency"),
    ).where(Session.created_at >= since)
    base = _scope_query(base, current_user)

    row = (await db.execute(base)).one()

    total_sessions = row.total_sessions or 0
    total_faces = int(row.total_faces)
    total_identified = int(row.total_identified)
    avg_latency = round(float(row.avg_latency), 3)
    identification_rate = round(
        (total_identified / total_faces * 100) if total_faces > 0 else 0, 1
    )

    # ── Active users (distinct user_ids with sessions) ───────────────────
    user_stmt = select(
        func.count(func.distinct(Session.user_id))
    ).where(Session.created_at >= since)
    user_stmt = _scope_query(user_stmt, current_user)
    active_users = (await db.execute(user_stmt)).scalar() or 0

    # ── Audit log count ──────────────────────────────────────────────────
    audit_stmt = select(func.count(AuditLog.id)).where(AuditLog.created_at >= since)
    if current_user.role == UserRole.org_admin:
        audit_stmt = audit_stmt.where(AuditLog.org_id == current_user.org_id)
    total_audit = (await db.execute(audit_stmt)).scalar() or 0

    return {
        "days": days,
        "total_sessions": total_sessions,
        "total_faces": total_faces,
        "total_identified": total_identified,
        "identification_rate": identification_rate,
        "avg_latency_s": avg_latency,
        "active_users": active_users,
        "total_audit_events": total_audit,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sessions over time — trend chart data
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/sessions-over-time")
async def sessions_over_time(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(require_role(["super_admin", "org_admin"])),
    db: AsyncSession = Depends(get_db),
):
    """Returns [{date, count, faces, identified}] grouped by day."""
    since = _date_filter(days)

    stmt = (
        select(
            func.date(Session.created_at).label("day"),
            func.count(Session.id).label("count"),
            func.coalesce(func.sum(Session.n_faces), 0).label("faces"),
            func.coalesce(func.sum(Session.n_identified), 0).label("identified"),
        )
        .where(Session.created_at >= since)
        .group_by(func.date(Session.created_at))
        .order_by(func.date(Session.created_at))
    )
    stmt = _scope_query(stmt, current_user)
    rows = (await db.execute(stmt)).all()

    return [
        {
            "date": str(r.day),
            "count": r.count,
            "faces": int(r.faces),
            "identified": int(r.identified),
        }
        for r in rows
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Emotion distribution — donut chart
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/emotion-distribution")
async def emotion_distribution(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(require_role(["super_admin", "org_admin"])),
    db: AsyncSession = Depends(get_db),
):
    """
    Parses results_json from sessions to count each emotion.
    Returns [{emotion, count}] sorted desc.
    """
    since = _date_filter(days)

    stmt = select(Session.results_json).where(
        Session.created_at >= since,
        Session.results_json.isnot(None),
    )
    stmt = _scope_query(stmt, current_user)
    rows = (await db.execute(stmt)).scalars().all()

    emotion_counts: dict[str, int] = {}
    for results_json in rows:
        results = []
        if isinstance(results_json, dict):
            results = results_json.get("results", [])
        elif isinstance(results_json, list):
            results = results_json

        for face in results:
            emotion = face.get("emotion", "unknown")
            if emotion:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    # Sort by count desc
    sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
    return [{"emotion": e, "count": c} for e, c in sorted_emotions]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Top identified persons — bar chart
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/top-identified")
async def top_identified(
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(require_role(["super_admin", "org_admin"])),
    db: AsyncSession = Depends(get_db),
):
    """
    Parses results_json to count how often each named person appears.
    Returns [{name, count}] sorted desc, top N.
    """
    since = _date_filter(days)

    stmt = select(Session.results_json).where(
        Session.created_at >= since,
        Session.results_json.isnot(None),
    )
    stmt = _scope_query(stmt, current_user)
    rows = (await db.execute(stmt)).scalars().all()

    person_counts: dict[str, int] = {}
    for results_json in rows:
        results = []
        if isinstance(results_json, dict):
            results = results_json.get("results", [])
        elif isinstance(results_json, list):
            results = results_json

        for face in results:
            name = face.get("name", "")
            if name and name.upper() != "UNKNOWN":
                person_counts[name] = person_counts.get(name, 0) + 1

    sorted_persons = sorted(person_counts.items(), key=lambda x: x[1], reverse=True)
    return [{"name": n, "count": c} for n, c in sorted_persons[:limit]]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Latency trend — sparkline / area chart
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/latency-trend")
async def latency_trend(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(require_role(["super_admin", "org_admin"])),
    db: AsyncSession = Depends(get_db),
):
    """Returns [{date, avg_latency_s, min_latency_s, max_latency_s}] by day."""
    since = _date_filter(days)

    stmt = (
        select(
            func.date(Session.created_at).label("day"),
            func.avg(Session.elapsed_s).label("avg_lat"),
            func.min(Session.elapsed_s).label("min_lat"),
            func.max(Session.elapsed_s).label("max_lat"),
        )
        .where(Session.created_at >= since)
        .group_by(func.date(Session.created_at))
        .order_by(func.date(Session.created_at))
    )
    stmt = _scope_query(stmt, current_user)
    rows = (await db.execute(stmt)).all()

    return [
        {
            "date": str(r.day),
            "avg_latency_s": round(float(r.avg_lat), 3),
            "min_latency_s": round(float(r.min_lat), 3),
            "max_latency_s": round(float(r.max_lat), 3),
        }
        for r in rows
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 6. User activity — leaderboard
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/user-activity")
async def user_activity(
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(require_role(["super_admin", "org_admin"])),
    db: AsyncSession = Depends(get_db),
):
    """Returns [{user_id, email, session_count, total_faces}] sorted by count."""
    since = _date_filter(days)

    stmt = (
        select(
            Session.user_id,
            User.email,
            func.count(Session.id).label("session_count"),
            func.coalesce(func.sum(Session.n_faces), 0).label("total_faces"),
        )
        .join(User, User.id == Session.user_id)
        .where(Session.created_at >= since)
        .group_by(Session.user_id, User.email)
        .order_by(func.count(Session.id).desc())
        .limit(limit)
    )
    stmt = _scope_query(stmt, current_user)
    rows = (await db.execute(stmt)).all()

    return [
        {
            "user_id": str(r.user_id),
            "email": r.email,
            "session_count": r.session_count,
            "total_faces": int(r.total_faces),
        }
        for r in rows
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 7. Hourly heatmap — when is the system used most?
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/hourly-heatmap")
async def hourly_heatmap(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(require_role(["super_admin", "org_admin"])),
    db: AsyncSession = Depends(get_db),
):
    """Returns [{hour: 0-23, count}] for a usage heatmap."""
    since = _date_filter(days)

    stmt = (
        select(
            func.extract("hour", Session.created_at).label("hour"),
            func.count(Session.id).label("count"),
        )
        .where(Session.created_at >= since)
        .group_by(func.extract("hour", Session.created_at))
        .order_by(func.extract("hour", Session.created_at))
    )
    stmt = _scope_query(stmt, current_user)
    rows = (await db.execute(stmt)).all()

    # Fill all 24 hours (some may have 0 sessions)
    hour_map = {int(r.hour): r.count for r in rows}
    return [{"hour": h, "count": hour_map.get(h, 0)} for h in range(24)]
