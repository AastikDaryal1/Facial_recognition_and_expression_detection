"""
api/models.py
─────────────
SQLAlchemy ORM models for the auth/RBAC system.

Models
------
UserRole     — enum: super_admin, org_admin, user
Organisation — companies / teams
User         — all users of every role
Person       — enrolled individuals whose faces are in the recognition dataset
Session      — each prediction run is logged as a session
AuditLog     — every mutating action is recorded here
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean, DateTime, Enum, ForeignKey,
    Integer, Float, String, Text, func,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class UserRole(str, enum.Enum):
    super_admin = "super_admin"
    org_admin   = "org_admin"
    user        = "user"


# ─────────────────────────────────────────────────────────────────────────────
# Organisation
# ─────────────────────────────────────────────────────────────────────────────

class Organisation(Base):
    __tablename__ = "organisations"

    id         : Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name       : Mapped[str]       = mapped_column(String(255), nullable=False)
    is_active  : Mapped[bool]      = mapped_column(Boolean, default=True, nullable=False)
    created_at : Mapped[datetime]  = mapped_column(DateTime(timezone=True), server_default=func.now())

    users    : Mapped[list["User"]]    = relationship("User",    back_populates="organisation")
    persons  : Mapped[list["Person"]]  = relationship("Person",  back_populates="organisation")
    sessions : Mapped[list["Session"]] = relationship("Session", back_populates="organisation")


# ─────────────────────────────────────────────────────────────────────────────
# User
# ─────────────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id            : Mapped[uuid.UUID]        = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email         : Mapped[str]              = mapped_column(String(255), unique=True, nullable=False)
    password_hash : Mapped[str]              = mapped_column(String(255), nullable=False)
    role          : Mapped[UserRole]         = mapped_column(Enum(UserRole), nullable=False, default=UserRole.user)
    org_id        : Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("organisations.id", ondelete="SET NULL"), nullable=True)
    is_active     : Mapped[bool]             = mapped_column(Boolean, default=True, nullable=False)
    invited_by    : Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at    : Mapped[datetime]         = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_login_at : Mapped[datetime | None]  = mapped_column(DateTime(timezone=True), nullable=True)

    organisation : Mapped["Organisation | None"] = relationship("Organisation", back_populates="users")
    sessions     : Mapped[list["Session"]]       = relationship("Session",      back_populates="user")
    audit_logs   : Mapped[list["AuditLog"]]      = relationship("AuditLog",     back_populates="actor", foreign_keys="AuditLog.actor_id")


# ─────────────────────────────────────────────────────────────────────────────
# Person — enrolled individual whose face is in the recognition dataset
# ─────────────────────────────────────────────────────────────────────────────

class Person(Base):
    __tablename__ = "persons"

    id           : Mapped[uuid.UUID]        = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id       : Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("organisations.id", ondelete="SET NULL"), nullable=True)
    added_by     : Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    # Identity
    full_name    : Mapped[str]              = mapped_column(String(255), nullable=False)
    employee_id  : Mapped[str | None]       = mapped_column(String(100), nullable=True)   # optional company ID
    department   : Mapped[str | None]       = mapped_column(String(100), nullable=True)

    # Dataset tracking
    # gcs_path stays NULL until GCS upload is wired up in a later phase.
    # Format will be: dataset/{org_id}/{person_id}/
    gcs_path     : Mapped[str | None]       = mapped_column(String(500), nullable=True)
    photo_count  : Mapped[int]              = mapped_column(Integer, default=0)           # how many photos uploaded
    is_enrolled  : Mapped[bool]             = mapped_column(Boolean, default=False)       # True once model has been trained on this person
    is_active    : Mapped[bool]             = mapped_column(Boolean, default=True)        # soft delete

    created_at   : Mapped[datetime]         = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at   : Mapped[datetime]         = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    organisation : Mapped["Organisation | None"] = relationship("Organisation", back_populates="persons")
    creator      : Mapped["User | None"]         = relationship("User", foreign_keys=[added_by])


# ─────────────────────────────────────────────────────────────────────────────
# Session  — each prediction run (image scan) is stored here
# ─────────────────────────────────────────────────────────────────────────────

class Session(Base):
    __tablename__ = "sessions"

    id           : Mapped[uuid.UUID]        = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id      : Mapped[uuid.UUID]        = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    org_id       : Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("organisations.id", ondelete="SET NULL"), nullable=True)

    # Scan summary
    n_faces      : Mapped[int]   = mapped_column(Integer, default=0)
    n_identified : Mapped[int]   = mapped_column(Integer, default=0)
    elapsed_s    : Mapped[float] = mapped_column(Float,   default=0.0)

    # Full JSON results from inference (list of FaceResult dicts)
    results_json : Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Optional note added by org_admin
    note         : Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at   : Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at   : Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user         : Mapped["User"]              = relationship("User",         back_populates="sessions")
    organisation : Mapped["Organisation|None"] = relationship("Organisation", back_populates="sessions")


# ─────────────────────────────────────────────────────────────────────────────
# AuditLog  — immutable record of every mutating action
# ─────────────────────────────────────────────────────────────────────────────

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id          : Mapped[uuid.UUID]        = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    actor_id    : Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    org_id      : Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)

    # What happened
    action      : Mapped[str]              = mapped_column(String(100), nullable=False)   # e.g. "user.invite", "user.deactivate"
    target_type : Mapped[str | None]       = mapped_column(String(50),  nullable=True)    # e.g. "user", "organisation", "session"
    target_id   : Mapped[str | None]       = mapped_column(String(255), nullable=True)    # UUID of affected record
    detail      : Mapped[dict | None]      = mapped_column(JSONB, nullable=True)          # extra context

    created_at  : Mapped[datetime]         = mapped_column(DateTime(timezone=True), server_default=func.now())

    actor       : Mapped["User | None"]    = relationship("User", back_populates="audit_logs", foreign_keys=[actor_id])