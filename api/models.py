"""
api/models.py
─────────────
SQLAlchemy ORM models for VisionX multi-tenant system.

Models
------
UserRole         — enum: super_admin, org_admin, member
MasterRole       — roles table (replaces enum in DB)
MasterAction     — actions table for RBAC
Permission       — role → action mapping
Organisation     — companies / teams (tenants)
User             — all users of every role
PasswordResetLog — password reset history (max 5 per user)
Person           — enrolled individuals in face recognition dataset
Session          — each prediction run is logged as a session
AuditLog         — every mutating action is recorded here
"""

import enum
import uuid
from datetime import datetime, date

from sqlalchemy import (
    Boolean, DateTime, Date, Enum, ForeignKey,
    Integer, Float, String, Text, func,
    JSON, Uuid, UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class UserRole(str, enum.Enum):
    super_admin = "super_admin"
    org_admin   = "org_admin"
    member      = "member"          # renamed from 'user' to avoid confusion


# ─────────────────────────────────────────────────────────────────────────────
# MasterRole
# ─────────────────────────────────────────────────────────────────────────────

class MasterRole(Base):
    __tablename__ = "master_roles"

    id          : Mapped[uuid.UUID]      = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    role_name   : Mapped[str]            = mapped_column(String(100), unique=True, nullable=False)
    description : Mapped[str | None]     = mapped_column(String(255), nullable=True)

    # Audit columns
    is_active   : Mapped[bool]           = mapped_column(Boolean, default=True,  nullable=False)
    is_deleted  : Mapped[bool]           = mapped_column(Boolean, default=False, nullable=False)
    created_at  : Mapped[datetime]       = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_by  : Mapped[uuid.UUID|None] = mapped_column(Uuid, nullable=True)
    created_ip  : Mapped[str|None]       = mapped_column(String(45), nullable=True)
    updated_at  : Mapped[datetime|None]  = mapped_column(DateTime(timezone=True), nullable=True)
    updated_by  : Mapped[uuid.UUID|None] = mapped_column(Uuid, nullable=True)
    updated_ip  : Mapped[str|None]       = mapped_column(String(45), nullable=True)

    permissions : Mapped[list["Permission"]] = relationship("Permission", back_populates="role")


# ─────────────────────────────────────────────────────────────────────────────
# MasterAction
# ─────────────────────────────────────────────────────────────────────────────

class MasterAction(Base):
    __tablename__ = "master_actions"

    id          : Mapped[uuid.UUID]      = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    action_name : Mapped[str]            = mapped_column(String(100), unique=True, nullable=False)
    description : Mapped[str | None]     = mapped_column(String(255), nullable=True)

    # Audit columns
    is_active   : Mapped[bool]           = mapped_column(Boolean, default=True,  nullable=False)
    is_deleted  : Mapped[bool]           = mapped_column(Boolean, default=False, nullable=False)
    created_at  : Mapped[datetime]       = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_by  : Mapped[uuid.UUID|None] = mapped_column(Uuid, nullable=True)
    created_ip  : Mapped[str|None]       = mapped_column(String(45), nullable=True)
    updated_at  : Mapped[datetime|None]  = mapped_column(DateTime(timezone=True), nullable=True)
    updated_by  : Mapped[uuid.UUID|None] = mapped_column(Uuid, nullable=True)
    updated_ip  : Mapped[str|None]       = mapped_column(String(45), nullable=True)

    permissions : Mapped[list["Permission"]] = relationship("Permission", back_populates="action")


# ─────────────────────────────────────────────────────────────────────────────
# Permission
# ─────────────────────────────────────────────────────────────────────────────

class Permission(Base):
    __tablename__  = "permissions"
    __table_args__ = (UniqueConstraint("role_id", "action_id"),)

    id        : Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    role_id   : Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("master_roles.id",   ondelete="CASCADE"), nullable=False)
    action_id : Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("master_actions.id", ondelete="CASCADE"), nullable=False)

    # Audit columns
    is_active  : Mapped[bool]           = mapped_column(Boolean, default=True, nullable=False)
    created_at : Mapped[datetime]       = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_by : Mapped[uuid.UUID|None] = mapped_column(Uuid, nullable=True)
    created_ip : Mapped[str|None]       = mapped_column(String(45), nullable=True)

    role   : Mapped["MasterRole"]   = relationship("MasterRole",   back_populates="permissions")
    action : Mapped["MasterAction"] = relationship("MasterAction", back_populates="permissions")


# ─────────────────────────────────────────────────────────────────────────────
# Organisation
# ─────────────────────────────────────────────────────────────────────────────

class Organisation(Base):
    __tablename__ = "organisations"

    id   : Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name : Mapped[str]       = mapped_column(String(255), nullable=False)

    # Audit columns
    is_active  : Mapped[bool]           = mapped_column(Boolean, default=True,  nullable=False)
    is_deleted : Mapped[bool]           = mapped_column(Boolean, default=False, nullable=False)
    created_at : Mapped[datetime]       = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_by : Mapped[uuid.UUID|None] = mapped_column(Uuid, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_ip : Mapped[str|None]       = mapped_column(String(45), nullable=True)
    updated_at : Mapped[datetime|None]  = mapped_column(DateTime(timezone=True), nullable=True)
    updated_by : Mapped[uuid.UUID|None] = mapped_column(Uuid, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    updated_ip : Mapped[str|None]       = mapped_column(String(45), nullable=True)
    deleted_at : Mapped[datetime|None]  = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_by : Mapped[uuid.UUID|None] = mapped_column(Uuid, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    deleted_ip : Mapped[str|None]       = mapped_column(String(45), nullable=True)

    users    : Mapped[list["User"]]    = relationship("User",    back_populates="organisation", foreign_keys="User.org_id")
    persons  : Mapped[list["Person"]]  = relationship("Person",  back_populates="organisation")
    sessions : Mapped[list["Session"]] = relationship("Session", back_populates="organisation")


# ─────────────────────────────────────────────────────────────────────────────
# User
# ─────────────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id            : Mapped[uuid.UUID]        = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    org_id        : Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("organisations.id", ondelete="SET NULL"), nullable=True)
    role : Mapped[str] = mapped_column(String(50), nullable=False, default="member")

    # Identity — document specifies: name, email, contact, password, dob
    full_name     : Mapped[str]              = mapped_column(String(255), nullable=False, default="")
    email         : Mapped[str]              = mapped_column(String(255), unique=True, nullable=False)
    password_hash : Mapped[str]              = mapped_column(String(255), nullable=False)
    contact       : Mapped[str | None]       = mapped_column(String(20),  nullable=True)
    dob           : Mapped[date | None]      = mapped_column(Date, nullable=True)

    # Invite + login tracking
    invited_by    : Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    last_login_at : Mapped[datetime | None]  = mapped_column(DateTime(timezone=True), nullable=True)

    # Audit columns
    is_active  : Mapped[bool]           = mapped_column(Boolean, default=True,  nullable=False)
    is_deleted : Mapped[bool]           = mapped_column(Boolean, default=False, nullable=False)
    created_at : Mapped[datetime]       = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_by : Mapped[uuid.UUID|None] = mapped_column(Uuid, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_ip : Mapped[str|None]       = mapped_column(String(45), nullable=True)
    updated_at : Mapped[datetime|None]  = mapped_column(DateTime(timezone=True), nullable=True)
    updated_by : Mapped[uuid.UUID|None] = mapped_column(Uuid, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    updated_ip : Mapped[str|None]       = mapped_column(String(45), nullable=True)
    deleted_at : Mapped[datetime|None]  = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_by : Mapped[uuid.UUID|None] = mapped_column(Uuid, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    deleted_ip : Mapped[str|None]       = mapped_column(String(45), nullable=True)

    organisation   : Mapped["Organisation | None"]      = relationship("Organisation", back_populates="users", foreign_keys=[org_id])
    sessions       : Mapped[list["Session"]]            = relationship("Session",      back_populates="user")
    audit_logs     : Mapped[list["AuditLog"]]           = relationship("AuditLog",     back_populates="actor", foreign_keys="AuditLog.actor_id")
    reset_logs     : Mapped[list["PasswordResetLog"]]   = relationship("PasswordResetLog", back_populates="user")


# ─────────────────────────────────────────────────────────────────────────────
# PasswordResetLog
# ─────────────────────────────────────────────────────────────────────────────

class PasswordResetLog(Base):
    __tablename__ = "password_reset_logs"

    id           : Mapped[uuid.UUID]        = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id      : Mapped[uuid.UUID]        = mapped_column(Uuid, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    reset_token  : Mapped[str | None]       = mapped_column(String(500), nullable=True)
    token_expiry : Mapped[datetime | None]  = mapped_column(DateTime(timezone=True), nullable=True)

    # Audit columns
    is_active  : Mapped[bool]       = mapped_column(Boolean, default=True,  nullable=False)
    is_deleted : Mapped[bool]       = mapped_column(Boolean, default=False, nullable=False)
    created_at : Mapped[datetime]   = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_ip : Mapped[str|None]   = mapped_column(String(45), nullable=True)

    user : Mapped["User"] = relationship("User", back_populates="reset_logs")


# ─────────────────────────────────────────────────────────────────────────────
# Person
# ─────────────────────────────────────────────────────────────────────────────

class Person(Base):
    __tablename__ = "persons"

    id          : Mapped[uuid.UUID]        = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    org_id      : Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("organisations.id", ondelete="CASCADE"), nullable=True)
    full_name   : Mapped[str]              = mapped_column(String(255), nullable=False)
    employee_id : Mapped[str | None]       = mapped_column(String(100), nullable=True)
    department  : Mapped[str | None]       = mapped_column(String(100), nullable=True)
    gcs_path    : Mapped[str | None]       = mapped_column(String(500), nullable=True)
    photo_count : Mapped[int]              = mapped_column(Integer, default=0)
    is_enrolled : Mapped[bool]             = mapped_column(Boolean, default=False)

    # Audit columns
    is_active  : Mapped[bool]           = mapped_column(Boolean, default=True,  nullable=False)
    is_deleted : Mapped[bool]           = mapped_column(Boolean, default=False, nullable=False)
    created_at : Mapped[datetime]       = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_by : Mapped[uuid.UUID|None] = mapped_column(Uuid, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_ip : Mapped[str|None]       = mapped_column(String(45), nullable=True)
    updated_at : Mapped[datetime|None]  = mapped_column(DateTime(timezone=True), nullable=True)
    updated_by : Mapped[uuid.UUID|None] = mapped_column(Uuid, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    updated_ip : Mapped[str|None]       = mapped_column(String(45), nullable=True)
    deleted_at : Mapped[datetime|None]  = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_by : Mapped[uuid.UUID|None] = mapped_column(Uuid, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    deleted_ip : Mapped[str|None]       = mapped_column(String(45), nullable=True)

    organisation : Mapped["Organisation | None"] = relationship("Organisation", back_populates="persons")
    creator      : Mapped["User | None"]         = relationship("User", foreign_keys=[created_by])


# ─────────────────────────────────────────────────────────────────────────────
# Session
# ─────────────────────────────────────────────────────────────────────────────

class Session(Base):
    __tablename__ = "sessions"

    id              : Mapped[uuid.UUID]        = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id         : Mapped[uuid.UUID]        = mapped_column(Uuid, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    org_id          : Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("organisations.id", ondelete="SET NULL"), nullable=True)
    n_faces         : Mapped[int]              = mapped_column(Integer, default=0)
    n_identified    : Mapped[int]              = mapped_column(Integer, default=0)
    elapsed_s       : Mapped[float]            = mapped_column(Float,   default=0.0)
    results_json    : Mapped[dict | None]      = mapped_column(JSON, nullable=True)
    annotated_image : Mapped[str | None]       = mapped_column(Text, nullable=True)
    note            : Mapped[str | None]       = mapped_column(Text, nullable=True)

    # Audit columns — sessions are never edited/deleted, only created
    created_at : Mapped[datetime]     = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_ip : Mapped[str | None]   = mapped_column(String(45), nullable=True)

    user         : Mapped["User"]              = relationship("User",         back_populates="sessions")
    organisation : Mapped["Organisation|None"] = relationship("Organisation", back_populates="sessions")


# ─────────────────────────────────────────────────────────────────────────────
# AuditLog
# ─────────────────────────────────────────────────────────────────────────────

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id          : Mapped[uuid.UUID]        = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    actor_id    : Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    org_id      : Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("organisations.id", ondelete="SET NULL"), nullable=True)
    action      : Mapped[str]              = mapped_column(String(100), nullable=False)
    target_type : Mapped[str | None]       = mapped_column(String(50),  nullable=True)
    target_id   : Mapped[str | None]       = mapped_column(String(255), nullable=True)
    detail      : Mapped[dict | None]      = mapped_column(JSON, nullable=True)
    ip_address  : Mapped[str | None]       = mapped_column(String(45), nullable=True)

    # Audit logs are immutable — only created_at needed
    created_at : Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    actor : Mapped["User | None"] = relationship("User", back_populates="audit_logs", foreign_keys=[actor_id])