import uuid
import enum
from datetime import datetime
from sqlalchemy import String, Boolean, Enum as SAEnum, ForeignKey, Text, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from database.session import Base


class UserRole(str, enum.Enum):
    super_admin = "super_admin"
    authority_admin = "authority_admin"
    user_admin = "user_admin"


class User(Base):
    __tablename__ = "users"

    id:            Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email:         Mapped[str]       = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str]       = mapped_column(String(255), nullable=False)
    full_name:     Mapped[str]       = mapped_column(String(255), nullable=False)
    role:          Mapped[UserRole]  = mapped_column(SAEnum(UserRole), nullable=False)
    is_active:     Mapped[bool]      = mapped_column(Boolean, default=True)
    created_at:    Mapped[datetime]  = mapped_column(default=datetime.utcnow)
    last_login:    Mapped[datetime | None] = mapped_column(nullable=True)

    persons    = relationship("Person", back_populates="enrolled_by_user")
    audit_logs = relationship("AuditLog", back_populates="actor")


class Person(Base):
    __tablename__ = "persons"

    id:          Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    full_name:   Mapped[str]       = mapped_column(String(255), nullable=False)
    metadata_:   Mapped[dict]      = mapped_column("metadata", JSON, default=dict)
    enrolled_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), nullable=True)
    status:      Mapped[str]       = mapped_column(String(50), default="pending")
    created_at:  Mapped[datetime]  = mapped_column(default=datetime.utcnow)

    enrolled_by_user = relationship("User", back_populates="persons")
    embeddings       = relationship("FaceEmbedding", back_populates="person", cascade="all, delete-orphan")


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id:         Mapped[uuid.UUID]   = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    person_id:  Mapped[uuid.UUID]   = mapped_column(ForeignKey("persons.id", ondelete="CASCADE"))
    embedding:  Mapped[list[float]] = mapped_column(Vector(512), nullable=False)
    image_path: Mapped[str]         = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime]    = mapped_column(default=datetime.utcnow)

    person = relationship("Person", back_populates="embeddings")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id:         Mapped[uuid.UUID]    = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    actor_id:   Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    action:     Mapped[str]          = mapped_column(String(100), nullable=False, index=True)
    target_id:  Mapped[str | None]   = mapped_column(String(255), nullable=True)
    ip_address: Mapped[str | None]   = mapped_column(String(45), nullable=True)
    detail:     Mapped[dict]         = mapped_column(JSON, default=dict)
    timestamp:  Mapped[datetime]     = mapped_column(default=datetime.utcnow, index=True)

    actor = relationship("User", back_populates="audit_logs")


class UserSession(Base):
    __tablename__ = "user_sessions"

    id:                  Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id:             Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    refresh_token_hash:  Mapped[str]       = mapped_column(String(255), nullable=False)
    expires_at:          Mapped[datetime]  = mapped_column(nullable=False)
    created_at:          Mapped[datetime]  = mapped_column(default=datetime.utcnow)


# HNSW index for fast face similarity search — add after class definitions
hnsw_index = Index(
    "idx_face_embeddings_hnsw",
    FaceEmbedding.embedding,
    postgresql_using="hnsw",
    postgresql_with={"m": 16, "ef_construction": 64},
    postgresql_ops={"embedding": "vector_cosine_ops"},
)


# Audit action constants — import these anywhere you write a log
class AuditAction:
    LOGIN_OK      = "LOGIN_SUCCESS"
    LOGIN_FAIL    = "LOGIN_FAILED"
    ENROLL_PERSON = "ENROLL_PERSON"
    DELETE_PERSON = "DELETE_PERSON"
    ROLE_CHANGE   = "ROLE_CHANGE"
    INVITE_SENT   = "INVITE_SENT"
    FACE_LOGIN_OK = "FACE_LOGIN_SUCCESS"
    FACE_LOGIN_FAIL = "FACE_LOGIN_FAILED"