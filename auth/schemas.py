"""
auth/schemas.py
────────────────
Pydantic request/response models for the auth module.

These are the shapes of JSON bodies that FastAPI validates automatically.
If a request doesn't match the schema, FastAPI returns 422 before your
route code even runs.
"""

from pydantic import BaseModel, EmailStr
from database.models import UserRole


class LoginRequest(BaseModel):
    """Body for POST /auth/login"""
    email: EmailStr
    password: str


class SignupRequest(BaseModel):
    """
    Body for POST /auth/signup.
    The 'token' field is the invite token sent to the user's email.
    """
    token: str
    full_name: str
    password: str


class InviteRequest(BaseModel):
    """Body for POST /auth/invite (super_admin only)"""
    email: EmailStr
    role: UserRole


class FaceLoginRequest(BaseModel):
    """
    Body for POST /auth/face-login.
    image_base64 is a base64-encoded JPEG/PNG of the person's face.
    """
    image_base64: str


class TokenResponse(BaseModel):
    """Returned on successful login — contains both token types."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    role: str


class UserResponse(BaseModel):
    """Safe user representation — never includes password_hash."""
    id: object          # UUID — serialised as string by FastAPI
    email: str
    full_name: str
    role: UserRole
    is_active: bool

    model_config = {"from_attributes": True}
