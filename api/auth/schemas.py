"""
api/auth/schemas.py
───────────────────
Pydantic request and response models for auth endpoints.
"""

from pydantic import BaseModel, EmailStr
from api.models import UserRole


class SignupRequest(BaseModel):
    email    : EmailStr
    password : str
    org_name : str          # creates a new org at signup (for super_admin seeding)


class LoginRequest(BaseModel):
    email    : EmailStr
    password : str


class InviteSignupRequest(BaseModel):
    email        : EmailStr
    password     : str
    invite_token : str      # token issued by /auth/invite


class InviteRequest(BaseModel):
    email  : EmailStr
    role   : UserRole        # org_admin or user only (super_admin invites via CLI)
    org_id : str | None = None  # required when super_admin invites; ignored for org_admin


class TokenResponse(BaseModel):
    access_token  : str
    refresh_token : str
    token_type    : str = "bearer"


class UserOut(BaseModel):
    id    : str
    email : str
    role  : UserRole
    org_id: str | None

    model_config = {"from_attributes": True}