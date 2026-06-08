"""
api/auth/schemas.py
───────────────────
Pydantic request and response models for auth endpoints.
Role is handled as a plain string: "super_admin", "org_admin", "member"
"""

from pydantic import BaseModel, EmailStr


class SignupRequest(BaseModel):
    full_name : str
    email     : EmailStr
    password  : str
    org_name  : str          # creates a new org at signup (for super_admin seeding)


class LoginRequest(BaseModel):
    email    : EmailStr
    password : str


class InviteSignupRequest(BaseModel):
    full_name    : str
    email        : EmailStr
    password     : str
    contact      : str | None = None
    invite_token : str        # token issued by /auth/invite


class InviteRequest(BaseModel):
    email  : EmailStr
    role   : str              # "org_admin" or "member" only
    org_id : str | None = None


class TokenResponse(BaseModel):
    access_token  : str
    refresh_token : str
    token_type    : str = "bearer"


class UserOut(BaseModel):
    id        : str
    full_name : str
    email     : str
    role      : str
    org_id    : str | None

    model_config = {"from_attributes": True}