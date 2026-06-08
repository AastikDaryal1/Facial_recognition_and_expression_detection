"""
api/routers/users.py
─────────────────────
User management endpoints with full role-based access control.
Role is a plain string column: "super_admin", "org_admin", "member"
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_current_user, require_role
from api.models import User
from api.routers.audit import write_audit_log
from db.base import get_db

router = APIRouter()

VALID_ROLES = {"super_admin", "org_admin", "member"}


class UserOut(BaseModel):
    id        : str
    full_name : str
    email     : str
    role      : str
    org_id    : str | None
    is_active : bool

    model_config = {"from_attributes": True}


class ChangeRoleRequest(BaseModel):
    role: str


def _user_out(user: User) -> dict:
    return {
        "id"        : str(user.id),
        "full_name" : user.full_name,
        "email"     : user.email,
        "role"      : user.role,          # plain string — no .value
        "org_id"    : str(user.org_id) if user.org_id else None,
        "is_active" : user.is_active,
    }


async def _get_user_or_404(user_id: str, db: AsyncSession) -> User:
    try:
        uid = UUID(user_id)
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid user ID format.")
    user = await db.get(User, uid)
    if not user:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found.")
    return user


def _check_org_access(current_user: User, target_user: User) -> None:
    if (
        current_user.role == "org_admin"
        and target_user.org_id != current_user.org_id
    ):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "You can only manage users within your own organisation.",
        )


@router.get("", response_model=list[UserOut])
async def list_users(
    current_user : User = Depends(require_role(["super_admin", "org_admin"])),
    db           : AsyncSession = Depends(get_db),
):
    if current_user.role == "super_admin":
        result = await db.execute(select(User))
    else:
        result = await db.execute(
            select(User).where(User.org_id == current_user.org_id)
        )
    return [_user_out(u) for u in result.scalars().all()]


@router.get("/{user_id}", response_model=UserOut)
async def get_user(
    user_id      : str,
    current_user : User = Depends(require_role(["super_admin", "org_admin"])),
    db           : AsyncSession = Depends(get_db),
):
    target = await _get_user_or_404(user_id, db)
    _check_org_access(current_user, target)
    return _user_out(target)


@router.patch("/{user_id}/role", response_model=UserOut)
async def change_user_role(
    user_id      : str,
    payload      : ChangeRoleRequest,
    current_user : User = Depends(require_role(["super_admin"])),
    db           : AsyncSession = Depends(get_db),
):
    if payload.role not in VALID_ROLES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Invalid role. Must be one of: {VALID_ROLES}")

    target = await _get_user_or_404(user_id, db)
    if str(target.id) == str(current_user.id):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "You cannot change your own role.")

    old_role    = target.role          # plain string — no .value
    target.role = payload.role
    await write_audit_log(
        db=db, actor_id=current_user.id, org_id=current_user.org_id,
        action="user.role_change", target_type="user", target_id=str(target.id),
        detail={"email": target.email, "old_role": old_role, "new_role": payload.role},
    )
    await db.commit()
    await db.refresh(target)
    return _user_out(target)


@router.patch("/{user_id}/deactivate", response_model=UserOut)
async def deactivate_user(
    user_id      : str,
    current_user : User = Depends(require_role(["super_admin", "org_admin"])),
    db           : AsyncSession = Depends(get_db),
):
    target = await _get_user_or_404(user_id, db)
    _check_org_access(current_user, target)
    if str(target.id) == str(current_user.id):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "You cannot deactivate your own account.")
    if not target.is_active:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User is already deactivated.")

    target.is_active = False
    await write_audit_log(
        db=db, actor_id=current_user.id, org_id=current_user.org_id,
        action="user.deactivate", target_type="user", target_id=str(target.id),
        detail={"email": target.email, "role": target.role},   # plain string
    )
    await db.commit()
    await db.refresh(target)
    return _user_out(target)


@router.patch("/{user_id}/activate", response_model=UserOut)
async def activate_user(
    user_id      : str,
    current_user : User = Depends(require_role(["super_admin", "org_admin"])),
    db           : AsyncSession = Depends(get_db),
):
    target = await _get_user_or_404(user_id, db)
    _check_org_access(current_user, target)
    if target.is_active:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User is already active.")

    target.is_active = True
    await write_audit_log(
        db=db, actor_id=current_user.id, org_id=current_user.org_id,
        action="user.activate", target_type="user", target_id=str(target.id),
        detail={"email": target.email, "role": target.role},   # plain string
    )
    await db.commit()
    await db.refresh(target)
    return _user_out(target)


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id      : str,
    current_user : User = Depends(require_role(["super_admin"])),
    db           : AsyncSession = Depends(get_db),
):
    target = await _get_user_or_404(user_id, db)
    if str(target.id) == str(current_user.id):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "You cannot delete your own account.")

    await write_audit_log(
        db=db, actor_id=current_user.id, org_id=current_user.org_id,
        action="user.delete", target_type="user", target_id=str(target.id),
        detail={"email": target.email, "role": target.role},   # plain string
    )
    await db.delete(target)
    await db.commit()