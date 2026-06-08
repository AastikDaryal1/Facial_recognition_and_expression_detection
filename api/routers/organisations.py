"""
api/routers/organisations.py
─────────────────────────────
Organisation management endpoints with role-based access control.

Endpoints
---------
GET    /organisations              → list orgs (super_admin sees all, org_admin sees own)
GET    /organisations/{id}         → get single org detail
POST   /organisations              → create a new org (super_admin only)
PATCH  /organisations/{id}         → update org name (super_admin only)
PATCH  /organisations/{id}/deactivate → deactivate org (super_admin only)
PATCH  /organisations/{id}/activate   → reactivate org (super_admin only)
DELETE /organisations/{id}         → permanently delete org (super_admin only)
"""

from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_current_user, require_role
from api.models import Organisation, User
from api.routers.audit import write_audit_log
from db.base import get_db

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class OrgOut(BaseModel):
    id        : str
    name      : str
    is_active : bool

    model_config = {"from_attributes": True}


class CreateOrgRequest(BaseModel):
    name: str


class UpdateOrgRequest(BaseModel):
    name: str


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _org_out(org: Organisation) -> dict:
    return {
        "id"       : str(org.id),
        "name"     : org.name,
        "is_active": org.is_active,
    }


async def _get_org_or_404(org_id: str, db: AsyncSession) -> Organisation:
    try:
        oid = UUID(org_id)
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid organisation ID format.")
    org = await db.get(Organisation, oid)
    if not org:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Organisation not found.")
    return org


def _check_org_access(current_user: User, org: Organisation) -> None:
    """org_admin can only view/touch their own org."""
    if (
        current_user.role == "org_admin"
        and org.id != current_user.org_id
    ):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "You can only access your own organisation."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("", response_model=list[OrgOut])
async def list_organisations(
    current_user : User = Depends(require_role(["super_admin", "org_admin"])),
    db           : AsyncSession = Depends(get_db),
):
    """
    List organisations.
    - super_admin  → sees ALL organisations
    - org_admin    → sees ONLY their own organisation
    """
    if current_user.role == "super_admin":
        result = await db.execute(select(Organisation))
    else:
        result = await db.execute(
            select(Organisation).where(Organisation.id == current_user.org_id)
        )
    orgs = result.scalars().all()
    return [_org_out(o) for o in orgs]


@router.get("/{org_id}", response_model=OrgOut)
async def get_organisation(
    org_id       : str,
    current_user : User = Depends(require_role(["super_admin", "org_admin"])),
    db           : AsyncSession = Depends(get_db),
):
    """Get a single organisation's details."""
    org = await _get_org_or_404(org_id, db)
    _check_org_access(current_user, org)
    return _org_out(org)


@router.post("", response_model=OrgOut, status_code=status.HTTP_201_CREATED)
async def create_organisation(
    payload      : CreateOrgRequest,
    current_user : User = Depends(require_role(["super_admin"])),
    db           : AsyncSession = Depends(get_db),
):
    """
    Create a new organisation.
    - super_admin only
    """
    org = Organisation(name=payload.name)
    db.add(org)
    await db.flush()  # get org.id

    await write_audit_log(
        db          = db,
        actor_id    = current_user.id,
        org_id      = org.id,
        action      = "org.create",
        target_type = "organisation",
        target_id   = str(org.id),
        detail      = {"name": org.name, "created_by": current_user.email},
    )
    await db.commit()
    await db.refresh(org)
    return _org_out(org)


@router.patch("/{org_id}", response_model=OrgOut)
async def update_organisation(
    org_id       : str,
    payload      : UpdateOrgRequest,
    current_user : User = Depends(require_role(["super_admin"])),
    db           : AsyncSession = Depends(get_db),
):
    """
    Update an organisation's name.
    - super_admin only
    """
    org = await _get_org_or_404(org_id, db)
    org.name = payload.name
    await db.commit()
    await db.refresh(org)
    return _org_out(org)


@router.patch("/{org_id}/deactivate", response_model=OrgOut)
async def deactivate_organisation(
    org_id       : str,
    current_user : User = Depends(require_role(["super_admin"])),
    db           : AsyncSession = Depends(get_db),
):
    """
    Deactivate an organisation (all users in it lose access).
    - super_admin only
    """
    org = await _get_org_or_404(org_id, db)
    if not org.is_active:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Organisation is already deactivated.")
    org.is_active = False
    await write_audit_log(
        db          = db,
        actor_id    = current_user.id,
        org_id      = org.id,
        action      = "org.deactivate",
        target_type = "organisation",
        target_id   = str(org.id),
        detail      = {"name": org.name},
    )
    await db.commit()
    await db.refresh(org)
    return _org_out(org)


@router.patch("/{org_id}/activate", response_model=OrgOut)
async def activate_organisation(
    org_id       : str,
    current_user : User = Depends(require_role(["super_admin"])),
    db           : AsyncSession = Depends(get_db),
):
    """
    Reactivate a deactivated organisation.
    - super_admin only
    """
    org = await _get_org_or_404(org_id, db)
    if org.is_active:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Organisation is already active.")
    org.is_active = True
    await db.commit()
    await db.refresh(org)
    return _org_out(org)


@router.delete("/{org_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_organisation(
    org_id       : str,
    current_user : User = Depends(require_role(["super_admin"])),
    db           : AsyncSession = Depends(get_db),
):
    """
    Permanently delete an organisation.
    - super_admin only
    - All users in this org will have org_id set to NULL (due to FK ON DELETE SET NULL)
    """
    org = await _get_org_or_404(org_id, db)
    await db.delete(org)
    await db.commit()