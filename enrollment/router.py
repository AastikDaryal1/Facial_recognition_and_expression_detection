"""
enrollment/router.py
─────────────────────
Endpoints for managing enrolled persons (faces).
"""

import os
import shutil
import tempfile
import uuid
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user, require_role
from audit.service import write_audit_log
from config.settings import settings
from database.models import AuditAction, Person, User
from database.session import get_db
from enrollment.schemas import PersonCreate, PersonResponse
from enrollment.service import delete_person_data
from tasks.training_job import run_enrollment_pipeline

router = APIRouter(tags=["Enrollment"])

@router.post("", response_model=PersonResponse, status_code=status.HTTP_201_CREATED)
async def create_person(
    body: PersonCreate,
    current_user: User = Depends(require_role(["user_admin", "authority_admin", "super_admin"])),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new person record with status="pending".
    """
    new_person = Person(
        full_name=body.full_name,
        metadata_=body.metadata,
        enrolled_by=current_user.id,
        status="pending"
    )
    db.add(new_person)
    await db.commit()
    await db.refresh(new_person)
    return PersonResponse.model_validate(new_person)

@router.post("/{person_id}/photos")
async def upload_photos(
    person_id: uuid.UUID,
    files: List[UploadFile] = File(...),
    current_user: User = Depends(require_role(["user_admin", "authority_admin", "super_admin"])),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload photos for enrollment. Triggers Celery pipeline.
    """
    if len(files) < settings.min_photos_required:
        raise HTTPException(
            status_code=400,
            detail=f"At least {settings.min_photos_required} photos are required."
        )

    # Verify person exists
    result = await db.execute(select(Person).where(Person.id == person_id))
    person = result.scalar_one_or_none()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    # Save to temp dir
    temp_dir = tempfile.mkdtemp(prefix=f"enroll_{person_id}_")
    saved_paths = []
    
    for file in files:
        if not file.filename:
            continue
        safe_filename = os.path.basename(file.filename)
        file_path = os.path.join(temp_dir, safe_filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_paths.append(file_path)

    # Trigger Celery
    run_enrollment_pipeline.delay(str(person_id), saved_paths)

    return {"message": "processing", "person_id": str(person_id)}

@router.delete("/{person_id}")
async def delete_person(
    person_id: uuid.UUID,
    current_user: User = Depends(require_role(["user_admin", "authority_admin", "super_admin"])),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a person and all their data.
    """
    # Initialize GCSStorage if possible, though it's sync. We can import it here.
    from storage.gcs_storage import GCSStorage
    storage = None
    try:
        storage = GCSStorage()
    except Exception:
        pass

    deleted = await delete_person_data(person_id, db, storage)
    if not deleted:
        raise HTTPException(status_code=404, detail="Person not found")

    await write_audit_log(
        db,
        action=AuditAction.DELETE_PERSON,
        actor_id=current_user.id,
        target_id=str(person_id),
    )
    
    # Ideally trigger a retrain Celery job here
    from tasks.training_job import retrain_svm
    retrain_svm.delay()

    return {"message": "deleted"}

@router.get("", response_model=List[PersonResponse])
async def list_persons(
    status: str | None = None,
    current_user: User = Depends(require_role(["authority_admin", "super_admin"])),
    db: AsyncSession = Depends(get_db),
):
    """
    List all persons. Optionally filter by status.
    """
    stmt = select(Person)
    if status:
        stmt = stmt.where(Person.status == status)
        
    result = await db.execute(stmt)
    persons = result.scalars().all()
    return [PersonResponse.model_validate(p) for p in persons]
