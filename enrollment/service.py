"""
enrollment/service.py
──────────────────────
Business logic for managing enrolled persons.
"""

import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from database.models import Person, FaceEmbedding
from storage.gcs_storage import GCSStorage

async def get_person(person_id: uuid.UUID, db: AsyncSession) -> Person | None:
    result = await db.execute(select(Person).where(Person.id == person_id))
    return result.scalar_one_or_none()

async def delete_person_data(person_id: uuid.UUID, db: AsyncSession, storage: GCSStorage = None) -> bool:
    """
    Deletes a person's embeddings, images from GCS, and the person record.
    Returns True if deleted, False if not found.
    """
    person = await get_person(person_id, db)
    if not person:
        return False

    # Note: FaceEmbedding has cascade delete, so deleting Person deletes embeddings.
    # But we might need to delete from GCS.
    # The embeddings table might contain paths.
    result = await db.execute(select(FaceEmbedding.image_path).where(FaceEmbedding.person_id == person_id))
    image_paths = result.scalars().all()
    
    if storage:
        for path in image_paths:
            if path:
                try:
                    storage.delete_blob(path)
                except Exception:
                    pass
    
    await db.delete(person)
    await db.commit()
    return True
