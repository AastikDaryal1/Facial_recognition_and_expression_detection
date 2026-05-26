"""
enrollment/schemas.py
──────────────────────
Pydantic models for the enrollment (person management) module.
"""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class PersonCreate(BaseModel):
    """Body for POST /persons — create a new person record."""
    full_name: str
    metadata: dict[str, Any] = {}


class PersonResponse(BaseModel):
    """Safe person representation returned by the API."""
    id: uuid.UUID
    full_name: str
    metadata: dict[str, Any]
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}

    # Map ORM column name 'metadata_' → JSON key 'metadata'
    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        if hasattr(obj, "metadata_"):
            obj.__dict__.setdefault("metadata", obj.metadata_)
        return super().model_validate(obj, *args, **kwargs)
