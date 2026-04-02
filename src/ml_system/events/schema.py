from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from pydantic import BaseModel, Field, field_validator


def _ensure_utc(v: datetime) -> datetime:
    if v.tzinfo is None:
        return v.replace(tzinfo=timezone.utc)
    return v.astimezone(timezone.utc)


class RawEvent(BaseModel):
    """Bronze-layer event. Feature windows use occurred_at only (never ingested_at)."""

    model_config = {"frozen": True}

    event_id: str = Field(..., description="Idempotency key; duplicate event_ids are dropped on ingest")
    entity_id: str
    event_type: str
    occurred_at: datetime
    ingested_at: datetime
    payload: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("occurred_at", "ingested_at")
    @classmethod
    def utc(cls, v: datetime) -> datetime:
        return _ensure_utc(v)
