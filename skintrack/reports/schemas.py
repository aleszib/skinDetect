"""Initial report schemas for SkinTrack."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PhotoRecord(BaseModel):
    """Normalized import record for one image."""

    model_config = ConfigDict(extra="forbid")

    id: str
    original_filename: str
    file_hash: str
    stored_path: str
    width: int | None = None
    height: int | None = None
    taken_at: datetime | None = None
    timestamp_source: str
    timestamp_confidence: str
    timestamp_notes: list[str] = Field(default_factory=list)
    imported_at: datetime
    quality_status: str | None = None


class ChangeFlag(BaseModel):
    """Conservative report for a visually notable change."""

    model_config = ConfigDict(extra="forbid")

    id: str
    photo_ids: list[str] = Field(default_factory=list)
    track_id: str | None = None
    timestamp_start: datetime | None = None
    timestamp_end: datetime | None = None
    registration_confidence: float | None = None
    severity: str
    reason: str
    evidence: list[str] = Field(default_factory=list)
    text_summary: str
    annotated_image_path: str
    confidence: float | None = None
    uncertainty_notes: list[str] = Field(default_factory=list)
    created_at: datetime

