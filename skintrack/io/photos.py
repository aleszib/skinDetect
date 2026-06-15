"""Local folder photo import and manifest generation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from skintrack.metadata.timestamps import TimestampResult, extract_timestamp
from skintrack.quality.images import ImageQualityAssessment, assess_image_quality

SUPPORTED_IMAGE_EXTENSIONS: Final[set[str]] = {".jpg", ".jpeg", ".png"}
HEIC_EXTENSION: Final[str] = ".heic"

ImportStatus = Literal["imported", "skipped", "unreadable", "unsupported"]


class PhotoImportRecord(BaseModel):
    """A single file discovered during photo import."""

    model_config = ConfigDict(extra="forbid")

    original_path: str
    original_filename: str
    file_hash: str | None = None
    hash_algorithm: str = "sha256"
    taken_at: datetime | None = None
    timestamp_source: str
    timestamp_confidence: str
    timestamp_notes: list[str] = Field(default_factory=list)
    quality: ImageQualityAssessment | None = None
    width: int | None = None
    height: int | None = None
    import_status: ImportStatus
    notes: list[str] = Field(default_factory=list)


class ImportCounts(BaseModel):
    """Status counts for a photo import scan."""

    model_config = ConfigDict(extra="forbid")

    total_files: int
    imported: int
    skipped: int
    unreadable: int
    unsupported: int
    low_quality: int


class PhotoImportManifest(BaseModel):
    """JSON manifest describing a local photo import scan."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    created_at: datetime
    input_directory: str
    recursive: bool = True
    photo_records: list[PhotoImportRecord] = Field(default_factory=list)
    counts: ImportCounts
    warnings: list[str] = Field(default_factory=list)


def build_photo_import_manifest(
    input_directory: str | Path,
    *,
    recursive: bool = True,
) -> PhotoImportManifest:
    """Scan a local directory and build a JSON-serializable photo manifest."""

    root = Path(input_directory).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root}")

    records: list[PhotoImportRecord] = []
    warnings: list[str] = []

    for path in _iter_input_files(root, recursive=recursive):
        records.append(_build_record(path, warnings=warnings))

    counts = _compute_counts(records)
    return PhotoImportManifest(
        created_at=datetime.now(timezone.utc),
        input_directory=str(root),
        recursive=recursive,
        photo_records=records,
        counts=counts,
        warnings=warnings,
    )


def write_photo_import_manifest(
    input_directory: str | Path,
    output_path: str | Path,
    *,
    recursive: bool = True,
) -> PhotoImportManifest:
    """Build and write a photo import manifest to disk."""

    manifest = build_photo_import_manifest(input_directory, recursive=recursive)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(manifest_to_json(manifest), encoding="utf-8")
    return manifest


def _iter_input_files(root: Path, *, recursive: bool) -> list[Path]:
    if recursive:
        candidates = [path for path in root.rglob("*") if path.is_file()]
    else:
        candidates = [path for path in root.iterdir() if path.is_file()]

    return sorted(candidates, key=lambda path: str(path))


def _build_record(path: Path, *, warnings: list[str]) -> PhotoImportRecord:
    extension = path.suffix.lower()
    original_path = str(path.resolve())
    timestamp = _extract_timestamp_for_record(path)
    notes = list(timestamp.notes)
    file_hash, hash_notes = _safe_hash_file(path)
    notes.extend(hash_notes)

    if extension in SUPPORTED_IMAGE_EXTENSIONS:
        quality = assess_image_quality(path)
        return _build_supported_image_record(
            path,
            original_path,
            timestamp,
            file_hash,
            notes,
            quality=quality,
            warnings=warnings,
        )

    if extension == HEIC_EXTENSION:
        warning = "HEIC files are not supported in this release."
        _append_warning(warnings, warning)
        notes.append(warning)
        return PhotoImportRecord(
            original_path=original_path,
            original_filename=path.name,
            file_hash=file_hash,
            taken_at=timestamp.taken_at,
            timestamp_source="unsupported",
            timestamp_confidence="unknown",
            timestamp_notes=notes,
            quality=None,
            import_status="unsupported",
            notes=["File extension .heic is not supported yet."],
        )

    warning = f"Unsupported file extension {extension or '[none]'}: {path.name}"
    _append_warning(warnings, warning)
    notes.append(warning)
    return PhotoImportRecord(
        original_path=original_path,
        original_filename=path.name,
        file_hash=file_hash,
        taken_at=timestamp.taken_at,
        timestamp_source="unsupported",
        timestamp_confidence="unknown",
        timestamp_notes=notes,
        quality=None,
        import_status="unsupported",
        notes=["Skipped because the file extension is not supported."],
    )


def _build_supported_image_record(
    path: Path,
    original_path: str,
    timestamp: TimestampResult,
    file_hash: str | None,
    notes: list[str],
    *,
    quality: ImageQualityAssessment,
    warnings: list[str],
) -> PhotoImportRecord:
    notes.extend(quality.warnings)
    for warning in quality.warnings:
        _append_warning(warnings, f"{path.name}: {warning}")

    if not quality.readable:
        unreadable_note = quality.warnings[0] if quality.warnings else "Image is unreadable."
        return PhotoImportRecord(
            original_path=original_path,
            original_filename=path.name,
            file_hash=file_hash,
            taken_at=timestamp.taken_at,
            timestamp_source=timestamp.timestamp_source,
            timestamp_confidence=timestamp.timestamp_confidence,
            timestamp_notes=notes,
            quality=quality,
            width=None,
            height=None,
            import_status="unreadable",
            notes=[unreadable_note],
        )

    return PhotoImportRecord(
        original_path=original_path,
        original_filename=path.name,
        file_hash=file_hash,
        taken_at=timestamp.taken_at,
        timestamp_source=timestamp.timestamp_source,
        timestamp_confidence=timestamp.timestamp_confidence,
        timestamp_notes=notes,
        quality=quality,
        width=quality.width,
        height=quality.height,
        import_status="imported",
    )


def _extract_timestamp_for_record(path: Path) -> TimestampResult:
    return extract_timestamp(path)


def _safe_hash_file(path: Path) -> tuple[str | None, list[str]]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest(), []
    except OSError as exc:
        return None, [f"File hash could not be computed: {exc.__class__.__name__}."]


def _compute_counts(records: list[PhotoImportRecord]) -> ImportCounts:
    status_counts = Counter(record.import_status for record in records)
    low_quality_count = sum(
        1
        for record in records
        if record.quality is not None and record.quality.status == "low_quality"
    )
    return ImportCounts(
        total_files=len(records),
        imported=status_counts.get("imported", 0),
        skipped=status_counts.get("skipped", 0),
        unreadable=status_counts.get("unreadable", 0),
        unsupported=status_counts.get("unsupported", 0),
        low_quality=low_quality_count,
    )


def manifest_to_json(manifest: PhotoImportManifest) -> str:
    """Serialize a manifest as stable pretty-printed JSON."""

    return json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=False)


def _append_warning(warnings: list[str], warning: str) -> None:
    if warning not in warnings:
        warnings.append(warning)
