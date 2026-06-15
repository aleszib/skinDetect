"""Timestamp extraction for imported photos."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from PIL import Image, UnidentifiedImageError

_EXIF_TAG_IDS: Final[tuple[int, ...]] = (36867, 36868, 306)

_FILENAME_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"IMG_(\d{8})_(\d{6})$", re.IGNORECASE), "%Y%m%d%H%M%S"),
    (
        re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})$", re.IGNORECASE),
        "%Y-%m-%d %H-%M-%S",
    ),
    (re.compile(r"(\d{8})-(\d{6})$", re.IGNORECASE), "%Y%m%d%H%M%S"),
)


@dataclass(frozen=True)
class TimestampResult:
    """Normalized timestamp provenance for one imported photo."""

    taken_at: datetime | None
    timestamp_source: str
    timestamp_confidence: str
    notes: tuple[str, ...] = field(default_factory=tuple)


def extract_timestamp(photo_path: str | Path) -> TimestampResult:
    """Extract a timestamp from EXIF, filename, or filesystem metadata."""

    path = Path(photo_path)

    exif_result = _extract_exif_timestamp(path)
    if exif_result is not None:
        return exif_result

    filename_result = _extract_filename_timestamp(path)
    if filename_result is not None:
        return filename_result

    file_mtime_result = _extract_file_mtime_timestamp(path)
    if file_mtime_result is not None:
        return file_mtime_result

    return TimestampResult(
        taken_at=None,
        timestamp_source="unknown",
        timestamp_confidence="unknown",
        notes=("No EXIF, filename, or filesystem timestamp could be obtained.",),
    )


def _extract_exif_timestamp(path: Path) -> TimestampResult | None:
    if not path.exists():
        return None

    try:
        with Image.open(path) as image:
            exif = image.getexif()
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return None

    if not exif:
        return None

    for tag_id in _EXIF_TAG_IDS:
        raw_value = exif.get(tag_id)
        if raw_value is None:
            continue

        parsed = _parse_exif_datetime(raw_value)
        if parsed is not None:
            return TimestampResult(
                taken_at=parsed,
                timestamp_source="exif",
                timestamp_confidence="high",
                notes=(f"Parsed EXIF tag {tag_id}.",),
            )

    return None


def _extract_filename_timestamp(path: Path) -> TimestampResult | None:
    stem = path.stem

    for pattern, strptime_format in _FILENAME_PATTERNS:
        match = pattern.search(stem)
        if not match:
            continue

        timestamp_text = " ".join(match.groups()) if " " in strptime_format else "".join(
            match.groups()
        )
        try:
            taken_at = datetime.strptime(timestamp_text, strptime_format)
        except ValueError:
            continue

        return TimestampResult(
            taken_at=taken_at,
            timestamp_source="filename",
            timestamp_confidence="medium",
            notes=(f"Matched filename pattern {pattern.pattern}.",),
        )

    return None


def _extract_file_mtime_timestamp(path: Path) -> TimestampResult | None:
    try:
        stat_result = path.stat()
    except FileNotFoundError:
        return None
    except OSError:
        return None

    taken_at = datetime.fromtimestamp(stat_result.st_mtime, tz=timezone.utc).replace(tzinfo=None)
    return TimestampResult(
        taken_at=taken_at,
        timestamp_source="file_mtime",
        timestamp_confidence="low",
        notes=("Used filesystem modification time as a fallback.",),
    )


def _parse_exif_datetime(raw_value: object) -> datetime | None:
    if isinstance(raw_value, bytes):
        raw_text = raw_value.decode("utf-8", errors="ignore").strip()
    else:
        raw_text = str(raw_value).strip()

    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw_text, fmt)
        except ValueError:
            continue

    return None

