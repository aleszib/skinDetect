"""Timestamp extraction tests."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from skintrack.metadata.timestamps import extract_timestamp


def test_filename_timestamp_parsing_img_pattern(tmp_path: Path) -> None:
    result = extract_timestamp(tmp_path / "IMG_20240517_143022.jpg")

    assert result.taken_at == datetime(2024, 5, 17, 14, 30, 22)
    assert result.timestamp_source == "filename"
    assert result.timestamp_confidence == "medium"


def test_filename_timestamp_parsing_dash_pattern(tmp_path: Path) -> None:
    result = extract_timestamp(tmp_path / "2024-05-17_14-30-22.png")

    assert result.taken_at == datetime(2024, 5, 17, 14, 30, 22)
    assert result.timestamp_source == "filename"
    assert result.timestamp_confidence == "medium"


def test_file_mtime_fallback_is_low_confidence(tmp_path: Path) -> None:
    image_path = tmp_path / "plain.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)

    expected = datetime(2024, 5, 17, 14, 30, 22)
    epoch_seconds = expected.replace(tzinfo=timezone.utc).timestamp()
    os.utime(image_path, (epoch_seconds, epoch_seconds))

    result = extract_timestamp(image_path)

    assert result.taken_at == expected
    assert result.timestamp_source == "file_mtime"
    assert result.timestamp_confidence == "low"


def test_exif_timestamp_parsing(tmp_path: Path) -> None:
    image_path = tmp_path / "exif.jpg"
    exif = Image.Exif()
    exif[36867] = "2024:05:17 14:30:22"
    Image.new("RGB", (8, 8), color="white").save(image_path, exif=exif)

    result = extract_timestamp(image_path)

    assert result.taken_at == datetime(2024, 5, 17, 14, 30, 22)
    assert result.timestamp_source == "exif"
    assert result.timestamp_confidence == "high"


def test_unknown_timestamp_for_missing_file(tmp_path: Path) -> None:
    result = extract_timestamp(tmp_path / "plain.png")

    assert result.taken_at is None
    assert result.timestamp_source == "unknown"
    assert result.timestamp_confidence == "unknown"

