"""Photo import manifest tests."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image

from skintrack.io.photos import (
    build_photo_import_manifest,
    manifest_to_json,
)
from skintrack.metadata.timestamps import TimestampResult


def _make_image(path: Path, size: tuple[int, int] = (12, 8), color: str = "white") -> None:
    image = Image.new("RGB", size, color=color)
    image.save(path)


def test_importing_folder_with_supported_images(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    _make_image(images / "one.jpg", (12, 8), "red")
    _make_image(images / "two.png", (14, 10), "blue")

    manifest = build_photo_import_manifest(images)

    assert manifest.counts.imported == 2
    assert manifest.counts.total_files == 2
    assert all(record.import_status == "imported" for record in manifest.photo_records)


def test_unsupported_files_are_marked_unsupported(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "note.txt").write_text("not an image", encoding="utf-8")
    (images / "photo.heic").write_bytes(b"heic-data")

    manifest = build_photo_import_manifest(images)

    assert manifest.counts.unsupported == 2
    assert {record.original_filename for record in manifest.photo_records} == {
        "note.txt",
        "photo.heic",
    }
    assert all(record.import_status == "unsupported" for record in manifest.photo_records)
    assert any("HEIC files are not supported" in warning for warning in manifest.warnings)


def test_unreadable_image_files_are_handled_without_crashing(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "broken.jpg").write_text("not a real jpeg", encoding="utf-8")

    manifest = build_photo_import_manifest(images)

    assert manifest.counts.unreadable == 1
    record = manifest.photo_records[0]
    assert record.import_status == "unreadable"
    assert record.width is None
    assert record.height is None


def test_timestamp_extraction_is_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    images = tmp_path / "images"
    images.mkdir()
    image_path = images / "timestamp.jpg"
    _make_image(image_path)

    called: list[Path] = []

    def fake_extract_timestamp(path: Path) -> TimestampResult:
        called.append(path)
        return TimestampResult(
            taken_at=datetime(2024, 5, 17, 14, 30, 22),
            timestamp_source="filename",
            timestamp_confidence="medium",
            notes=("stubbed",),
        )

    monkeypatch.setattr("skintrack.io.photos.extract_timestamp", fake_extract_timestamp)

    manifest = build_photo_import_manifest(images)

    assert called == [image_path]
    record = manifest.photo_records[0]
    assert record.taken_at == datetime(2024, 5, 17, 14, 30, 22)
    assert record.timestamp_source == "filename"


def test_file_hash_is_stable(tmp_path: Path) -> None:
    image_path = tmp_path / "hash.jpg"
    _make_image(image_path)

    manifest_one = build_photo_import_manifest(tmp_path)
    manifest_two = build_photo_import_manifest(tmp_path)

    assert manifest_one.photo_records[0].file_hash == manifest_two.photo_records[0].file_hash


def test_dimensions_are_captured_for_valid_images(tmp_path: Path) -> None:
    image_path = tmp_path / "size.png"
    _make_image(image_path, size=(31, 17))

    manifest = build_photo_import_manifest(tmp_path)
    record = manifest.photo_records[0]

    assert record.width == 31
    assert record.height == 17


def test_manifest_json_has_expected_schema_fields(tmp_path: Path) -> None:
    image_path = tmp_path / "schema.jpg"
    _make_image(image_path)

    manifest = build_photo_import_manifest(tmp_path)
    payload = json.loads(manifest_to_json(manifest))

    assert payload["schema_version"] == "1.0"
    assert "created_at" in payload
    assert "input_directory" in payload
    assert "photo_records" in payload
    assert "warnings" in payload
    assert "counts" in payload
    assert payload["counts"]["total_files"] == 1


def test_cli_command_produces_manifest_file(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    _make_image(images / "cli.jpg")
    output = tmp_path / "manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "skintrack.cli",
            "import-photos",
            str(images),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert output.exists()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["counts"]["imported"] == 1
    assert payload["photo_records"][0]["import_status"] == "imported"
