"""Manual candidate-region validation tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

from skintrack.io.photos import write_photo_import_manifest
from skintrack.regions.manual import validate_candidate_regions


def _make_image(path: Path, *, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", size, color=color)
    draw = ImageDraw.Draw(image)
    width, height = size
    draw.line((0, 0, width - 1, height - 1), fill=(20, 20, 20), width=3)
    draw.rectangle((20, 15, width // 2, height // 2), outline=(255, 255, 255), width=4)
    draw.ellipse((width // 3, height // 3, width - 25, height - 20), outline=(0, 0, 0), width=3)
    image.save(path)


def _build_manifest(tmp_path: Path) -> tuple[Path, Path]:
    photos = tmp_path / "images"
    photos.mkdir()
    _make_image(photos / "img001.jpg", size=(200, 150), color=(220, 210, 200))
    _make_image(photos / "img002.jpg", size=(180, 180), color=(200, 220, 210))

    manifest_path = tmp_path / "manifest.json"
    write_photo_import_manifest(photos, manifest_path)
    return photos, manifest_path


def _write_regions(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_valid_rectangle_polygon_and_point_radius_regions_are_accepted(
    tmp_path: Path,
) -> None:
    photos, manifest_path = _build_manifest(tmp_path)
    regions_path = _write_regions(
        tmp_path / "candidate_regions.json",
        {
            "schema_version": "manual-candidate-regions-v1",
            "created_at": "2026-06-16T00:00:00Z",
            "regions": [
                {
                    "candidate_id": "candidate-001",
                    "photo": {"path": "images/img001.jpg"},
                    "region_type": "rectangle",
                    "coordinates": {"x": 30, "y": 25, "width": 50, "height": 40},
                    "label": "candidate_region",
                },
                {
                    "candidate_id": "candidate-002",
                    "photo": {"filename": "img001.jpg"},
                    "region_type": "polygon",
                    "coordinates": {
                        "points": [{"x": 40, "y": 35}, {"x": 90, "y": 40}, {"x": 70, "y": 95}]
                    },
                    "label": "candidate_region",
                },
                {
                    "candidate_id": "candidate-003",
                    "photo": {"path": "images/img002.jpg"},
                    "region_type": "point_radius",
                    "coordinates": {"x": 80, "y": 80, "radius": 25},
                    "label": "candidate_region",
                },
            ],
        },
    )

    result = validate_candidate_regions(manifest_path, regions_path, overlay_dir=tmp_path / "overlays")

    assert result.schema_version == "validated-candidate-regions-v1"
    assert result.candidate_count == 3
    assert result.valid_count == 3
    assert result.invalid_count == 0
    assert result.skipped_count == 0
    assert all(candidate.status == "valid" for candidate in result.candidates)

    rectangle = next(candidate for candidate in result.candidates if candidate.candidate_id == "candidate-001")
    polygon = next(candidate for candidate in result.candidates if candidate.candidate_id == "candidate-002")
    point_radius = next(candidate for candidate in result.candidates if candidate.candidate_id == "candidate-003")

    assert rectangle.bounding_box is not None
    assert rectangle.bounding_box.width == 50
    assert polygon.bounding_box is not None
    assert polygon.bounding_box.width > 0
    assert point_radius.bounding_box is not None
    assert point_radius.bounding_box.width == 50
    assert rectangle.overlay_visualization is not None
    assert Path(rectangle.overlay_visualization).exists()
    assert Path(point_radius.overlay_visualization).exists()

    with Image.open(photos / "img001.jpg") as original, Image.open(rectangle.overlay_visualization) as overlay:
        difference = ImageChops.difference(original.convert("RGB"), overlay.convert("RGB"))
        assert difference.getbbox() is not None


def test_missing_photo_reference_is_skipped_without_crashing(tmp_path: Path) -> None:
    _, manifest_path = _build_manifest(tmp_path)
    regions_path = _write_regions(
        tmp_path / "candidate_regions.json",
        {
            "schema_version": "manual-candidate-regions-v1",
            "created_at": "2026-06-16T00:00:00Z",
            "regions": [
                {
                    "candidate_id": "candidate-100",
                    "photo": {"filename": "missing.jpg"},
                    "region_type": "rectangle",
                    "coordinates": {"x": 10, "y": 10, "width": 20, "height": 20},
                }
            ],
        },
    )

    result = validate_candidate_regions(manifest_path, regions_path)

    assert result.candidate_count == 1
    assert result.skipped_count == 1
    assert result.valid_count == 0
    assert result.invalid_count == 0
    assert result.candidates[0].status == "skipped"
    assert any("resolved" in warning.lower() for warning in result.candidates[0].warnings)


def test_invalid_regions_are_reported(tmp_path: Path) -> None:
    _, manifest_path = _build_manifest(tmp_path)
    regions_path = _write_regions(
        tmp_path / "candidate_regions.json",
        {
            "schema_version": "manual-candidate-regions-v1",
            "created_at": "2026-06-16T00:00:00Z",
            "regions": [
                {
                    "candidate_id": "candidate-200",
                    "photo": {"path": "images/img001.jpg"},
                    "region_type": "rectangle",
                    "coordinates": {"x": 180, "y": 130, "width": 40, "height": 40},
                },
                {
                    "candidate_id": "candidate-201",
                    "photo": {"path": "images/img001.jpg"},
                    "region_type": "polygon",
                    "coordinates": {"points": [{"x": 20, "y": 20}, {"x": 30, "y": 30}]},
                },
                {
                    "candidate_id": "candidate-202",
                    "photo": {"path": "images/img001.jpg"},
                    "region_type": "ellipse",
                    "coordinates": {"x": 10, "y": 10, "width": 20, "height": 20},
                },
                {
                    "candidate_id": "candidate-203",
                    "photo": {"path": "images/img002.jpg"},
                    "region_type": "rectangle",
                    "coordinates": {"x": 10, "y": 10, "width": 20, "height": 20},
                },
                {
                    "candidate_id": "candidate-203",
                    "photo": {"path": "images/img002.jpg"},
                    "region_type": "rectangle",
                    "coordinates": {"x": 15, "y": 15, "width": 20, "height": 20},
                },
            ],
        },
    )

    result = validate_candidate_regions(manifest_path, regions_path)

    statuses = {}
    for candidate in result.candidates:
        statuses.setdefault(candidate.candidate_id, candidate.status)
    assert statuses["candidate-200"] == "invalid"
    assert statuses["candidate-201"] == "invalid"
    assert statuses["candidate-202"] == "invalid"
    assert result.invalid_count == 5
    assert any("outside the image bounds" in warning.lower() for warning in result.candidates[0].warnings)
    assert any("at least 3 points" in warning.lower() for warning in result.candidates[1].warnings)
    assert any("unsupported region type" in warning.lower() for warning in result.candidates[2].warnings)
    assert any("duplicate candidate_id" in warning.lower() for warning in result.candidates[3].warnings)
    assert any("duplicate candidate_id" in warning.lower() for warning in result.candidates[4].warnings)


def test_cli_command_writes_validated_candidate_region_json_and_overlays(tmp_path: Path) -> None:
    _, manifest_path = _build_manifest(tmp_path)
    regions_path = _write_regions(
        tmp_path / "candidate_regions.json",
        {
            "schema_version": "manual-candidate-regions-v1",
            "created_at": "2026-06-16T00:00:00Z",
            "regions": [
                {
                    "candidate_id": "candidate-301",
                    "photo": {"path": "images/img001.jpg"},
                    "region_type": "rectangle",
                    "coordinates": {"x": 20, "y": 20, "width": 40, "height": 30},
                }
            ],
        },
    )
    output_path = tmp_path / "validated_candidate_regions.json"
    overlay_dir = tmp_path / "candidate_overlays"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "skintrack.cli",
            "validate-candidate-regions",
            str(regions_path),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--overlay-dir",
            str(overlay_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "validated-candidate-regions-v1"
    assert payload["candidate_count"] == 1
    assert payload["valid_count"] == 1
    assert payload["candidates"][0]["status"] == "valid"
    assert payload["candidates"][0]["bounding_box"]["width"] == 40
    assert payload["candidates"][0]["overlay_visualization"]
    assert overlay_dir.exists()
    assert any(path.suffix.lower() in {".jpg", ".jpeg", ".png"} for path in overlay_dir.iterdir())
