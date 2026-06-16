"""Candidate-region projection tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from skintrack.io.photos import write_photo_import_manifest
from skintrack.overlap.candidates import rank_overlap_candidates
from skintrack.registration.geometric import register_candidate_pairs
from skintrack.regions.manual import validate_candidate_regions
from skintrack.regions.projection import project_candidate_regions


def _make_feature_rich_image(
    path: Path,
    *,
    size: tuple[int, int] = (320, 240),
    background: tuple[int, int, int] = (235, 235, 235),
    shift: tuple[int, int] = (0, 0),
) -> None:
    image = Image.new("RGB", size, color=background)
    draw = ImageDraw.Draw(image)
    width, height = size

    for x in range(0, width, 32):
        draw.line((x, 0, x, height - 1), fill=(80, 80, 80), width=1)
    for y in range(0, height, 32):
        draw.line((0, y, width - 1, y), fill=(100, 100, 100), width=1)

    draw.rectangle((35, 30, 145, 135), outline=(20, 20, 20), width=4)
    draw.ellipse((170, 40, 260, 130), outline=(30, 30, 30), width=4)
    draw.line((25, height - 25, width - 20, 20), fill=(10, 10, 10), width=4)
    draw.line((20, 20, width - 30, height - 35), fill=(60, 40, 120), width=3)

    if shift != (0, 0):
        image = image.transform(
            size,
            Image.Transform.AFFINE,
            (1, 0, shift[0], 0, 1, shift[1]),
            fillcolor=background,
        )

    image.save(path)


def _make_blank_image(path: Path, *, size: tuple[int, int] = (240, 180), fill: str = "gray") -> None:
    Image.new("RGB", size, color=fill).save(path)


def _build_pipeline_artifacts(
    tmp_path: Path,
    *,
    use_blank_registration: bool = False,
) -> tuple[Path, Path, Path, Path, Path]:
    images = tmp_path / "images"
    images.mkdir()
    if use_blank_registration:
        _make_blank_image(images / "IMG_20240517_120000.jpg", fill="gray")
        _make_blank_image(images / "IMG_20240518_120000.jpg", fill="silver")
    else:
        _make_feature_rich_image(images / "IMG_20240517_120000.jpg")
        _make_feature_rich_image(images / "IMG_20240518_120000.jpg", shift=(18, 12))

    manifest_path = tmp_path / "manifest.json"
    write_photo_import_manifest(images, manifest_path)

    candidate_manifest = rank_overlap_candidates(manifest_path)
    candidate_path = tmp_path / "overlap_candidates.json"
    candidate_path.write_text(candidate_manifest.model_dump_json(indent=2), encoding="utf-8")

    registrations = register_candidate_pairs(manifest_path, candidate_path)
    registrations_path = tmp_path / "registrations.json"
    registrations_path.write_text(registrations.model_dump_json(indent=2), encoding="utf-8")

    return images, manifest_path, candidate_path, registrations_path, tmp_path / "validated_candidate_regions.json"


def _write_candidate_regions(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _build_validated_regions_manifest(
    manifest_path: Path,
    regions_path: Path,
    output_path: Path,
) -> Path:
    result = validate_candidate_regions(manifest_path, regions_path)
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return output_path


def test_rectangle_polygon_and_point_radius_projection_on_synthetic_translation(
    tmp_path: Path,
) -> None:
    _, manifest_path, _, registrations_path, validated_path = _build_pipeline_artifacts(tmp_path)
    regions_path = _write_candidate_regions(
        tmp_path / "candidate_regions.json",
        {
            "schema_version": "manual-candidate-regions-v1",
            "created_at": "2026-06-16T00:00:00Z",
            "regions": [
                {
                    "candidate_id": "candidate-001",
                    "photo": {"path": "images/IMG_20240517_120000.jpg"},
                    "region_type": "rectangle",
                    "coordinates": {"x": 30, "y": 25, "width": 50, "height": 40},
                },
                {
                    "candidate_id": "candidate-002",
                    "photo": {"path": "images/IMG_20240517_120000.jpg"},
                    "region_type": "polygon",
                    "coordinates": {
                        "points": [{"x": 42, "y": 38}, {"x": 92, "y": 42}, {"x": 72, "y": 94}]
                    },
                },
                {
                    "candidate_id": "candidate-003",
                    "photo": {"filename": "IMG_20240518_120000.jpg"},
                    "region_type": "point_radius",
                    "coordinates": {"x": 155, "y": 110, "radius": 22},
                },
            ],
        },
    )
    _build_validated_regions_manifest(manifest_path, regions_path, validated_path)

    overlay_dir = tmp_path / "projected_overlays"
    result = project_candidate_regions(
        manifest_path,
        validated_path,
        registrations_path,
        overlay_dir=overlay_dir,
    )

    assert result.schema_version == "projected-candidate-regions-v1"
    assert result.projection_count == 3
    assert len(result.projections) == 3

    rectangle = next(item for item in result.projections if item.candidate_id == "candidate-001")
    polygon = next(item for item in result.projections if item.candidate_id == "candidate-002")
    point_radius = next(item for item in result.projections if item.candidate_id == "candidate-003")

    assert rectangle.projection_status in {"projected", "weak_projection"}
    assert rectangle.target_photo is not None
    assert rectangle.target_photo.filename == "IMG_20240518_120000.jpg"
    assert rectangle.projected_region is not None
    assert rectangle.projected_region.type == "polygon"
    assert rectangle.projected_bounding_box is not None
    assert abs(rectangle.projected_bounding_box.x - 12) <= 10
    assert abs(rectangle.projected_bounding_box.y - 13) <= 10
    assert rectangle.warnings is not None

    assert polygon.projection_status in {"projected", "weak_projection"}
    assert polygon.target_photo is not None
    assert polygon.target_photo.filename == "IMG_20240518_120000.jpg"
    assert polygon.projected_region is not None
    assert len(polygon.projected_region.points) == 3
    assert polygon.projected_bounding_box is not None
    assert polygon.projected_bounding_box.width > 0

    assert point_radius.projection_status in {"projected", "weak_projection"}
    assert point_radius.target_photo is not None
    assert point_radius.target_photo.filename == "IMG_20240517_120000.jpg"
    assert point_radius.projected_region is not None
    assert point_radius.projected_bounding_box is not None
    assert point_radius.transform_direction == "source_to_target"
    assert point_radius.registration_status in {"registered", "weak_match"}

    assert rectangle.overlay_visualization is not None
    assert polygon.overlay_visualization is not None
    assert point_radius.overlay_visualization is not None
    assert any(path.suffix.lower() in {".jpg", ".jpeg", ".png"} for path in overlay_dir.iterdir())


def test_reverse_direction_projection_uses_the_pair_opposite_side(tmp_path: Path) -> None:
    _, manifest_path, _, registrations_path, validated_path = _build_pipeline_artifacts(tmp_path)
    regions_path = _write_candidate_regions(
        tmp_path / "candidate_regions.json",
        {
            "schema_version": "manual-candidate-regions-v1",
            "created_at": "2026-06-16T00:00:00Z",
            "regions": [
                {
                    "candidate_id": "candidate-101",
                    "photo": {"filename": "IMG_20240518_120000.jpg"},
                    "region_type": "point_radius",
                    "coordinates": {"x": 155, "y": 110, "radius": 22},
                }
            ],
        },
    )
    _build_validated_regions_manifest(manifest_path, regions_path, validated_path)

    result = project_candidate_regions(manifest_path, validated_path, registrations_path)

    projection = result.projections[0]
    assert projection.projection_status in {"projected", "weak_projection"}
    assert projection.target_photo is not None
    assert projection.target_photo.filename == "IMG_20240517_120000.jpg"
    assert projection.transform_direction == "source_to_target"
    assert projection.registration_status in {"registered", "weak_match"}


def test_missing_registration_produces_safe_failure(tmp_path: Path) -> None:
    _, manifest_path, _, _, validated_path = _build_pipeline_artifacts(tmp_path)
    regions_path = _write_candidate_regions(
        tmp_path / "candidate_regions.json",
        {
            "schema_version": "manual-candidate-regions-v1",
            "created_at": "2026-06-16T00:00:00Z",
            "regions": [
                {
                    "candidate_id": "candidate-201",
                    "photo": {"path": "images/IMG_20240517_120000.jpg"},
                    "region_type": "rectangle",
                    "coordinates": {"x": 30, "y": 25, "width": 50, "height": 40},
                }
            ],
        },
    )
    _build_validated_regions_manifest(manifest_path, regions_path, validated_path)

    registrations_path = tmp_path / "registrations.json"
    registrations_path.write_text(
        json.dumps(
            {
                "schema_version": "registrations-v1",
                "created_at": "2026-06-16T00:00:00Z",
                "source_manifest": str(manifest_path),
                "source_candidates": "overlap_candidates.json",
                "registration_count": 0,
                "registrations": [],
                "warnings": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = project_candidate_regions(manifest_path, validated_path, registrations_path)

    projection = result.projections[0]
    assert projection.projection_status == "skipped"
    assert projection.target_photo is None
    assert any("matching registration" in warning.lower() for warning in projection.warnings)


def test_failed_registration_is_handled_without_crashing(tmp_path: Path) -> None:
    _, manifest_path, _, registrations_path, validated_path = _build_pipeline_artifacts(
        tmp_path,
        use_blank_registration=True,
    )
    regions_path = _write_candidate_regions(
        tmp_path / "candidate_regions.json",
        {
            "schema_version": "manual-candidate-regions-v1",
            "created_at": "2026-06-16T00:00:00Z",
            "regions": [
                {
                    "candidate_id": "candidate-301",
                    "photo": {"path": "images/IMG_20240517_120000.jpg"},
                    "region_type": "rectangle",
                    "coordinates": {"x": 20, "y": 18, "width": 40, "height": 30},
                }
            ],
        },
    )
    _build_validated_regions_manifest(manifest_path, regions_path, validated_path)

    result = project_candidate_regions(manifest_path, validated_path, registrations_path)

    projection = result.projections[0]
    assert projection.projection_status in {"failed", "skipped"}
    assert projection.registration_status in {"failed", "weak_match", "skipped", None}
    assert projection.warnings


def test_cli_command_writes_projected_candidate_region_json_and_overlays(tmp_path: Path) -> None:
    _, manifest_path, _, registrations_path, validated_path = _build_pipeline_artifacts(
        tmp_path,
    )
    regions_path = _write_candidate_regions(
        tmp_path / "candidate_regions.json",
        {
            "schema_version": "manual-candidate-regions-v1",
            "created_at": "2026-06-16T00:00:00Z",
            "regions": [
                {
                    "candidate_id": "candidate-401",
                    "photo": {"path": "images/IMG_20240517_120000.jpg"},
                    "region_type": "rectangle",
                    "coordinates": {"x": 30, "y": 25, "width": 50, "height": 40},
                }
            ],
        },
    )
    _build_validated_regions_manifest(manifest_path, regions_path, validated_path)

    output_path = tmp_path / "projected_candidate_regions.json"
    overlay_dir = tmp_path / "projected_region_overlays"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "skintrack.cli",
            "project-candidate-regions",
            "--validated-regions",
            str(validated_path),
            "--registrations",
            str(registrations_path),
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
    assert payload["schema_version"] == "projected-candidate-regions-v1"
    assert payload["projection_count"] == 1
    assert payload["projections"][0]["projection_status"] in {"projected", "weak_projection"}
    assert payload["projections"][0]["projected_region"]["type"] == "polygon"
    assert payload["projections"][0]["target_photo"]["filename"] == "IMG_20240518_120000.jpg"
    assert payload["projections"][0]["warnings"] is not None
    assert overlay_dir.exists()
    assert any(path.suffix.lower() in {".jpg", ".jpeg", ".png"} for path in overlay_dir.iterdir())
