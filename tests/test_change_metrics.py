"""Candidate change metric tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from skintrack.change.metrics import measure_candidate_changes
from skintrack.io.photos import write_photo_import_manifest


def _make_metric_image(
    path: Path,
    *,
    size: tuple[int, int] = (320, 240),
    background: tuple[int, int, int] = (232, 226, 220),
    region_kind: str = "rectangle",
    region: dict[str, object] | None = None,
    fill: tuple[int, int, int] = (120, 80, 80),
) -> None:
    image = Image.new("RGB", size, color=background)
    draw = ImageDraw.Draw(image)
    width, height = size

    for x in range(0, width, 32):
        draw.line((x, 0, x, height - 1), fill=(80, 80, 80), width=1)
    for y in range(0, height, 32):
        draw.line((0, y, width - 1, y), fill=(100, 100, 100), width=1)
    draw.rectangle((20, 18, width - 20, height - 18), outline=(30, 30, 30), width=2)

    region = region or {}
    if region_kind == "rectangle":
        x = int(region["x"])
        y = int(region["y"])
        region_width = int(region["width"])
        region_height = int(region["height"])
        draw.rectangle((x, y, x + region_width, y + region_height), fill=fill)
    elif region_kind == "polygon":
        points = region["points"]
        polygon_points = []
        for point in points:  # type: ignore[assignment]
            if isinstance(point, dict):
                polygon_points.append((int(point["x"]), int(point["y"])))
            else:
                polygon_points.append((int(point[0]), int(point[1])))  # type: ignore[index]
        draw.polygon(polygon_points, fill=fill)
    elif region_kind == "point_radius":
        x = int(region["x"])
        y = int(region["y"])
        radius = int(region["radius"])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)

    image.save(path)


def _build_import_manifest(tmp_path: Path) -> tuple[Path, Path]:
    images = tmp_path / "images"
    images.mkdir()
    manifest_path = tmp_path / "manifest.json"
    return images, manifest_path


def _write_track_manifest(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _build_track_payload(
    *,
    candidate_id: str,
    observations: list[dict[str, object]],
    tracking_confidence: float = 0.92,
    tracking_status: str = "tracked",
) -> dict[str, object]:
    first_seen = min(observation["observed_at"] for observation in observations)
    last_seen = max(observation["observed_at"] for observation in observations)
    return {
        "schema_version": "candidate-tracks-v1",
        "created_at": "2026-06-16T00:00:00Z",
        "source_manifest": "manifest.json",
        "source_validated_regions": "validated_candidate_regions.json",
        "source_projections": "projected_candidate_regions.json",
        "track_count": 1,
        "tracks": [
            {
                "track_id": f"track-{candidate_id}",
                "candidate_id": candidate_id,
                "tracking_status": tracking_status,
                "tracking_confidence": tracking_confidence,
                "observation_count": len(observations),
                "first_seen_at": first_seen,
                "last_seen_at": last_seen,
                "observations": observations,
                "warnings": [],
            }
        ],
        "warnings": [],
    }


def _base_observation(
    *,
    filename: str,
    observed_at: str,
    region_type: str,
    region: dict[str, object],
    observation_source: str = "manual",
    observation_status: str = "manual_valid",
    confidence: float = 1.0,
    warnings: list[str] | None = None,
) -> dict[str, object]:
    return {
        "photo": {"filename": filename},
        "observed_at": observed_at,
        "timestamp_source": "filename",
        "timestamp_confidence": "medium",
        "observation_source": observation_source,
        "region_type": region_type,
        "region": region,
        "observation_status": observation_status,
        "confidence": confidence,
        "warnings": warnings or [],
    }


def _forbidden_words_absent(text: str) -> None:
    forbidden = [
        "melanoma",
        "cancer",
        "dangerous",
        "diagnostic",
        "confirmed same lesion",
        "suspicious melanoma",
    ]
    lower = text.lower()
    assert all(word not in lower for word in forbidden)


def test_measurable_rectangle_change_metrics_from_manual_track(tmp_path: Path) -> None:
    images, manifest_path = _build_import_manifest(tmp_path)
    _make_metric_image(
        images / "IMG_20240517_120000.jpg",
        region_kind="rectangle",
        region={"x": 40, "y": 30, "width": 40, "height": 30},
        fill=(90, 40, 40),
    )
    _make_metric_image(
        images / "IMG_20240519_120000.jpg",
        region_kind="rectangle",
        region={"x": 40, "y": 30, "width": 60, "height": 40},
        fill=(210, 150, 120),
    )
    write_photo_import_manifest(images, manifest_path)

    track_path = _write_track_manifest(
        tmp_path / "candidate_tracks.json",
        _build_track_payload(
            candidate_id="candidate-001",
            observations=[
                _base_observation(
                    filename="IMG_20240517_120000.jpg",
                    observed_at="2024-05-17T12:00:00",
                    region_type="rectangle",
                    region={"x": 40, "y": 30, "width": 40, "height": 30},
                ),
                _base_observation(
                    filename="IMG_20240519_120000.jpg",
                    observed_at="2024-05-19T12:00:00",
                    region_type="rectangle",
                    region={"x": 40, "y": 30, "width": 60, "height": 40},
                ),
            ],
        ),
    )

    result = measure_candidate_changes(manifest_path, track_path)

    assert result.schema_version == "candidate-change-metrics-v1"
    assert result.track_count == 1
    assert result.measurable_count == 1
    measurement = result.results[0]
    assert measurement.measurement_status == "measurable"
    assert measurement.geometry_metrics is not None
    assert measurement.geometry_metrics.first_area_px == 1200
    assert measurement.geometry_metrics.last_area_px == 2400
    assert measurement.geometry_metrics.absolute_area_change_px == 1200
    assert measurement.geometry_metrics.relative_area_change == 1.0
    assert measurement.appearance_metrics is not None
    assert measurement.appearance_metrics.brightness_change is not None
    assert measurement.appearance_metrics.brightness_change > 0.0
    assert measurement.appearance_metrics.mean_rgb_distance is not None
    assert measurement.appearance_metrics.mean_rgb_distance > 0.0
    assert measurement.neutral_summary
    _forbidden_words_absent(measurement.neutral_summary)
    _forbidden_words_absent(measurement.measurement_status)


def test_polygon_region_metrics_are_supported(tmp_path: Path) -> None:
    images, manifest_path = _build_import_manifest(tmp_path)
    polygon = {
        "points": [{"x": 60, "y": 40}, {"x": 120, "y": 48}, {"x": 95, "y": 104}]
    }
    _make_metric_image(
        images / "IMG_20240517_120000.jpg",
        region_kind="polygon",
        region=polygon,
        fill=(80, 120, 80),
    )
    _make_metric_image(
        images / "IMG_20240519_120000.jpg",
        region_kind="polygon",
        region=polygon,
        fill=(150, 180, 150),
    )
    write_photo_import_manifest(images, manifest_path)

    track_path = _write_track_manifest(
        tmp_path / "candidate_tracks.json",
        _build_track_payload(
            candidate_id="candidate-002",
            observations=[
                _base_observation(
                    filename="IMG_20240517_120000.jpg",
                    observed_at="2024-05-17T12:00:00",
                    region_type="polygon",
                    region=polygon,
                ),
                _base_observation(
                    filename="IMG_20240519_120000.jpg",
                    observed_at="2024-05-19T12:00:00",
                    region_type="polygon",
                    region=polygon,
                ),
            ],
        ),
    )

    result = measure_candidate_changes(manifest_path, track_path)

    measurement = result.results[0]
    assert measurement.measurement_status == "measurable"
    assert measurement.geometry_metrics is not None
    assert measurement.geometry_metrics.first_area_px is not None
    assert measurement.geometry_metrics.first_area_px > 0
    assert measurement.geometry_metrics.first_bounding_box is not None
    assert measurement.appearance_metrics is not None
    assert measurement.appearance_metrics.first_mean_rgb is not None


def test_fewer_than_two_observations_returns_not_measurable(tmp_path: Path) -> None:
    images, manifest_path = _build_import_manifest(tmp_path)
    _make_metric_image(
        images / "IMG_20240517_120000.jpg",
        region_kind="rectangle",
        region={"x": 35, "y": 25, "width": 40, "height": 30},
        fill=(100, 80, 80),
    )
    write_photo_import_manifest(images, manifest_path)

    track_path = _write_track_manifest(
        tmp_path / "candidate_tracks.json",
        _build_track_payload(
            candidate_id="candidate-003",
            observations=[
                _base_observation(
                    filename="IMG_20240517_120000.jpg",
                    observed_at="2024-05-17T12:00:00",
                    region_type="rectangle",
                    region={"x": 35, "y": 25, "width": 40, "height": 30},
                )
            ],
        ),
    )

    result = measure_candidate_changes(manifest_path, track_path)

    measurement = result.results[0]
    assert measurement.measurement_status == "not_measurable"
    assert measurement.measurement_confidence == 0.0
    assert measurement.geometry_metrics is None
    assert any("at least two usable observations" in warning.lower() for warning in measurement.warnings)


def test_missing_image_file_is_handled_without_crashing(tmp_path: Path) -> None:
    images, manifest_path = _build_import_manifest(tmp_path)
    first = images / "IMG_20240517_120000.jpg"
    second = images / "IMG_20240519_120000.jpg"
    _make_metric_image(first, region_kind="rectangle", region={"x": 30, "y": 22, "width": 40, "height": 30})
    _make_metric_image(second, region_kind="rectangle", region={"x": 30, "y": 22, "width": 50, "height": 36})
    write_photo_import_manifest(images, manifest_path)
    second.unlink()

    track_path = _write_track_manifest(
        tmp_path / "candidate_tracks.json",
        _build_track_payload(
            candidate_id="candidate-004",
            observations=[
                _base_observation(
                    filename=first.name,
                    observed_at="2024-05-17T12:00:00",
                    region_type="rectangle",
                    region={"x": 30, "y": 22, "width": 40, "height": 30},
                ),
                _base_observation(
                    filename=second.name,
                    observed_at="2024-05-19T12:00:00",
                    region_type="rectangle",
                    region={"x": 30, "y": 22, "width": 50, "height": 36},
                ),
            ],
        ),
    )

    result = measure_candidate_changes(manifest_path, track_path)

    measurement = result.results[0]
    assert measurement.measurement_status == "not_measurable"
    assert any("missing" in warning.lower() for warning in measurement.warnings)


def test_weak_projection_reduces_measurement_confidence(tmp_path: Path) -> None:
    images, manifest_path = _build_import_manifest(tmp_path)
    _make_metric_image(
        images / "IMG_20240517_120000.jpg",
        region_kind="rectangle",
        region={"x": 40, "y": 30, "width": 40, "height": 30},
        fill=(110, 70, 70),
    )
    _make_metric_image(
        images / "IMG_20240519_120000.jpg",
        region_kind="rectangle",
        region={"x": 44, "y": 34, "width": 42, "height": 32},
        fill=(170, 120, 120),
    )
    write_photo_import_manifest(images, manifest_path)

    strong_track_path = _write_track_manifest(
        tmp_path / "strong_tracks.json",
        _build_track_payload(
            candidate_id="candidate-005",
            observations=[
                _base_observation(
                    filename="IMG_20240517_120000.jpg",
                    observed_at="2024-05-17T12:00:00",
                    region_type="rectangle",
                    region={"x": 40, "y": 30, "width": 40, "height": 30},
                ),
                _base_observation(
                    filename="IMG_20240519_120000.jpg",
                    observed_at="2024-05-19T12:00:00",
                    region_type="rectangle",
                    region={"x": 44, "y": 34, "width": 42, "height": 32},
                    observation_source="projection",
                    observation_status="projected",
                    confidence=0.88,
                ),
            ],
            tracking_confidence=0.88,
        ),
    )
    weak_track_path = _write_track_manifest(
        tmp_path / "weak_tracks.json",
        _build_track_payload(
            candidate_id="candidate-005",
            observations=[
                _base_observation(
                    filename="IMG_20240517_120000.jpg",
                    observed_at="2024-05-17T12:00:00",
                    region_type="rectangle",
                    region={"x": 40, "y": 30, "width": 40, "height": 30},
                ),
                _base_observation(
                    filename="IMG_20240519_120000.jpg",
                    observed_at="2024-05-19T12:00:00",
                    region_type="rectangle",
                    region={"x": 44, "y": 34, "width": 42, "height": 32},
                    observation_source="projection",
                    observation_status="weak_projection",
                    confidence=0.34,
                    warnings=["Observation is based on a weak projection."],
                ),
            ],
            tracking_confidence=0.62,
        ),
    )

    strong_result = measure_candidate_changes(manifest_path, strong_track_path)
    weak_result = measure_candidate_changes(manifest_path, weak_track_path)

    strong_measurement = strong_result.results[0]
    weak_measurement = weak_result.results[0]

    assert weak_measurement.measurement_confidence < strong_measurement.measurement_confidence
    assert weak_measurement.measurement_status == "weak_evidence"
    assert weak_measurement.neutral_summary
    assert weak_measurement.warnings
    _forbidden_words_absent(weak_measurement.neutral_summary)


def test_cli_command_writes_change_metrics_json(tmp_path: Path) -> None:
    images, manifest_path = _build_import_manifest(tmp_path)
    _make_metric_image(
        images / "IMG_20240517_120000.jpg",
        region_kind="rectangle",
        region={"x": 40, "y": 30, "width": 40, "height": 30},
    )
    _make_metric_image(
        images / "IMG_20240519_120000.jpg",
        region_kind="rectangle",
        region={"x": 40, "y": 30, "width": 60, "height": 40},
    )
    write_photo_import_manifest(images, manifest_path)

    track_path = _write_track_manifest(
        tmp_path / "candidate_tracks.json",
        _build_track_payload(
            candidate_id="candidate-006",
            observations=[
                _base_observation(
                    filename="IMG_20240517_120000.jpg",
                    observed_at="2024-05-17T12:00:00",
                    region_type="rectangle",
                    region={"x": 40, "y": 30, "width": 40, "height": 30},
                ),
                _base_observation(
                    filename="IMG_20240519_120000.jpg",
                    observed_at="2024-05-19T12:00:00",
                    region_type="rectangle",
                    region={"x": 40, "y": 30, "width": 60, "height": 40},
                ),
            ],
        ),
    )
    output_path = tmp_path / "change_metrics.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "skintrack.cli",
            "measure-candidate-changes",
            "--manifest",
            str(manifest_path),
            "--tracks",
            str(track_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "candidate-change-metrics-v1"
    assert payload["track_count"] == 1
    assert payload["measurable_count"] == 1
    assert payload["results"][0]["measurement_status"] == "measurable"
    assert payload["results"][0]["geometry_metrics"]["absolute_area_change_px"] == 1200
