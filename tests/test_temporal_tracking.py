"""Temporal candidate tracking tests."""

from __future__ import annotations

import copy
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
from skintrack.tracking.temporal import track_candidate_regions


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


def _build_pipeline_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    images = tmp_path / "images"
    images.mkdir()
    _make_feature_rich_image(images / "IMG_20240517_120000.jpg")
    _make_feature_rich_image(images / "IMG_20240518_120000.jpg", shift=(18, 12))
    _make_feature_rich_image(images / "IMG_20240519_120000.jpg", background=(180, 180, 180))

    manifest_path = tmp_path / "manifest.json"
    write_photo_import_manifest(images, manifest_path)

    regions_path = tmp_path / "candidate_regions.json"
    regions_path.write_text(
        json.dumps(
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
                        "photo": {"path": "images/IMG_20240518_120000.jpg"},
                        "region_type": "polygon",
                        "coordinates": {
                            "points": [{"x": 42, "y": 38}, {"x": 92, "y": 42}, {"x": 72, "y": 94}]
                        },
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    validated_path = tmp_path / "validated_candidate_regions.json"
    validated = validate_candidate_regions(manifest_path, regions_path)
    validated_path.write_text(validated.model_dump_json(indent=2), encoding="utf-8")

    overlap_path = tmp_path / "overlap_candidates.json"
    overlap = rank_overlap_candidates(manifest_path)
    overlap_path.write_text(overlap.model_dump_json(indent=2), encoding="utf-8")

    registrations_path = tmp_path / "registrations.json"
    registrations = register_candidate_pairs(manifest_path, overlap_path)
    registrations_path.write_text(registrations.model_dump_json(indent=2), encoding="utf-8")

    projections_path = tmp_path / "projected_candidate_regions.json"
    projections = project_candidate_regions(manifest_path, validated_path, registrations_path)
    projections_path.write_text(projections.model_dump_json(indent=2), encoding="utf-8")

    return manifest_path, validated_path, projections_path, regions_path


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_missing_timestamp_manifest(manifest_path: Path, filename: str) -> None:
    payload = _load_json(manifest_path)
    for record in payload["photo_records"]:
        if record["original_filename"] == filename:
            record["taken_at"] = None
            record["timestamp_source"] = "unknown"
            record["timestamp_confidence"] = "unknown"
            break
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _modify_projection_status(
    projections_path: Path,
    *,
    candidate_id: str,
    status: str,
    confidence: float | None = None,
    duplicate: bool = False,
) -> None:
    payload = _load_json(projections_path)
    projections = payload["projections"]
    for projection in projections:
        if projection["candidate_id"] != candidate_id:
            continue
        projection["projection_status"] = status
        if confidence is not None:
            projection["projection_confidence"] = confidence
        if status not in {"projected", "weak_projection"}:
            projection["projected_region"] = None
        if duplicate:
            projections.append(copy.deepcopy(projection))
        break
    projections_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_manual_and_projected_observations_form_one_track_and_sort_by_timestamp(
    tmp_path: Path,
) -> None:
    manifest_path, validated_path, projections_path, _ = _build_pipeline_artifacts(tmp_path)

    result = track_candidate_regions(manifest_path, validated_path, projections_path)

    assert result.schema_version == "candidate-tracks-v1"
    assert result.track_count == 2
    assert {track.candidate_id for track in result.tracks} == {"candidate-001", "candidate-002"}

    track = next(item for item in result.tracks if item.candidate_id == "candidate-001")
    timestamps = [observation.observed_at for observation in track.observations if observation.observed_at is not None]

    assert track.observation_count >= 2
    assert track.tracking_status in {"tracked", "partial"}
    assert any(observation.observation_source == "manual" for observation in track.observations)
    assert any(observation.observation_source == "projection" for observation in track.observations)
    assert timestamps == sorted(timestamps)
    assert track.first_seen_at == min(timestamps)
    assert track.last_seen_at == max(timestamps)


def test_missing_timestamps_do_not_crash_and_produce_warnings(tmp_path: Path) -> None:
    manifest_path, validated_path, projections_path, _ = _build_pipeline_artifacts(tmp_path)
    _build_missing_timestamp_manifest(manifest_path, "IMG_20240517_120000.jpg")

    result = track_candidate_regions(manifest_path, validated_path, projections_path)

    track = next(item for item in result.tracks if item.candidate_id == "candidate-001")
    assert track.observation_count >= 1
    assert any("timestamp" in warning.lower() for warning in track.warnings)
    assert track.tracking_confidence < 1.0


def test_weak_projections_reduce_tracking_confidence(tmp_path: Path) -> None:
    manifest_path, validated_path, projections_path, _ = _build_pipeline_artifacts(tmp_path)
    weak_projections_path = tmp_path / "weak_projected_candidate_regions.json"
    weak_payload = _load_json(projections_path)
    for projection in weak_payload["projections"]:
        if projection["candidate_id"] == "candidate-001":
            projection["projection_status"] = "weak_projection"
            projection["projection_confidence"] = min(float(projection["projection_confidence"]), 0.38)
            break
    weak_projections_path.write_text(json.dumps(weak_payload, indent=2), encoding="utf-8")

    strong_result = track_candidate_regions(manifest_path, validated_path, projections_path)
    weak_result = track_candidate_regions(manifest_path, validated_path, weak_projections_path)

    strong_track = next(item for item in strong_result.tracks if item.candidate_id == "candidate-001")
    weak_track = next(item for item in weak_result.tracks if item.candidate_id == "candidate-001")

    assert weak_track.tracking_confidence < strong_track.tracking_confidence
    assert weak_track.tracking_status in {"partial", "tracked"}


def test_failed_or_skipped_projections_are_handled_safely(tmp_path: Path) -> None:
    manifest_path, validated_path, projections_path, _ = _build_pipeline_artifacts(tmp_path)
    _modify_projection_status(
        projections_path,
        candidate_id="candidate-002",
        status="failed",
        confidence=0.0,
    )

    result = track_candidate_regions(manifest_path, validated_path, projections_path)

    track = next(item for item in result.tracks if item.candidate_id == "candidate-002")
    assert track.tracking_status in {"partial", "untracked"}
    assert any("failed" in warning.lower() or "not tracked" in warning.lower() for warning in track.warnings)


def test_duplicate_observations_are_reported_as_ambiguous(tmp_path: Path) -> None:
    manifest_path, validated_path, projections_path, _ = _build_pipeline_artifacts(tmp_path)
    _modify_projection_status(
        projections_path,
        candidate_id="candidate-001",
        status="projected",
        confidence=0.72,
        duplicate=True,
    )

    result = track_candidate_regions(manifest_path, validated_path, projections_path)

    track = next(item for item in result.tracks if item.candidate_id == "candidate-001")
    assert track.tracking_status == "ambiguous"
    assert any("duplicate" in warning.lower() for warning in track.warnings)


def test_cli_command_writes_candidate_track_json(tmp_path: Path) -> None:
    manifest_path, validated_path, projections_path, _ = _build_pipeline_artifacts(tmp_path)
    output_path = tmp_path / "candidate_tracks.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "skintrack.cli",
            "track-candidate-regions",
            "--manifest",
            str(manifest_path),
            "--validated-regions",
            str(validated_path),
            "--projections",
            str(projections_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "candidate-tracks-v1"
    assert payload["track_count"] == 2
    assert payload["tracks"]
    assert payload["tracks"][0]["observations"]
    assert payload["tracks"][0]["tracking_status"] in {"tracked", "partial", "ambiguous", "untracked"}
