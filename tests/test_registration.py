"""Geometric registration tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from skintrack.io.photos import write_photo_import_manifest
from skintrack.overlap.candidates import rank_overlap_candidates
from skintrack.registration.geometric import register_candidate_pairs


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


def _build_manifest_and_candidates(tmp_path: Path) -> tuple[Path, Path]:
    images = tmp_path / "images"
    images.mkdir()
    _make_feature_rich_image(images / "IMG_20240517_120000.jpg")
    _make_feature_rich_image(images / "IMG_20240518_120000.jpg", shift=(18, 12))
    _make_feature_rich_image(images / "IMG_20240519_120000.jpg", background=(180, 180, 180))

    manifest_path = tmp_path / "manifest.json"
    write_photo_import_manifest(images, manifest_path)
    candidate_manifest = rank_overlap_candidates(manifest_path)
    candidate_path = tmp_path / "overlap_candidates.json"
    candidate_path.write_text(candidate_manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest_path, candidate_path


def test_successful_registration_on_synthetic_translation(tmp_path: Path) -> None:
    manifest_path, candidate_path = _build_manifest_and_candidates(tmp_path)

    result = register_candidate_pairs(manifest_path, candidate_path)

    assert result.schema_version == "registrations-v1"
    assert result.registration_count == 3
    translated_pair = next(
        registration
        for registration in result.registrations
        if {
            registration.photo_a.filename,
            registration.photo_b.filename,
        }
        == {"IMG_20240517_120000.jpg", "IMG_20240518_120000.jpg"}
    )

    assert translated_pair.registration_status == "registered"
    assert translated_pair.registration_confidence > 0.5
    assert translated_pair.method == "orb_affine_ransac"
    assert translated_pair.match_count >= 8
    assert translated_pair.inlier_count >= 4
    assert translated_pair.inlier_ratio > 0.3
    assert translated_pair.transform is not None
    assert translated_pair.transform.type == "affine"
    assert abs(abs(translated_pair.transform.matrix[0][2]) - 18) < 8
    assert abs(abs(translated_pair.transform.matrix[1][2]) - 12) < 8
    assert translated_pair.overlap is not None
    assert translated_pair.overlap.type == "polygon"
    assert translated_pair.overlap.estimated_overlap_fraction is not None
    assert translated_pair.overlap.estimated_overlap_fraction > 0.5


def test_featureless_images_fail_without_crashing(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    _make_blank_image(images / "IMG_20240517_120000.jpg", fill="gray")
    _make_blank_image(images / "IMG_20240518_120000.jpg", fill="silver")

    manifest_path = tmp_path / "manifest.json"
    write_photo_import_manifest(images, manifest_path)
    candidate_path = tmp_path / "overlap_candidates.json"
    candidate_manifest = rank_overlap_candidates(manifest_path)
    candidate_path.write_text(candidate_manifest.model_dump_json(indent=2), encoding="utf-8")

    result = register_candidate_pairs(manifest_path, candidate_path)

    assert result.registration_count == 1
    registration = result.registrations[0]
    assert registration.registration_status in {"failed", "weak_match"}
    assert registration.registration_confidence <= 0.5
    if registration.registration_status == "failed":
        assert registration.transform is None
        assert registration.overlap is None
    assert registration.warnings


def test_missing_image_file_is_handled_safely(tmp_path: Path) -> None:
    manifest_path, candidate_path = _build_manifest_and_candidates(tmp_path)
    missing_file = tmp_path / "images" / "IMG_20240518_120000.jpg"
    missing_file.unlink()

    result = register_candidate_pairs(manifest_path, candidate_path)

    missing_registration = next(
        registration
        for registration in result.registrations
        if registration.photo_b.filename == "IMG_20240518_120000.jpg"
    )

    assert missing_registration.registration_status == "failed"
    assert any("could not be loaded" in warning.lower() for warning in missing_registration.warnings)


def test_manifest_and_candidate_json_are_consumed_correctly(tmp_path: Path) -> None:
    manifest_path, candidate_path = _build_manifest_and_candidates(tmp_path)

    result = register_candidate_pairs(manifest_path, candidate_path)

    assert result.source_manifest == str(manifest_path)
    assert result.source_candidates == str(candidate_path)
    assert result.registration_count == len(result.registrations) == 3
    assert result.registrations[0].candidate_score is not None
    assert result.registrations[0].warnings is not None


def test_cli_command_writes_registrations_json_and_debug_images(tmp_path: Path) -> None:
    manifest_path, candidate_path = _build_manifest_and_candidates(tmp_path)
    output_path = tmp_path / "registrations.json"
    debug_dir = tmp_path / "debug_registration"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "skintrack.cli",
            "register-candidate-pairs",
            str(candidate_path),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--debug-dir",
            str(debug_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "registrations-v1"
    assert payload["registration_count"] == 3
    assert payload["registrations"]
    assert "registration_status" in payload["registrations"][0]
    assert "candidate_score" in payload["registrations"][0]
    assert debug_dir.exists()
    assert any(path.suffix.lower() in {".jpg", ".jpeg", ".png"} for path in debug_dir.iterdir())
