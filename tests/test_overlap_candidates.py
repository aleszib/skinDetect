"""Overlap candidate ranking tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from skintrack.io.photos import build_photo_import_manifest, write_photo_import_manifest
from skintrack.overlap.candidates import rank_overlap_candidates


def _make_structured_image(path: Path, *, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", size, color=color)
    draw = ImageDraw.Draw(image)
    width, height = size
    draw.line((0, 0, width - 1, height - 1), fill="white", width=max(1, width // 32))
    draw.rectangle((width // 4, height // 4, width // 2, height // 2), outline="black", width=3)
    image.save(path)


def test_ranking_produces_pairwise_candidates_from_three_images(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    _make_structured_image(images / "IMG_20240517_120000.jpg", size=(256, 256), color=(140, 120, 110))
    _make_structured_image(images / "IMG_20240518_120000.jpg", size=(256, 256), color=(138, 122, 112))
    _make_structured_image(images / "portrait_20240630_120000.jpg", size=(64, 64), color=(18, 18, 18))

    manifest = build_photo_import_manifest(images)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    result = rank_overlap_candidates(manifest_path)

    assert result.schema_version == "overlap-candidates-v1"
    assert result.candidate_count == 3
    assert len(result.pairs) == 3
    assert result.pairs[0].score >= result.pairs[1].score >= result.pairs[2].score
    assert {
        result.pairs[0].photo_a.filename,
        result.pairs[0].photo_b.filename,
    } == {"IMG_20240517_120000.jpg", "IMG_20240518_120000.jpg"}
    assert result.pairs[0].reasons
    assert result.pairs[0].penalties or result.pairs[1].penalties or result.pairs[2].penalties


def test_unreadable_and_unsupported_records_are_down_ranked_when_included(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    _make_structured_image(images / "IMG_20240517_120000.jpg", size=(256, 256), color=(140, 120, 110))
    _make_structured_image(images / "IMG_20240518_120000.jpg", size=(256, 256), color=(138, 122, 112))
    (images / "broken.jpg").write_text("not a real jpeg", encoding="utf-8")
    (images / "note.txt").write_text("not an image", encoding="utf-8")

    manifest = build_photo_import_manifest(images)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    result = rank_overlap_candidates(manifest_path, include_nonimported=True)

    broken_pairs = [
        pair
        for pair in result.pairs
        if {"broken.jpg", "note.txt"} & {pair.photo_a.filename, pair.photo_b.filename}
    ]

    assert broken_pairs
    assert all(pair.status == "not_candidate" for pair in broken_pairs)
    assert all(pair.score <= 0.2 for pair in broken_pairs)
    assert any(pair.penalties for pair in broken_pairs)


def test_low_quality_records_are_penalized(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    _make_structured_image(images / "IMG_20240517_120000.jpg", size=(256, 256), color=(140, 120, 110))
    _make_structured_image(images / "IMG_20240518_120000.jpg", size=(256, 256), color=(138, 122, 112))
    _make_structured_image(images / "IMG_20240519_120000.jpg", size=(48, 48), color=(15, 15, 15))

    manifest = build_photo_import_manifest(images)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    result = rank_overlap_candidates(manifest_path)

    usable_pair = next(
        pair
        for pair in result.pairs
        if {
            pair.photo_a.filename,
            pair.photo_b.filename,
        }
        == {"IMG_20240517_120000.jpg", "IMG_20240518_120000.jpg"}
    )
    low_quality_pair = next(
        pair
        for pair in result.pairs
        if "IMG_20240519_120000.jpg" in {pair.photo_a.filename, pair.photo_b.filename}
    )

    assert usable_pair.score > low_quality_pair.score
    assert low_quality_pair.penalties
    assert any("low quality" in penalty for penalty in low_quality_pair.penalties)


def test_cli_command_writes_overlap_candidate_json_file(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    _make_structured_image(images / "IMG_20240517_120000.jpg", size=(256, 256), color=(140, 120, 110))
    _make_structured_image(images / "IMG_20240518_120000.jpg", size=(256, 256), color=(138, 122, 112))
    _make_structured_image(images / "IMG_20240519_120000.jpg", size=(48, 48), color=(15, 15, 15))

    manifest_path = tmp_path / "manifest.json"
    write_photo_import_manifest(images, manifest_path)

    output_path = tmp_path / "overlap_candidates.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "skintrack.cli",
            "rank-overlap-candidates",
            str(manifest_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "overlap-candidates-v1"
    assert payload["candidate_count"] == 3
    assert payload["pairs"]
    assert payload["pairs"][0]["reasons"]
    assert "penalties" in payload["pairs"][0]
