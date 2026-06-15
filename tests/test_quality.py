"""Heuristic image quality assessment tests."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from skintrack.io.photos import build_photo_import_manifest
from skintrack.quality.images import assess_image_quality


def _make_structured_image(path: Path, *, size: tuple[int, int], fill: str = "gray") -> None:
    image = Image.new("RGB", size, color=fill)
    draw = ImageDraw.Draw(image)
    width, height = size
    draw.line((0, 0, width - 1, height - 1), fill="black", width=max(1, width // 32))
    draw.rectangle((width // 4, height // 4, width // 2, height // 2), outline="white", width=3)
    image.save(path)


def test_valid_image_is_readable_and_usable(tmp_path: Path) -> None:
    image_path = tmp_path / "usable.jpg"
    _make_structured_image(image_path, size=(256, 256))

    result = assess_image_quality(image_path)

    assert result.readable is True
    assert result.status == "usable"
    assert result.exposure_status == "ok"
    assert result.blur_status == "ok"
    assert result.size_status == "ok"
    assert result.width == 256
    assert result.height == 256
    assert result.brightness_score is not None
    assert result.sharpness_score is not None


def test_too_small_image_is_marked_low_quality(tmp_path: Path) -> None:
    image_path = tmp_path / "small.jpg"
    _make_structured_image(image_path, size=(48, 48))

    result = assess_image_quality(image_path)

    assert result.readable is True
    assert result.status == "low_quality"
    assert result.size_status == "too_small"
    assert any("too small" in warning for warning in result.warnings)


def test_very_dark_image_is_marked_too_dark(tmp_path: Path) -> None:
    image_path = tmp_path / "dark.jpg"
    Image.new("RGB", (256, 256), color=(15, 15, 15)).save(image_path)

    result = assess_image_quality(image_path)

    assert result.readable is True
    assert result.status == "low_quality"
    assert result.exposure_status == "too_dark"
    assert any("too dark" in warning for warning in result.warnings)


def test_very_bright_image_is_marked_too_bright(tmp_path: Path) -> None:
    image_path = tmp_path / "bright.jpg"
    Image.new("RGB", (256, 256), color=(250, 250, 250)).save(image_path)

    result = assess_image_quality(image_path)

    assert result.readable is True
    assert result.status == "low_quality"
    assert result.exposure_status == "too_bright"
    assert any("too bright" in warning for warning in result.warnings)


def test_blurred_image_is_marked_possibly_blurry(tmp_path: Path) -> None:
    sharp_path = tmp_path / "sharp.jpg"
    blurred_path = tmp_path / "blurred.jpg"
    _make_structured_image(sharp_path, size=(256, 256))
    with Image.open(sharp_path) as sharp_image:
        sharp_image.filter(ImageFilter.GaussianBlur(radius=4)).save(blurred_path)

    sharp_result = assess_image_quality(sharp_path)
    blurred_result = assess_image_quality(blurred_path)

    assert sharp_result.sharpness_score is not None
    assert blurred_result.sharpness_score is not None
    assert sharp_result.sharpness_score > blurred_result.sharpness_score
    assert sharp_result.status == "usable"
    assert blurred_result.blur_status == "possibly_blurry"
    assert blurred_result.status == "low_quality"


def test_unreadable_corrupt_image_is_handled_without_crashing(tmp_path: Path) -> None:
    image_path = tmp_path / "broken.jpg"
    image_path.write_text("not an image", encoding="utf-8")

    result = assess_image_quality(image_path)

    assert result.readable is False
    assert result.status == "unreadable"
    assert result.brightness_score is None
    assert result.sharpness_score is None
    assert result.warnings


def test_manifest_records_quality_information_and_warnings(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    _make_structured_image(images / "usable.jpg", size=(256, 256))
    _make_structured_image(images / "small.jpg", size=(48, 48))

    manifest = build_photo_import_manifest(images)

    usable_record = next(
        record for record in manifest.photo_records if record.original_filename == "usable.jpg"
    )
    small_record = next(
        record for record in manifest.photo_records if record.original_filename == "small.jpg"
    )

    assert usable_record.quality is not None
    assert usable_record.quality.status == "usable"
    assert small_record.quality is not None
    assert small_record.quality.status == "low_quality"
    assert manifest.counts.low_quality == 1
    assert any("too small" in warning for warning in manifest.warnings)
