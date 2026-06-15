"""Heuristic local image quality assessment."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

ExposureStatus = Literal["ok", "too_dark", "too_bright", "unknown"]
BlurStatus = Literal["ok", "possibly_blurry", "unknown"]
SizeStatus = Literal["ok", "too_small"]
QualityStatus = Literal["usable", "low_quality", "unreadable"]

_MIN_USEFUL_DIMENSION: Final[int] = 128
_DARK_BRIGHTNESS_THRESHOLD: Final[float] = 0.25
_BRIGHT_BRIGHTNESS_THRESHOLD: Final[float] = 0.85
_BLURRY_SHARPNESS_THRESHOLD: Final[float] = 3.5
_ANALYSIS_SAMPLE_SIZE: Final[tuple[int, int]] = (64, 64)


class ImageQualityAssessment(BaseModel):
    """Heuristic technical assessment for a local image file."""

    model_config = ConfigDict(extra="forbid")

    readable: bool
    width: int | None = None
    height: int | None = None
    brightness_score: float | None = None
    exposure_status: ExposureStatus = "unknown"
    sharpness_score: float | None = None
    blur_status: BlurStatus = "unknown"
    size_status: SizeStatus = "too_small"
    status: QualityStatus = "unreadable"
    warnings: list[str] = Field(default_factory=list)


def assess_image_quality(image_path: str | Path) -> ImageQualityAssessment:
    """Assess readability, exposure, blur, and size using local Pillow logic.

    The thresholds are heuristic and intended only for technical image triage,
    not clinical assessment.
    """

    path = Path(image_path)
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            image.load()
            width, height = image.size
            return _assess_readable_image(image, width=width, height=height)
    except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as exc:
        return ImageQualityAssessment(
            readable=False,
            width=None,
            height=None,
            brightness_score=None,
            exposure_status="unknown",
            sharpness_score=None,
            blur_status="unknown",
            size_status="too_small",
            status="unreadable",
            warnings=[f"Image could not be opened: {exc.__class__.__name__}."],
        )


def _assess_readable_image(
    image: Image.Image,
    *,
    width: int,
    height: int,
) -> ImageQualityAssessment:
    warnings: list[str] = []

    grayscale = image.convert("L").resize(_ANALYSIS_SAMPLE_SIZE, Image.Resampling.BILINEAR)
    brightness_score = _compute_brightness_score(grayscale)
    sharpness_score = _compute_sharpness_score(grayscale)

    size_status = "too_small" if min(width, height) < _MIN_USEFUL_DIMENSION else "ok"
    if size_status == "too_small":
        warnings.append(
            f"Image dimensions {width}x{height} may be too small for reliable comparison."
        )

    exposure_status = _classify_exposure(brightness_score)
    if exposure_status == "too_dark":
        warnings.append("Image may be too dark for reliable comparison.")
    elif exposure_status == "too_bright":
        warnings.append("Image may be too bright for reliable comparison.")

    blur_status = "possibly_blurry" if sharpness_score < _BLURRY_SHARPNESS_THRESHOLD else "ok"
    if blur_status == "possibly_blurry":
        warnings.append("Image may be blurry for reliable comparison.")

    status = "usable"
    if warnings:
        status = "low_quality"

    return ImageQualityAssessment(
        readable=True,
        width=width,
        height=height,
        brightness_score=brightness_score,
        exposure_status=exposure_status,
        sharpness_score=sharpness_score,
        blur_status=blur_status,
        size_status=size_status,
        status=status,
        warnings=warnings,
    )


def _compute_brightness_score(image: Image.Image) -> float:
    pixels = list(image.getdata())
    if not pixels:
        return 0.0
    return sum(pixels) / (len(pixels) * 255.0)


def _compute_sharpness_score(image: Image.Image) -> float:
    width, height = image.size
    pixels = list(image.getdata())
    if width < 2 or height < 2 or not pixels:
        return 0.0

    total = 0.0
    count = 0
    for y in range(height):
        row_offset = y * width
        for x in range(width):
            center = pixels[row_offset + x]
            neighbor_sum = 0
            neighbor_count = 0
            if x > 0:
                neighbor_sum += abs(center - pixels[row_offset + x - 1])
                neighbor_count += 1
            if x < width - 1:
                neighbor_sum += abs(center - pixels[row_offset + x + 1])
                neighbor_count += 1
            if y > 0:
                neighbor_sum += abs(center - pixels[row_offset - width + x])
                neighbor_count += 1
            if y < height - 1:
                neighbor_sum += abs(center - pixels[row_offset + width + x])
                neighbor_count += 1

            if neighbor_count:
                total += neighbor_sum / neighbor_count
                count += 1

    return total / count if count else 0.0


def _classify_exposure(brightness_score: float) -> ExposureStatus:
    if brightness_score < _DARK_BRIGHTNESS_THRESHOLD:
        return "too_dark"
    if brightness_score > _BRIGHT_BRIGHTNESS_THRESHOLD:
        return "too_bright"
    return "ok"
