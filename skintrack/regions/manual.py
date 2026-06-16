"""Manual candidate-region intake and neutral technical overlay generation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from skintrack.io.photos import PhotoImportManifest, PhotoImportRecord

RegionValidationStatus = Literal["valid", "invalid", "skipped"]

_OVERLAY_LABEL: Final[str] = "candidate_region"
_OVERLAY_FILL: Final[tuple[int, int, int, int]] = (0, 180, 255, 52)
_OVERLAY_OUTLINE: Final[tuple[int, int, int, int]] = (0, 120, 180, 255)


class CandidateRegionPhotoRef(BaseModel):
    """Reference to a photo in a manual candidate-region manifest."""

    model_config = ConfigDict(extra="forbid")

    path: str | None = None
    filename: str | None = None


class CandidateRegionInput(BaseModel):
    """One manual candidate region supplied by a user or project lead."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    photo: CandidateRegionPhotoRef
    region_type: str
    coordinates: dict[str, Any]
    label: str | None = "candidate_region"
    notes: str | list[str] | None = None


class CandidateRegionManifest(BaseModel):
    """JSON manifest of manually supplied candidate regions."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "manual-candidate-regions-v1"
    created_at: datetime
    regions: list[CandidateRegionInput] = Field(default_factory=list)


class ValidatedCandidateRegionPhoto(BaseModel):
    """Resolved photo reference with dimensions from the import manifest."""

    model_config = ConfigDict(extra="forbid")

    path: str | None = None
    filename: str | None = None
    width: int | None = None
    height: int | None = None


class ValidatedBoundingBox(BaseModel):
    """Bounding box for a validated candidate region."""

    model_config = ConfigDict(extra="forbid")

    x: int
    y: int
    width: int
    height: int


class CandidateRegionResult(BaseModel):
    """Validated candidate region entry."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    photo: ValidatedCandidateRegionPhoto
    region_type: str
    coordinates: dict[str, Any]
    bounding_box: ValidatedBoundingBox | None = None
    status: RegionValidationStatus
    label: str | None = None
    notes: str | list[str] | None = None
    overlay_visualization: str | None = None
    warnings: list[str] = Field(default_factory=list)


class CandidateRegionValidationManifest(BaseModel):
    """Validated candidate-region manifest with optional overlay paths."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "validated-candidate-regions-v1"
    created_at: datetime
    source_manifest: str
    source_regions: str
    candidate_count: int
    valid_count: int
    invalid_count: int
    skipped_count: int
    candidates: list[CandidateRegionResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _ResolvedPhoto:
    record: PhotoImportRecord | None
    display_path: str | None
    display_filename: str | None


def load_photo_import_manifest(manifest_path: str | Path) -> PhotoImportManifest:
    """Load the photo import manifest JSON from disk."""

    path = Path(manifest_path)
    return PhotoImportManifest.model_validate_json(path.read_text(encoding="utf-8"))


def load_candidate_region_manifest(regions_path: str | Path) -> CandidateRegionManifest:
    """Load the manual candidate-region manifest JSON from disk."""

    path = Path(regions_path)
    return CandidateRegionManifest.model_validate_json(path.read_text(encoding="utf-8"))


def validate_candidate_regions(
    manifest_path: str | Path,
    regions_path: str | Path,
    *,
    overlay_dir: str | Path | None = None,
) -> CandidateRegionValidationManifest:
    """Validate manual candidate regions against imported photos."""

    manifest = load_photo_import_manifest(manifest_path)
    regions = load_candidate_region_manifest(regions_path)
    return build_validated_candidate_region_manifest(
        manifest,
        regions,
        source_manifest=manifest_path,
        source_regions=regions_path,
        overlay_dir=overlay_dir,
    )


def write_validated_candidate_region_manifest(
    manifest_path: str | Path,
    regions_path: str | Path,
    output_path: str | Path,
    *,
    overlay_dir: str | Path | None = None,
) -> CandidateRegionValidationManifest:
    """Validate candidate regions and write the output manifest to disk."""

    validated_manifest = validate_candidate_regions(
        manifest_path,
        regions_path,
        overlay_dir=overlay_dir,
    )
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(validated_manifest.model_dump_json(indent=2), encoding="utf-8")
    return validated_manifest


def build_validated_candidate_region_manifest(
    manifest: PhotoImportManifest,
    regions: CandidateRegionManifest,
    *,
    source_manifest: str | Path,
    source_regions: str | Path,
    overlay_dir: str | Path | None = None,
) -> CandidateRegionValidationManifest:
    """Validate a parsed candidate-region manifest against a photo import manifest."""

    lookup = _build_record_lookup(manifest)
    overlay_root = _prepare_overlay_dir(overlay_dir)
    duplicate_ids = {
        candidate_id
        for candidate_id, count in Counter(region.candidate_id for region in regions.regions).items()
        if count > 1
    }

    results: list[CandidateRegionResult] = []
    warnings: list[str] = []

    for region in regions.regions:
        result = _validate_single_region(
            region,
            lookup,
            duplicate_ids=duplicate_ids,
            regions_root=Path(source_regions).expanduser().resolve().parent,
            overlay_root=overlay_root,
        )
        results.append(result)
        warnings.extend(result.warnings)

    if not results:
        warnings.append("No candidate regions were provided.")

    counts = Counter(result.status for result in results)
    return CandidateRegionValidationManifest(
        created_at=datetime.now(timezone.utc),
        source_manifest=str(Path(source_manifest)),
        source_regions=str(Path(source_regions)),
        candidate_count=len(results),
        valid_count=counts.get("valid", 0),
        invalid_count=counts.get("invalid", 0),
        skipped_count=counts.get("skipped", 0),
        candidates=results,
        warnings=_deduplicate(warnings),
    )


def _validate_single_region(
    region: CandidateRegionInput,
    lookup: dict[str, PhotoImportRecord],
    *,
    duplicate_ids: set[str],
    regions_root: Path,
    overlay_root: Path | None,
) -> CandidateRegionResult:
    warnings: list[str] = []
    photo = _resolve_photo(region.photo, lookup, regions_root)

    if not region.candidate_id:
        warnings.append("Missing candidate_id.")
        return _invalid_result(region, photo, warnings, overlay_visualization=None)

    if region.candidate_id in duplicate_ids:
        warnings.append(f"Duplicate candidate_id: {region.candidate_id}.")
        return _invalid_result(region, photo, warnings, overlay_visualization=None)

    if photo.record is None:
        warnings.append("Photo reference could not be resolved from the manifest.")
        return _skipped_result(region, photo, warnings, overlay_visualization=None)

    if photo.record.import_status != "imported" or photo.record.quality is None:
        warnings.append("Source photo was not imported or has no quality record.")
        return _skipped_result(region, photo, warnings, overlay_visualization=None)

    if not photo.record.quality.readable:
        warnings.append("Source photo is unreadable.")
        return _skipped_result(region, photo, warnings, overlay_visualization=None)

    width = photo.record.width
    height = photo.record.height
    if width is None or height is None:
        warnings.append("Source photo dimensions are unavailable.")
        return _skipped_result(region, photo, warnings, overlay_visualization=None)

    region_type = region.region_type
    coordinates, bounding_box, validation_warnings = _validate_coordinates(
        region_type,
        region.coordinates,
        width=width,
        height=height,
    )
    warnings.extend(validation_warnings)
    if coordinates is None or bounding_box is None:
        return _invalid_result(region, photo, warnings, overlay_visualization=None)

    overlay_visualization = None
    if overlay_root is not None:
        overlay_visualization, overlay_warning = _write_overlay(
            region,
            photo.record.original_path,
            overlay_root,
            coordinates=coordinates,
            region_type=region_type,
        )
        if overlay_warning is not None:
            warnings.append(overlay_warning)

    return CandidateRegionResult(
        candidate_id=region.candidate_id,
        photo=ValidatedCandidateRegionPhoto(
            path=photo.display_path,
            filename=photo.display_filename,
            width=width,
            height=height,
        ),
        region_type=region_type,
        coordinates=coordinates,
        bounding_box=bounding_box,
        status="valid",
        label=region.label,
        notes=region.notes,
        overlay_visualization=overlay_visualization,
        warnings=_deduplicate(warnings),
    )


def _validate_coordinates(
    region_type: str,
    coordinates: dict[str, Any],
    *,
    width: int,
    height: int,
) -> tuple[dict[str, Any] | None, ValidatedBoundingBox | None, list[str]]:
    warnings: list[str] = []

    if region_type == "rectangle":
        rect = _parse_rectangle(coordinates)
        if rect is None:
            warnings.append("Rectangle coordinates are missing or invalid.")
            return None, None, warnings
        x, y, rect_width, rect_height = rect
        if rect_width <= 0 or rect_height <= 0:
            warnings.append("Rectangle width and height must be positive.")
            return None, None, warnings
        if x < 0 or y < 0 or x + rect_width > width or y + rect_height > height:
            warnings.append("Rectangle coordinates are outside the image bounds.")
            return None, None, warnings
        normalized = {
            "x": x,
            "y": y,
            "width": rect_width,
            "height": rect_height,
        }
        return normalized, ValidatedBoundingBox(**normalized), warnings

    if region_type == "polygon":
        points = _parse_polygon_points(coordinates)
        if points is None:
            warnings.append("Polygon coordinates are missing or invalid.")
            return None, None, warnings
        if len(points) < 3:
            warnings.append("Polygon regions must have at least 3 points.")
            return None, None, warnings
        if any(x < 0 or y < 0 or x >= width or y >= height for x, y in points):
            warnings.append("Polygon points are outside the image bounds.")
            return None, None, warnings
        if abs(_polygon_area(points)) <= 0.0:
            warnings.append("Polygon region area is zero.")
            return None, None, warnings
        normalized_points = [{"x": x, "y": y} for x, y in points]
        bbox = _bounding_box_from_points(points)
        return {"points": normalized_points}, bbox, warnings

    if region_type == "point_radius":
        point_radius = _parse_point_radius(coordinates)
        if point_radius is None:
            warnings.append("Point-radius coordinates are missing or invalid.")
            return None, None, warnings
        x, y, radius = point_radius
        if radius <= 0:
            warnings.append("Point-radius must have a positive radius.")
            return None, None, warnings
        if x < 0 or y < 0 or x >= width or y >= height:
            warnings.append("Point-radius center is outside the image bounds.")
            return None, None, warnings
        if x - radius < 0 or y - radius < 0 or x + radius >= width or y + radius >= height:
            warnings.append("Point-radius circle must fit fully inside the image bounds.")
            return None, None, warnings
        bbox = {
            "x": x - radius,
            "y": y - radius,
            "width": radius * 2,
            "height": radius * 2,
        }
        return {"x": x, "y": y, "radius": radius}, ValidatedBoundingBox(**bbox), warnings

    warnings.append(f"Unsupported region type: {region_type}.")
    return None, None, warnings


def _parse_rectangle(coordinates: dict[str, Any]) -> tuple[int, int, int, int] | None:
    try:
        return (
            int(coordinates["x"]),
            int(coordinates["y"]),
            int(coordinates["width"]),
            int(coordinates["height"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _parse_polygon_points(coordinates: dict[str, Any]) -> list[tuple[int, int]] | None:
    try:
        points = coordinates["points"]
    except KeyError:
        return None

    if not isinstance(points, list):
        return None

    parsed: list[tuple[int, int]] = []
    for point in points:
        if not isinstance(point, dict):
            return None
        try:
            parsed.append((int(point["x"]), int(point["y"])))
        except (KeyError, TypeError, ValueError):
            return None
    return parsed


def _parse_point_radius(coordinates: dict[str, Any]) -> tuple[int, int, int] | None:
    try:
        return (
            int(coordinates["x"]),
            int(coordinates["y"]),
            int(coordinates["radius"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _polygon_area(points: list[tuple[int, int]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _bounding_box_from_points(points: list[tuple[int, int]]) -> ValidatedBoundingBox:
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)
    return ValidatedBoundingBox(
        x=min_x,
        y=min_y,
        width=max_x - min_x,
        height=max_y - min_y,
    )


def _resolve_photo(
    photo_ref: CandidateRegionPhotoRef,
    lookup: dict[str, PhotoImportRecord],
    regions_root: Path,
) -> _ResolvedPhoto:
    candidates: list[str] = []
    if photo_ref.path:
        candidates.append(photo_ref.path)
        path = Path(photo_ref.path)
        if not path.is_absolute():
            candidates.append(str((regions_root / path).resolve()))
            candidates.append(path.name)
    if photo_ref.filename:
        candidates.append(photo_ref.filename)

    for candidate in candidates:
        record = lookup.get(candidate)
        if record is not None:
            return _ResolvedPhoto(
                record=record,
                display_path=photo_ref.path or record.original_path,
                display_filename=photo_ref.filename or record.original_filename,
            )

    return _ResolvedPhoto(
        record=None,
        display_path=photo_ref.path,
        display_filename=photo_ref.filename,
    )


def _build_record_lookup(manifest: PhotoImportManifest) -> dict[str, PhotoImportRecord]:
    lookup: dict[str, PhotoImportRecord] = {}
    for record in manifest.photo_records:
        lookup.setdefault(record.original_path, record)
        lookup.setdefault(str(Path(record.original_path).expanduser().resolve()), record)
        lookup.setdefault(record.original_filename, record)
        lookup.setdefault(Path(record.original_path).name, record)
    return lookup


def _prepare_overlay_dir(overlay_dir: str | Path | None) -> Path | None:
    if overlay_dir is None:
        return None
    path = Path(overlay_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_overlay(
    region: CandidateRegionInput,
    image_path: str,
    overlay_root: Path,
    *,
    coordinates: dict[str, Any],
    region_type: str,
) -> tuple[str | None, str | None]:
    output_path = overlay_root / f"{_sanitize_name(region.candidate_id)}_overlay.jpg"
    try:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGBA")
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            if region_type == "rectangle":
                x = coordinates["x"]
                y = coordinates["y"]
                width = coordinates["width"]
                height = coordinates["height"]
                draw.rectangle(
                    [x, y, x + width, y + height],
                    outline=_OVERLAY_OUTLINE,
                    fill=_OVERLAY_FILL,
                    width=3,
                )
            elif region_type == "polygon":
                points = [(point["x"], point["y"]) for point in coordinates["points"]]
                draw.polygon(points, outline=_OVERLAY_OUTLINE, fill=_OVERLAY_FILL)
                draw.line(points + [points[0]], fill=_OVERLAY_OUTLINE, width=3)
            elif region_type == "point_radius":
                x = coordinates["x"]
                y = coordinates["y"]
                radius = coordinates["radius"]
                draw.ellipse(
                    [x - radius, y - radius, x + radius, y + radius],
                    outline=_OVERLAY_OUTLINE,
                    fill=_OVERLAY_FILL,
                    width=3,
                )

            combined = Image.alpha_composite(image, overlay).convert("RGB")
            text_draw = ImageDraw.Draw(combined)
            text = _OVERLAY_LABEL
            text_box = (12, 12, 12 + 180, 12 + 28)
            text_draw.rounded_rectangle(text_box, radius=8, fill=(255, 255, 255), outline=(0, 0, 0))
            text_draw.text((22, 18), text, fill=(0, 0, 0))
            combined.save(output_path, quality=92)
        return str(output_path), None
    except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as exc:
        return None, f"Overlay could not be created: {exc.__class__.__name__}."


def _invalid_result(
    region: CandidateRegionInput,
    photo: _ResolvedPhoto,
    warnings: list[str],
    *,
    overlay_visualization: str | None,
) -> CandidateRegionResult:
    return CandidateRegionResult(
        candidate_id=region.candidate_id,
        photo=ValidatedCandidateRegionPhoto(
            path=photo.display_path,
            filename=photo.display_filename,
            width=photo.record.width if photo.record is not None else None,
            height=photo.record.height if photo.record is not None else None,
        ),
        region_type=region.region_type,
        coordinates=region.coordinates,
        bounding_box=None,
        status="invalid",
        label=region.label,
        notes=region.notes,
        overlay_visualization=overlay_visualization,
        warnings=_deduplicate(warnings),
    )


def _skipped_result(
    region: CandidateRegionInput,
    photo: _ResolvedPhoto,
    warnings: list[str],
    *,
    overlay_visualization: str | None,
) -> CandidateRegionResult:
    return CandidateRegionResult(
        candidate_id=region.candidate_id,
        photo=ValidatedCandidateRegionPhoto(
            path=photo.display_path,
            filename=photo.display_filename,
            width=photo.record.width if photo.record is not None else None,
            height=photo.record.height if photo.record is not None else None,
        ),
        region_type=region.region_type,
        coordinates=region.coordinates,
        bounding_box=None,
        status="skipped",
        label=region.label,
        notes=region.notes,
        overlay_visualization=overlay_visualization,
        warnings=_deduplicate(warnings),
    )


def _deduplicate(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _sanitize_name(name: str) -> str:
    safe = [char if char.isalnum() or char in {"-", "_"} else "_" for char in Path(name).stem]
    sanitized = "".join(safe).strip("_")
    return sanitized or "candidate"
