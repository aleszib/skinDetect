"""Project validated manual candidate regions through registered image pairs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil, floor, pi, cos, sin
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from skintrack.io.photos import PhotoImportManifest, PhotoImportRecord
from skintrack.registration.geometric import RegistrationManifest, RegistrationResult, RegistrationTransform
from skintrack.regions.manual import (
    CandidateRegionResult,
    CandidateRegionValidationManifest,
    ValidatedCandidateRegionPhoto,
)

ProjectionStatus = Literal["projected", "weak_projection", "failed", "skipped"]
TransformDirection = Literal["source_to_target", "unknown"]
ProjectedRegionKind = Literal["polygon"]

_MIN_PROJECTION_CONFIDENCE: Final[float] = 0.35
_WEAK_PROJECTION_CONFIDENCE: Final[float] = 0.6
_CIRCLE_SAMPLE_POINTS: Final[int] = 16
_SOURCE_OUTLINE: Final[tuple[int, int, int, int]] = (0, 130, 180, 255)
_SOURCE_FILL: Final[tuple[int, int, int, int]] = (0, 130, 180, 50)
_PROJECTED_OUTLINE: Final[tuple[int, int, int, int]] = (255, 140, 0, 255)
_PROJECTED_FILL: Final[tuple[int, int, int, int]] = (255, 140, 0, 45)


class ProjectedCandidateRegionPhoto(BaseModel):
    """Photo reference with image dimensions for projection output."""

    model_config = ConfigDict(extra="forbid")

    path: str | None = None
    filename: str | None = None
    width: int | None = None
    height: int | None = None


class ProjectedRegionGeometry(BaseModel):
    """Projected candidate region geometry."""

    model_config = ConfigDict(extra="forbid")

    type: ProjectedRegionKind
    points: list[list[float]]


class ProjectedBoundingBox(BaseModel):
    """Bounding box for a projected region."""

    model_config = ConfigDict(extra="forbid")

    x: int
    y: int
    width: int
    height: int


class ProjectedCandidateRegionResult(BaseModel):
    """One projected candidate-region entry."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    source_photo: ProjectedCandidateRegionPhoto
    target_photo: ProjectedCandidateRegionPhoto | None = None
    source_region_type: str
    source_region: dict[str, Any]
    projected_region: ProjectedRegionGeometry | None = None
    projected_bounding_box: ProjectedBoundingBox | None = None
    projection_status: ProjectionStatus
    projection_confidence: float = 0.0
    registration_status: str | None = None
    registration_confidence: float | None = None
    transform_direction: TransformDirection = "unknown"
    overlay_visualization: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ProjectedCandidateRegionManifest(BaseModel):
    """Validated candidate-region projections through registered image pairs."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "projected-candidate-regions-v1"
    created_at: datetime
    source_manifest: str
    source_regions: str
    source_registrations: str
    projection_count: int
    projections: list[ProjectedCandidateRegionResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _ResolvedPhoto:
    record: PhotoImportRecord | None
    display_path: str | None
    display_filename: str | None


@dataclass(frozen=True)
class _RegistrationMatch:
    registration: RegistrationResult
    source_is_photo_a: bool
    source_record: PhotoImportRecord
    target_record: PhotoImportRecord
    transform_matrix: np.ndarray | None


def load_photo_import_manifest(manifest_path: str | Path) -> PhotoImportManifest:
    """Load the photo import manifest JSON from disk."""

    path = Path(manifest_path)
    return PhotoImportManifest.model_validate_json(path.read_text(encoding="utf-8"))


def load_validated_candidate_region_manifest(regions_path: str | Path) -> CandidateRegionValidationManifest:
    """Load the validated candidate-region manifest JSON from disk."""

    path = Path(regions_path)
    return CandidateRegionValidationManifest.model_validate_json(path.read_text(encoding="utf-8"))


def load_registration_manifest(registrations_path: str | Path) -> RegistrationManifest:
    """Load the registration manifest JSON from disk."""

    path = Path(registrations_path)
    return RegistrationManifest.model_validate_json(path.read_text(encoding="utf-8"))


def project_candidate_regions(
    manifest_path: str | Path,
    regions_path: str | Path,
    registrations_path: str | Path,
    *,
    overlay_dir: str | Path | None = None,
) -> ProjectedCandidateRegionManifest:
    """Project validated candidate regions into matched registered photos."""

    manifest = load_photo_import_manifest(manifest_path)
    regions = load_validated_candidate_region_manifest(regions_path)
    registrations = load_registration_manifest(registrations_path)
    return build_projected_candidate_region_manifest(
        manifest,
        regions,
        registrations,
        source_manifest=manifest_path,
        source_regions=regions_path,
        source_registrations=registrations_path,
        overlay_dir=overlay_dir,
    )


def write_projected_candidate_region_manifest(
    manifest_path: str | Path,
    regions_path: str | Path,
    registrations_path: str | Path,
    output_path: str | Path,
    *,
    overlay_dir: str | Path | None = None,
) -> ProjectedCandidateRegionManifest:
    """Project candidate regions and write the output manifest to disk."""

    projected_manifest = project_candidate_regions(
        manifest_path,
        regions_path,
        registrations_path,
        overlay_dir=overlay_dir,
    )
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(projected_manifest.model_dump_json(indent=2), encoding="utf-8")
    return projected_manifest


def build_projected_candidate_region_manifest(
    manifest: PhotoImportManifest,
    regions: CandidateRegionValidationManifest,
    registrations: RegistrationManifest,
    *,
    source_manifest: str | Path,
    source_regions: str | Path,
    source_registrations: str | Path,
    overlay_dir: str | Path | None = None,
) -> ProjectedCandidateRegionManifest:
    """Project validated regions through the best matching registrations."""

    record_lookup = _build_record_lookup(manifest)
    overlay_root = _prepare_overlay_dir(overlay_dir)
    results: list[ProjectedCandidateRegionResult] = []
    warnings: list[str] = []

    for region in regions.candidates:
        result = _project_single_region(
            region,
            record_lookup,
            registrations.registrations,
            regions_root=Path(source_regions).expanduser().resolve().parent,
            overlay_root=overlay_root,
        )
        results.append(result)
        warnings.extend(result.warnings)

    if not results:
        warnings.append("No validated candidate regions were available for projection.")

    return ProjectedCandidateRegionManifest(
        created_at=datetime.now(timezone.utc),
        source_manifest=str(Path(source_manifest)),
        source_regions=str(Path(source_regions)),
        source_registrations=str(Path(source_registrations)),
        projection_count=len(results),
        projections=results,
        warnings=_deduplicate(warnings),
    )


def _project_single_region(
    region: CandidateRegionResult,
    record_lookup: dict[str, PhotoImportRecord],
    registrations: list[RegistrationResult],
    *,
    regions_root: Path,
    overlay_root: Path | None,
) -> ProjectedCandidateRegionResult:
    warnings: list[str] = []

    if region.status != "valid":
        warnings.append("Source candidate region is not valid and was skipped.")
        return _skipped_projection(region, warnings)

    source_resolved = _resolve_photo(region.photo, record_lookup, regions_root)
    if source_resolved.record is None:
        warnings.append("Source photo could not be resolved from the import manifest.")
        return _skipped_projection(region, warnings, source_photo=source_resolved)

    source_match, match_warnings, fallback_match = _select_registration_match(
        source_resolved.record,
        registrations,
        record_lookup,
    )
    warnings.extend(match_warnings)

    if source_match is None:
        if fallback_match is not None:
            warnings.append("Matching registration was found but could not be used for projection.")
            return _failed_projection(
                region,
                source_resolved,
                fallback_match,
                warnings,
                overlay_root=overlay_root,
            )
        warnings.append("No matching registration was available for the source photo.")
        return _skipped_projection(region, warnings, source_photo=source_resolved)

    if source_match.registration.registration_status not in {"registered", "weak_match"}:
        warnings.append("Matching registration is not strong enough for projection.")
        return _failed_projection(
            region,
            source_resolved,
            source_match,
            warnings,
            overlay_root=overlay_root,
        )

    source_geometry = _region_points(region.region_type, region.coordinates)
    if source_geometry is None:
        warnings.append("Source candidate region geometry could not be interpreted.")
        return _failed_projection(
            region,
            source_resolved,
            source_match,
            warnings,
            overlay_root=overlay_root,
        )

    target_matrix = source_match.transform_matrix
    if source_match.source_is_photo_a:
        target_matrix = _invert_affine(source_match.transform_matrix)
        if target_matrix is None:
            warnings.append("Registration transform could not be inverted safely.")
            return _failed_projection(
                region,
                source_resolved,
                source_match,
                warnings,
                overlay_root=overlay_root,
            )

    projected_points = _transform_points(source_geometry, target_matrix)
    if not projected_points:
        warnings.append("Projection produced no points.")
        return _failed_projection(
            region,
            source_resolved,
            source_match,
            warnings,
            overlay_root=overlay_root,
        )

    target_width = source_match.target_record.width
    target_height = source_match.target_record.height
    if target_width is None or target_height is None:
        warnings.append("Target image dimensions are unavailable.")
        return _failed_projection(
            region,
            source_resolved,
            source_match,
            warnings,
            overlay_root=overlay_root,
        )

    projected_bbox = _bounding_box_from_float_points(projected_points)
    if projected_bbox is None:
        warnings.append("Projected region bounding box could not be calculated.")
        return _failed_projection(
            region,
            source_resolved,
            source_match,
            warnings,
            overlay_root=overlay_root,
        )

    inside_ratio = _points_inside_fraction(projected_points, target_width, target_height)
    overlap = _bbox_intersects_image(projected_bbox, target_width, target_height)
    if not overlap:
        warnings.append("Projected region falls outside the target image bounds.")
        return _failed_projection(
            region,
            source_resolved,
            source_match,
            warnings,
            overlay_root=overlay_root,
            projected_points=projected_points,
            projected_bbox=projected_bbox,
        )

    projection_confidence = _compute_projection_confidence(
        source_match.registration.registration_confidence,
        inside_ratio=inside_ratio,
        registration_status=source_match.registration.registration_status,
    )

    projected_status = _classify_projection(
        projection_confidence,
        source_match.registration.registration_status,
        inside_ratio=inside_ratio,
    )

    if source_match.registration.registration_status == "weak_match":
        warnings.append("Projection is based on a weak registration.")
    if inside_ratio < 1.0:
        warnings.append("Projected region extends beyond the target image bounds.")
    if projected_status == "weak_projection":
        warnings.append("Projection should be treated cautiously.")

    overlay_visualization = None
    if overlay_root is not None and projected_status in {"projected", "weak_projection"}:
        overlay_visualization, overlay_warning = _write_overlay(
            region,
            source_match.source_record.original_path,
            source_match.target_record.original_path,
            overlay_root,
            source_region=source_geometry,
            projected_region=projected_points,
            source_is_photo_a=source_match.source_is_photo_a,
            source_photo=source_resolved.record,
            target_photo=source_match.target_record,
            status=projected_status,
        )
        if overlay_warning is not None:
            warnings.append(overlay_warning)

    target_record = source_match.target_record
    return ProjectedCandidateRegionResult(
        candidate_id=region.candidate_id,
        source_photo=_photo_ref(source_resolved.record),
        target_photo=_photo_ref(target_record),
        source_region_type=region.region_type,
        source_region=_copy_mapping(region.coordinates),
        projected_region=ProjectedRegionGeometry(
            type="polygon",
            points=[[float(x), float(y)] for x, y in projected_points],
        ),
        projected_bounding_box=projected_bbox,
        projection_status=projected_status,
        projection_confidence=projection_confidence,
        registration_status=source_match.registration.registration_status,
        registration_confidence=source_match.registration.registration_confidence,
        transform_direction="source_to_target",
        overlay_visualization=overlay_visualization,
        warnings=_deduplicate(warnings),
    )


def _select_registration_match(
    source_record: PhotoImportRecord,
    registrations: list[RegistrationResult],
    record_lookup: dict[str, PhotoImportRecord],
) -> tuple[_RegistrationMatch | None, list[str], _RegistrationMatch | None]:
    warnings: list[str] = []
    candidates: list[_RegistrationMatch] = []
    failed_matches = 0
    fallback_match: _RegistrationMatch | None = None

    for registration in registrations:
        reg_a = _resolve_registration_record(registration.photo_a, record_lookup)
        reg_b = _resolve_registration_record(registration.photo_b, record_lookup)
        if reg_a is None or reg_b is None:
            warnings.append("A registration entry could not be resolved from the manifest.")
            continue

        if source_record is reg_a:
            source_is_photo_a = True
            target_record = reg_b
        elif source_record is reg_b:
            source_is_photo_a = False
            target_record = reg_a
        else:
            continue

        if fallback_match is None:
            fallback_match = _RegistrationMatch(
                registration=registration,
                source_is_photo_a=source_is_photo_a,
                source_record=source_record,
                target_record=target_record,
                transform_matrix=None,
            )

        if registration.transform is None:
            failed_matches += 1
            warnings.append("A matching registration has no usable transform.")
            continue

        matrix = _registration_matrix(registration.transform)
        if matrix is None:
            failed_matches += 1
            warnings.append("A matching registration transform could not be parsed.")
            continue

        candidates.append(
            _RegistrationMatch(
                registration=registration,
                source_is_photo_a=source_is_photo_a,
                source_record=source_record,
                target_record=target_record,
                transform_matrix=matrix,
            )
        )

    if not candidates:
        if failed_matches:
            warnings.append("Matching registrations were found, but none were usable for projection.")
        return None, _deduplicate(warnings), fallback_match

    candidates.sort(
        key=lambda item: (
            _registration_rank(item.registration.registration_status),
            item.registration.registration_confidence,
            item.registration.inlier_ratio,
            item.registration.inlier_count,
            item.registration.match_count,
        ),
        reverse=True,
    )
    if len(candidates) > 1:
        warnings.append("Multiple registrations matched the source photo; selected the strongest pair.")
    return candidates[0], _deduplicate(warnings), fallback_match


def _registration_rank(status: str) -> int:
    return {"registered": 3, "weak_match": 2, "failed": 1, "skipped": 0}.get(status, 0)


def _compute_projection_confidence(
    registration_confidence: float,
    *,
    inside_ratio: float,
    registration_status: str,
) -> float:
    status_factor = 1.0 if registration_status == "registered" else 0.7
    inside_factor = 0.5 + 0.5 * max(0.0, min(1.0, inside_ratio))
    score = max(0.0, min(1.0, registration_confidence * status_factor * inside_factor))
    return round(score, 3)


def _classify_projection(
    confidence: float,
    registration_status: str,
    *,
    inside_ratio: float,
) -> ProjectionStatus:
    if confidence <= 0.0:
        return "failed"
    if inside_ratio <= 0.0:
        return "failed"
    if registration_status == "weak_match" or confidence < _WEAK_PROJECTION_CONFIDENCE:
        return "weak_projection"
    if confidence < _MIN_PROJECTION_CONFIDENCE:
        return "failed"
    return "projected"


def _failed_projection(
    region: CandidateRegionResult,
    source_photo: _ResolvedPhoto,
    source_match: _RegistrationMatch,
    warnings: list[str],
    *,
    overlay_root: Path | None,
    projected_points: list[tuple[float, float]] | None = None,
    projected_bbox: ProjectedBoundingBox | None = None,
) -> ProjectedCandidateRegionResult:
    overlay_visualization = None
    if overlay_root is not None:
        overlay_visualization, overlay_warning = _write_overlay(
            region,
            source_match.source_record.original_path,
            source_match.target_record.original_path,
            overlay_root,
            source_region=_region_points(region.region_type, region.coordinates) or [],
            projected_region=projected_points or [],
            source_is_photo_a=source_match.source_is_photo_a,
            source_photo=source_match.source_record,
            target_photo=source_match.target_record,
            status="failed",
        )
        if overlay_warning is not None:
            warnings.append(overlay_warning)

    return ProjectedCandidateRegionResult(
        candidate_id=region.candidate_id,
        source_photo=_photo_ref(source_photo.record) if source_photo.record is not None else _photo_ref(source_match.source_record),
        target_photo=_photo_ref(source_match.target_record),
        source_region_type=region.region_type,
        source_region=_copy_mapping(region.coordinates),
        projected_region=None,
        projected_bounding_box=projected_bbox,
        projection_status="failed",
        projection_confidence=0.0,
        registration_status=source_match.registration.registration_status,
        registration_confidence=source_match.registration.registration_confidence,
        transform_direction="source_to_target",
        overlay_visualization=overlay_visualization,
        warnings=_deduplicate(warnings),
    )


def _skipped_projection(
    region: CandidateRegionResult,
    warnings: list[str],
    *,
    source_photo: _ResolvedPhoto | None = None,
) -> ProjectedCandidateRegionResult:
    resolved = source_photo or _ResolvedPhoto(
        record=None,
        display_path=region.photo.path,
        display_filename=region.photo.filename,
    )
    record = resolved.record
    return ProjectedCandidateRegionResult(
        candidate_id=region.candidate_id,
        source_photo=ProjectedCandidateRegionPhoto(
            path=resolved.display_path,
            filename=resolved.display_filename,
            width=record.width if record is not None else None,
            height=record.height if record is not None else None,
        ),
        target_photo=None,
        source_region_type=region.region_type,
        source_region=_copy_mapping(region.coordinates),
        projected_region=None,
        projected_bounding_box=None,
        projection_status="skipped",
        projection_confidence=0.0,
        registration_status=None,
        registration_confidence=None,
        transform_direction="unknown",
        overlay_visualization=None,
        warnings=_deduplicate(warnings),
    )


def _photo_ref(record: PhotoImportRecord) -> ProjectedCandidateRegionPhoto:
    return ProjectedCandidateRegionPhoto(
        path=record.original_path,
        filename=record.original_filename,
        width=record.width,
        height=record.height,
    )


def _build_record_lookup(manifest: PhotoImportManifest) -> dict[str, PhotoImportRecord]:
    lookup: dict[str, PhotoImportRecord] = {}
    for record in manifest.photo_records:
        lookup.setdefault(record.original_path, record)
        lookup.setdefault(str(Path(record.original_path).expanduser().resolve()), record)
        lookup.setdefault(record.original_filename, record)
        lookup.setdefault(Path(record.original_path).name, record)
    return lookup


def _resolve_photo(
    photo: ValidatedCandidateRegionPhoto,
    lookup: dict[str, PhotoImportRecord],
    regions_root: Path,
) -> _ResolvedPhoto:
    candidates: list[str] = []
    if photo.path:
        candidates.append(photo.path)
        path = Path(photo.path)
        if not path.is_absolute():
            candidates.append(str((regions_root / path).resolve()))
            candidates.append(path.name)
    if photo.filename:
        candidates.append(photo.filename)

    for candidate in candidates:
        record = lookup.get(candidate)
        if record is not None:
            return _ResolvedPhoto(
                record=record,
                display_path=photo.path or record.original_path,
                display_filename=photo.filename or record.original_filename,
            )

    return _ResolvedPhoto(record=None, display_path=photo.path, display_filename=photo.filename)


def _resolve_registration_record(
    photo_ref: ProjectedCandidateRegionPhoto,
    lookup: dict[str, PhotoImportRecord],
) -> PhotoImportRecord | None:
    candidates: list[str] = []
    if photo_ref.path:
        candidates.append(photo_ref.path)
        path = Path(photo_ref.path)
        if not path.is_absolute():
            candidates.append(str(path.expanduser().resolve()))
            candidates.append(path.name)
    if photo_ref.filename:
        candidates.append(photo_ref.filename)

    for candidate in candidates:
        record = lookup.get(candidate)
        if record is not None:
            return record
    return None


def _registration_matrix(transform: RegistrationTransform) -> np.ndarray | None:
    try:
        matrix = np.array(transform.matrix, dtype=float)
    except (TypeError, ValueError):
        return None

    if matrix.shape != (3, 3):
        return None
    return matrix


def _invert_affine(matrix: np.ndarray) -> np.ndarray | None:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None


def _region_points(region_type: str, coordinates: dict[str, Any]) -> list[tuple[float, float]] | None:
    if region_type == "rectangle":
        rect = _parse_rectangle(coordinates)
        if rect is None:
            return None
        x, y, width, height = rect
        return [
            (float(x), float(y)),
            (float(x + width), float(y)),
            (float(x + width), float(y + height)),
            (float(x), float(y + height)),
        ]

    if region_type == "polygon":
        points = _parse_polygon_points(coordinates)
        if points is None:
            return None
        return [(float(x), float(y)) for x, y in points]

    if region_type == "point_radius":
        point_radius = _parse_point_radius(coordinates)
        if point_radius is None:
            return None
        x, y, radius = point_radius
        points: list[tuple[float, float]] = []
        for index in range(_CIRCLE_SAMPLE_POINTS):
            angle = 2.0 * pi * index / _CIRCLE_SAMPLE_POINTS
            points.append((float(x + radius * cos(angle)), float(y + radius * sin(angle))))
        return points

    return None


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


def _transform_points(
    points: list[tuple[float, float]],
    matrix: np.ndarray,
) -> list[tuple[float, float]]:
    if not points:
        return []
    coords = np.array([[x, y, 1.0] for x, y in points], dtype=float).T
    transformed = matrix @ coords
    if transformed.shape[0] != 3:
        return []

    w = transformed[2]
    if np.any(np.isclose(w, 0.0)):
        return []

    xs = transformed[0] / w
    ys = transformed[1] / w
    return [(float(x), float(y)) for x, y in zip(xs, ys, strict=False)]


def _bounding_box_from_float_points(points: list[tuple[float, float]]) -> ProjectedBoundingBox | None:
    if not points:
        return None
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    min_x = floor(min(xs))
    min_y = floor(min(ys))
    max_x = ceil(max(xs))
    max_y = ceil(max(ys))
    width = max(1, max_x - min_x)
    height = max(1, max_y - min_y)
    return ProjectedBoundingBox(x=min_x, y=min_y, width=width, height=height)


def _points_inside_fraction(points: list[tuple[float, float]], width: int, height: int) -> float:
    if not points:
        return 0.0
    inside = sum(1 for x, y in points if 0.0 <= x < width and 0.0 <= y < height)
    return inside / len(points)


def _bbox_intersects_image(bbox: ProjectedBoundingBox, width: int, height: int) -> bool:
    left = bbox.x
    top = bbox.y
    right = bbox.x + bbox.width
    bottom = bbox.y + bbox.height
    return not (right <= 0 or bottom <= 0 or left >= width or top >= height)


def _prepare_overlay_dir(overlay_dir: str | Path | None) -> Path | None:
    if overlay_dir is None:
        return None
    path = Path(overlay_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_overlay(
    region: CandidateRegionResult,
    source_image_path: str,
    target_image_path: str,
    overlay_root: Path,
    *,
    source_region: list[tuple[float, float]],
    projected_region: list[tuple[float, float]],
    source_is_photo_a: bool,
    source_photo: PhotoImportRecord,
    target_photo: PhotoImportRecord,
    status: str,
) -> tuple[str | None, str | None]:
    source_name = _sanitize_name(region.candidate_id)
    target_name = _sanitize_name(target_photo.original_filename)
    source_filename = _sanitize_name(source_photo.original_filename)
    output_path = overlay_root / f"{source_name}_{source_filename}_to_{target_name}_projection.jpg"

    try:
        with Image.open(source_image_path) as source_image, Image.open(target_image_path) as target_image:
            source_image = ImageOps.exif_transpose(source_image).convert("RGBA")
            target_image = ImageOps.exif_transpose(target_image).convert("RGBA")
            gap = 24
            canvas_width = source_image.width + gap + target_image.width
            canvas_height = max(source_image.height, target_image.height)
            canvas = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 255))
            canvas.paste(source_image, (0, 0))
            canvas.paste(target_image, (source_image.width + gap, 0))

            draw = ImageDraw.Draw(canvas)
            _draw_region(
                draw,
                region.region_type,
                region.coordinates,
                offset=(0, 0),
                outline=_SOURCE_OUTLINE,
                fill=_SOURCE_FILL,
            )
            _draw_polygon(
                draw,
                projected_region,
                offset=(source_image.width + gap, 0),
                outline=_PROJECTED_OUTLINE,
                fill=_PROJECTED_FILL,
            )

            separator_x = source_image.width + gap // 2
            draw.line((separator_x, 0, separator_x, canvas_height), fill=(180, 180, 180, 255), width=2)

            label_fill = (255, 255, 255, 230)
            label_outline = (0, 0, 0, 255)
            _draw_label(draw, (12, 12), "source_manual_region", label_fill, label_outline)
            _draw_label(
                draw,
                (source_image.width + gap + 12, 12),
                "projected_region",
                label_fill,
                label_outline,
            )
            _draw_label(draw, (12, canvas_height - 40), f"technical_projection: {status}", label_fill, label_outline)
            if source_is_photo_a:
                _draw_label(draw, (12, 44), "source is photo_a", label_fill, label_outline)
            else:
                _draw_label(draw, (12, 44), "source is photo_b", label_fill, label_outline)

            canvas.convert("RGB").save(output_path, quality=92)
        return str(output_path), None
    except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as exc:
        return None, f"Projection overlay could not be created: {exc.__class__.__name__}."


def _draw_region(
    draw: ImageDraw.ImageDraw,
    region_type: str,
    coordinates: dict[str, Any],
    *,
    offset: tuple[int, int],
    outline: tuple[int, int, int, int],
    fill: tuple[int, int, int, int],
) -> None:
    ox, oy = offset
    if region_type == "rectangle":
        rect = _parse_rectangle(coordinates)
        if rect is None:
            return
        x, y, width, height = rect
        draw.rectangle((x + ox, y + oy, x + ox + width, y + oy + height), outline=outline, fill=fill, width=3)
        return

    if region_type == "polygon":
        points = _parse_polygon_points(coordinates)
        if points is None:
            return
        shifted = [(x + ox, y + oy) for x, y in points]
        _draw_polygon(draw, shifted, offset=(0, 0), outline=outline, fill=fill)
        return

    if region_type == "point_radius":
        point_radius = _parse_point_radius(coordinates)
        if point_radius is None:
            return
        x, y, radius = point_radius
        draw.ellipse(
            (x - radius + ox, y - radius + oy, x + radius + ox, y + radius + oy),
            outline=outline,
            fill=fill,
            width=3,
        )


def _draw_polygon(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    *,
    offset: tuple[int, int],
    outline: tuple[int, int, int, int],
    fill: tuple[int, int, int, int],
) -> None:
    if not points:
        return
    ox, oy = offset
    shifted = [(x + ox, y + oy) for x, y in points]
    draw.polygon(shifted, outline=outline, fill=fill)
    draw.line(shifted + [shifted[0]], fill=outline, width=3)


def _draw_label(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
) -> None:
    x, y = position
    text_box = (x, y, x + max(160, 8 * len(text) + 20), y + 28)
    draw.rounded_rectangle(text_box, radius=8, fill=fill, outline=outline)
    draw.text((x + 10, y + 6), text, fill=(0, 0, 0, 255))


def _invalid_region_result(
    region: CandidateRegionResult,
    warnings: list[str],
) -> ProjectedCandidateRegionResult:
    return ProjectedCandidateRegionResult(
        candidate_id=region.candidate_id,
        source_photo=ProjectedCandidateRegionPhoto(
            path=region.photo.path,
            filename=region.photo.filename,
            width=region.photo.width,
            height=region.photo.height,
        ),
        target_photo=None,
        source_region_type=region.region_type,
        source_region=_copy_mapping(region.coordinates),
        projected_region=None,
        projected_bounding_box=None,
        projection_status="skipped",
        projection_confidence=0.0,
        registration_status=None,
        registration_confidence=None,
        transform_direction="unknown",
        overlay_visualization=None,
        warnings=_deduplicate(warnings),
    )


def _skipped_projection(
    region: CandidateRegionResult,
    warnings: list[str],
    *,
    source_photo: _ResolvedPhoto | None = None,
) -> ProjectedCandidateRegionResult:
    if source_photo is None:
        source_photo = _ResolvedPhoto(record=None, display_path=region.photo.path, display_filename=region.photo.filename)
    return ProjectedCandidateRegionResult(
        candidate_id=region.candidate_id,
        source_photo=ProjectedCandidateRegionPhoto(
            path=source_photo.display_path,
            filename=source_photo.display_filename,
            width=source_photo.record.width if source_photo.record is not None else region.photo.width,
            height=source_photo.record.height if source_photo.record is not None else region.photo.height,
        ),
        target_photo=None,
        source_region_type=region.region_type,
        source_region=_copy_mapping(region.coordinates),
        projected_region=None,
        projected_bounding_box=None,
        projection_status="skipped",
        projection_confidence=0.0,
        registration_status=None,
        registration_confidence=None,
        transform_direction="unknown",
        overlay_visualization=None,
        warnings=_deduplicate(warnings),
    )


def _copy_mapping(value: dict[str, Any]) -> dict[str, Any]:
    return {key: _copy_value(item) for key, item in value.items()}


def _copy_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _copy_mapping(value)
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    return value


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
    return sanitized or "projection"
