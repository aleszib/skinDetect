"""Conservative technical change metrics for tracked candidate observations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from skintrack.io.photos import PhotoImportManifest, PhotoImportRecord
from skintrack.tracking.temporal import (
    CandidateTrack,
    CandidateTrackManifest,
    CandidateTrackObservation,
)

MeasurementStatus = Literal["measurable", "weak_evidence", "not_measurable", "skipped"]


class ChangeTrackPhoto(BaseModel):
    """Reference to a photo used in change measurement output."""

    model_config = ConfigDict(extra="forbid")

    path: str | None = None
    filename: str | None = None


class ChangeBoundingBox(BaseModel):
    """Bounding box for a measured candidate region."""

    model_config = ConfigDict(extra="forbid")

    x: int
    y: int
    width: int
    height: int


class ChangeTimeSpan(BaseModel):
    """Time span covered by a candidate track measurement."""

    model_config = ConfigDict(extra="forbid")

    first_observed_at: datetime | None = None
    last_observed_at: datetime | None = None
    days: float | None = None


class ChangeGeometryMetrics(BaseModel):
    """Pixel geometry change measurements for a track."""

    model_config = ConfigDict(extra="forbid")

    first_area_px: int | None = None
    last_area_px: int | None = None
    absolute_area_change_px: int | None = None
    relative_area_change: float | None = None
    first_bounding_box: ChangeBoundingBox | None = None
    last_bounding_box: ChangeBoundingBox | None = None
    first_bounding_box_area_px: int | None = None
    last_bounding_box_area_px: int | None = None
    bounding_box_area_change_px: int | None = None
    relative_bounding_box_area_change: float | None = None


class ChangeAppearanceMetrics(BaseModel):
    """Simple appearance change measurements for a track."""

    model_config = ConfigDict(extra="forbid")

    first_mean_brightness: float | None = None
    last_mean_brightness: float | None = None
    brightness_change: float | None = None
    first_mean_rgb: list[float] | None = None
    last_mean_rgb: list[float] | None = None
    mean_rgb_distance: float | None = None


class ChangeEvidence(BaseModel):
    """Evidence and provenance that support a change measurement."""

    model_config = ConfigDict(extra="forbid")

    tracking_confidence: float | None = None
    lowest_observation_confidence: float | None = None
    observation_confidences: list[float] = Field(default_factory=list)
    observation_sources: list[str] = Field(default_factory=list)
    timestamp_sources: list[str] = Field(default_factory=list)
    timestamp_confidences: list[str] = Field(default_factory=list)
    projection_confidences: list[float | None] = Field(default_factory=list)
    quality_warnings: list[str] = Field(default_factory=list)
    timestamp_warnings: list[str] = Field(default_factory=list)
    projection_warnings: list[str] = Field(default_factory=list)
    measurement_warnings: list[str] = Field(default_factory=list)


class CandidateChangeResult(BaseModel):
    """One conservative technical change measurement result."""

    model_config = ConfigDict(extra="forbid")

    track_id: str
    candidate_id: str
    measurement_status: MeasurementStatus
    measurement_confidence: float
    observation_count: int
    time_span: ChangeTimeSpan
    geometry_metrics: ChangeGeometryMetrics | None = None
    appearance_metrics: ChangeAppearanceMetrics | None = None
    evidence: ChangeEvidence
    neutral_summary: str
    warnings: list[str] = Field(default_factory=list)


class CandidateChangeMetricsManifest(BaseModel):
    """JSON manifest for technical change metrics over candidate tracks."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "candidate-change-metrics-v1"
    created_at: datetime
    source_manifest: str
    source_tracks: str
    track_count: int
    measurable_count: int
    results: list[CandidateChangeResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _ResolvedPhoto:
    record: PhotoImportRecord | None
    resolved_path: Path | None
    display_path: str | None
    display_filename: str | None


@dataclass(frozen=True)
class _ObservationMeasurement:
    observation: CandidateTrackObservation
    resolved_photo: _ResolvedPhoto
    image_path: Path
    area_px: int
    bounding_box: ChangeBoundingBox
    mean_brightness: float
    mean_rgb: list[float]
    quality_warnings: list[str]
    timestamp_warnings: list[str]
    projection_warnings: list[str]
    image_warnings: list[str]

    @property
    def confidence(self) -> float:
        return self.observation.confidence


def load_photo_import_manifest(manifest_path: str | Path) -> PhotoImportManifest:
    """Load a photo import manifest from JSON."""

    path = Path(manifest_path)
    return PhotoImportManifest.model_validate_json(path.read_text(encoding="utf-8"))


def load_candidate_track_manifest(tracks_path: str | Path) -> CandidateTrackManifest:
    """Load a candidate-track manifest from JSON."""

    path = Path(tracks_path)
    return CandidateTrackManifest.model_validate_json(path.read_text(encoding="utf-8"))


def measure_candidate_changes(
    manifest_path: str | Path,
    tracks_path: str | Path,
) -> CandidateChangeMetricsManifest:
    """Load a manifest and candidate tracks, then compute technical change metrics."""

    manifest = load_photo_import_manifest(manifest_path)
    tracks = load_candidate_track_manifest(tracks_path)
    return build_candidate_change_metrics_manifest(
        manifest,
        tracks,
        source_manifest=manifest_path,
        source_tracks=tracks_path,
    )


def write_candidate_change_metrics_manifest(
    manifest_path: str | Path,
    tracks_path: str | Path,
    output_path: str | Path,
) -> CandidateChangeMetricsManifest:
    """Compute candidate change metrics and write the result to disk."""

    change_manifest = measure_candidate_changes(manifest_path, tracks_path)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(change_manifest.model_dump_json(indent=2), encoding="utf-8")
    return change_manifest


def build_candidate_change_metrics_manifest(
    manifest: PhotoImportManifest,
    tracks: CandidateTrackManifest,
    *,
    source_manifest: str | Path,
    source_tracks: str | Path,
) -> CandidateChangeMetricsManifest:
    """Compute conservative change metrics for each candidate track."""

    record_lookup = _build_record_lookup(manifest)
    results: list[CandidateChangeResult] = []
    warnings: list[str] = []

    for track in tracks.tracks:
        result = _measure_track(track, manifest, record_lookup)
        results.append(result)
        warnings.extend(result.warnings)

    measurable_count = sum(result.measurement_status == "measurable" for result in results)
    if not results:
        warnings.append("No candidate tracks were provided.")
    elif measurable_count == 0:
        warnings.append("No candidate tracks reached measurable status.")

    return CandidateChangeMetricsManifest(
        created_at=datetime.now(timezone.utc),
        source_manifest=str(Path(source_manifest)),
        source_tracks=str(Path(source_tracks)),
        track_count=len(results),
        measurable_count=measurable_count,
        results=results,
        warnings=_deduplicate(warnings),
    )


def _measure_track(
    track: CandidateTrack,
    manifest: PhotoImportManifest,
    record_lookup: dict[str, PhotoImportRecord],
) -> CandidateChangeResult:
    track_warnings = list(track.warnings)
    observations = _sort_observations(track.observations)
    usable_measurements: list[_ObservationMeasurement] = []
    observation_confidences: list[float] = []
    observation_sources: list[str] = []
    timestamp_sources: list[str] = []
    timestamp_confidences: list[str] = []
    projection_confidences: list[float | None] = []
    quality_warnings: list[str] = []
    timestamp_warnings: list[str] = []
    projection_warnings: list[str] = []
    measurement_warnings: list[str] = []
    observation_measurement_warnings: list[str] = []

    for observation in observations:
        measurement, warnings = _measure_observation(
            observation,
            manifest,
            record_lookup,
        )
        observation_confidences.append(round(float(observation.confidence), 3))
        observation_sources.append(observation.observation_source)
        timestamp_sources.append(observation.timestamp_source)
        timestamp_confidences.append(observation.timestamp_confidence)
        projection_confidences.append(
            _projection_confidence_for_observation(observation)
        )
        measurement_warnings.extend(warnings)
        if measurement is None:
            continue
        usable_measurements.append(measurement)
        quality_warnings.extend(measurement.quality_warnings)
        timestamp_warnings.extend(measurement.timestamp_warnings)
        projection_warnings.extend(measurement.projection_warnings)
        observation_measurement_warnings.extend(measurement.image_warnings)

    time_span = _build_time_span(observations)
    lowest_observation_confidence = (
        min(observation_confidences) if observation_confidences else None
    )

    if len(usable_measurements) < 2:
        status = "skipped" if not usable_measurements else "not_measurable"
        neutral_summary = _neutral_summary(status)
        warnings = _deduplicate(
            track_warnings
            + measurement_warnings
            + [
                "At least two usable observations are required for technical change measurement."
            ]
        )
        return CandidateChangeResult(
            track_id=track.track_id,
            candidate_id=track.candidate_id,
            measurement_status=status,
            measurement_confidence=0.0,
            observation_count=len(observations),
            time_span=time_span,
            geometry_metrics=None,
            appearance_metrics=None,
            evidence=ChangeEvidence(
                tracking_confidence=round(float(track.tracking_confidence), 3),
                lowest_observation_confidence=lowest_observation_confidence,
                observation_confidences=observation_confidences,
                observation_sources=observation_sources,
                timestamp_sources=timestamp_sources,
                timestamp_confidences=timestamp_confidences,
                projection_confidences=projection_confidences,
                quality_warnings=_deduplicate(quality_warnings),
                timestamp_warnings=_deduplicate(timestamp_warnings),
                projection_warnings=_deduplicate(projection_warnings),
                measurement_warnings=_deduplicate(measurement_warnings + observation_measurement_warnings),
            ),
            neutral_summary=neutral_summary,
            warnings=_deduplicate(warnings),
        )

    first_measurement = usable_measurements[0]
    last_measurement = usable_measurements[-1]

    geometry_metrics = _build_geometry_metrics(first_measurement, last_measurement)
    appearance_metrics = _build_appearance_metrics(first_measurement, last_measurement)
    warnings = _deduplicate(
        track_warnings
        + measurement_warnings
        + observation_measurement_warnings
        + _measurement_status_warnings(first_measurement, last_measurement)
    )
    measurement_confidence = _compute_measurement_confidence(
        track,
        usable_measurements,
        geometry_metrics,
        appearance_metrics,
        warning_count=len(warnings),
    )
    status = _classify_measurement_status(measurement_confidence, warnings)

    neutral_summary = _neutral_summary(status)

    return CandidateChangeResult(
        track_id=track.track_id,
        candidate_id=track.candidate_id,
        measurement_status=status,
        measurement_confidence=measurement_confidence,
        observation_count=len(observations),
        time_span=time_span,
        geometry_metrics=geometry_metrics,
        appearance_metrics=appearance_metrics,
        evidence=ChangeEvidence(
            tracking_confidence=round(float(track.tracking_confidence), 3),
            lowest_observation_confidence=lowest_observation_confidence,
            observation_confidences=observation_confidences,
            observation_sources=observation_sources,
            timestamp_sources=timestamp_sources,
            timestamp_confidences=timestamp_confidences,
            projection_confidences=projection_confidences,
            quality_warnings=_deduplicate(quality_warnings),
            timestamp_warnings=_deduplicate(timestamp_warnings),
            projection_warnings=_deduplicate(projection_warnings),
            measurement_warnings=_deduplicate(measurement_warnings + observation_measurement_warnings),
        ),
        neutral_summary=neutral_summary,
        warnings=warnings,
    )


def _measure_observation(
    observation: CandidateTrackObservation,
    manifest: PhotoImportManifest,
    record_lookup: dict[str, PhotoImportRecord],
) -> tuple[_ObservationMeasurement | None, list[str]]:
    warnings = list(observation.warnings)
    resolved = _resolve_photo(observation.photo.path, observation.photo.filename, record_lookup)

    image_path = _resolve_image_path(resolved, manifest)
    if image_path is None:
        warnings.append("Observation image could not be resolved from the import manifest.")
        return None, _deduplicate(warnings)

    if not image_path.exists():
        warnings.append("Observation image file is missing.")
        return None, _deduplicate(warnings)

    if resolved.record is not None:
        if resolved.record.import_status != "imported":
            warnings.append("Observation photo was not imported and cannot be measured safely.")
            return None, _deduplicate(warnings)
        if resolved.record.quality is not None:
            warnings.extend(resolved.record.quality.warnings)
            if not resolved.record.quality.readable:
                warnings.append("Observation photo is unreadable.")
                return None, _deduplicate(warnings)

    try:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            rgb = np.asarray(image, dtype=np.float32)
    except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as exc:
        warnings.append(f"Observation image could not be opened: {exc.__class__.__name__}.")
        return None, _deduplicate(warnings)

    region_info = _normalize_region(observation.region_type, observation.region)
    warnings.extend(region_info.warnings)
    if region_info.kind is None:
        warnings.append("Observation region could not be interpreted.")
        return None, _deduplicate(warnings)

    mask, bbox, region_warnings = _build_region_mask(rgb.shape[1], rgb.shape[0], region_info)
    warnings.extend(region_warnings)
    if mask is None or bbox is None:
        warnings.append("Observation region could not be measured.")
        return None, _deduplicate(warnings)

    area_px = int(mask.sum())
    if area_px <= 0:
        warnings.append("Observation region has no measurable area.")
        return None, _deduplicate(warnings)

    masked_pixels = rgb[mask]
    if masked_pixels.size == 0:
        warnings.append("Observation region contains no measurable pixels.")
        return None, _deduplicate(warnings)

    mean_rgb = masked_pixels.mean(axis=0).tolist()
    mean_brightness = float(np.dot(masked_pixels.mean(axis=0) / 255.0, np.array([0.2126, 0.7152, 0.0722])))

    quality_warnings = list(resolved.record.quality.warnings) if resolved.record and resolved.record.quality else []
    timestamp_warnings = _timestamp_warnings(observation)
    projection_warnings = _projection_warnings(observation)

    return (
        _ObservationMeasurement(
            observation=observation,
            resolved_photo=resolved,
            image_path=image_path,
            area_px=area_px,
            bounding_box=bbox,
            mean_brightness=round(mean_brightness, 4),
            mean_rgb=[round(float(value), 3) for value in mean_rgb],
            quality_warnings=_deduplicate(quality_warnings),
            timestamp_warnings=_deduplicate(timestamp_warnings),
            projection_warnings=_deduplicate(projection_warnings),
            image_warnings=[],
        ),
        _deduplicate(warnings),
    )


def _build_region_mask(
    width: int,
    height: int,
    region_info: _RegionInfo,
) -> tuple[np.ndarray | None, ChangeBoundingBox | None, list[str]]:
    warnings = list(region_info.warnings)
    mask = np.zeros((height, width), dtype=bool)

    if region_info.kind == "rectangle":
        x = int(round(region_info.payload["x"]))
        y = int(round(region_info.payload["y"]))
        rect_width = int(round(region_info.payload["width"]))
        rect_height = int(round(region_info.payload["height"]))
        if rect_width <= 0 or rect_height <= 0:
            warnings.append("Rectangle region has a non-positive size.")
            return None, None, warnings
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(width, x + rect_width)
        y1 = min(height, y + rect_height)
        if x1 <= x0 or y1 <= y0:
            warnings.append("Rectangle region does not intersect the image bounds.")
            return None, None, warnings
        mask[y0:y1, x0:x1] = True
        bbox = ChangeBoundingBox(x=x0, y=y0, width=x1 - x0, height=y1 - y0)
        if x0 != x or y0 != y or x1 != x + rect_width or y1 != y + rect_height:
            warnings.append("Rectangle region was clamped to the image bounds.")
        return mask, bbox, warnings

    if region_info.kind == "polygon":
        points = region_info.payload["points"]
        if len(points) < 3:
            warnings.append("Polygon regions require at least 3 points.")
            return None, None, warnings
        image_mask = Image.new("L", (width, height), 0)
        mask_draw = ImageDraw.Draw(image_mask)
        mask_points = [(int(round(x)), int(round(y))) for x, y in points]
        mask_draw.polygon(mask_points, fill=1)
        mask = np.array(image_mask, dtype=bool)
        if not mask.any():
            warnings.append("Polygon region does not intersect the image bounds.")
            return None, None, warnings
        bbox = _mask_bounding_box(mask)
        if bbox is None:
            warnings.append("Polygon region bounding box could not be calculated.")
            return None, None, warnings
        if any(x < 0 or y < 0 or x >= width or y >= height for x, y in mask_points):
            warnings.append("Polygon region was clipped to the image bounds.")
        return mask, bbox, warnings

    if region_info.kind == "point_radius":
        x = int(round(region_info.payload["x"]))
        y = int(round(region_info.payload["y"]))
        radius = int(round(region_info.payload["radius"]))
        if radius <= 0:
            warnings.append("Point-radius region has a non-positive radius.")
            return None, None, warnings
        image_mask = Image.new("L", (width, height), 0)
        mask_draw = ImageDraw.Draw(image_mask)
        mask_draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=1)
        mask = np.array(image_mask, dtype=bool)
        if not mask.any():
            warnings.append("Point-radius region does not intersect the image bounds.")
            return None, None, warnings
        bbox = _mask_bounding_box(mask)
        if bbox is None:
            warnings.append("Point-radius region bounding box could not be calculated.")
            return None, None, warnings
        if x - radius < 0 or y - radius < 0 or x + radius >= width or y + radius >= height:
            warnings.append("Point-radius region was clipped to the image bounds.")
        return mask, bbox, warnings

    warnings.append(f"Unsupported region type for measurement: {region_info.kind}.")
    return None, None, warnings


def _mask_bounding_box(mask: np.ndarray) -> ChangeBoundingBox | None:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return ChangeBoundingBox(x=x0, y=y0, width=x1 - x0, height=y1 - y0)


def _build_geometry_metrics(
    first: _ObservationMeasurement,
    last: _ObservationMeasurement,
) -> ChangeGeometryMetrics:
    first_bbox_area = first.bounding_box.width * first.bounding_box.height
    last_bbox_area = last.bounding_box.width * last.bounding_box.height
    absolute_area_change_px = last.area_px - first.area_px
    relative_area_change = (
        (absolute_area_change_px / first.area_px) if first.area_px else None
    )
    bbox_area_change_px = last_bbox_area - first_bbox_area
    relative_bbox_change = (
        (bbox_area_change_px / first_bbox_area) if first_bbox_area else None
    )
    return ChangeGeometryMetrics(
        first_area_px=first.area_px,
        last_area_px=last.area_px,
        absolute_area_change_px=absolute_area_change_px,
        relative_area_change=round(float(relative_area_change), 4) if relative_area_change is not None else None,
        first_bounding_box=first.bounding_box,
        last_bounding_box=last.bounding_box,
        first_bounding_box_area_px=first_bbox_area,
        last_bounding_box_area_px=last_bbox_area,
        bounding_box_area_change_px=bbox_area_change_px,
        relative_bounding_box_area_change=(
            round(float(relative_bbox_change), 4) if relative_bbox_change is not None else None
        ),
    )


def _build_appearance_metrics(
    first: _ObservationMeasurement,
    last: _ObservationMeasurement,
) -> ChangeAppearanceMetrics:
    brightness_change = last.mean_brightness - first.mean_brightness
    rgb_distance = sqrt(
        sum((last_value - first_value) ** 2 for first_value, last_value in zip(first.mean_rgb, last.mean_rgb))
    )
    return ChangeAppearanceMetrics(
        first_mean_brightness=round(first.mean_brightness, 4),
        last_mean_brightness=round(last.mean_brightness, 4),
        brightness_change=round(brightness_change, 4),
        first_mean_rgb=[round(float(value), 3) for value in first.mean_rgb],
        last_mean_rgb=[round(float(value), 3) for value in last.mean_rgb],
        mean_rgb_distance=round(float(rgb_distance), 4),
    )


def _compute_measurement_confidence(
    track: CandidateTrack,
    measurements: list[_ObservationMeasurement],
    geometry_metrics: ChangeGeometryMetrics,
    appearance_metrics: ChangeAppearanceMetrics,
    *,
    warning_count: int,
) -> float:
    if not measurements:
        return 0.0

    usable_count = len(measurements)
    observation_pair_factor = min(1.0, usable_count / 2.0)
    average_observation_confidence = sum(m.confidence for m in measurements) / usable_count
    tracking_factor = max(0.0, min(1.0, track.tracking_confidence))
    timestamp_fraction = sum(1 for m in measurements if m.observation.observed_at is not None) / usable_count
    low_quality_fraction = sum(1 for m in measurements if m.quality_warnings) / usable_count
    weak_projection_fraction = sum(
        1 for m in measurements if m.observation.observation_status == "weak_projection"
    ) / usable_count
    missing_path_fraction = sum(1 for m in measurements if not m.image_path.exists()) / usable_count

    score = (
        0.35 * average_observation_confidence
        + 0.25 * tracking_factor
        + 0.15 * observation_pair_factor
        + 0.1 * timestamp_fraction
        + 0.1 * (1.0 - min(1.0, low_quality_fraction * 0.6))
        + 0.05 * _comparable_metric_factor(geometry_metrics, appearance_metrics)
    )
    score -= 0.08 * weak_projection_fraction
    score -= 0.1 * missing_path_fraction
    score -= min(0.2, warning_count * 0.03)
    if timestamp_fraction < 1.0:
        score -= 0.05
    if low_quality_fraction > 0.0:
        score -= 0.05
    return round(max(0.0, min(1.0, score)), 3)


def _comparable_metric_factor(
    geometry_metrics: ChangeGeometryMetrics,
    appearance_metrics: ChangeAppearanceMetrics,
) -> float:
    factors = [1.0]
    if geometry_metrics.first_area_px and geometry_metrics.last_area_px:
        factors.append(1.0 if geometry_metrics.first_area_px > 0 and geometry_metrics.last_area_px > 0 else 0.0)
    if appearance_metrics.first_mean_rgb and appearance_metrics.last_mean_rgb:
        factors.append(1.0)
    return sum(factors) / len(factors)


def _classify_measurement_status(confidence: float, warnings: list[str]) -> MeasurementStatus:
    if confidence <= 0.0:
        return "skipped"
    if confidence < 0.35:
        return "not_measurable"
    if warnings:
        return "weak_evidence"
    if confidence < 0.75:
        return "weak_evidence"
    return "measurable"


def _neutral_summary(status: MeasurementStatus) -> str:
    if status == "measurable":
        return "Measured technical differences across two observations for review."
    if status == "weak_evidence":
        return "Technical differences were computed with limited evidence and should be reviewed cautiously."
    if status == "not_measurable":
        return "Not enough usable observations were available for technical change measurement."
    return "Change measurement was skipped because no usable observations were available."


def _timestamp_warnings(observation: CandidateTrackObservation) -> list[str]:
    warnings: list[str] = []
    if observation.observed_at is None:
        warnings.append("Timestamp is unavailable for this observation.")
    if observation.timestamp_confidence in {"unknown", "low"}:
        warnings.append("Timestamp confidence is limited for this observation.")
    return warnings


def _projection_warnings(observation: CandidateTrackObservation) -> list[str]:
    warnings: list[str] = []
    if observation.observation_source == "projection":
        if observation.observation_status == "weak_projection":
            warnings.append("Observation is based on a weak projection.")
        elif observation.observation_status not in {"projected", "weak_projection"}:
            warnings.append("Projection observation is not suitable for measurement.")
    return warnings


def _measurement_status_warnings(
    first: _ObservationMeasurement,
    last: _ObservationMeasurement,
) -> list[str]:
    warnings: list[str] = []
    if first.quality_warnings or last.quality_warnings:
        warnings.append("One or more observations have low-quality imaging warnings.")
    if first.timestamp_warnings or last.timestamp_warnings:
        warnings.append("One or more observations have timestamp warnings.")
    if first.projection_warnings or last.projection_warnings:
        warnings.append("One or more observations have projection warnings.")
    return warnings


def _build_time_span(observations: list[CandidateTrackObservation]) -> ChangeTimeSpan:
    timestamps = [observation.observed_at for observation in observations if observation.observed_at is not None]
    if not timestamps:
        return ChangeTimeSpan()
    first = min(timestamps)
    last = max(timestamps)
    days = None
    if len(timestamps) >= 2:
        days = round((last - first).total_seconds() / 86400.0, 4)
    return ChangeTimeSpan(first_observed_at=first, last_observed_at=last, days=days)


class _RegionInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def _normalize_region(region_type: str, region: dict[str, Any]) -> _RegionInfo:
    warnings: list[str] = []
    normalized_type = region_type or str(region.get("type") or "")

    if normalized_type == "rectangle":
        try:
            payload = {
                "x": float(region["x"]),
                "y": float(region["y"]),
                "width": float(region["width"]),
                "height": float(region["height"]),
            }
        except (KeyError, TypeError, ValueError):
            warnings.append("Rectangle region coordinates are invalid.")
            return _RegionInfo(kind=None, payload={}, warnings=warnings)
        return _RegionInfo(kind="rectangle", payload=payload, warnings=warnings)

    if normalized_type == "polygon":
        points = region.get("points")
        parsed_points: list[tuple[float, float]] = []
        if isinstance(points, list):
            for point in points:
                if isinstance(point, dict):
                    try:
                        parsed_points.append((float(point["x"]), float(point["y"])))
                    except (KeyError, TypeError, ValueError):
                        warnings.append("Polygon region coordinates are invalid.")
                        return _RegionInfo(kind=None, payload={}, warnings=warnings)
                elif isinstance(point, (list, tuple)) and len(point) >= 2:
                    try:
                        parsed_points.append((float(point[0]), float(point[1])))
                    except (TypeError, ValueError):
                        warnings.append("Polygon region coordinates are invalid.")
                        return _RegionInfo(kind=None, payload={}, warnings=warnings)
                else:
                    warnings.append("Polygon region coordinates are invalid.")
                    return _RegionInfo(kind=None, payload={}, warnings=warnings)
        if not parsed_points:
            warnings.append("Polygon region coordinates are missing.")
            return _RegionInfo(kind=None, payload={}, warnings=warnings)
        return _RegionInfo(kind="polygon", payload={"points": parsed_points}, warnings=warnings)

    if normalized_type == "point_radius":
        try:
            payload = {
                "x": float(region["x"]),
                "y": float(region["y"]),
                "radius": float(region["radius"]),
            }
        except (KeyError, TypeError, ValueError):
            warnings.append("Point-radius region coordinates are invalid.")
            return _RegionInfo(kind=None, payload={}, warnings=warnings)
        return _RegionInfo(kind="point_radius", payload=payload, warnings=warnings)

    warnings.append(f"Unsupported region type for measurement: {normalized_type or '[none]'}")
    return _RegionInfo(kind=None, payload={}, warnings=warnings)


def _resolve_photo(
    path: str | None,
    filename: str | None,
    lookup: dict[str, PhotoImportRecord],
) -> _ResolvedPhoto:
    candidates: list[str] = []
    if path:
        candidates.append(path)
        candidate_path = Path(path)
        if not candidate_path.is_absolute():
            candidates.append(str(candidate_path.expanduser().resolve()))
            candidates.append(candidate_path.name)
    if filename:
        candidates.append(filename)

    for candidate in candidates:
        record = lookup.get(candidate)
        if record is not None:
            return _ResolvedPhoto(
                record=record,
                resolved_path=Path(record.original_path),
                display_path=path or record.original_path,
                display_filename=filename or record.original_filename,
            )

    if path:
        candidate_path = Path(path).expanduser()
        if candidate_path.exists():
            return _ResolvedPhoto(
                record=None,
                resolved_path=candidate_path.resolve(),
                display_path=path,
                display_filename=filename,
            )
    if filename:
        for candidate in lookup.values():
            if candidate.original_filename == filename:
                return _ResolvedPhoto(
                    record=candidate,
                    resolved_path=Path(candidate.original_path),
                    display_path=path or candidate.original_path,
                    display_filename=filename,
                )

    return _ResolvedPhoto(record=None, resolved_path=None, display_path=path, display_filename=filename)


def _resolve_image_path(
    resolved: _ResolvedPhoto,
    manifest: PhotoImportManifest,
) -> Path | None:
    if resolved.resolved_path is not None:
        return resolved.resolved_path
    if resolved.display_path is None:
        return None
    candidate_path = Path(resolved.display_path).expanduser()
    if candidate_path.is_absolute():
        return candidate_path
    manifest_root = Path(manifest.input_directory)
    possible = manifest_root / candidate_path
    if possible.exists():
        return possible.resolve()
    if candidate_path.exists():
        return candidate_path.resolve()
    return possible.resolve()


def _build_record_lookup(manifest: PhotoImportManifest) -> dict[str, PhotoImportRecord]:
    lookup: dict[str, PhotoImportRecord] = {}
    for record in manifest.photo_records:
        lookup.setdefault(record.original_path, record)
        lookup.setdefault(str(Path(record.original_path).expanduser().resolve()), record)
        lookup.setdefault(record.original_filename, record)
        lookup.setdefault(Path(record.original_path).name, record)
    return lookup


def _sort_observations(observations: list[CandidateTrackObservation]) -> list[CandidateTrackObservation]:
    return sorted(
        observations,
        key=lambda observation: (
            observation.observed_at is None,
            observation.observed_at or datetime.max,
            observation.observation_source,
            -float(observation.confidence),
        ),
    )


def _projection_confidence_for_observation(observation: CandidateTrackObservation) -> float | None:
    if observation.observation_source != "projection":
        return None
    return round(float(observation.confidence), 3)


def _deduplicate(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
