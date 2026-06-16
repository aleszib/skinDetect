"""Temporal grouping for validated and projected candidate regions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from skintrack.io.photos import PhotoImportManifest, PhotoImportRecord
from skintrack.regions.manual import CandidateRegionResult, CandidateRegionValidationManifest
from skintrack.regions.projection import (
    ProjectedCandidateRegionManifest,
    ProjectedCandidateRegionResult,
)

TrackingStatus = Literal["tracked", "partial", "ambiguous", "untracked"]
ObservationSource = Literal["manual", "projection"]
ObservationStatus = Literal["manual_valid", "projected", "weak_projection"]

_MANUAL_OBSERVATION_PRIORITY: Final[int] = 2
_PROJECTED_OBSERVATION_PRIORITY: Final[int] = 1


class CandidateTrackPhoto(BaseModel):
    """Photo reference included in a candidate track observation."""

    model_config = ConfigDict(extra="forbid")

    path: str | None = None
    filename: str | None = None


class CandidateTrackObservation(BaseModel):
    """One temporal observation within a candidate track."""

    model_config = ConfigDict(extra="forbid")

    photo: CandidateTrackPhoto
    observed_at: datetime | None = None
    timestamp_source: str = "unknown"
    timestamp_confidence: str = "unknown"
    observation_source: ObservationSource
    region_type: str
    region: dict[str, Any]
    observation_status: ObservationStatus
    confidence: float
    warnings: list[str] = Field(default_factory=list)


class CandidateTrack(BaseModel):
    """Tracked candidate across time."""

    model_config = ConfigDict(extra="forbid")

    track_id: str
    candidate_id: str
    tracking_status: TrackingStatus
    tracking_confidence: float
    observation_count: int
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None
    observations: list[CandidateTrackObservation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CandidateTrackManifest(BaseModel):
    """Technical temporal grouping of candidate observations."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "candidate-tracks-v1"
    created_at: datetime
    source_manifest: str
    source_validated_regions: str
    source_projections: str
    track_count: int
    tracks: list[CandidateTrack] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _ResolvedPhoto:
    record: PhotoImportRecord | None
    display_path: str | None
    display_filename: str | None


@dataclass(frozen=True)
class _ObservationDraft:
    photo_key: str
    photo: CandidateTrackPhoto
    observed_at: datetime | None
    timestamp_source: str
    timestamp_confidence: str
    observation_source: ObservationSource
    region_type: str
    region: dict[str, Any]
    observation_status: ObservationStatus
    confidence: float
    warnings: list[str]
    priority: int
    has_timestamp: bool
    low_quality: bool


def load_photo_import_manifest(manifest_path: str | Path) -> PhotoImportManifest:
    """Load the photo import manifest JSON from disk."""

    path = Path(manifest_path)
    return PhotoImportManifest.model_validate_json(path.read_text(encoding="utf-8"))


def load_validated_candidate_region_manifest(
    regions_path: str | Path,
) -> CandidateRegionValidationManifest:
    """Load the validated candidate-region manifest JSON from disk."""

    path = Path(regions_path)
    return CandidateRegionValidationManifest.model_validate_json(path.read_text(encoding="utf-8"))


def load_projected_candidate_region_manifest(
    projections_path: str | Path,
) -> ProjectedCandidateRegionManifest:
    """Load the projected candidate-region manifest JSON from disk."""

    path = Path(projections_path)
    return ProjectedCandidateRegionManifest.model_validate_json(path.read_text(encoding="utf-8"))


def track_candidate_regions(
    manifest_path: str | Path,
    validated_regions_path: str | Path,
    projections_path: str | Path,
) -> CandidateTrackManifest:
    """Load inputs and build temporal candidate tracks."""

    manifest = load_photo_import_manifest(manifest_path)
    validated_regions = load_validated_candidate_region_manifest(validated_regions_path)
    projections = load_projected_candidate_region_manifest(projections_path)
    return build_candidate_track_manifest(
        manifest,
        validated_regions,
        projections,
        source_manifest=manifest_path,
        source_validated_regions=validated_regions_path,
        source_projections=projections_path,
    )


def write_candidate_track_manifest(
    manifest_path: str | Path,
    validated_regions_path: str | Path,
    projections_path: str | Path,
    output_path: str | Path,
) -> CandidateTrackManifest:
    """Build candidate tracks and write the manifest to disk."""

    candidate_tracks = track_candidate_regions(
        manifest_path,
        validated_regions_path,
        projections_path,
    )
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(candidate_tracks.model_dump_json(indent=2), encoding="utf-8")
    return candidate_tracks


def build_candidate_track_manifest(
    manifest: PhotoImportManifest,
    validated_regions: CandidateRegionValidationManifest,
    projections: ProjectedCandidateRegionManifest,
    *,
    source_manifest: str | Path,
    source_validated_regions: str | Path,
    source_projections: str | Path,
) -> CandidateTrackManifest:
    """Group manual and projected observations into conservative candidate tracks."""

    record_lookup = _build_record_lookup(manifest)
    builders: dict[str, _TrackBuilder] = {}
    warnings: list[str] = []

    candidate_ids = {
        region.candidate_id for region in validated_regions.candidates
    } | {projection.candidate_id for projection in projections.projections}

    for candidate_id in sorted(candidate_ids):
        builders[candidate_id] = _TrackBuilder(candidate_id=candidate_id)

    for region in validated_regions.candidates:
        builder = builders.setdefault(region.candidate_id, _TrackBuilder(candidate_id=region.candidate_id))
        if region.status != "valid":
            builder.warnings.append(
                f"Manual candidate region {region.candidate_id} has status {region.status} and was not tracked."
            )
            continue
        observation, obs_warnings = _build_manual_observation(region, record_lookup)
        builder.add_observation(observation, obs_warnings)

    for projection in projections.projections:
        builder = builders.setdefault(
            projection.candidate_id,
            _TrackBuilder(candidate_id=projection.candidate_id),
        )
        if projection.projection_status not in {"projected", "weak_projection"}:
            builder.warnings.append(
                f"Projection for {projection.candidate_id} had status {projection.projection_status} and was not tracked."
            )
            builder.warnings.extend(_deduplicate(projection.warnings))
            continue
        observation, obs_warnings = _build_projection_observation(projection, record_lookup)
        builder.add_observation(observation, obs_warnings)

    tracks = [builder.finalize() for builder in builders.values()]
    tracks.sort(key=lambda track: track.candidate_id)

    if not tracks:
        warnings.append("No candidate observations were available for tracking.")

    track_status_counts = Counter(track.tracking_status for track in tracks)
    if track_status_counts.get("ambiguous", 0) > 0:
        warnings.append("At least one candidate track is ambiguous and should be reviewed cautiously.")

    return CandidateTrackManifest(
        created_at=datetime.now(timezone.utc),
        source_manifest=str(Path(source_manifest)),
        source_validated_regions=str(Path(source_validated_regions)),
        source_projections=str(Path(source_projections)),
        track_count=len(tracks),
        tracks=tracks,
        warnings=_deduplicate(warnings),
    )


def _build_manual_observation(
    region: CandidateRegionResult,
    record_lookup: dict[str, PhotoImportRecord],
) -> tuple[_ObservationDraft | None, list[str]]:
    resolved = _resolve_photo(region.photo.path, region.photo.filename, record_lookup)
    warnings = list(region.warnings)
    if resolved.record is None:
        warnings.append("Could not resolve the source photo in the import manifest.")
    confidence = 1.0
    if resolved.record is not None:
        confidence = _observation_confidence_from_record(resolved.record, confidence=confidence)
    photo = _photo_from_resolved(resolved, fallback_path=region.photo.path, fallback_filename=region.photo.filename)
    observed_at, timestamp_source, timestamp_confidence, timestamp_warnings, has_timestamp = _timestamp_from_record(
        resolved.record,
    )
    warnings.extend(timestamp_warnings)

    draft = _ObservationDraft(
        photo_key=_photo_key(resolved, fallback_path=region.photo.path, fallback_filename=region.photo.filename),
        photo=photo,
        observed_at=observed_at,
        timestamp_source=timestamp_source,
        timestamp_confidence=timestamp_confidence,
        observation_source="manual",
        region_type=region.region_type,
        region=_copy_mapping(region.coordinates),
        observation_status="manual_valid",
        confidence=round(confidence, 3),
        warnings=_deduplicate(warnings),
        priority=_MANUAL_OBSERVATION_PRIORITY,
        has_timestamp=has_timestamp,
        low_quality=_is_low_quality(resolved.record),
    )
    return draft, _deduplicate(warnings)


def _build_projection_observation(
    projection: ProjectedCandidateRegionResult,
    record_lookup: dict[str, PhotoImportRecord],
) -> tuple[_ObservationDraft | None, list[str]]:
    warnings = list(projection.warnings)
    if projection.target_photo is None:
        warnings.append("Projected observation has no target photo.")
        return None, _deduplicate(warnings)

    resolved = _resolve_photo(
        projection.target_photo.path,
        projection.target_photo.filename,
        record_lookup,
    )
    if resolved.record is None:
        warnings.append("Could not resolve the projected target photo in the import manifest.")
    observed_at, timestamp_source, timestamp_confidence, timestamp_warnings, has_timestamp = _timestamp_from_record(
        resolved.record,
    )
    warnings.extend(timestamp_warnings)

    region = _projected_region_payload(projection)
    if region is None:
        warnings.append("Projected geometry was unavailable.")
        return None, _deduplicate(warnings)

    confidence = max(0.0, min(1.0, float(projection.projection_confidence)))
    if _is_low_quality(resolved.record):
        warnings.append("Projected observation uses a low-quality photo.")
        confidence = round(max(0.0, confidence - 0.08), 3)

    draft = _ObservationDraft(
        photo_key=_photo_key(
            resolved,
            fallback_path=projection.target_photo.path,
            fallback_filename=projection.target_photo.filename,
        ),
        photo=_photo_from_resolved(
            resolved,
            fallback_path=projection.target_photo.path,
            fallback_filename=projection.target_photo.filename,
        ),
        observed_at=observed_at,
        timestamp_source=timestamp_source,
        timestamp_confidence=timestamp_confidence,
        observation_source="projection",
        region_type="polygon",
        region=region,
        observation_status="weak_projection" if projection.projection_status == "weak_projection" else "projected",
        confidence=round(confidence, 3),
        warnings=_deduplicate(warnings),
        priority=_PROJECTED_OBSERVATION_PRIORITY,
        has_timestamp=has_timestamp,
        low_quality=_is_low_quality(resolved.record),
    )
    return draft, _deduplicate(warnings)


def _projected_region_payload(projection: ProjectedCandidateRegionResult) -> dict[str, Any] | None:
    if projection.projected_region is None:
        return None
    return {
        "type": projection.projected_region.type,
        "points": [[float(x), float(y)] for x, y in projection.projected_region.points],
    }


@dataclass
class _TrackBuilder:
    candidate_id: str
    observations: list[_ObservationDraft] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duplicate_count: int = 0

    def add_observation(self, observation: _ObservationDraft | None, warnings: list[str]) -> None:
        self.warnings.extend(warnings)
        if observation is None:
            return
        self.observations.append(observation)

    def finalize(self) -> CandidateTrack:
        merged, duplicate_warning_count = _deduplicate_observations(self.observations)
        if duplicate_warning_count > 0:
            self.duplicate_count += duplicate_warning_count
            self.warnings.append(
                f"Resolved {duplicate_warning_count} duplicate observation(s) for candidate {self.candidate_id}."
            )

        observations = _sort_observations(merged)
        successful = [observation for observation in observations if observation.observation_status in {"manual_valid", "projected", "weak_projection"}]
        track_confidence = _compute_tracking_confidence(observations)
        tracking_status = _classify_track(
            observations,
            successful,
            track_confidence,
            self.duplicate_count > 0,
        )
        timestamps = [observation.observed_at for observation in observations if observation.observed_at is not None]
        first_seen_at = min(timestamps) if timestamps else None
        last_seen_at = max(timestamps) if timestamps else None

        if observations and len(successful) < len(observations):
            self.warnings.append(
                f"{len(observations) - len(successful)} observation(s) were not suitable for inclusion."
            )
        if not observations:
            self.warnings.append(f"No usable observations were available for candidate {self.candidate_id}.")

        return CandidateTrack(
            track_id=_track_id(self.candidate_id),
            candidate_id=self.candidate_id,
            tracking_status=tracking_status,
            tracking_confidence=track_confidence,
            observation_count=len(observations),
            first_seen_at=first_seen_at,
            last_seen_at=last_seen_at,
            observations=[_draft_to_observation(observation) for observation in observations],
            warnings=_deduplicate(self.warnings),
        )


def _deduplicate_observations(
    observations: list[_ObservationDraft],
) -> tuple[list[_ObservationDraft], int]:
    selected: dict[str, _ObservationDraft] = {}
    duplicate_count = 0
    for observation in observations:
        existing = selected.get(observation.photo_key)
        if existing is None:
            selected[observation.photo_key] = observation
            continue
        duplicate_count += 1
        if _observation_rank(observation) > _observation_rank(existing):
            selected[observation.photo_key] = observation
        elif _observation_rank(observation) == _observation_rank(existing) and observation.confidence > existing.confidence:
            selected[observation.photo_key] = observation
    return list(selected.values()), duplicate_count


def _observation_rank(observation: _ObservationDraft) -> tuple[int, float]:
    return observation.priority, observation.confidence


def _sort_observations(observations: list[_ObservationDraft]) -> list[_ObservationDraft]:
    return sorted(
        observations,
        key=lambda observation: (
            observation.observed_at is None,
            observation.observed_at or datetime.max,
            -observation.priority,
            -observation.confidence,
            observation.observation_source,
        ),
    )


def _draft_to_observation(observation: _ObservationDraft) -> CandidateTrackObservation:
    return CandidateTrackObservation(
        photo=observation.photo,
        observed_at=observation.observed_at,
        timestamp_source=observation.timestamp_source,
        timestamp_confidence=observation.timestamp_confidence,
        observation_source=observation.observation_source,
        region_type=observation.region_type,
        region=_copy_mapping(observation.region),
        observation_status=observation.observation_status,
        confidence=observation.confidence,
        warnings=_deduplicate(observation.warnings),
    )


def _compute_tracking_confidence(observations: list[_ObservationDraft]) -> float:
    if not observations:
        return 0.0

    successful = [observation for observation in observations if observation.observation_status in {"manual_valid", "projected", "weak_projection"}]
    if not successful:
        return 0.0

    average_observation_confidence = sum(observation.confidence for observation in successful) / len(successful)
    observation_count_factor = min(1.0, len(successful) / 2.0)
    timestamp_fraction = sum(1 for observation in successful if observation.has_timestamp) / len(successful)
    quality_fraction = 1.0 - min(0.3, sum(1 for observation in successful if observation.low_quality) * 0.1)
    weak_penalty = 0.12 if any(observation.observation_status == "weak_projection" for observation in successful) else 0.0
    missing_timestamp_penalty = 0.08 if timestamp_fraction < 1.0 else 0.0
    score = (
        0.45 * average_observation_confidence
        + 0.25 * observation_count_factor
        + 0.2 * timestamp_fraction
        + 0.1 * quality_fraction
        - weak_penalty
        - missing_timestamp_penalty
    )
    return round(max(0.0, min(1.0, score)), 3)


def _classify_track(
    observations: list[_ObservationDraft],
    successful: list[_ObservationDraft],
    confidence: float,
    has_duplicates: bool,
) -> TrackingStatus:
    if not observations or not successful:
        return "untracked"
    if has_duplicates:
        return "ambiguous"
    if len(successful) >= 2 and confidence >= 0.75:
        return "tracked"
    return "partial"


def _observation_confidence_from_record(
    record: PhotoImportRecord,
    *,
    confidence: float,
) -> float:
    result = confidence
    if record.quality is not None and record.quality.status == "low_quality":
        result -= 0.08
    if record.taken_at is None:
        result -= 0.1
    return max(0.0, min(1.0, result))


def _timestamp_from_record(
    record: PhotoImportRecord | None,
) -> tuple[datetime | None, str, str, list[str], bool]:
    warnings: list[str] = []
    if record is None:
        warnings.append("Photo reference was not resolved in the import manifest.")
        return None, "unknown", "unknown", warnings, False

    if record.taken_at is None:
        warnings.append("Timestamp is unavailable for this observation.")
        return None, record.timestamp_source, record.timestamp_confidence, warnings, False

    return record.taken_at, record.timestamp_source, record.timestamp_confidence, warnings, True


def _is_low_quality(record: PhotoImportRecord | None) -> bool:
    return bool(record is not None and record.quality is not None and record.quality.status == "low_quality")


def _photo_from_resolved(
    resolved: _ResolvedPhoto,
    *,
    fallback_path: str | None,
    fallback_filename: str | None,
) -> CandidateTrackPhoto:
    if resolved.record is not None:
        return CandidateTrackPhoto(
            path=resolved.record.original_path,
            filename=resolved.record.original_filename,
        )
    return CandidateTrackPhoto(
        path=resolved.display_path or fallback_path,
        filename=resolved.display_filename or fallback_filename,
    )


def _photo_key(
    resolved: _ResolvedPhoto,
    *,
    fallback_path: str | None,
    fallback_filename: str | None,
) -> str:
    if resolved.record is not None:
        return resolved.record.original_path
    if resolved.display_path:
        return str(Path(resolved.display_path).expanduser())
    if fallback_path:
        return fallback_path
    if fallback_filename:
        return fallback_filename
    return "unknown-photo"


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
                display_path=path or record.original_path,
                display_filename=filename or record.original_filename,
            )

    return _ResolvedPhoto(record=None, display_path=path, display_filename=filename)


def _build_record_lookup(manifest: PhotoImportManifest) -> dict[str, PhotoImportRecord]:
    lookup: dict[str, PhotoImportRecord] = {}
    for record in manifest.photo_records:
        lookup.setdefault(record.original_path, record)
        lookup.setdefault(str(Path(record.original_path).expanduser().resolve()), record)
        lookup.setdefault(record.original_filename, record)
        lookup.setdefault(Path(record.original_path).name, record)
    return lookup


def _track_id(candidate_id: str) -> str:
    return f"track-{candidate_id}"


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
