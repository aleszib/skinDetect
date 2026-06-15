"""Lightweight overlap candidate ranking from local photo import manifests."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from skintrack.io.photos import PhotoImportManifest, PhotoImportRecord

CandidateStatus = Literal["candidate", "weak_candidate", "not_candidate"]

_MAX_TIMESTAMP_WINDOW_SECONDS = 14 * 24 * 60 * 60
_CANDIDATE_THRESHOLD = 0.7
_WEAK_CANDIDATE_THRESHOLD = 0.4


class OverlapCandidatePhotoRef(BaseModel):
    """Reference to one imported photo in a candidate pair."""

    model_config = ConfigDict(extra="forbid")

    path: str
    filename: str


class OverlapPairMetadata(BaseModel):
    """Heuristic similarity metadata for one candidate pair."""

    model_config = ConfigDict(extra="forbid")

    timestamp_delta_seconds: int | None = None
    dimension_similarity: float | None = None
    brightness_similarity: float | None = None
    filename_similarity: float | None = None
    quality_penalty: float = 0.0


class OverlapCandidatePair(BaseModel):
    """Ranked candidate pair for later geometric registration."""

    model_config = ConfigDict(extra="forbid")

    photo_a: OverlapCandidatePhotoRef
    photo_b: OverlapCandidatePhotoRef
    score: float
    status: CandidateStatus
    reasons: list[str] = Field(default_factory=list)
    penalties: list[str] = Field(default_factory=list)
    metadata: OverlapPairMetadata


class OverlapCandidateManifest(BaseModel):
    """JSON manifest of ranked overlap candidates."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "overlap-candidates-v1"
    created_at: datetime
    source_manifest: str
    candidate_count: int
    pairs: list[OverlapCandidatePair] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def load_overlap_candidate_manifest(manifest_path: str | Path) -> PhotoImportManifest:
    """Load a photo import manifest from JSON."""

    path = Path(manifest_path)
    return PhotoImportManifest.model_validate_json(path.read_text(encoding="utf-8"))


def build_overlap_candidate_manifest(
    manifest: PhotoImportManifest,
    *,
    source_manifest: str | Path,
    include_nonimported: bool = False,
) -> OverlapCandidateManifest:
    """Rank likely overlapping photo pairs from a photo import manifest."""

    records = _select_records(manifest.photo_records, include_nonimported=include_nonimported)
    pairs: list[OverlapCandidatePair] = []
    warnings: list[str] = []

    for record_a, record_b in combinations(records, 2):
        pair = _score_pair(record_a, record_b)
        pairs.append(pair)

    pairs.sort(key=lambda pair: (-pair.score, pair.photo_a.filename, pair.photo_b.filename))
    if not pairs:
        warnings.append("Not enough imported records to rank overlap candidates.")

    return OverlapCandidateManifest(
        created_at=datetime.now(timezone.utc),
        source_manifest=str(Path(source_manifest)),
        candidate_count=len(pairs),
        pairs=pairs,
        warnings=warnings,
    )


def rank_overlap_candidates(
    manifest_path: str | Path,
    *,
    include_nonimported: bool = False,
) -> OverlapCandidateManifest:
    """Load a photo import manifest and rank likely overlap candidates."""

    manifest = load_overlap_candidate_manifest(manifest_path)
    return build_overlap_candidate_manifest(
        manifest,
        source_manifest=manifest_path,
        include_nonimported=include_nonimported,
    )


def write_overlap_candidate_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    include_nonimported: bool = False,
) -> OverlapCandidateManifest:
    """Rank candidates and write the result to disk."""

    candidate_manifest = rank_overlap_candidates(
        manifest_path,
        include_nonimported=include_nonimported,
    )
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(candidate_manifest.model_dump_json(indent=2), encoding="utf-8")
    return candidate_manifest


def _select_records(
    records: Iterable[PhotoImportRecord],
    *,
    include_nonimported: bool,
) -> list[PhotoImportRecord]:
    selected: list[PhotoImportRecord] = []
    for record in records:
        if record.import_status == "imported":
            selected.append(record)
            continue
        if include_nonimported:
            selected.append(record)
    return selected


def _score_pair(record_a: PhotoImportRecord, record_b: PhotoImportRecord) -> OverlapCandidatePair:
    reasons: list[str] = []
    penalties: list[str] = []
    metadata = OverlapPairMetadata()
    score = 0.0

    quality_penalty = 0.0
    for record in (record_a, record_b):
        record_score, record_reasons, record_penalties, record_penalty = _record_quality_adjustment(
            record,
        )
        score += record_score
        reasons.extend(record_reasons)
        penalties.extend(record_penalties)
        quality_penalty += record_penalty

    metadata.quality_penalty = quality_penalty

    if _is_readable(record_a) and _is_readable(record_b):
        score += 0.15
        reasons.append("both images are readable")

    dimension_similarity = _dimension_similarity(record_a, record_b)
    metadata.dimension_similarity = dimension_similarity
    if dimension_similarity is not None:
        score += 0.2 * dimension_similarity
        if dimension_similarity >= 0.85:
            reasons.append("similar dimensions")
        elif dimension_similarity < 0.6:
            penalties.append("dimension mismatch reduces overlap likelihood")

    timestamp_delta_seconds = _timestamp_delta_seconds(record_a, record_b)
    metadata.timestamp_delta_seconds = timestamp_delta_seconds
    if timestamp_delta_seconds is not None:
        timestamp_similarity = max(
            0.0,
            1.0 - (timestamp_delta_seconds / _MAX_TIMESTAMP_WINDOW_SECONDS),
        )
        score += 0.25 * timestamp_similarity
        if timestamp_delta_seconds <= 24 * 60 * 60:
            reasons.append("timestamps are close")
        elif timestamp_delta_seconds >= 7 * 24 * 60 * 60:
            penalties.append("timestamps are far apart")

    brightness_similarity = _brightness_similarity(record_a, record_b)
    metadata.brightness_similarity = brightness_similarity
    if brightness_similarity is not None:
        score += 0.1 * brightness_similarity
        if brightness_similarity >= 0.8:
            reasons.append("similar brightness profile")

    filename_similarity = _filename_similarity(record_a, record_b)
    metadata.filename_similarity = filename_similarity
    if filename_similarity is not None:
        score += 0.05 * filename_similarity
        if filename_similarity >= 0.7:
            reasons.append("similar filename pattern")

    score = max(0.0, min(1.0, score - quality_penalty))
    status = _classify_score(score)

    if not reasons:
        reasons.append("limited evidence for likely overlap")
    if not penalties and status == "not_candidate":
        penalties.append("not enough evidence for reliable matching")

    return OverlapCandidatePair(
        photo_a=_photo_ref(record_a),
        photo_b=_photo_ref(record_b),
        score=score,
        status=status,
        reasons=reasons,
        penalties=penalties,
        metadata=metadata,
    )


def _record_quality_adjustment(
    record: PhotoImportRecord,
) -> tuple[float, list[str], list[str], float]:
    score = 0.0
    reasons: list[str] = []
    penalties: list[str] = []
    penalty = 0.0

    if record.import_status == "imported":
        score += 0.1
        reasons.append(f"{record.original_filename} is imported")
    elif record.import_status == "unreadable":
        penalty += 0.7
        penalties.append(f"{record.original_filename} is unreadable")
    elif record.import_status == "unsupported":
        penalty += 0.7
        penalties.append(f"{record.original_filename} is unsupported")
    else:
        penalty += 0.3
        penalties.append(f"{record.original_filename} is not imported")

    quality = record.quality
    if quality is None:
        penalty += 0.2
        penalties.append(f"{record.original_filename} has no quality assessment")
        return score, reasons, penalties, penalty

    if quality.readable:
        score += 0.05
        reasons.append(f"{record.original_filename} is readable")
    else:
        penalty += 0.4
        penalties.append(f"{record.original_filename} is unreadable")

    if quality.status == "usable":
        score += 0.1
        reasons.append(f"{record.original_filename} has usable quality")
    elif quality.status == "low_quality":
        penalty += 0.15
        penalties.append(f"{record.original_filename} is low quality")
    elif quality.status == "unreadable":
        penalty += 0.7
        penalties.append(f"{record.original_filename} is unreadable")

    return score, reasons, penalties, penalty


def _photo_ref(record: PhotoImportRecord) -> OverlapCandidatePhotoRef:
    return OverlapCandidatePhotoRef(path=record.original_path, filename=record.original_filename)


def _is_readable(record: PhotoImportRecord) -> bool:
    return record.import_status == "imported" and record.quality is not None and record.quality.readable


def _dimension_similarity(record_a: PhotoImportRecord, record_b: PhotoImportRecord) -> float | None:
    if record_a.width is None or record_a.height is None:
        return None
    if record_b.width is None or record_b.height is None:
        return None
    if record_a.width <= 0 or record_a.height <= 0 or record_b.width <= 0 or record_b.height <= 0:
        return None

    width_similarity = min(record_a.width, record_b.width) / max(record_a.width, record_b.width)
    height_similarity = min(record_a.height, record_b.height) / max(record_a.height, record_b.height)
    return (width_similarity + height_similarity) / 2


def _timestamp_delta_seconds(record_a: PhotoImportRecord, record_b: PhotoImportRecord) -> int | None:
    if record_a.taken_at is None or record_b.taken_at is None:
        return None
    delta = abs(record_a.taken_at - record_b.taken_at)
    return int(delta.total_seconds())


def _brightness_similarity(record_a: PhotoImportRecord, record_b: PhotoImportRecord) -> float | None:
    quality_a = record_a.quality
    quality_b = record_b.quality
    if quality_a is None or quality_b is None:
        return None
    if quality_a.brightness_score is None or quality_b.brightness_score is None:
        return None
    return max(0.0, 1.0 - abs(quality_a.brightness_score - quality_b.brightness_score))


def _filename_similarity(record_a: PhotoImportRecord, record_b: PhotoImportRecord) -> float | None:
    stem_a = Path(record_a.original_filename).stem
    stem_b = Path(record_b.original_filename).stem
    if not stem_a or not stem_b:
        return None
    return SequenceMatcher(None, stem_a.lower(), stem_b.lower()).ratio()


def _classify_score(score: float) -> CandidateStatus:
    if score >= _CANDIDATE_THRESHOLD:
        return "candidate"
    if score >= _WEAK_CANDIDATE_THRESHOLD:
        return "weak_candidate"
    return "not_candidate"
