"""Local geometric registration for overlap candidate pairs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from skintrack.io.photos import PhotoImportManifest, PhotoImportRecord
from skintrack.overlap.candidates import OverlapCandidateManifest, OverlapCandidatePair

RegistrationStatus = Literal["registered", "weak_match", "failed", "skipped"]
TransformKind = Literal["affine"]
OverlapKind = Literal["polygon"]

_ORB_FEATURE_LIMIT: Final[int] = 1000
_RATIO_TEST_THRESHOLD: Final[float] = 0.75
_MIN_GOOD_MATCHES: Final[int] = 4
_MIN_WEAK_MATCHES: Final[int] = 6
_MIN_REGISTERED_INLIERS: Final[int] = 8
_MIN_WEAK_INLIERS: Final[int] = 4
_MIN_REGISTERED_INLIER_RATIO: Final[float] = 0.45
_MIN_WEAK_INLIER_RATIO: Final[float] = 0.25
_RANSAC_REPROJECTION_THRESHOLD: Final[float] = 4.0
_MATCHES_TO_DRAW: Final[int] = 30


class RegistrationPhotoRef(BaseModel):
    """Reference to one imported photo in a registration result."""

    model_config = ConfigDict(extra="forbid")

    path: str
    filename: str


class RegistrationTransform(BaseModel):
    """Estimated transform from candidate photo B into photo A space."""

    model_config = ConfigDict(extra="forbid")

    type: TransformKind
    matrix: list[list[float]]


class RegistrationOverlap(BaseModel):
    """Estimated technical overlap polygon and fraction."""

    model_config = ConfigDict(extra="forbid")

    type: OverlapKind
    points: list[list[float]]
    estimated_overlap_fraction: float | None = None


class RegistrationResult(BaseModel):
    """One registration attempt for a candidate pair."""

    model_config = ConfigDict(extra="forbid")

    photo_a: RegistrationPhotoRef
    photo_b: RegistrationPhotoRef
    candidate_score: float | None = None
    registration_status: RegistrationStatus
    registration_confidence: float
    method: str
    match_count: int
    inlier_count: int
    inlier_ratio: float
    transform: RegistrationTransform | None = None
    overlap: RegistrationOverlap | None = None
    debug_visualization: str | None = None
    warnings: list[str] = Field(default_factory=list)


class RegistrationManifest(BaseModel):
    """JSON manifest of geometric registration attempts."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "registrations-v1"
    created_at: datetime
    source_manifest: str
    source_candidates: str
    registration_count: int
    registrations: list[RegistrationResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _LoadedImage:
    rgb: np.ndarray
    bgr: np.ndarray
    gray: np.ndarray
    width: int
    height: int


def load_photo_import_manifest(manifest_path: str | Path) -> PhotoImportManifest:
    """Load the photo import manifest JSON from disk."""

    path = Path(manifest_path)
    return PhotoImportManifest.model_validate_json(path.read_text(encoding="utf-8"))


def load_overlap_candidate_manifest(candidate_path: str | Path) -> OverlapCandidateManifest:
    """Load the overlap candidate manifest JSON from disk."""

    path = Path(candidate_path)
    return OverlapCandidateManifest.model_validate_json(path.read_text(encoding="utf-8"))


def build_registration_manifest(
    manifest: PhotoImportManifest,
    candidates: OverlapCandidateManifest,
    *,
    source_manifest: str | Path,
    source_candidates: str | Path,
    debug_dir: str | Path | None = None,
) -> RegistrationManifest:
    """Estimate registrations for ranked overlap candidate pairs."""

    record_lookup = _build_record_lookup(manifest)
    debug_root = _prepare_debug_dir(debug_dir)
    registrations: list[RegistrationResult] = []
    warnings: list[str] = []

    for pair in candidates.pairs:
        result = _register_pair(
            pair,
            record_lookup,
            debug_root=debug_root,
        )
        registrations.append(result)
        warnings.extend(result.warnings)

    if not registrations:
        warnings.append("No candidate pairs were available for registration.")

    status_counts = _status_counts(registrations)
    if status_counts["registered"] == 0:
        warnings.append("No candidate pairs reached registered status.")

    return RegistrationManifest(
        created_at=datetime.now(timezone.utc),
        source_manifest=str(Path(source_manifest)),
        source_candidates=str(Path(source_candidates)),
        registration_count=len(registrations),
        registrations=registrations,
        warnings=_deduplicate(warnings),
    )


def register_candidate_pairs(
    manifest_path: str | Path,
    candidate_path: str | Path,
    *,
    debug_dir: str | Path | None = None,
) -> RegistrationManifest:
    """Load manifest JSON and overlap candidates, then estimate registrations."""

    manifest = load_photo_import_manifest(manifest_path)
    candidates = load_overlap_candidate_manifest(candidate_path)
    return build_registration_manifest(
        manifest,
        candidates,
        source_manifest=manifest_path,
        source_candidates=candidate_path,
        debug_dir=debug_dir,
    )


def write_registration_manifest(
    manifest_path: str | Path,
    candidate_path: str | Path,
    output_path: str | Path,
    *,
    debug_dir: str | Path | None = None,
) -> RegistrationManifest:
    """Write a registration manifest to disk."""

    registration_manifest = register_candidate_pairs(
        manifest_path,
        candidate_path,
        debug_dir=debug_dir,
    )
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(registration_manifest.model_dump_json(indent=2), encoding="utf-8")
    return registration_manifest


def _register_pair(
    pair: OverlapCandidatePair,
    record_lookup: dict[str, PhotoImportRecord],
    *,
    debug_root: Path | None,
) -> RegistrationResult:
    warnings: list[str] = []
    record_a = _resolve_record(pair.photo_a, record_lookup)
    record_b = _resolve_record(pair.photo_b, record_lookup)

    if record_a is None or record_b is None:
        warnings.append("Could not resolve both photos from the manifest.")
        debug_path = _write_debug_visualization(
            pair,
            None,
            None,
            status="skipped",
            warnings=warnings,
            debug_root=debug_root,
        )
        return _terminal_result(
            pair,
            status="skipped",
            method="not_run",
            warnings=warnings,
            debug_visualization=debug_path,
        )

    if record_a.import_status != "imported" or record_b.import_status != "imported":
        warnings.append("One or both photos were not imported, so registration was skipped.")
        debug_path = _write_debug_visualization(
            pair,
            None,
            None,
            status="skipped",
            warnings=warnings,
            debug_root=debug_root,
        )
        return _terminal_result(
            pair,
            status="skipped",
            method="not_run",
            warnings=warnings,
            debug_visualization=debug_path,
        )

    if not _is_readable(record_a) or not _is_readable(record_b):
        warnings.append("One or both photos were unreadable, so registration was skipped.")
        debug_path = _write_debug_visualization(
            pair,
            None,
            None,
            status="skipped",
            warnings=warnings,
            debug_root=debug_root,
        )
        return _terminal_result(
            pair,
            status="skipped",
            method="not_run",
            warnings=warnings,
            debug_visualization=debug_path,
        )

    loaded_a, load_warnings_a = _load_image(record_a.original_path)
    loaded_b, load_warnings_b = _load_image(record_b.original_path)
    warnings.extend(load_warnings_a)
    warnings.extend(load_warnings_b)

    if loaded_a is None or loaded_b is None:
        warnings.append("One or both source image files could not be loaded.")
        debug_path = _write_debug_visualization(
            pair,
            loaded_a,
            loaded_b,
            status="failed",
            warnings=warnings,
            debug_root=debug_root,
        )
        return _terminal_result(
            pair,
            status="failed",
            method="orb_affine_ransac",
            warnings=warnings,
            debug_visualization=debug_path,
        )

    features_a = _detect_orb_features(loaded_a.gray)
    features_b = _detect_orb_features(loaded_b.gray)

    if features_a.keypoints is None or features_b.keypoints is None:
        warnings.append("Feature detection failed for one or both images.")
        debug_path = _write_debug_visualization(
            pair,
            loaded_a,
            loaded_b,
            status="failed",
            warnings=warnings,
            debug_root=debug_root,
        )
        return _terminal_result(
            pair,
            status="failed",
            method="orb_affine_ransac",
            warnings=warnings,
            debug_visualization=debug_path,
        )

    matches = _match_features(features_a.descriptors, features_b.descriptors)

    if len(matches) < _MIN_GOOD_MATCHES:
        warnings.append("Not enough feature matches for registration.")
        status = "weak_match" if len(matches) >= 2 else "failed"
        debug_path = _write_debug_visualization(
            pair,
            loaded_a,
            loaded_b,
            keypoints_a=features_a.keypoints,
            keypoints_b=features_b.keypoints,
            matches=matches,
            status=status,
            warnings=warnings,
            debug_root=debug_root,
        )
        return _terminal_result(
            pair,
            status=status,
            method="orb_affine_ransac",
            warnings=warnings,
            debug_visualization=debug_path,
            match_count=len(matches),
        )

    affine, inlier_mask = _estimate_affine(features_a.keypoints, features_b.keypoints, matches)
    if affine is None or inlier_mask is None:
        warnings.append("RANSAC could not estimate a stable transform.")
        debug_path = _write_debug_visualization(
            pair,
            loaded_a,
            loaded_b,
            keypoints_a=features_a.keypoints,
            keypoints_b=features_b.keypoints,
            matches=matches,
            status="failed",
            warnings=warnings,
            debug_root=debug_root,
        )
        return _terminal_result(
            pair,
            status="failed",
            method="orb_affine_ransac",
            warnings=warnings,
            debug_visualization=debug_path,
            match_count=len(matches),
        )

    inlier_count = int(np.count_nonzero(inlier_mask))
    inlier_ratio = inlier_count / len(matches)
    transform = _affine_to_transform(affine)
    overlap = _estimate_overlap(affine, loaded_a.width, loaded_a.height, loaded_b.width, loaded_b.height)
    confidence = _compute_registration_confidence(
        pair.score or 0.0,
        len(matches),
        inlier_count,
        inlier_ratio,
        overlap.estimated_overlap_fraction if overlap else None,
    )

    status = _classify_registration(
        confidence,
        len(matches),
        inlier_count,
        inlier_ratio,
        overlap.estimated_overlap_fraction if overlap else None,
    )

    if overlap is None:
        warnings.append("Could not estimate a reliable overlap polygon.")

    if status == "failed":
        warnings.append("Registration did not meet the conservative thresholds.")
    elif status == "weak_match":
        warnings.append("Registration is a weak match and should be treated cautiously.")

    debug_path = _write_debug_visualization(
        pair,
        loaded_a,
        loaded_b,
        keypoints_a=features_a.keypoints,
        keypoints_b=features_b.keypoints,
        matches=matches,
        inlier_mask=inlier_mask,
        affine=affine,
        status=status,
        warnings=warnings,
        debug_root=debug_root,
    )

    return RegistrationResult(
        photo_a=_photo_ref(record_a),
        photo_b=_photo_ref(record_b),
        candidate_score=pair.score,
        registration_status=status,
        registration_confidence=confidence,
        method="orb_affine_ransac",
        match_count=len(matches),
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        transform=transform if status != "failed" else None,
        overlap=overlap if status != "failed" else None,
        debug_visualization=debug_path,
        warnings=_deduplicate(warnings),
    )


@dataclass(frozen=True)
class _ORBFeatures:
    keypoints: list[cv2.KeyPoint] | None
    descriptors: np.ndarray | None


def _detect_orb_features(gray: np.ndarray) -> _ORBFeatures:
    orb = cv2.ORB_create(nfeatures=_ORB_FEATURE_LIMIT, fastThreshold=5)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if keypoints is None or descriptors is None or len(keypoints) == 0:
        return _ORBFeatures(keypoints=None, descriptors=None)
    return _ORBFeatures(keypoints=list(keypoints), descriptors=descriptors)


def _match_features(
    descriptors_a: np.ndarray | None,
    descriptors_b: np.ndarray | None,
) -> list[cv2.DMatch]:
    if descriptors_a is None or descriptors_b is None:
        return []
    if len(descriptors_a) == 0 or len(descriptors_b) == 0:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    good_matches: list[cv2.DMatch] = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < _RATIO_TEST_THRESHOLD * second.distance:
            good_matches.append(first)

    good_matches.sort(key=lambda match: match.distance)
    return good_matches


def _estimate_affine(
    keypoints_a: list[cv2.KeyPoint],
    keypoints_b: list[cv2.KeyPoint],
    matches: Iterable[cv2.DMatch],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    match_list = list(matches)
    if len(match_list) < _MIN_GOOD_MATCHES:
        return None, None

    points_a = np.float32([keypoints_a[match.queryIdx].pt for match in match_list]).reshape(-1, 1, 2)
    points_b = np.float32([keypoints_b[match.trainIdx].pt for match in match_list]).reshape(-1, 1, 2)
    affine, inlier_mask = cv2.estimateAffinePartial2D(
        points_b,
        points_a,
        method=cv2.RANSAC,
        ransacReprojThreshold=_RANSAC_REPROJECTION_THRESHOLD,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    return affine, inlier_mask


def _classify_registration(
    confidence: float,
    match_count: int,
    inlier_count: int,
    inlier_ratio: float,
    overlap_fraction: float | None,
) -> RegistrationStatus:
    if confidence <= 0.0 or match_count < _MIN_GOOD_MATCHES:
        return "failed"
    if overlap_fraction is not None and overlap_fraction <= 0.0:
        return "failed"
    if (
        match_count >= _MIN_WEAK_MATCHES
        and inlier_count >= _MIN_REGISTERED_INLIERS
        and inlier_ratio >= _MIN_REGISTERED_INLIER_RATIO
    ):
        return "registered"
    if (
        match_count >= _MIN_WEAK_MATCHES
        and inlier_count >= _MIN_WEAK_INLIERS
        and inlier_ratio >= _MIN_WEAK_INLIER_RATIO
    ):
        return "weak_match"
    return "failed"


def _compute_registration_confidence(
    candidate_score: float,
    match_count: int,
    inlier_count: int,
    inlier_ratio: float,
    overlap_fraction: float | None,
) -> float:
    overlap_bonus = 0.0 if overlap_fraction is None else min(0.2, overlap_fraction * 0.2)
    score = (
        0.2 * max(0.0, min(1.0, candidate_score))
        + 0.3 * min(1.0, match_count / 40.0)
        + 0.4 * max(0.0, min(1.0, inlier_ratio))
        + 0.1 * min(1.0, inlier_count / 20.0)
        + overlap_bonus
    )
    return round(max(0.0, min(1.0, score)), 3)


def _estimate_overlap(
    affine: np.ndarray,
    width_a: int,
    height_a: int,
    width_b: int,
    height_b: int,
) -> RegistrationOverlap | None:
    corners_b = np.array(
        [
            [[0.0, 0.0]],
            [[float(width_b - 1), 0.0]],
            [[float(width_b - 1), float(height_b - 1)]],
            [[0.0, float(height_b - 1)]],
        ],
        dtype=np.float32,
    )
    transformed = cv2.transform(corners_b, affine).reshape(-1, 2).astype(np.float32)
    if transformed.shape[0] < 3:
        return None

    rect_a = np.array(
        [
            [0.0, 0.0],
            [float(width_a - 1), 0.0],
            [float(width_a - 1), float(height_a - 1)],
            [0.0, float(height_a - 1)],
        ],
        dtype=np.float32,
    )
    try:
        intersection_area, intersection_polygon = cv2.intersectConvexConvex(rect_a, transformed)
    except cv2.error:
        return None

    if intersection_polygon is None or intersection_polygon.size == 0:
        return None

    transformed_area = float(cv2.contourArea(transformed))
    if transformed_area <= 0:
        return None

    points = [[float(x), float(y)] for x, y in intersection_polygon.reshape(-1, 2)]
    return RegistrationOverlap(
        type="polygon",
        points=points,
        estimated_overlap_fraction=round(max(0.0, min(1.0, float(intersection_area) / transformed_area)), 3),
    )


def _load_image(image_path: str | Path) -> tuple[_LoadedImage | None, list[str]]:
    path = Path(image_path)
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.load()
            rgb = np.asarray(image)
    except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as exc:
        return None, [f"Image could not be loaded: {exc.__class__.__name__}."]

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    return _LoadedImage(rgb=rgb, bgr=bgr, gray=gray, width=width, height=height), []


def _photo_ref(record: PhotoImportRecord) -> RegistrationPhotoRef:
    return RegistrationPhotoRef(path=record.original_path, filename=record.original_filename)


def _is_readable(record: PhotoImportRecord) -> bool:
    return record.import_status == "imported" and record.quality is not None and record.quality.readable


def _build_record_lookup(manifest: PhotoImportManifest) -> dict[str, PhotoImportRecord]:
    lookup: dict[str, PhotoImportRecord] = {}
    for record in manifest.photo_records:
        lookup[record.original_path] = record
        lookup[str(Path(record.original_path).expanduser().resolve())] = record
        lookup.setdefault(record.original_filename, record)
    return lookup


def _resolve_record(
    ref: RegistrationPhotoRef | object,
    record_lookup: dict[str, PhotoImportRecord],
) -> PhotoImportRecord | None:
    path = getattr(ref, "path", None)
    filename = getattr(ref, "filename", None)
    if isinstance(path, str) and path in record_lookup:
        return record_lookup[path]

    if isinstance(path, str):
        resolved = str(Path(path).expanduser().resolve())
        if resolved in record_lookup:
            return record_lookup[resolved]

    if isinstance(filename, str) and filename in record_lookup:
        return record_lookup[filename]

    return None


def _prepare_debug_dir(debug_dir: str | Path | None) -> Path | None:
    if debug_dir is None:
        return None
    path = Path(debug_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_debug_visualization(
    pair: OverlapCandidatePair,
    loaded_a: _LoadedImage | None,
    loaded_b: _LoadedImage | None,
    *,
    keypoints_a: list[cv2.KeyPoint] | None = None,
    keypoints_b: list[cv2.KeyPoint] | None = None,
    matches: list[cv2.DMatch] | None = None,
    inlier_mask: np.ndarray | None = None,
    affine: np.ndarray | None = None,
    status: str,
    warnings: list[str],
    debug_root: Path | None,
) -> str | None:
    if debug_root is None:
        return None

    output_path = debug_root / f"{_sanitize_name(pair.photo_a.filename)}__{_sanitize_name(pair.photo_b.filename)}_registration.jpg"

    if loaded_a is None or loaded_b is None or keypoints_a is None or keypoints_b is None or not matches:
        canvas = _blank_debug_canvas(
            [
                f"registration {status}",
                pair.photo_a.filename,
                pair.photo_b.filename,
                *warnings[:4],
            ]
        )
        canvas.save(output_path, quality=90)
        return str(output_path)

    matches_to_draw = matches[:_MATCHES_TO_DRAW]
    if inlier_mask is not None and len(inlier_mask) == len(matches):
        inlier_matches = [match for match, flag in zip(matches, inlier_mask.ravel().tolist()) if flag]
        if inlier_matches:
            matches_to_draw = inlier_matches[:_MATCHES_TO_DRAW]

    canvas = cv2.drawMatches(
        loaded_a.bgr,
        keypoints_a,
        loaded_b.bgr,
        keypoints_b,
        matches_to_draw,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(120, 120, 120),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    if affine is not None:
        transformed = np.array(
            [
                [0.0, 0.0],
                [float(loaded_b.width - 1), 0.0],
                [float(loaded_b.width - 1), float(loaded_b.height - 1)],
                [0.0, float(loaded_b.height - 1)],
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        warped = cv2.transform(transformed, affine).reshape(-1, 2).astype(np.int32)
        cv2.polylines(canvas, [warped], isClosed=True, color=(0, 255, 255), thickness=2)

    overlay_lines = [
        f"registration {status}",
        f"candidate score: {pair.score:.2f}",
        f"matches: {len(matches)}",
        f"warnings: {len(warnings)}",
    ]
    y = 24
    for line in overlay_lines:
        cv2.putText(
            canvas,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        y += 28

    cv2.imwrite(str(output_path), canvas)
    return str(output_path)


def _blank_debug_canvas(lines: list[str]) -> Image.Image:
    image = Image.new("RGB", (1200, 360), color="white")
    draw = ImageDraw.Draw(image)
    y = 20
    for line in lines:
        draw.text((20, y), line, fill="black")
        y += 28
    return image


def _terminal_result(
    pair: OverlapCandidatePair,
    *,
    status: RegistrationStatus,
    method: str,
    warnings: list[str],
    debug_visualization: str | None,
    match_count: int = 0,
) -> RegistrationResult:
    return RegistrationResult(
        photo_a=RegistrationPhotoRef(path=pair.photo_a.path, filename=pair.photo_a.filename),
        photo_b=RegistrationPhotoRef(path=pair.photo_b.path, filename=pair.photo_b.filename),
        candidate_score=pair.score,
        registration_status=status,
        registration_confidence=0.0,
        method=method,
        match_count=match_count,
        inlier_count=0,
        inlier_ratio=0.0,
        transform=None,
        overlap=None,
        debug_visualization=debug_visualization,
        warnings=_deduplicate(warnings),
    )


def _affine_to_transform(affine: np.ndarray) -> RegistrationTransform:
    matrix = [
        [float(affine[0, 0]), float(affine[0, 1]), float(affine[0, 2])],
        [float(affine[1, 0]), float(affine[1, 1]), float(affine[1, 2])],
        [0.0, 0.0, 1.0],
    ]
    return RegistrationTransform(type="affine", matrix=matrix)


def _status_counts(registrations: list[RegistrationResult]) -> dict[str, int]:
    counts = {"registered": 0, "weak_match": 0, "failed": 0, "skipped": 0}
    for result in registrations:
        counts[result.registration_status] += 1
    return counts


def _deduplicate(values: Iterable[str]) -> list[str]:
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
    return sanitized or "registration"
