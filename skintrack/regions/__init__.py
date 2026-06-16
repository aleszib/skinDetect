"""Manual candidate-region intake helpers."""

from __future__ import annotations

from .manual import (
    CandidateRegionInput,
    CandidateRegionManifest,
    CandidateRegionPhotoRef,
    CandidateRegionResult,
    CandidateRegionValidationManifest,
    RegionValidationStatus,
    load_candidate_region_manifest,
    load_photo_import_manifest,
    validate_candidate_regions,
    write_validated_candidate_region_manifest,
)
from .projection import (
    ProjectedBoundingBox,
    ProjectedCandidateRegionManifest,
    ProjectedCandidateRegionPhoto,
    ProjectedCandidateRegionResult,
    ProjectedRegionGeometry,
    ProjectionStatus,
    TransformDirection,
    load_registration_manifest,
    load_validated_candidate_region_manifest,
    project_candidate_regions,
    write_projected_candidate_region_manifest,
)

__all__ = [
    "CandidateRegionInput",
    "CandidateRegionManifest",
    "CandidateRegionPhotoRef",
    "CandidateRegionResult",
    "CandidateRegionValidationManifest",
    "ProjectedBoundingBox",
    "ProjectedCandidateRegionManifest",
    "ProjectedCandidateRegionPhoto",
    "ProjectedCandidateRegionResult",
    "ProjectedRegionGeometry",
    "RegionValidationStatus",
    "ProjectionStatus",
    "TransformDirection",
    "load_candidate_region_manifest",
    "load_photo_import_manifest",
    "load_registration_manifest",
    "load_validated_candidate_region_manifest",
    "validate_candidate_regions",
    "project_candidate_regions",
    "write_validated_candidate_region_manifest",
    "write_projected_candidate_region_manifest",
]
