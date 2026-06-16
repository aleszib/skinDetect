# SkinTrack Change Monitor

SkinTrack Change Monitor is a backend-first longitudinal skin-photo change-detection scaffold.
It is designed to accept local photos, preserve timestamp provenance, prepare report contracts,
and support conservative change review workflows.

## Guardrails

- No melanoma diagnosis claims.
- No web UI in PR-001.
- No cloud upload or external image handling.
- No real patient photos in the repository.
- Reports must use conservative language such as "change detected" and "review recommended".
- This project is not a melanoma detector or diagnosis tool.

## Photo import

PR-002 adds a local photo import CLI that scans a folder and writes a JSON manifest.
Recursive scanning is the default.

PR-003 adds heuristic image quality assessment to that import step. Each supported image record
now includes a `quality` section with technical checks for readability, brightness, sharpness,
size, and overall usability.

PR-004 adds overlap candidate ranking on top of the import manifest. It ranks likely pairs for
later geometric registration, but it does not prove that two photos share the same skin region.

PR-005 adds backend geometric registration for the ranked pairs. It estimates technical image
alignment only and can optionally write debug visualization images for inspection.

Example:

```bash
python -m skintrack.cli import-photos ./photos --output ./artifacts/manifest.json
python -m skintrack.cli rank-overlap-candidates ./artifacts/manifest.json --output ./artifacts/overlap_candidates.json
python -m skintrack.cli register-candidate-pairs ./artifacts/overlap_candidates.json --manifest ./artifacts/manifest.json --output ./artifacts/registrations.json --debug-dir ./artifacts/debug_registration
```

The manifest records:

- schema version and creation time;
- input directory and whether the scan was recursive;
- per-file status, hash, original path, filename, timestamp provenance, and image dimensions when available;
- a `quality` object for supported images with readability, brightness, blur, size, and overall status;
- counts for imported, skipped, unreadable, and unsupported files;
- counts for low-quality images that are technically readable but may be too dark, too bright,
  too blurry, or too small for reliable comparison;
- overlap candidate ranking output with scores, statuses, reasons, penalties, and heuristic
  similarity metadata;
- geometric registration output with transform estimates, inlier counts, overlap polygons,
  confidence, warnings, and optional debug visualization paths;
- warnings, including unsupported-format notes such as HEIC not being supported yet.

## Developer setup

Create a virtual environment and install the local package plus dev tools:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]
```

## Checks

```bash
python -m pytest
ruff check .
```

## Safety reminder

Do not commit real patient photos or other sensitive clinical images. This repository is for
backend change detection and review support, not diagnosis.

Quality scores are heuristic technical checks only. They are not medical assessment and they do
not diagnose melanoma or any other condition. Low image quality may prevent reliable future
overlap or change analysis.

Overlap candidate ranking is only a lightweight local heuristic. It does not prove geometric
overlap. PR-005 geometric registration estimates technical alignment only and remains
conservative. No melanoma diagnosis or cancer risk scoring is performed.

Geometric registration in PR-005 is also conservative. It estimates technical alignment for
later review and does not identify lesions, diagnose melanoma, or estimate cancer risk. Debug
visualization images are technical inspection aids only; they are not the final user-facing
annotated area-of-concern images.
