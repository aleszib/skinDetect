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

Example:

```bash
python -m skintrack.cli import-photos ./photos --output ./artifacts/manifest.json
```

The manifest records:

- schema version and creation time;
- input directory and whether the scan was recursive;
- per-file status, hash, original path, filename, timestamp provenance, and image dimensions when available;
- a `quality` object for supported images with readability, brightness, blur, size, and overall status;
- counts for imported, skipped, unreadable, and unsupported files;
- counts for low-quality images that are technically readable but may be too dark, too bright,
  too blurry, or too small for reliable comparison;
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
