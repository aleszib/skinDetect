# AGENTS.md

## Mission
This repository implements **SkinTrack Change Monitor**: a privacy-preserving longitudinal skin-photo analysis backend that imports smartphone skin photos, extracts timestamps, detects overlapping regions across time, tracks visible lesion candidates, and produces conservative change reports for human review.

The system must produce two linked outputs for every flagged change:

1. a concise textual explanation of what changed, with evidence and uncertainty; and
2. an annotated image where the relevant area is visibly marked.

The system is **not** a melanoma diagnosis tool. It must not claim that a lesion is benign, malignant, melanoma, cancer, or safe. It may say that a visible area changed and that human/dermatology review is recommended.

## Product boundary
The intended first product is a backend-first analysis engine. A web app is explicitly deferred until the backend pipeline is implemented, tested, and able to generate reliable structured reports and annotated images.

### First release scope
The first backend release should support:

- importing a series of smartphone photos from local files;
- extracting timestamps from EXIF, filename, or file metadata;
- recording timestamp source and confidence;
- assessing basic image quality;
- finding likely overlapping photo regions;
- estimating geometric registration and overlap masks;
- detecting or accepting lesion candidates;
- tracking candidates over time within registered overlaps;
- computing conservative change metrics;
- generating a structured text report plus annotated image for each flagged change;
- preserving uncertainty and failure reasons.

### Explicit non-goals for early milestones
- Do not build the web interface before the backend milestones say it is allowed.
- Do not implement user accounts or cloud upload before privacy architecture is defined.
- Do not diagnose melanoma or any other disease.
- Do not claim clinical validation.
- Do not train a melanoma classifier as an early task.
- Do not use public datasets in a way that implies smartphone-photo clinical validity unless the dataset match and validation limits are documented.
- Do not send images to external APIs unless explicitly approved by the human lead.

## Medical and safety language rules
The app may use language such as:

- "Notable visual change detected."
- "Change flag for human review."
- "Dermatology review recommended."
- "Insufficient overlap or low image quality prevents reliable comparison."

The app must not use output language such as:

- "melanoma detected";
- "cancer detected";
- "benign";
- "malignant";
- "safe";
- "no concern";
- "diagnosis";
- "medical advice".

If templates or reports include safety text, they must clearly state that the tool does not replace clinical assessment.

## Privacy and security rules
Skin photos are sensitive health-related personal data. Treat all images, metadata, derived masks, and reports as sensitive.

- Do not commit real patient or user photos.
- Do not commit identifiable test images.
- Use synthetic images or explicitly approved de-identified fixtures only.
- Strip or ignore GPS/location metadata unless a future privacy design explicitly allows it.
- Do not log full EXIF payloads by default if they may contain identifying metadata.
- Do not upload images to external services.
- Do not add telemetry.
- Do not store secrets in the repository.
- Do not paste private images or metadata into prompts.

## Preferred technical stack
Backend-first Python stack:

- Python 3.11+ or 3.12+
- pyproject.toml-based packaging
- pytest for tests
- ruff for linting/format checks
- Pillow and/or exifread for image metadata
- OpenCV and/or scikit-image for image registration and image processing
- Pydantic for structured report schemas
- Typer for early CLI workflows
- FastAPI later for the backend API
- SQLite for local prototype persistence, PostgreSQL later if needed

Do not add a frontend framework until the backend is ready for the web-app milestone.

## Repository structure guidance
Recommended structure:

```text
skintrack/
  __init__.py
  cli.py
  config.py
  metadata/
    timestamps.py
  io/
    photos.py
  quality/
    assess.py
  registration/
    overlap.py
    geometry.py
  lesions/
    candidates.py
    tracking.py
  reports/
    schemas.py
    render.py
    annotations.py
  safety/
    language.py

tests/
  test_*.py
  fixtures/

docs/
  PROJECT_MEMORY.md
  ARCHITECTURE.md
  MILESTONES.md
  DECISION_LOG.md

prompts/
  PR-001-codex-work-order.md
```

## Data and report contracts
Every imported photo should preserve:

- original filename;
- file hash;
- stored path or local reference;
- extracted capture timestamp;
- timestamp source;
- timestamp confidence;
- minimal safe metadata required for processing;
- image quality status.

Every flagged change should preserve:

- involved photo IDs;
- timestamp range;
- registered overlap confidence;
- lesion/area track ID if available;
- change metrics;
- confidence;
- textual explanation;
- annotated image path;
- failure/uncertainty notes.

Annotated images must mark the relevant area. They may also show the overlap boundary, candidate lesion boundary, direction of change, and a small label. They must not visually label an area as melanoma or cancer.

## Testing requirements
Every implementation PR must include tests unless the PR is explicitly documentation-only.

Minimum expected tests over the project:

- package import smoke test;
- timestamp extraction from EXIF DateTimeOriginal;
- timestamp extraction from filename patterns;
- file metadata fallback marked as lower confidence;
- unreadable/missing timestamp handling;
- image quality tests using synthetic images;
- registration tests using synthetic transformed images;
- overlap rejection tests for unrelated images;
- report schema validation tests;
- annotated image rendering tests proving a marker/overlay is produced;
- safety-language tests preventing diagnostic claims in user-facing templates;
- CLI smoke tests for import/report commands.

A skipped test is not a passing test. A test that was not run is not evidence.

## Runtime and Codex CLI rules
The human operator intends to use Codex CLI with `--yolo` and model `gpt-5.4-Mini` inside a hardened WSL Ubuntu environment.

The execution agent may install missing local development tools inside that hardened WSL Ubuntu environment if required to complete the task. It must document all installed packages and setup commands in the final report.

The agent must not:

- access production secrets;
- use private user photos unless explicitly provided as approved local fixtures;
- write to protected branches;
- merge its own PR;
- ask the human to perform routine dependency setup unless blocked by a deliberate safety boundary.

## Workflow
For every coding task:

1. Read this AGENTS.md first.
2. Inspect the repository before editing.
3. Start from current main unless the prompt says otherwise.
4. Create a feature branch.
5. Keep the task PR-sized.
6. Commit only related files.
7. Include or update tests.
8. Run focused tests and lint checks where configured.
9. Push the branch and open a pull request if repository access is available.
10. Do not merge.

## Final report format
Every execution task must end with:

```text
Branch:
Commit:
Pull request:

Summary:
- ...

Files changed:
- ...

Tests run:
- command: result

Annotated-output impact:
- ...

Medical-safety confirmations:
- No diagnostic claim added.
- Reports remain change-detection / human-review only.

Privacy confirmations:
- No real user photos committed.
- No external image upload added.

Local setup performed:
- ...

Known limitations / skipped tests:
- ...

Recommended next task:
- ...
```
