# Decision Log

## Decision 001: Backend before web app
Status: Accepted
Date: 2026-06-15

The project will build the backend analysis pipeline first. The web app will be added only after the backend can import photos, extract timestamps, handle overlaps/registration, produce change flags, and generate text plus annotated-image reports.

Rationale: A web UI built too early can hide backend uncertainty and encourage premature product claims.

## Decision 002: Change-detection tool, not melanoma diagnosis
Status: Accepted
Date: 2026-06-15

The project will not initially diagnose melanoma, cancer, malignancy, or benignity. It will flag visual changes and recommend human/dermatology review when appropriate.

Rationale: Diagnostic claims require clinical validation and may trigger medical-device regulatory obligations. The safer early product is a longitudinal change-detection and evidence-organization tool.

## Decision 003: Every flag must include text plus annotated image
Status: Accepted
Date: 2026-06-15

A flag is incomplete unless it produces both a human-readable explanation and an annotated image marking the relevant area.

Rationale: The user must see what region triggered the flag. This also supports review, debugging, and future validation.

## Decision 004: Python backend preferred
Status: Accepted
Date: 2026-06-15

Use a Python-first backend unless a later technical finding strongly argues otherwise.

Rationale: Python has strong libraries for image processing, scientific testing, data handling, APIs, and ML integration if needed later.

## Decision 005: Synthetic test images first
Status: Accepted
Date: 2026-06-15

Use synthetic test images for early tests. Do not commit real user skin photos.

Rationale: This protects privacy and makes tests deterministic.

## Decision 006: Codex CLI yolo only inside hardened WSL Ubuntu
Status: Accepted
Date: 2026-06-15

The execution agent may run with high autonomy only inside the operator's hardened WSL Ubuntu environment.

Rationale: This follows the OAP high-autonomy runtime pattern: useful autonomy inside a disposable or bounded environment, with no production secrets or irreplaceable data.
