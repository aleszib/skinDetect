# PR-001 Codex Work Order: Backend scaffold, timestamp extraction, report contracts, tests

You are working on the **SkinTrack Change Monitor** repository.

The human operator will run you using Codex CLI with `--yolo` and model `gpt-5.4-Mini` inside a hardened WSL Ubuntu environment. You may install local development dependencies inside that environment if needed, but you must document what you installed.

## Governing instructions
Read `AGENTS.md` first and follow it strictly.

## Current state
This may be a new or nearly empty repository. If files already exist, inspect them first and preserve useful existing structure unless it conflicts with `AGENTS.md`.

## Goal
Create the initial backend scaffold and first tested functionality for SkinTrack Change Monitor.

This PR must establish:

1. Python backend package structure.
2. Project configuration.
3. Timestamp extraction from images and filenames.
4. Initial data/report schemas, including annotated image output path.
5. Safety-language guardrails.
6. Tests from the first PR.

## Scope
Implement only the scaffold and foundational backend contracts. This is not the overlap/registration PR and not the web UI PR.

## Non-goals
- Do not build a web app.
- Do not implement melanoma diagnosis.
- Do not add ML model training.
- Do not add cloud upload.
- Do not commit real skin photos.
- Do not implement full lesion tracking yet.
- Do not make clinical validation claims.

## Required files / structure
Create or update a structure close to:

```text
skintrack/
  __init__.py
  cli.py
  metadata/
    __init__.py
    timestamps.py
  reports/
    __init__.py
    schemas.py
  safety/
    __init__.py
    language.py

tests/
  test_package_import.py
  test_timestamps.py
  test_report_schemas.py
  test_safety_language.py

pyproject.toml
README.md
```

It is acceptable to add small helper modules if justified.

## Functional requirements

### Timestamp extraction
Implement a timestamp extraction module that tries, in order:

1. EXIF DateTimeOriginal or equivalent capture-date fields.
2. Recognized filename patterns such as:
   - `IMG_20240517_143022.jpg`
   - `2024-05-17_14-30-22.jpg`
   - `20240517-143022.png`
3. File modification time fallback.
4. Unknown timestamp if none can be obtained.

The result must include:

- `taken_at`, nullable if unknown;
- `timestamp_source`, such as `exif`, `filename`, `file_mtime`, or `unknown`;
- `timestamp_confidence`, such as `high`, `medium`, `low`, or `unknown`;
- optional notes.

Do not silently treat file modification time as a high-confidence capture time.

### Report schemas
Define initial Pydantic schemas for at least:

- `PhotoRecord`
- `ChangeFlag`

`ChangeFlag` must include `annotated_image_path` or a similarly named field that points to the image where the changed area is marked.

Do not implement full annotation rendering in this PR. Just define the contract that later PRs must satisfy.

### Safety-language guardrails
Create a small safety module that defines forbidden user-facing diagnostic phrases, including at least:

- melanoma detected
- cancer detected
- benign
- malignant
- safe
- no concern
- diagnosis

Provide a function that can validate a candidate user-facing text string and report whether it contains forbidden diagnostic language.

This is not a complete medical-safety system; it is an early guardrail.

### README
Add a short README explaining:

- project purpose;
- backend-first plan;
- no diagnostic claims;
- basic developer setup;
- test command.

## Tests required
Use pytest. Add tests for:

1. Package import smoke test.
2. Filename timestamp parsing for at least two supported patterns.
3. File mtime fallback marked as low confidence.
4. Unknown timestamp behavior if appropriate.
5. Report schema includes an annotated image path for change flags.
6. Safety-language validation rejects forbidden diagnostic phrases.
7. Safety-language validation accepts conservative phrases such as "Notable visual change detected; dermatology review recommended."

If EXIF timestamp testing is easy with Pillow, include it. If not, leave a clear TODO or xfail with reason, but do not claim it passed.

## Tooling
Add pyproject configuration for:

- pytest;
- ruff;
- pydantic;
- Pillow or another lightweight image metadata library if needed.

Keep dependencies minimal.

## Commands to run
At minimum run:

```bash
python -m pytest
ruff check .
```

If `ruff` is not installed, install it in the local environment or use the project dependency workflow you create. Report exactly what happened.

## Branch and PR workflow
- Start from current main.
- Create branch: `feature/pr-001-backend-scaffold-timestamps`
- Commit only related files.
- Push and open a pull request if repository remote access is configured.
- Do not merge.

## Final report required
End with:

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
- ChangeFlag schema includes annotated image output field.
- Rendering will be implemented in a later PR.

Medical-safety confirmations:
- No diagnostic claim added.
- Safety-language guardrail tests added.

Privacy confirmations:
- No real user photos committed.
- No external image upload added.

Local setup performed:
- ...

Known limitations / skipped tests:
- ...

Recommended next task:
- PR-002 Photo import CLI or PR-003 Image quality assessment, depending on current repository state.
```
