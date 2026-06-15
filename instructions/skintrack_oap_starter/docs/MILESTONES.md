# Milestones and PR-Sized Execution Plan

## Milestone 0: Governance and scaffold
Goal: create durable repository rules and a working backend skeleton.

### PR-001: Backend scaffold, timestamp extraction, report contracts, tests
- Create Python package structure.
- Add pyproject.toml.
- Add pytest and ruff configuration.
- Implement timestamp extraction from EXIF, filename, and file metadata fallback.
- Define initial Pydantic schemas for PhotoRecord and ChangeFlag.
- Include annotated_image_path in ChangeFlag.
- Add safety-language guardrails.
- Add tests.

Definition of done:
- `pytest` passes.
- `ruff check` passes if configured.
- Timestamp source/confidence tests pass.
- Report schema test proves annotated_image_path exists.
- Safety-language test rejects forbidden diagnostic phrases.

## Milestone 1: Image ingestion and quality

### PR-002: Photo import CLI
- Add Typer CLI for importing a folder of images.
- Hash files.
- Generate PhotoRecord JSON.
- Preserve timestamp source/confidence.
- Add tests with synthetic images.

### PR-003: Image quality assessment
- Detect blur, exposure extremes, unreadable files.
- Store quality flags.
- Add tests using synthetic images.

## Milestone 2: Overlap and registration

### PR-004: Candidate overlap ranking
- Rank photo pairs for likely overlap.
- Start with simple image similarity and metadata heuristics.
- Add deterministic tests.

### PR-005: Geometric registration
- Implement feature matching and robust transform estimation.
- Produce overlap mask or polygon.
- Store confidence and failure reason.
- Add synthetic transformed-image tests.

## Milestone 3: Candidate areas and tracking

### PR-006: Lesion candidate interface
- Implement conservative lesion-candidate representation.
- Allow manual/synthetic candidate input for testing.
- Do not claim melanoma detection.
- Add tests.

### PR-007: Temporal tracking
- Track candidates across registered images.
- Store track confidence.
- Reject weak matches.
- Add tests.

## Milestone 4: Change detection and annotated reports

### PR-008: Change metrics
- Compute relative area/color/border proxy changes.
- Add threshold configuration.
- Preserve uncertainty.
- Add tests.

### PR-009: Annotated image rendering
- Render marked areas on output images.
- Include overlap/lesion boundaries where available.
- Add tests proving annotation changes pixels and returns an output file.

### PR-010: Human-readable report generation
- Generate text summary plus JSON report.
- Use conservative review language only.
- Add tests for report content and forbidden diagnostic phrases.

## Milestone 5: Backend API

### PR-011: FastAPI skeleton
- Add API only after backend report generation works.
- Endpoints for import, job status, photo timeline, overlaps, flags, annotated image retrieval.
- Add API tests.

### PR-012: Background jobs
- Add job queue only if long-running processing requires it.
- Add tests.

## Milestone 6: Web app
Allowed only after Milestones 0-5 are sufficiently complete.

### PR-013: Minimal web UI
- Upload/select photos.
- Show timeline.
- Show flagged changes.
- Display annotated images.
- Export report.
- Add end-to-end smoke tests.

## Milestone 7: Validation and release honesty

### PR-014: Validation fixtures and benchmark harness
- Add synthetic and approved de-identified validation sets.
- Document known limitations.
- Avoid clinical claims.

### PR-015: Release-readiness review
- Produce release brief.
- Confirm docs, tests, privacy, and diagnostic-language boundaries.
