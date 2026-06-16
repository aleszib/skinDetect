# Project Memory: SkinTrack Change Monitor

Prepared: 2026-06-15

## Current project truth
This is a backend-first project for longitudinal skin-photo change detection. It is not a diagnostic melanoma classifier.

The human lead wants an app that accepts a series of skin photos, usually from a smartphone, obtains timestamps when possible from filenames or file properties such as date taken, detects overlapping regions across images, and reports suspicious visual changes. The final output should include both a written description of the change and an annotated image marking the relevant area.

The backend should preferably be Python-based. A web app interface is desired, but only after the backend is implemented and tested.

The intended coding workflow is OAP-style: strategic planning and durable instructions first, then Codex CLI execution in PR-sized tasks. The human plans to run Codex CLI with `--yolo` and model `gpt-5.4-Mini` in a hardened WSL Ubuntu environment.

## Product name
Working name: **SkinTrack Change Monitor**

## Product mission
A privacy-preserving longitudinal skin-photo analysis tool that helps users and clinicians organize smartphone skin images, identify overlapping regions across time, track visible lesion candidates, and flag notable visual changes for human review. It does not diagnose melanoma and does not replace dermatological assessment.

## Core user story
A user provides a sequence of skin photos taken at different times. The system extracts or infers capture timestamps, finds photos that show overlapping skin regions, registers the overlapping area, identifies visible lesion candidates, compares them over time, and produces a conservative report with annotated images marking regions that changed.

## Non-negotiable invariants
- Backend before web UI.
- Tests from the first coding PR.
- No diagnostic claims.
- No melanoma/cancer classification in early milestones.
- Output must include text plus annotated image for flagged change areas.
- Every measurement must preserve uncertainty.
- Timestamp source and confidence must be stored.
- Registration confidence must be stored.
- Private photos must not be committed.
- External image upload is forbidden unless explicitly approved later.

## Current intended backend pipeline
```text
photo import
  -> timestamp extraction
  -> image quality checks
  -> overlap candidate ranking
  -> geometric registration
  -> overlap mask creation
  -> lesion candidate detection or manual candidate intake
  -> lesion tracking across registered images
  -> change metrics
  -> change flags
  -> text report + annotated image
```

PR-004 adds overlap candidate ranking only. It is a lightweight local heuristic for identifying
likely photo pairs before geometric registration; it does not confirm actual overlap.

PR-005 adds geometric registration for ranked candidate pairs using local feature matching and
RANSAC-based transform estimation. It can also write technical debug visualization images for
inspection, but it does not perform lesion detection, change flagging, or medical annotation.

PR-006 adds manual candidate-region intake for rectangle, polygon, and point-radius regions, plus
neutral technical overlay images. It validates regions against imported image dimensions and does
not automatically detect lesions or make diagnostic claims.

PR-007 adds candidate-region projection through registered image pairs. It estimates where a
validated manual candidate region lands in the paired photo and can emit neutral technical
projection overlays. It does not decide whether anything changed, does not identify lesions, and
does not make diagnostic claims.

## Output contract
For every change flag, the system should produce:

- a short written explanation;
- timestamp range;
- photo IDs or filenames involved;
- change metrics;
- overlap/registration confidence;
- uncertainty notes;
- recommendation for human/dermatology review if appropriate;
- annotated image path.

The annotated image should visibly mark the region of change and may include overlap boundary and lesion boundary overlays.

## Strategic safety boundary
This project should initially be framed as a change-detection and review-support tool. It should not be framed as an autonomous diagnostic tool. Future clinical or regulatory expansion must be treated as a separate milestone requiring validation, documentation, and legal/regulatory review.

## References to preserve in future work
- AAD ABCDE guidance emphasizes evolving/changing lesions as a warning sign.
- FDA device software/mobile medical app guidance may be relevant if diagnostic or clinical claims are made.
- EU MDCG 2019-11 qualification/classification guidance may be relevant under MDR/IVDR if the software is intended for medical purposes.

## Next recommended action
Run PR-001: scaffold the Python backend, add project configuration, define initial data/report schemas including annotated image output, implement timestamp extraction, and add tests.
