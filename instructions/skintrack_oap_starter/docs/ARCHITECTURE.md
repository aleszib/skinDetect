# Architecture: SkinTrack Change Monitor

## Product category
Backend-first longitudinal skin-photo change-detection system.

## Architectural principle
Separate analysis truth from presentation. The backend must produce structured evidence and annotated outputs before any web interface is built.

## Components

### 1. Photo ingestion
Responsibilities:

- accept local image files;
- calculate file hash;
- safely read minimal metadata;
- extract timestamp from EXIF, filename, or file metadata;
- assign timestamp source and confidence;
- store a normalized photo record.

### 2. Image quality assessment
Responsibilities:

- detect unreadable files;
- estimate blur;
- estimate underexposure/overexposure;
- mark images that should not be used for change claims.

### 3. Overlap candidate search
Responsibilities:

- rank photo pairs by likelihood of shared skin area;
- avoid expensive registration for obviously unrelated images;
- preserve pair scores and reasons.

### 4. Geometric registration
Responsibilities:

- detect local features or use a future learned matcher;
- match features;
- estimate transform with robust outlier rejection;
- compute overlap mask/polygon;
- report confidence and failure reason.

### 5. Lesion candidate detection
Responsibilities:

- identify visible lesion candidates conservatively; or
- accept manually supplied candidate masks/points in early versions;
- compute basic features such as area, color, border proxies, and centroid.

### 6. Temporal tracking
Responsibilities:

- connect candidates across registered images;
- preserve track confidence;
- avoid false tracking when registration is weak.

### 7. Change engine
Responsibilities:

- compare candidate observations over time;
- compute relative metrics;
- flag notable change using conservative thresholds;
- distinguish measurement failure from biological/visual change.

### 8. Reporting and annotation
Responsibilities:

- generate structured JSON reports;
- generate a human-readable text summary;
- render annotated images marking changed areas;
- avoid diagnostic language.

### 9. API layer, later
Responsibilities:

- expose backend functions through FastAPI;
- handle jobs/status/results;
- remain backend-driven.

### 10. Web app, later
Responsibilities:

- upload or select photos;
- show timeline;
- show overlap and registration results;
- show flagged changes with annotated images;
- export report.

The web app must not be started until backend milestones explicitly allow it.

## Initial storage strategy
Early prototype can use local files plus JSON outputs. SQLite may be introduced for local persistence. PostgreSQL can be introduced later if multi-user or server deployment becomes a real requirement.

## Data contracts

### PhotoRecord
- id
- original_filename
- file_hash
- stored_path
- width
- height
- taken_at
- timestamp_source
- timestamp_confidence
- imported_at
- quality_status

### PhotoOverlap
- photo_a_id
- photo_b_id
- transform
- overlap_mask_path
- overlap_score
- registration_confidence
- status
- failure_reason

### LesionCandidate
- id
- photo_id
- mask_path
- centroid
- area_px
- color_features
- border_features
- confidence

### LesionTrackObservation
- track_id
- photo_id
- candidate_id
- registered_location
- measurement

### ChangeFlag
- id
- track_id
- severity
- reason
- evidence
- text_summary
- annotated_image_path
- confidence
- created_at

## Annotated image requirements
Annotated images should be deterministic enough to test. Minimum annotation:

- draw a visible outline or marker around the changed region;
- optionally show overlap boundary;
- include a short non-diagnostic label such as "visual change";
- save to a predictable output path;
- return path in report JSON.

## Test strategy
Use synthetic images wherever possible. Synthetic images allow deterministic tests for timestamps, transformations, lesion-like blobs, overlays, and report generation without committing real skin photos.
