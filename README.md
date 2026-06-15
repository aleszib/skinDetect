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

