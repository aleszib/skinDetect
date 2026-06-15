"""Package import smoke test."""

from __future__ import annotations


def test_package_import_smoke() -> None:
    import skintrack

    assert skintrack.__version__ == "0.1.0"

