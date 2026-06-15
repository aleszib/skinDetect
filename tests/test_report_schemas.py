"""Report schema contract tests."""

from __future__ import annotations

from datetime import datetime

from skintrack.reports.schemas import ChangeFlag


def test_change_flag_includes_annotated_image_path() -> None:
    flag = ChangeFlag(
        id="flag-001",
        severity="review",
        reason="Visible area changed between overlapping photos.",
        text_summary="Notable visual change detected; dermatology review recommended.",
        annotated_image_path="outputs/flag-001.png",
        created_at=datetime(2024, 5, 18, 12, 0, 0),
    )

    assert "annotated_image_path" in ChangeFlag.model_fields
    assert flag.annotated_image_path == "outputs/flag-001.png"
