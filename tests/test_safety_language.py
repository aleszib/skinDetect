"""Safety-language validation tests."""

from __future__ import annotations

from skintrack.safety.language import validate_user_facing_text


def test_forbidden_diagnostic_phrases_are_rejected() -> None:
    result = validate_user_facing_text("This looks benign and safe.")

    assert result.contains_forbidden_language is True
    assert "benign" in result.matched_phrases
    assert "safe" in result.matched_phrases


def test_conservative_language_is_allowed() -> None:
    result = validate_user_facing_text(
        "Notable visual change detected; dermatology review recommended."
    )

    assert result.contains_forbidden_language is False
    assert result.matched_phrases == ()

