"""Conservative safety-language guardrails for user-facing text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

FORBIDDEN_DIAGNOSTIC_PHRASES: Final[tuple[str, ...]] = (
    "melanoma detected",
    "cancer detected",
    "benign",
    "malignant",
    "safe",
    "no concern",
    "diagnosis",
)

_FORBIDDEN_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = tuple(
    (phrase, re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE))
    for phrase in FORBIDDEN_DIAGNOSTIC_PHRASES
)


@dataclass(frozen=True)
class SafetyLanguageCheck:
    """Result of a text safety-language validation pass."""

    text: str
    contains_forbidden_language: bool
    matched_phrases: tuple[str, ...]


def validate_user_facing_text(text: str) -> SafetyLanguageCheck:
    """Check whether user-facing text contains forbidden diagnostic language."""

    matched_phrases = tuple(
        phrase
        for phrase, pattern in _FORBIDDEN_PATTERNS
        if pattern.search(text)
    )
    return SafetyLanguageCheck(
        text=text,
        contains_forbidden_language=bool(matched_phrases),
        matched_phrases=matched_phrases,
    )

