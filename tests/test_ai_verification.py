"""Tests for `ai_verification` (PROJECT_BRIEF.md §26).

Planned checks (to be implemented in Stage 7, once `ai_verification` has a real
implementation -- currently a Stage 1 placeholder that raises
NotImplementedError):
- predictions are non-negative
- module rounding is always upward (ceiling), never downward
- physical verification uses the predicted battery capacity, not the reference/optimal capacity

Tests in this file must use small deterministic fixtures and must not
require internet access, per PROJECT_BRIEF.md §26.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Placeholder — real implementation lands in Stage 7.")
def test_placeholder() -> None:
    """Replaced with real assertions once the corresponding src module is implemented."""
    raise NotImplementedError
