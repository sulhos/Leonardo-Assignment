"""Tests for `wind_model` (PROJECT_BRIEF.md §26).

Planned checks (to be implemented in Stage 2, once `wind_model` has a real
implementation -- currently a Stage 1 placeholder that raises
NotImplementedError):
- no negative wind generation
- wind output never exceeds rated capacity
- correct behaviour below cut-in and above cut-out speeds

Tests in this file must use small deterministic fixtures and must not
require internet access, per PROJECT_BRIEF.md §26.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Placeholder — real implementation lands in Stage 2.")
def test_placeholder() -> None:
    """Replaced with real assertions once the corresponding src module is implemented."""
    raise NotImplementedError
