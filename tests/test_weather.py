"""Tests for `weather` (PROJECT_BRIEF.md §26).

Planned checks (to be implemented in Stage 2, once `weather` has a real
implementation -- currently a Stage 1 placeholder that raises
NotImplementedError):
- exactly 8,760 records for a non-leap local year
- unique timestamps, no missing timestamps
- correct timezone handling (including DST for Vasteras)
- no unexpected NaN values after processing
- correct unit conversions

Uses small deterministic fixtures; must not require internet access (PROJECT_BRIEF.md §26).

Tests in this file must use small deterministic fixtures and must not
require internet access, per PROJECT_BRIEF.md §26.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Placeholder — real implementation lands in Stage 2.")
def test_placeholder() -> None:
    """Replaced with real assertions once the corresponding src module is implemented."""
    raise NotImplementedError
