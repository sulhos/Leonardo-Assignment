"""Tests for `splitting` (PROJECT_BRIEF.md §26).

Planned checks (to be implemented in Stage 5, once `splitting` has a real
implementation -- currently a Stage 1 placeholder that raises
NotImplementedError):
- splits are reproducible given a fixed seed
- no overlap between grouped train/test scenarios

Tests in this file must use small deterministic fixtures and must not
require internet access, per PROJECT_BRIEF.md §26.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Placeholder — real implementation lands in Stage 5.")
def test_placeholder() -> None:
    """Replaced with real assertions once the corresponding src module is implemented."""
    raise NotImplementedError
