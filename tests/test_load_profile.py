"""Tests for `load_profile` (PROJECT_BRIEF.md §26).

Planned checks (to be implemented in Stage 2, once `load_profile` has a real
implementation -- currently a Stage 1 placeholder that raises
NotImplementedError):
- no negative demand
- target annual consumption achieved within tolerance
- peak demand respected
- identical seed produces an identical profile

Tests in this file must use small deterministic fixtures and must not
require internet access, per PROJECT_BRIEF.md §26.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Placeholder — real implementation lands in Stage 2.")
def test_placeholder() -> None:
    """Replaced with real assertions once the corresponding src module is implemented."""
    raise NotImplementedError
