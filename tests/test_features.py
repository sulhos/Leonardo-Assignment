"""Tests for `features` (PROJECT_BRIEF.md §26).

Planned checks (to be implemented in Stage 4/5, once `features` has a real
implementation -- currently a Stage 1 placeholder that raises
NotImplementedError):
- no target leakage: optimal capacity/module count/cost never appear as input features
- feature scalers are fit on training data only

Tests in this file must use small deterministic fixtures and must not
require internet access, per PROJECT_BRIEF.md §26.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Placeholder — real implementation lands in Stage 4/5.")
def test_placeholder() -> None:
    """Replaced with real assertions once the corresponding src module is implemented."""
    raise NotImplementedError
