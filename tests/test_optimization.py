"""Tests for `optimization` (PROJECT_BRIEF.md §26).

Planned checks (to be implemented in Stage 3, once `optimization` has a real
implementation -- currently a Stage 1 placeholder that raises
NotImplementedError):
- a feasible minimum candidate is selected correctly
- an infeasible range is reported correctly (not silently accepted)
- increasing battery capacity does not cause unexplained increases in unserved energy
- modular capacity (n * module_capacity_kwh) is calculated correctly

Tests in this file must use small deterministic fixtures and must not
require internet access, per PROJECT_BRIEF.md §26.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Placeholder — real implementation lands in Stage 3.")
def test_placeholder() -> None:
    """Replaced with real assertions once the corresponding src module is implemented."""
    raise NotImplementedError
