"""Tests for `battery` (PROJECT_BRIEF.md §26).

Planned checks (to be implemented in Stage 2, once `battery` has a real
implementation -- currently a Stage 1 placeholder that raises
NotImplementedError):
- SOC remains within configured bounds
- charge and discharge power limits are respected
- no simultaneous charging and discharging
- no energy is created (charge/discharge efficiency correctly applied)
- a zero-capacity battery behaves correctly (no charge/discharge possible)

Tests in this file must use small deterministic fixtures and must not
require internet access, per PROJECT_BRIEF.md §26.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Placeholder — real implementation lands in Stage 2.")
def test_placeholder() -> None:
    """Replaced with real assertions once the corresponding src module is implemented."""
    raise NotImplementedError
