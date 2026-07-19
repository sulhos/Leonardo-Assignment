"""Tests for `dispatch` (PROJECT_BRIEF.md §26).

Planned checks (to be implemented in Stage 2, once `dispatch` has a real
implementation -- currently a Stage 1 placeholder that raises
NotImplementedError):
- dispatch priority order is respected (direct supply -> charge -> discharge -> curtail/unserved)
- initial-SOC-bias mitigation method converges and is tested explicitly

Tests in this file must use small deterministic fixtures and must not
require internet access, per PROJECT_BRIEF.md §26.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Placeholder — real implementation lands in Stage 2.")
def test_placeholder() -> None:
    """Replaced with real assertions once the corresponding src module is implemented."""
    raise NotImplementedError
