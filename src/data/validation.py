"""Shared data-quality validation checks used across weather, load, PV, and wind
outputs (PROJECT_BRIEF.md §26).

Responsibilities (implemented starting Stage 2):

- Confirm exactly 8,760 hourly records for a non-leap local year.
- Confirm unique, gap-free hourly timestamps.
- Confirm no unexpected NaN values after processing.
- Confirm value ranges are physically plausible (no negative demand/generation,
  outputs within configured limits, etc.) for use by physics-layer tests.

This module intentionally contains no implementation yet -- it is a Stage 1
placeholder. Functions raise NotImplementedError until Stage 2.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def validate_hourly_index(series: pd.Series, expected_length: int = 8760) -> None:
    """Raise if the series index is not a complete, unique, gap-free hourly
    index of the expected length."""
    raise NotImplementedError("Implemented in Stage 2.")


def validate_no_nans(df: pd.DataFrame) -> None:
    """Raise if any column contains unexpected NaN values."""
    raise NotImplementedError("Implemented in Stage 2.")


def validate_bounds(series: pd.Series, lower: float, upper: float, name: str) -> None:
    """Raise if any value in `series` falls outside [lower, upper]."""
    raise NotImplementedError("Implemented in Stage 2.")
