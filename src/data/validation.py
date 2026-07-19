"""Shared data-quality validation checks used across weather, load, PV, and wind
outputs (PROJECT_BRIEF.md §26).
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when a data-quality check fails."""


def validate_hourly_index(index: pd.DatetimeIndex, expected_length: int | None = None) -> None:
    """Raise if `index` is not a complete, unique, gap-free hourly index.

    Args:
        index: The DatetimeIndex to check.
        expected_length: If given, the index must have exactly this many
            entries (e.g. 8760 for a non-leap local year).
    """
    if index.has_duplicates:
        raise DataValidationError(f"Index has duplicate timestamps: {index[index.duplicated()][:5].tolist()}")

    if not index.is_monotonic_increasing:
        raise DataValidationError("Index is not sorted ascending.")

    diffs = index.to_series().diff().dropna().unique()
    if len(diffs) > 1 or (len(diffs) == 1 and pd.Timedelta(diffs[0]) != pd.Timedelta(hours=1)):
        raise DataValidationError(f"Index is not a uniform hourly series; found step sizes: {diffs}")

    if expected_length is not None and len(index) != expected_length:
        raise DataValidationError(f"Expected {expected_length} records, got {len(index)}.")


def validate_no_nans(df: pd.DataFrame, columns: list[str] | None = None) -> None:
    """Raise if any of `columns` (or all columns, if not given) contain NaNs."""
    subset = df[columns] if columns is not None else df
    nan_counts = subset.isna().sum()
    bad = nan_counts[nan_counts > 0]
    if not bad.empty:
        raise DataValidationError(f"Unexpected NaN values found: {bad.to_dict()}")


def validate_bounds(series: pd.Series, lower: float, upper: float, name: str) -> None:
    """Raise if any value in `series` falls outside [lower, upper]."""
    below = series < lower
    above = series > upper
    if below.any() or above.any():
        raise DataValidationError(
            f"{name} has values outside [{lower}, {upper}]: "
            f"min={series.min()}, max={series.max()}"
        )
