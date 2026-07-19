"""Reproducible synthetic hourly load-profile generator (PROJECT_BRIEF.md §8).

Responsibilities (implemented in Stage 2):

- Generate an 8,760-hour local load profile with morning/evening peaks, lower
  nighttime demand, weekday/weekend differences, seasonal variation, and
  reproducible stochastic variation, given a fixed random seed.
- Support configurable annual-consumption level and peak-to-average ratio.
- Support generating multiple, meaningfully distinct load-profile "families"
  for scenario generation (Stage 4) -- not simple scalar multiples of one
  base profile.
- Save the baseline profile as CSV.

This module intentionally contains no implementation yet -- it is a Stage 1
placeholder. Functions raise NotImplementedError until Stage 2.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_load_profile(
    annual_consumption_kwh: float,
    peak_load_kw: float,
    year: int,
    timezone: str,
    random_seed: int,
    profile_family: str = "residential_baseline",
) -> pd.Series:
    """Generate one reproducible 8,760-hour local load profile.

    Args:
        annual_consumption_kwh: Target total annual energy consumption.
        peak_load_kw: Maximum allowed hourly demand.
        year: Local calendar year to generate timestamps for.
        timezone: IANA timezone name for the local index.
        random_seed: Fixed seed for reproducible stochastic variation.
        profile_family: Named pattern family (timing/seasonality/peak shape).

    Returns:
        Hourly load in kW, indexed by local timestamp, length 8,760.
    """
    raise NotImplementedError("Implemented in Stage 2.")


def generate_load_profile_family(
    n_profiles: int,
    random_seed: int,
    **kwargs,
) -> list[pd.Series]:
    """Generate multiple load profiles with meaningful variation in timing,
    seasonality, peaks, and stochastic behaviour, for ML scenario generation."""
    raise NotImplementedError("Implemented in Stage 4.")


def load_duration_curve(load: pd.Series) -> pd.Series:
    """Return the load-duration curve (load values sorted descending, reindexed
    by duration fraction)."""
    raise NotImplementedError("Implemented in Stage 2.")
