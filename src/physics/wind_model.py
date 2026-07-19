"""Hourly mechanistic wind-generation model (PROJECT_BRIEF.md §10).

Responsibilities (implemented in Stage 2, after the NASA POWER wind reference
height is confirmed per addendum §1.4):

- Adjust wind speed from the confirmed API reference height to the configurable
  turbine hub height via the power-law relation with a configurable shear
  exponent.
- Apply a transparent, normalized turbine power curve with configurable
  cut-in, rated, and cut-out wind speeds.
- Enforce 0 <= output <= rated power at every hour.
- Document the limitations of reanalysis wind data, spatial resolution, height
  extrapolation, generic power curves, and missing turbulence/wake effects.

This module intentionally contains no implementation yet -- it is a Stage 1
placeholder. Functions raise NotImplementedError until Stage 2.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def adjust_wind_speed_to_hub_height(
    wind_speed: pd.Series,
    reference_height_m: float,
    hub_height_m: float,
    shear_exponent: float,
) -> pd.Series:
    """Apply the power-law wind-shear adjustment:
    v_hub = v_ref * (hub_height / reference_height) ** shear_exponent
    """
    raise NotImplementedError("Implemented in Stage 2.")


def turbine_power_curve(
    wind_speed_hub: pd.Series,
    rated_power_kw: float,
    cut_in_mps: float,
    rated_mps: float,
    cut_out_mps: float,
) -> pd.Series:
    """Apply the normalized cubic turbine power curve:

    P(v) = 0                                              for v < cut_in
    P(v) = P_rated * (v**3 - cut_in**3) / (rated**3 - cut_in**3)  for cut_in <= v < rated
    P(v) = P_rated                                        for rated <= v <= cut_out
    P(v) = 0                                              for v > cut_out
    """
    raise NotImplementedError("Implemented in Stage 2.")


def compute_wind_generation(
    weather: pd.DataFrame,
    rated_power_kw: float,
    reference_height_m: float,
    hub_height_m: float,
    shear_exponent: float,
    cut_in_mps: float,
    rated_mps: float,
    cut_out_mps: float,
) -> pd.Series:
    """High-level entry point: hourly wind generation in kW for a fixed
    turbine rated power, satisfying 0 <= output <= rated_power_kw."""
    raise NotImplementedError("Implemented in Stage 2.")
