"""Hourly mechanistic PV generation model (PROJECT_BRIEF.md §9).

Responsibilities (implemented in Stage 2, using pvlib where appropriate, after
NASA POWER solar-irradiance units are confirmed per addendum §1.4):

- Compute solar position and plane-of-array irradiance for a configurable
  tilt/azimuth.
- Apply module/cell-temperature effects, inverter/conversion efficiency, and
  configurable system losses.
- Enforce nighttime zero generation and output clipping at the configured
  system AC limit.
- Guarantee 0 <= PV output <= configured maximum PV AC output at every hour.
- If a simplified PVWatts-style model is used instead of full pvlib chain,
  document the equations, units, assumptions, and limitations inline.

This module intentionally contains no implementation yet -- it is a Stage 1
placeholder. Functions raise NotImplementedError until Stage 2.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_pv_generation(
    weather: pd.DataFrame,
    capacity_kwp: float,
    latitude: float,
    longitude: float,
    tilt_deg: float,
    azimuth_deg: float,
    system_losses_fraction: float,
    inverter_efficiency: float,
) -> pd.Series:
    """Compute hourly AC PV generation in kW for a fixed PV capacity.

    Args:
        weather: Cleaned hourly weather data (must include verified-unit
            irradiance and temperature columns).
        capacity_kwp: Fixed PV DC capacity.
        latitude: Site latitude in decimal degrees.
        longitude: Site longitude in decimal degrees.
        tilt_deg: Panel tilt angle.
        azimuth_deg: Panel azimuth angle.
        system_losses_fraction: Configurable non-inverter system loss fraction.
        inverter_efficiency: Configurable inverter/conversion efficiency.

    Returns:
        Hourly AC PV output in kW, satisfying 0 <= output <= capacity_kwp
        (adjusted for the configured AC limit, if different from DC capacity).
    """
    raise NotImplementedError("Implemented in Stage 2.")
