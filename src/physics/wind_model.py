"""Hourly mechanistic wind-generation model (PROJECT_BRIEF.md §10).

Wind speed is measured at 10 m (`WS10M`, confirmed via the NASA POWER API
validation in `data/README.md` addendum §1.4) and adjusted to the configured
turbine hub height via the power-law relation, then passed through a
transparent, normalized cubic turbine power curve.

**Limitations** (documented per PROJECT_BRIEF.md §10, not hidden):
- Reanalysis-derived wind speed (NASA POWER blends MERRA-2 reanalysis and
  other sources) has coarse spatial resolution (~0.5-1 degree) and may not
  capture site-specific terrain effects.
- The power-law height extrapolation with a single generic shear exponent is
  an approximation; actual shear depends on terrain roughness, atmospheric
  stability, and time of day, none of which are modelled here.
- The power curve is a generic normalized cubic curve, not a specific
  manufacturer's certified power curve.
- Turbulence intensity and wake effects (irrelevant for a single turbine, but
  relevant if capacity were split across multiple units) are not modelled.
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
    if reference_height_m <= 0 or hub_height_m <= 0:
        raise ValueError("reference_height_m and hub_height_m must be positive.")
    adjusted = wind_speed * (hub_height_m / reference_height_m) ** shear_exponent
    adjusted.name = "wind_speed_hub_mps"
    return adjusted


def turbine_power_curve(
    wind_speed_hub: pd.Series,
    rated_power_kw: float,
    cut_in_mps: float,
    rated_mps: float,
    cut_out_mps: float,
) -> pd.Series:
    """Apply the normalized cubic turbine power curve:

    P(v) = 0                                                       for v < cut_in
    P(v) = P_rated * (v**3 - cut_in**3) / (rated**3 - cut_in**3)    for cut_in <= v < rated
    P(v) = P_rated                                                 for rated <= v <= cut_out
    P(v) = 0                                                       for v > cut_out
    """
    if not (0 <= cut_in_mps < rated_mps <= cut_out_mps):
        raise ValueError(
            f"Expected 0 <= cut_in < rated <= cut_out, got "
            f"cut_in={cut_in_mps}, rated={rated_mps}, cut_out={cut_out_mps}."
        )

    v = wind_speed_hub.to_numpy()
    power = np.zeros_like(v, dtype="float64")

    ramp_mask = (v >= cut_in_mps) & (v < rated_mps)
    power[ramp_mask] = rated_power_kw * (v[ramp_mask] ** 3 - cut_in_mps**3) / (rated_mps**3 - cut_in_mps**3)

    rated_mask = (v >= rated_mps) & (v <= cut_out_mps)
    power[rated_mask] = rated_power_kw

    # v > cut_out and v < cut_in remain 0.

    power = np.clip(power, 0.0, rated_power_kw)
    return pd.Series(power, index=wind_speed_hub.index, name="wind_kw")


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
    if "WS10M" not in weather.columns:
        raise ValueError("weather is missing required column: WS10M")

    wind_speed_hub = adjust_wind_speed_to_hub_height(
        weather["WS10M"], reference_height_m, hub_height_m, shear_exponent
    )
    wind_kw = turbine_power_curve(wind_speed_hub, rated_power_kw, cut_in_mps, rated_mps, cut_out_mps)

    logger.info(
        "Computed wind generation: rated=%.1f kW, annual=%.1f kWh, capacity_factor=%.3f",
        rated_power_kw, wind_kw.sum(), wind_kw.sum() / (rated_power_kw * len(wind_kw)) if rated_power_kw > 0 else 0.0,
    )
    return wind_kw
