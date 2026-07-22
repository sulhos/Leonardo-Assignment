"""Hourly mechanistic wind-generation model, audited against the course
lecture's exact equations (Campana, *Deploying LLMs for the Management of
Microgrids*, OptiCE slide deck).

Lecture equations implemented here (Task 1b):

1. Height extrapolation, power-law profile (primary, `adjust_wind_speed_to_hub_height`):
   `v(z) = v(z_r) * (z / z_r)^alpha`, with `alpha` a config-driven shear
   exponent (`wind_shear_exponent`, generic open-terrain default 0.14, per
   the lecture's `alpha ~= 0.143`).
2. Log-law alternative (`adjust_wind_speed_to_hub_height_log_law`):
   `v(z2) = v(z1) * ln((z2 - d)/z0) / ln((z1 - d)/z0)`, implemented and
   tested but not used by the main pipeline (see function docstring for why).
3. Turbine output via **interpolation on a manufacturer-style power curve**
   (`turbine_power_curve`): `P_wind(t) = P_curve(v_hub(t))`, evaluated by
   linear interpolation (`numpy.interp`, equivalent to MATLAB's `interp1`
   default) on a `[speed_mps, power_fraction]` table loaded from config
   (`wind.power_curve_speed_power_fraction`), not a closed-form cubic ramp
   formula. This project's power curve table is a discretized (0.5 m/s
   resolution) sampling of a generic cubic ramp between `cut_in_mps` and
   `rated_mps` -- a documented generic curve shape, not a specific
   certified manufacturer curve -- switched here from a closed-form cubic
   evaluation to the lecture's table-interpolation evaluation mechanism.

**Limitations** (documented per PROJECT_BRIEF.md §10, not hidden):
- Reanalysis-derived wind speed (NASA POWER blends MERRA-2 reanalysis and
  other sources) has coarse spatial resolution (~0.5-1 degree) and may not
  capture site-specific terrain effects.
- The power-law height extrapolation with a single generic shear exponent is
  an approximation; actual shear depends on terrain roughness, atmospheric
  stability, and time of day, none of which are modelled here.
- The power curve is a generic normalized curve, not a specific
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
    """Power-law wind-shear adjustment (lecture eq., Task 1b, primary method):
    v_hub = v_ref * (hub_height / reference_height) ** shear_exponent
    """
    if reference_height_m <= 0 or hub_height_m <= 0:
        raise ValueError("reference_height_m and hub_height_m must be positive.")
    adjusted = wind_speed * (hub_height_m / reference_height_m) ** shear_exponent
    adjusted.name = "wind_speed_hub_mps"
    return adjusted


def adjust_wind_speed_to_hub_height_log_law(
    wind_speed: pd.Series,
    reference_height_m: float,
    hub_height_m: float,
    roughness_length_m: float,
    zero_plane_displacement_m: float = 0.0,
) -> pd.Series:
    """Logarithmic-profile wind-shear adjustment (lecture eq., Task 1b,
    documented alternative to the power law):
    v(z2) = v(z1) * ln((z2 - d)/z0) / ln((z1 - d)/z0)

    Not used by the main pipeline: it requires a site-specific roughness
    length `z0` (typically 0.03-1.0 m depending on terrain) that NASA
    POWER's reanalysis product does not provide and this project has not
    independently characterized for Jinan, whereas the power-law exponent
    is a single widely-tabulated generic value (`alpha ~= 0.14` for open
    terrain) requiring no additional site survey. Implemented and tested
    for completeness/lecture fidelity.
    """
    if reference_height_m <= zero_plane_displacement_m or hub_height_m <= zero_plane_displacement_m:
        raise ValueError("reference_height_m and hub_height_m must exceed zero_plane_displacement_m.")
    if roughness_length_m <= 0:
        raise ValueError("roughness_length_m must be positive.")

    log_ratio = np.log((hub_height_m - zero_plane_displacement_m) / roughness_length_m) / np.log(
        (reference_height_m - zero_plane_displacement_m) / roughness_length_m
    )
    adjusted = wind_speed * log_ratio
    adjusted.name = "wind_speed_hub_mps"
    return adjusted


def turbine_power_curve(
    wind_speed_hub: pd.Series,
    rated_power_kw: float,
    power_curve_speed_power_fraction: list[list[float]],
) -> pd.Series:
    """Turbine output via linear interpolation on a manufacturer-style power
    curve (lecture eq., Task 1b): `P_wind(t) = P_rated * interp1(v_hub(t))`.

    Args:
        wind_speed_hub: Hourly hub-height wind speed, m/s.
        rated_power_kw: Turbine rated power, kW.
        power_curve_speed_power_fraction: `[speed_mps, power_fraction]`
            pairs (config-driven, `wind.power_curve_speed_power_fraction`),
            sorted ascending by speed, power_fraction in `[0, 1]`. Outside
            the table's speed range, `numpy.interp` holds the boundary
            value constant (matching the table's own leading/trailing
            zero-power entries at 0 m/s and well above cut-out).
    """
    curve = np.asarray(power_curve_speed_power_fraction, dtype=float)
    if curve.ndim != 2 or curve.shape[1] != 2:
        raise ValueError("power_curve_speed_power_fraction must be a list of [speed, power_fraction] pairs.")
    speeds, fractions = curve[:, 0], curve[:, 1]
    if not np.all(np.diff(speeds) >= 0):
        raise ValueError("power_curve_speed_power_fraction speeds must be sorted ascending.")
    if np.any((fractions < 0) | (fractions > 1)):
        raise ValueError("power_curve_speed_power_fraction power_fraction values must be in [0, 1].")

    v = wind_speed_hub.to_numpy()
    power_fraction = np.interp(v, speeds, fractions)
    power = rated_power_kw * power_fraction
    power = np.clip(power, 0.0, rated_power_kw)
    return pd.Series(power, index=wind_speed_hub.index, name="wind_kw")


def compute_wind_generation(
    weather: pd.DataFrame,
    rated_power_kw: float,
    reference_height_m: float,
    hub_height_m: float,
    shear_exponent: float,
    power_curve_speed_power_fraction: list[list[float]],
) -> pd.Series:
    """High-level entry point: hourly wind generation in kW for a fixed
    turbine rated power, satisfying 0 <= output <= rated_power_kw."""
    if "WS10M" not in weather.columns:
        raise ValueError("weather is missing required column: WS10M")

    wind_speed_hub = adjust_wind_speed_to_hub_height(
        weather["WS10M"], reference_height_m, hub_height_m, shear_exponent
    )
    wind_kw = turbine_power_curve(wind_speed_hub, rated_power_kw, power_curve_speed_power_fraction)

    logger.info(
        "Computed wind generation: rated=%.1f kW, annual=%.1f kWh, capacity_factor=%.3f",
        rated_power_kw, wind_kw.sum(), wind_kw.sum() / (rated_power_kw * len(wind_kw)) if rated_power_kw > 0 else 0.0,
    )
    return wind_kw
