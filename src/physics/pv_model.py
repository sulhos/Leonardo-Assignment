"""Hourly mechanistic PV generation model (PROJECT_BRIEF.md §9).

Uses pvlib for solar position and plane-of-array (POA) irradiance. Because
NASA POWER's hourly endpoint provides GHI, DNI, and DHI directly
(`ALLSKY_SFC_SW_DWN`/`_DNI`/`_DIFF`, all in Wh/m^2 -- numerically usable as
hourly-average W/m^2, see `data/README.md` addendum §1.4 findings), no
GHI-decomposition model (Erbs/DISC/etc.) is needed; `pvlib.irradiance.get_total_irradiance`
is used directly with the measured components.

Cell temperature uses the standard NOCT (Nominal Operating Cell Temperature)
model -- a simple, transparent equation rather than a full SAM module
database lookup, since only fixed-capacity aggregate PV output is needed,
not a specific commercial module's electrical characteristics:

    T_cell = T_ambient + (NOCT - 20) / 800 * POA_global

DC power uses a linear temperature-coefficient derating around STC (25 C):

    P_dc = capacity_kwp * (POA_global / 1000) * (1 + gamma_pdc * (T_cell - 25))

**Modelling assumptions** (distinct from NASA POWER's measured data):
- `NOCT_C = 45` (typical crystalline-silicon NOCT).
- `GAMMA_PDC_PER_C = -0.004` (typical crystalline-silicon power temperature
  coefficient, -0.4%/C).
- DC:AC ratio of 1:1 -- the configured `capacity_kwp` is treated as both the
  DC nameplate and the AC system limit for output clipping.

**Limitations:** this is a fixed-tilt model with no shading, snow, soiling
schedule, or module mismatch beyond the single `system_losses_fraction`
lump-sum derate; tilt/azimuth are configured per-site modelling choices, not
manufacturer specifications.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pvlib

logger = logging.getLogger(__name__)

NOCT_C = 45.0
GAMMA_PDC_PER_C = -0.004
STC_TEMPERATURE_C = 25.0
STC_IRRADIANCE_WM2 = 1000.0
NOCT_REFERENCE_IRRADIANCE_WM2 = 800.0
NOCT_REFERENCE_AMBIENT_C = 20.0


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
        weather: Cleaned hourly weather data (local-timezone index) with
            columns `ALLSKY_SFC_SW_DWN` (GHI), `ALLSKY_SFC_SW_DNI` (DNI),
            `ALLSKY_SFC_SW_DIFF` (DHI), all Wh/m^2 (hourly, numerically
            usable as average W/m^2), and `T2M` (ambient temperature, C).
        capacity_kwp: Fixed PV DC capacity (also used as the AC output cap).
        latitude: Site latitude in decimal degrees.
        longitude: Site longitude in decimal degrees.
        tilt_deg: Panel tilt angle, degrees from horizontal.
        azimuth_deg: Panel azimuth angle, degrees from north (180 = south-facing).
        system_losses_fraction: Configurable lump-sum non-inverter system
            loss fraction (soiling, wiring, mismatch, etc.).
        inverter_efficiency: Configurable inverter/conversion efficiency.

    Returns:
        Hourly AC PV output in kW, satisfying 0 <= output <= capacity_kwp.
    """
    required_columns = {"ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DIFF", "T2M"}
    missing = required_columns - set(weather.columns)
    if missing:
        raise ValueError(f"weather is missing required columns: {missing}")

    times = weather.index
    solar_position = pvlib.solarposition.get_solarposition(times, latitude, longitude)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt_deg,
        surface_azimuth=azimuth_deg,
        solar_zenith=solar_position["apparent_zenith"],
        solar_azimuth=solar_position["azimuth"],
        dni=weather["ALLSKY_SFC_SW_DNI"],
        ghi=weather["ALLSKY_SFC_SW_DWN"],
        dhi=weather["ALLSKY_SFC_SW_DIFF"],
    )
    poa_global = poa["poa_global"].clip(lower=0.0)

    ambient_c = weather["T2M"]
    cell_temp_c = ambient_c + (NOCT_C - NOCT_REFERENCE_AMBIENT_C) / NOCT_REFERENCE_IRRADIANCE_WM2 * poa_global

    dc_kw = (
        capacity_kwp
        * (poa_global / STC_IRRADIANCE_WM2)
        * (1 + GAMMA_PDC_PER_C * (cell_temp_c - STC_TEMPERATURE_C))
    )
    dc_kw = dc_kw.clip(lower=0.0)

    ac_kw = dc_kw * (1 - system_losses_fraction) * inverter_efficiency
    ac_kw = ac_kw.clip(lower=0.0, upper=capacity_kwp)

    ac_kw.name = "pv_kw"
    logger.info(
        "Computed PV generation: capacity=%.1f kWp, annual=%.1f kWh, capacity_factor=%.3f",
        capacity_kwp, ac_kw.sum(), ac_kw.sum() / (capacity_kwp * len(ac_kw)) if capacity_kwp > 0 else 0.0,
    )
    return ac_kw
