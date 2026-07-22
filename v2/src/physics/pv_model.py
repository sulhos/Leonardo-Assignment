"""Hourly mechanistic PV generation model, audited against the course
lecture's exact equations (Campana, *Deploying LLMs for the Management of
Microgrids*, OptiCE slide deck).

Lecture equations implemented here:

1. Clearness index and extraterrestrial irradiance (`clearness_index`,
   `extraterrestrial_ghi`): `k_t = GHI / EGHI`,
   `EGHI = G_sc * (1 + 0.033*cos(2*pi*n_H/365.25)) * sin(alpha_s)`.
2. Erbs decomposition (`erbs_decomposition`): estimates DHI from GHI alone
   via the standard three-piece cubic in `k_t`.
3. Liu & Jordan isotropic-sky transposition to the tilted plane
   (`liu_jordan_transposition`):
   `GTI = R_b*(GHI - DHI) + DHI*(1 + cos(beta))/2 + rho_g*GHI*(1 - cos(beta))/2`,
   `R_b = cos(theta) / cos(90 - alpha_s)`.
4. NOCT cell temperature, linear temperature derating, and the incidence
   angle modifier (`compute_pv_generation`):
   `T_c = T_a + (NOCT - 20)/800 * GTI`,
   `eta_PV(T) = eta_PV_ref * (1 - beta_ref*(T_c - 25))`,
   `IAM = 1 - 0.05*(1/cos(theta) - 1)`,
   `P_PV = eta_PV(T) * A_PV * GTI * IAM`.

**Main pipeline uses measured DNI/DHI, not Erbs.** NASA POWER's hourly
endpoint provides GHI, DNI, and DHI independently
(`ALLSKY_SFC_SW_DWN`/`_DNI`/`_DIFF`, all Wh/m^2 -- numerically usable as
hourly-average W/m^2, see `data/README.md` addendum §1.4), so
`compute_pv_generation` transposes using the measured DNI/DHI directly via
`pvlib.irradiance.get_total_irradiance(..., model="isotropic")`, rather
than decomposing DHI from GHI via Erbs first (Task 1a's `erbs_decomposition`/
`clearness_index`/`extraterrestrial_ghi` are implemented and tested
standalone, for a GHI-only weather source, but are not called by the main
pipeline).

**Verified numerically that pvlib's isotropic model reproduces the
lecture's Liu & Jordan transposition exactly**, component by component
(sky-diffuse `DHI*(1+cos(beta))/2` and ground-reflected
`rho_g*GHI*(1-cos(beta))/2` match pvlib's `poa_sky_diffuse`/
`poa_ground_diffuse` to floating-point precision; the beam term matches
`DNI*cos(theta)` exactly, which is what `R_b*(GHI-DHI)` computes *if*
`GHI - DHI == DNI*cos(zenith)` exactly). On real NASA POWER data, that
closure does not hold exactly (residuals up to several tens of W/m^2,
inherent to independently-measured/modelled GHI/DNI/DHI channels), so using
the directly measured DNI for the beam term -- what pvlib does -- is more
accurate than re-deriving beam irradiance from `GHI - DHI`, which is only
necessary when DNI itself is unavailable and must be inferred. Both
formulas are the same Liu & Jordan isotropic model; they differ only in
which measured channel supplies the beam term. See
`tests/test_pv_model.py::test_pvlib_isotropic_matches_lecture_liu_jordan_formula_componentwise`.

**Modelling assumptions** (distinct from NASA POWER's measured data):
- `NOCT_C = 45` (typical crystalline-silicon NOCT).
- `BETA_REF_PER_C = 0.004` (typical crystalline-silicon power temperature
  coefficient magnitude, 0.4%/C loss above 25C -- the lecture's `beta_ref`).
- DC:AC ratio of 1:1 -- the configured `capacity_kwp` is treated as both the
  DC nameplate and the AC system limit for output clipping.
- `GROUND_ALBEDO = 0.25` (pvlib's own default, a generic mid-range ground
  reflectance -- the lecture's `rho_g`).

**Limitations:** this is a fixed-tilt model with no shading, snow, soiling
schedule, or module mismatch beyond the single `system_losses_fraction`
lump-sum derate; tilt/azimuth are configured per-site modelling choices, not
manufacturer specifications. The IAM formula is the lecture's simple linear
form, applied to total POA irradiance (not split between beam/diffuse
components as some IAM treatments do), clipped to `[0, 1]`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pvlib

logger = logging.getLogger(__name__)

NOCT_C = 45.0
BETA_REF_PER_C = 0.004
STC_TEMPERATURE_C = 25.0
STC_IRRADIANCE_WM2 = 1000.0
NOCT_REFERENCE_IRRADIANCE_WM2 = 800.0
NOCT_REFERENCE_AMBIENT_C = 20.0
GROUND_ALBEDO = 0.25
SOLAR_CONSTANT_WM2 = 1367.0
IAM_COEFFICIENT = 0.05


def extraterrestrial_ghi(day_of_year: np.ndarray | float, solar_elevation_deg: np.ndarray | float) -> np.ndarray:
    """Extraterrestrial horizontal irradiance (lecture eq., Task 1a):
    `EGHI = G_sc * (1 + 0.033*cos(2*pi*n_H/365.25)) * sin(alpha_s)`,
    clipped to non-negative (sun below the horizon -> zero, not negative)."""
    n_h = np.asarray(day_of_year, dtype=float)
    elevation_rad = np.radians(np.asarray(solar_elevation_deg, dtype=float))
    eccentricity_factor = 1 + 0.033 * np.cos(2 * np.pi * n_h / 365.25)
    eghi = SOLAR_CONSTANT_WM2 * eccentricity_factor * np.sin(elevation_rad)
    return np.clip(eghi, 0.0, None)


def clearness_index(ghi: np.ndarray, eghi: np.ndarray) -> np.ndarray:
    """Clearness index `k_t = GHI / EGHI` (lecture eq., Task 1a). Zero
    wherever EGHI is zero (night), rather than dividing by zero."""
    ghi_arr = np.asarray(ghi, dtype=float)
    eghi_arr = np.asarray(eghi, dtype=float)
    return np.divide(ghi_arr, eghi_arr, out=np.zeros_like(ghi_arr, dtype=float), where=eghi_arr > 1e-6)


def erbs_decomposition(ghi: np.ndarray, k_t: np.ndarray) -> np.ndarray:
    """Erbs decomposition, estimating DHI from GHI alone via the clearness
    index (lecture eq., Task 1a) -- for a GHI-only weather source; the main
    pipeline uses NASA POWER's measured DHI directly instead (see module
    docstring).

        DHI = (1 - 0.09*k_t) * GHI                                              for 0 < k_t <= 0.22
        DHI = (0.9511 - 0.1604*k_t + 4.388*k_t^2 - 16.638*k_t^3 + 12.336*k_t^4) * GHI  for 0.22 < k_t <= 0.8
        DHI = 0.165 * GHI                                                       for k_t > 0.8
    """
    ghi_arr = np.asarray(ghi, dtype=float)
    k_t_arr = np.asarray(k_t, dtype=float)

    low = k_t_arr <= 0.22
    mid = (k_t_arr > 0.22) & (k_t_arr <= 0.8)
    high = k_t_arr > 0.8

    diffuse_fraction = np.zeros_like(k_t_arr)
    diffuse_fraction[low] = 1 - 0.09 * k_t_arr[low]
    kt_mid = k_t_arr[mid]
    diffuse_fraction[mid] = (
        0.9511 - 0.1604 * kt_mid + 4.388 * kt_mid**2 - 16.638 * kt_mid**3 + 12.336 * kt_mid**4
    )
    diffuse_fraction[high] = 0.165

    dhi = diffuse_fraction * ghi_arr
    return np.clip(dhi, 0.0, ghi_arr)


def liu_jordan_transposition(
    ghi: np.ndarray, dhi: np.ndarray, aoi_deg: np.ndarray, zenith_deg: np.ndarray,
    tilt_deg: float, ground_albedo: float = GROUND_ALBEDO,
) -> np.ndarray:
    """Liu & Jordan isotropic-sky transposition to the tilted plane
    (lecture eq., Task 1a):

        GTI = R_b*(GHI - DHI) + DHI*(1 + cos(beta))/2 + rho_g*GHI*(1 - cos(beta))/2
        R_b = cos(theta) / cos(90 - alpha_s)

    where `theta` is the angle of incidence and `90 - alpha_s` is the solar
    zenith angle (`alpha_s` = solar elevation), so `R_b = cos(theta)/cos(zenith)`.
    `R_b` is clipped to non-negative (sun behind the panel plane contributes
    no beam component). Requires `GHI - DHI` to be the true horizontal beam
    component (i.e. self-consistent GHI/DHI/DNI); see module docstring for
    why the main pipeline uses `pvlib.irradiance.get_total_irradiance`
    with measured DNI instead.
    """
    ghi_arr = np.asarray(ghi, dtype=float)
    dhi_arr = np.asarray(dhi, dtype=float)
    aoi_rad = np.radians(np.asarray(aoi_deg, dtype=float))
    zenith_rad = np.radians(np.asarray(zenith_deg, dtype=float))
    tilt_rad = np.radians(tilt_deg)

    cos_zenith = np.cos(zenith_rad)
    r_b = np.divide(
        np.cos(aoi_rad), cos_zenith, out=np.zeros_like(cos_zenith, dtype=float), where=cos_zenith > 1e-6,
    )
    r_b = np.clip(r_b, 0.0, None)

    beam = r_b * (ghi_arr - dhi_arr)
    sky_diffuse = dhi_arr * (1 + np.cos(tilt_rad)) / 2
    ground_diffuse = ground_albedo * ghi_arr * (1 - np.cos(tilt_rad)) / 2
    gti = beam + sky_diffuse + ground_diffuse
    return np.clip(gti, 0.0, None)


def incidence_angle_modifier(aoi_deg: np.ndarray) -> np.ndarray:
    """Incidence angle modifier (lecture eq., Task 1a):
    `IAM = 1 - 0.05*(1/cos(theta) - 1)`, clipped to `[0, 1]` (grazing
    incidence would otherwise drive IAM sharply negative, which is
    unphysical -- no panel gains transmittance beyond its normal-incidence
    value)."""
    aoi_rad = np.radians(np.asarray(aoi_deg, dtype=float))
    cos_aoi = np.cos(aoi_rad)
    inv_cos = np.divide(1.0, cos_aoi, out=np.full_like(cos_aoi, np.inf), where=cos_aoi > 1e-6)
    iam = 1 - IAM_COEFFICIENT * (inv_cos - 1)
    return np.clip(iam, 0.0, 1.0)


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
    aoi_deg = pvlib.irradiance.aoi(
        tilt_deg, azimuth_deg, solar_position["apparent_zenith"], solar_position["azimuth"],
    )

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt_deg,
        surface_azimuth=azimuth_deg,
        solar_zenith=solar_position["apparent_zenith"],
        solar_azimuth=solar_position["azimuth"],
        dni=weather["ALLSKY_SFC_SW_DNI"],
        ghi=weather["ALLSKY_SFC_SW_DWN"],
        dhi=weather["ALLSKY_SFC_SW_DIFF"],
        albedo=GROUND_ALBEDO,
    )
    poa_global = poa["poa_global"].clip(lower=0.0)

    ambient_c = weather["T2M"]
    cell_temp_c = ambient_c + (NOCT_C - NOCT_REFERENCE_AMBIENT_C) / NOCT_REFERENCE_IRRADIANCE_WM2 * poa_global

    iam = pd.Series(incidence_angle_modifier(aoi_deg.values), index=times)

    dc_kw = (
        capacity_kwp
        * (poa_global / STC_IRRADIANCE_WM2)
        * (1 - BETA_REF_PER_C * (cell_temp_c - STC_TEMPERATURE_C))
        * iam
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
