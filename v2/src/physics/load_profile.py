"""Reproducible synthetic hourly load-profile generator (PROJECT_BRIEF.md §8).

Builds an 8,760-hour (non-leap local year) load profile from:
- a 24-hour base shape (family-specific -- see `_PROFILE_FAMILIES`),
- a weekday/weekend multiplier (family-specific),
- a seasonal multiplier (elevated in winter and summer, trough in
  spring/autumn -- a combined heating+cooling proxy; amplitude is
  family-specific),
- reproducible multiplicative noise (fixed `random_seed`),
then scales the result so total annual energy matches `annual_consumption_kwh`
exactly (hourly kW values are numerically equal to hourly kWh energy).

Two profile families are implemented: "residential_baseline" (sharp
morning/evening peaks, deep overnight trough, mild weekend dip, strong
seasonality) and "industrial_baseline" (flatter two-shift day, substantial
weekend reduction, weak seasonality -- added for the industrial-scale pivot,
PROJECT_BRIEF.md Addendum 2).

`generate_load_profile_family` (multiple, meaningfully distinct profiles
*within* one family for ML scenario generation) is a Stage 4 concern and is
not implemented here.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 24-hour base shape (relative units, un-normalized): a smooth double-peak
# curve with a morning peak around 07:00-09:00, an evening peak around
# 18:00-21:00, and a nighttime trough. Values are illustrative, not derived
# from a specific dataset -- documented here as a modelling assumption.
_RESIDENTIAL_HOUR_OF_DAY_WEIGHTS = np.array(
    [
        0.35, 0.30, 0.28, 0.28, 0.32, 0.45,  # 00-05: overnight trough
        0.70, 0.95, 1.00, 0.80, 0.65, 0.60,  # 06-11: morning peak & taper
        0.62, 0.60, 0.58, 0.60, 0.68, 0.85,  # 12-17: midday plateau, pre-evening rise
        1.05, 1.10, 1.00, 0.85, 0.65, 0.45,  # 18-23: evening peak & taper
    ]
)

# 24-hour base shape for a two-shift industrial facility: a much higher
# baseline (equipment/lighting/HVAC running even overnight) and a flatter
# day, rather than sharp residential-style peaks. Illustrative, not derived
# from measured data -- documented here as a modelling assumption (added for
# the industrial-scale pivot; see PROJECT_BRIEF.md Addendum 2).
_INDUSTRIAL_HOUR_OF_DAY_WEIGHTS = np.array(
    [
        0.55, 0.52, 0.50, 0.50, 0.52, 0.58,  # 00-05: reduced overnight baseline (not near-zero)
        0.75, 0.92, 1.00, 1.00, 1.00, 1.00,  # 06-11: shift 1 ramp-up & run
        1.00, 0.98, 0.95, 0.98, 1.00, 1.00,  # 12-17: shift 2 run
        0.98, 0.95, 0.90, 0.85, 0.75, 0.65,  # 18-23: shift 2 taper toward night baseline
    ]
)

# Per-profile-family shape parameters. `weekend_multiplier` reflects each
# family's typical weekend operating pattern: residential demand dips only
# mildly on weekends, while a shift-based industrial facility (the pattern
# assumed here; continuous-process or weekdays-only facilities would use a
# different value) drops substantially to a skeleton-crew/maintenance level.
# `seasonal_amplitude` is deliberately much smaller for industrial: process
# loads are largely weather-insensitive, unlike residential heating/cooling.
_PROFILE_FAMILIES: dict[str, dict] = {
    "residential_baseline": {
        "hour_of_day_weights": _RESIDENTIAL_HOUR_OF_DAY_WEIGHTS,
        "weekend_multiplier": 0.90,
        "weekend_morning_shift_hours": 1.5,
        "seasonal_amplitude": 0.15,
        "noise_std_fraction": 0.06,
    },
    "industrial_baseline": {
        "hour_of_day_weights": _INDUSTRIAL_HOUR_OF_DAY_WEIGHTS,
        "weekend_multiplier": 0.45,  # shift-based facility, skeleton weekend crew
        "weekend_morning_shift_hours": 0.0,  # no residential-style peak shift to model
        "seasonal_amplitude": 0.05,
        "noise_std_fraction": 0.04,  # aggregated industrial load is smoother than a single household
    },
}


def _seasonal_multiplier(day_of_year: np.ndarray, days_in_year: int, amplitude: float) -> np.ndarray:
    """Elevated in winter and summer, trough in spring/autumn -- combined
    heating+cooling proxy. Modelling assumption, not derived from measured
    data; documented for the technical report's assumptions section."""
    angle = 2 * np.pi * (day_of_year - 1) / days_in_year
    # cos(2*angle) peaks near day 0 (winter) and mid-year (summer), troughs
    # at the equinox-ish quarter points.
    return 1.0 + amplitude * np.cos(2 * angle)


def _weekday_weekend_multiplier(dow: np.ndarray, weekend_multiplier: float) -> np.ndarray:
    """dow: 0=Monday .. 6=Sunday. Weekends get a flat multiplier reduction."""
    is_weekend = dow >= 5
    return np.where(is_weekend, weekend_multiplier, 1.0)


def _base_hourly_shape(
    hour: np.ndarray, dow: np.ndarray, hour_of_day_weights: np.ndarray, weekend_morning_shift_hours: float,
) -> np.ndarray:
    """24-hour base shape, with the weekend morning peak shifted later
    (residential only; `weekend_morning_shift_hours=0` disables this)."""
    weekend_shifted_hour = (hour - weekend_morning_shift_hours) % 24
    is_weekend = dow >= 5
    effective_hour = np.where(is_weekend, weekend_shifted_hour, hour)
    # Linear interpolation across the 24 discrete weights for a smoother curve.
    return np.interp(effective_hour, np.arange(24), hour_of_day_weights, period=24)


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
        peak_load_kw: Maximum allowed hourly demand; the generated profile is
            checked against this and raises if it cannot be satisfied without
            distorting the annual-energy target (i.e. the target combination
            is not achievable with this generator's shape).
        year: Local calendar year to generate timestamps for.
        timezone: IANA timezone name for the local index.
        random_seed: Fixed seed for reproducible stochastic variation.
        profile_family: Named pattern family -- "residential_baseline"
            (sharp morning/evening peaks, deep overnight trough, mild
            weekend dip, strong heating/cooling seasonality) or
            "industrial_baseline" (flatter two-shift day, substantial
            weekend reduction, weak seasonality; see PROJECT_BRIEF.md
            Addendum 2).

    Returns:
        Hourly load in kW, indexed by local timestamp, length 8,760
        (non-leap year) or 8,784 (leap year).
    """
    if profile_family not in _PROFILE_FAMILIES:
        raise NotImplementedError(
            f"profile_family={profile_family!r} not implemented; "
            f"available families: {sorted(_PROFILE_FAMILIES)}."
        )
    family = _PROFILE_FAMILIES[profile_family]

    index = pd.date_range(
        start=pd.Timestamp(year=year, month=1, day=1, tz=timezone),
        end=pd.Timestamp(year=year, month=12, day=31, hour=23, tz=timezone),
        freq="h",
    )

    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    day_of_year = index.dayofyear.to_numpy()
    days_in_year = 366 if index.is_leap_year[0] else 365

    base = _base_hourly_shape(hour, dow, family["hour_of_day_weights"], family["weekend_morning_shift_hours"])
    weekday_mult = _weekday_weekend_multiplier(dow, family["weekend_multiplier"])
    seasonal_mult = _seasonal_multiplier(day_of_year, days_in_year, family["seasonal_amplitude"])

    rng = np.random.default_rng(random_seed)
    noise = rng.normal(loc=1.0, scale=family["noise_std_fraction"], size=len(index))
    noise = np.clip(noise, 0.5, 1.5)  # bound extreme noise draws

    relative_demand = base * weekday_mult * seasonal_mult * noise
    relative_demand = np.clip(relative_demand, a_min=1e-6, a_max=None)  # no negative/zero demand

    scale = annual_consumption_kwh / relative_demand.sum()
    load_kw = relative_demand * scale

    achieved_peak = load_kw.max()
    # Relative tolerance guards against float64 rounding noise when the same
    # (seed, annual_consumption_kwh) is regenerated more than once and compared
    # against a peak_load_kw sourced from a prior run's own achieved-peak
    # feature (see src/ai/physical_verification.py's docstring) -- this is a
    # numerical-hygiene safety net, not a loosening of the actual constraint.
    if achieved_peak > peak_load_kw * (1 + 1e-9):
        raise ValueError(
            f"Generated peak load {achieved_peak:.2f} kW exceeds the requested "
            f"peak_load_kw={peak_load_kw:.2f} for annual_consumption_kwh="
            f"{annual_consumption_kwh:.0f}. This combination is not achievable "
            "with this generator's shape (peak-to-average ratio too low for the "
            "requested peak); adjust the scenario inputs."
        )

    series = pd.Series(load_kw, index=index, name="load_kw")
    logger.info(
        "Generated load profile: annual=%.1f kWh, peak=%.2f kW, mean=%.3f kW, n=%d",
        series.sum(), series.max(), series.mean(), len(series),
    )
    return series


def generate_load_profile_family(
    n_profiles: int,
    random_seed: int,
    **kwargs,
) -> list[pd.Series]:
    """Generate multiple load profiles with meaningful variation in timing,
    seasonality, peaks, and stochastic behaviour, for ML scenario generation."""
    raise NotImplementedError("Implemented in Stage 4.")


def load_duration_curve(load: pd.Series) -> pd.Series:
    """Return the load-duration curve: load values sorted descending,
    reindexed by duration fraction (0 = start of year, 1 = full year)."""
    sorted_values = load.sort_values(ascending=False).reset_index(drop=True)
    duration_fraction = (sorted_values.index + 1) / len(sorted_values)
    sorted_values.index = duration_fraction
    sorted_values.index.name = "duration_fraction"
    return sorted_values
