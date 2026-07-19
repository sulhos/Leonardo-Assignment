"""Reproducible synthetic hourly load-profile generator (PROJECT_BRIEF.md §8).

Builds an 8,760-hour (non-leap local year) load profile from:
- a 24-hour base shape with morning and evening peaks and lower nighttime
  demand,
- a weekday/weekend multiplier,
- a seasonal multiplier (elevated in winter and summer, trough in
  spring/autumn -- a combined heating+cooling proxy),
- reproducible multiplicative noise (fixed `random_seed`),
then scales the result so total annual energy matches `annual_consumption_kwh`
exactly (hourly kW values are numerically equal to hourly kWh energy).

`generate_load_profile_family` (multiple, meaningfully distinct profiles for
ML scenario generation) is a Stage 4 concern and is not implemented here.
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
_HOUR_OF_DAY_WEIGHTS = np.array(
    [
        0.35, 0.30, 0.28, 0.28, 0.32, 0.45,  # 00-05: overnight trough
        0.70, 0.95, 1.00, 0.80, 0.65, 0.60,  # 06-11: morning peak & taper
        0.62, 0.60, 0.58, 0.60, 0.68, 0.85,  # 12-17: midday plateau, pre-evening rise
        1.05, 1.10, 1.00, 0.85, 0.65, 0.45,  # 18-23: evening peak & taper
    ]
)

_WEEKEND_MULTIPLIER = 0.90  # weekends: flatter, slightly lower overall demand
_WEEKEND_MORNING_SHIFT_HOURS = 1.5  # weekend morning peak shifts later

_NOISE_STD_FRACTION = 0.06  # reproducible multiplicative hourly noise, std as a fraction of the base weight


def _seasonal_multiplier(day_of_year: np.ndarray, days_in_year: int) -> np.ndarray:
    """Elevated in winter and summer, trough in spring/autumn -- combined
    heating+cooling proxy. Modelling assumption, not derived from measured
    data; documented for the technical report's assumptions section."""
    angle = 2 * np.pi * (day_of_year - 1) / days_in_year
    # cos(2*angle) peaks near day 0 (winter) and mid-year (summer), troughs
    # at the equinox-ish quarter points.
    return 1.0 + 0.15 * np.cos(2 * angle)


def _weekday_weekend_multiplier(dow: np.ndarray) -> np.ndarray:
    """dow: 0=Monday .. 6=Sunday. Weekends get a flat multiplier reduction."""
    is_weekend = dow >= 5
    return np.where(is_weekend, _WEEKEND_MULTIPLIER, 1.0)


def _base_hourly_shape(hour: np.ndarray, dow: np.ndarray) -> np.ndarray:
    """24-hour base shape, with the weekend morning peak shifted later."""
    weekend_shifted_hour = (hour - _WEEKEND_MORNING_SHIFT_HOURS) % 24
    is_weekend = dow >= 5
    effective_hour = np.where(is_weekend, weekend_shifted_hour, hour)
    # Linear interpolation across the 24 discrete weights for a smoother curve.
    return np.interp(effective_hour, np.arange(24), _HOUR_OF_DAY_WEIGHTS, period=24)


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
        profile_family: Named pattern family; currently only
            "residential_baseline" is implemented.

    Returns:
        Hourly load in kW, indexed by local timestamp, length 8,760
        (non-leap year) or 8,784 (leap year).
    """
    if profile_family != "residential_baseline":
        raise NotImplementedError(
            f"profile_family={profile_family!r} not implemented; "
            "multiple families are a Stage 4 (scenario generation) concern."
        )

    index = pd.date_range(
        start=pd.Timestamp(year=year, month=1, day=1, tz=timezone),
        end=pd.Timestamp(year=year, month=12, day=31, hour=23, tz=timezone),
        freq="h",
    )

    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    day_of_year = index.dayofyear.to_numpy()
    days_in_year = 366 if index.is_leap_year[0] else 365

    base = _base_hourly_shape(hour, dow)
    weekday_mult = _weekday_weekend_multiplier(dow)
    seasonal_mult = _seasonal_multiplier(day_of_year, days_in_year)

    rng = np.random.default_rng(random_seed)
    noise = rng.normal(loc=1.0, scale=_NOISE_STD_FRACTION, size=len(index))
    noise = np.clip(noise, 0.5, 1.5)  # bound extreme noise draws

    relative_demand = base * weekday_mult * seasonal_mult * noise
    relative_demand = np.clip(relative_demand, a_min=1e-6, a_max=None)  # no negative/zero demand

    scale = annual_consumption_kwh / relative_demand.sum()
    load_kw = relative_demand * scale

    achieved_peak = load_kw.max()
    if achieved_peak > peak_load_kw:
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
