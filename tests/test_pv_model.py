"""Tests for `src.physics.pv_model` (PROJECT_BRIEF.md §26).

Uses a small synthetic (not downloaded) weather fixture: a single local day
at the Jinan coordinates with a symmetric daytime GHI/DNI/DHI bell curve and
zero irradiance at night, so tests do not require internet access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.physics.pv_model import compute_pv_generation

LATITUDE = 36.65
LONGITUDE = 117.12
TILT_DEG = 36.65
AZIMUTH_DEG = 180.0


def _synthetic_day_weather() -> pd.DataFrame:
    index = pd.date_range("2023-06-15 00:00", periods=24, freq="h", tz="Asia/Shanghai")
    hour = index.hour.to_numpy()
    # Daylight roughly 05:00-19:00 at this latitude in June; bell-shaped GHI peaking at noon.
    daylight = (hour >= 5) & (hour <= 19)
    ghi = np.where(daylight, 800 * np.sin(np.pi * (hour - 5) / 14), 0.0)
    ghi = np.clip(ghi, 0, None)
    dni = ghi * 0.75  # rough direct fraction, illustrative only
    dhi = ghi * 0.30
    temp = 25.0 + 5 * np.sin(np.pi * (hour - 5) / 14)
    temp = np.where(daylight, temp, 20.0)
    return pd.DataFrame(
        {
            "ALLSKY_SFC_SW_DWN": ghi,
            "ALLSKY_SFC_SW_DNI": dni,
            "ALLSKY_SFC_SW_DIFF": dhi,
            "T2M": temp,
            "WS10M": 3.0,
        },
        index=index,
    )


def _compute(weather: pd.DataFrame, capacity_kwp: float = 20.0) -> pd.Series:
    return compute_pv_generation(
        weather=weather,
        capacity_kwp=capacity_kwp,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        tilt_deg=TILT_DEG,
        azimuth_deg=AZIMUTH_DEG,
        system_losses_fraction=0.14,
        inverter_efficiency=0.96,
    )


def test_no_negative_pv_generation() -> None:
    pv = _compute(_synthetic_day_weather())
    assert (pv >= 0).all()


def test_pv_zero_at_night() -> None:
    weather = _synthetic_day_weather()
    pv = _compute(weather)
    night_mask = weather["ALLSKY_SFC_SW_DWN"] == 0
    assert night_mask.sum() > 0
    assert (pv[night_mask] == 0).all()


def test_pv_never_exceeds_capacity() -> None:
    capacity_kwp = 20.0
    pv = _compute(_synthetic_day_weather(), capacity_kwp=capacity_kwp)
    assert (pv <= capacity_kwp).all()


def test_pv_positive_during_peak_daylight() -> None:
    pv = _compute(_synthetic_day_weather())
    assert pv.loc[pv.index.hour == 12].iloc[0] > 0


def test_missing_required_column_raises() -> None:
    weather = _synthetic_day_weather().drop(columns=["T2M"])
    with pytest.raises(ValueError, match="missing required columns"):
        _compute(weather)
