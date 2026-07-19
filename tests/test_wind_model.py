"""Tests for `src.physics.wind_model` (PROJECT_BRIEF.md §26)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.physics.wind_model import (
    adjust_wind_speed_to_hub_height,
    compute_wind_generation,
    turbine_power_curve,
)

RATED_POWER_KW = 10.0
CUT_IN_MPS = 2.5
RATED_MPS = 11.0
CUT_OUT_MPS = 25.0


def _wind_speed_series(values: list[float]) -> pd.Series:
    index = pd.date_range("2023-06-15 00:00", periods=len(values), freq="h", tz="Asia/Shanghai")
    return pd.Series(values, index=index, name="WS10M")


def test_power_curve_zero_below_cut_in() -> None:
    ws = _wind_speed_series([0.0, 1.0, 2.4])
    power = turbine_power_curve(ws, RATED_POWER_KW, CUT_IN_MPS, RATED_MPS, CUT_OUT_MPS)
    assert (power == 0).all()


def test_power_curve_zero_above_cut_out() -> None:
    ws = _wind_speed_series([25.1, 30.0, 40.0])
    power = turbine_power_curve(ws, RATED_POWER_KW, CUT_IN_MPS, RATED_MPS, CUT_OUT_MPS)
    assert (power == 0).all()


def test_power_curve_rated_between_rated_and_cutout() -> None:
    ws = _wind_speed_series([11.0, 15.0, 25.0])
    power = turbine_power_curve(ws, RATED_POWER_KW, CUT_IN_MPS, RATED_MPS, CUT_OUT_MPS)
    assert (power == RATED_POWER_KW).all()


def test_power_curve_never_negative_or_above_rated() -> None:
    ws = _wind_speed_series(list(np.arange(0, 40, 0.5)))
    power = turbine_power_curve(ws, RATED_POWER_KW, CUT_IN_MPS, RATED_MPS, CUT_OUT_MPS)
    assert (power >= 0).all()
    assert (power <= RATED_POWER_KW).all()


def test_power_curve_monotonic_in_ramp_region() -> None:
    ramp_speeds = np.linspace(CUT_IN_MPS, RATED_MPS, 20, endpoint=False)
    ws = _wind_speed_series(list(ramp_speeds))
    power = turbine_power_curve(ws, RATED_POWER_KW, CUT_IN_MPS, RATED_MPS, CUT_OUT_MPS)
    assert (power.diff().dropna() >= 0).all()


def test_power_curve_rejects_invalid_speed_thresholds() -> None:
    ws = _wind_speed_series([5.0])
    with pytest.raises(ValueError, match="cut_in < rated"):
        turbine_power_curve(ws, RATED_POWER_KW, cut_in_mps=12.0, rated_mps=11.0, cut_out_mps=25.0)


def test_hub_height_adjustment_increases_speed_for_taller_hub() -> None:
    ws = _wind_speed_series([5.0, 6.0])
    adjusted = adjust_wind_speed_to_hub_height(ws, reference_height_m=10.0, hub_height_m=50.0, shear_exponent=0.14)
    assert (adjusted > ws).all()


def test_hub_height_adjustment_identity_when_heights_equal() -> None:
    ws = _wind_speed_series([5.0, 6.0])
    adjusted = adjust_wind_speed_to_hub_height(ws, reference_height_m=10.0, hub_height_m=10.0, shear_exponent=0.14)
    assert np.allclose(adjusted.to_numpy(), ws.to_numpy())


def test_compute_wind_generation_end_to_end_bounds() -> None:
    index = pd.date_range("2023-01-01", periods=8760, freq="h", tz="Asia/Shanghai")
    weather = pd.DataFrame({"WS10M": (np.arange(8760) % 30).astype(float)}, index=index)
    wind_kw = compute_wind_generation(
        weather=weather,
        rated_power_kw=RATED_POWER_KW,
        reference_height_m=10.0,
        hub_height_m=15.0,
        shear_exponent=0.14,
        cut_in_mps=CUT_IN_MPS,
        rated_mps=RATED_MPS,
        cut_out_mps=CUT_OUT_MPS,
    )
    assert (wind_kw >= 0).all()
    assert (wind_kw <= RATED_POWER_KW).all()


def test_compute_wind_generation_requires_ws10m_column() -> None:
    weather = pd.DataFrame({"other": [1.0, 2.0]}, index=pd.date_range("2023-01-01", periods=2, freq="h", tz="Asia/Shanghai"))
    with pytest.raises(ValueError, match="WS10M"):
        compute_wind_generation(
            weather=weather,
            rated_power_kw=RATED_POWER_KW,
            reference_height_m=10.0,
            hub_height_m=15.0,
            shear_exponent=0.14,
            cut_in_mps=CUT_IN_MPS,
            rated_mps=RATED_MPS,
            cut_out_mps=CUT_OUT_MPS,
        )
