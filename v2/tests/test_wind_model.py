"""Tests for `src.physics.wind_model` (PROJECT_BRIEF.md §26)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.physics.wind_model import (
    adjust_wind_speed_to_hub_height,
    adjust_wind_speed_to_hub_height_log_law,
    compute_wind_generation,
    turbine_power_curve,
)

RATED_POWER_KW = 10.0
CUT_IN_MPS = 2.5
RATED_MPS = 11.0
CUT_OUT_MPS = 25.0
POWER_CURVE = [
    [0.0, 0.0], [2.5, 0.0], [3.0, 0.0086], [3.5, 0.0207], [4.0, 0.0368], [4.5, 0.0574],
    [5.0, 0.0832], [5.5, 0.1146], [6.0, 0.1523], [6.5, 0.1969], [7.0, 0.2489], [7.5, 0.3088],
    [8.0, 0.3774], [8.5, 0.455], [9.0, 0.5423], [9.5, 0.6399], [10.0, 0.7484], [10.5, 0.8682],
    [11.0, 1.0], [25.0, 1.0], [25.01, 0.0], [50.0, 0.0],
]


def _wind_speed_series(values: list[float]) -> pd.Series:
    index = pd.date_range("2023-06-15 00:00", periods=len(values), freq="h", tz="Asia/Shanghai")
    return pd.Series(values, index=index, name="WS10M")


def test_power_curve_zero_below_cut_in() -> None:
    ws = _wind_speed_series([0.0, 1.0, 2.4])
    power = turbine_power_curve(ws, RATED_POWER_KW, POWER_CURVE)
    assert (power == 0).all()


def test_power_curve_zero_above_cut_out() -> None:
    ws = _wind_speed_series([25.1, 30.0, 40.0])
    power = turbine_power_curve(ws, RATED_POWER_KW, POWER_CURVE)
    assert (power == 0).all()


def test_power_curve_rated_between_rated_and_cutout() -> None:
    ws = _wind_speed_series([11.0, 15.0, 25.0])
    power = turbine_power_curve(ws, RATED_POWER_KW, POWER_CURVE)
    assert (power == RATED_POWER_KW).all()


def test_power_curve_never_negative_or_above_rated() -> None:
    ws = _wind_speed_series(list(np.arange(0, 40, 0.5)))
    power = turbine_power_curve(ws, RATED_POWER_KW, POWER_CURVE)
    assert (power >= 0).all()
    assert (power <= RATED_POWER_KW).all()


def test_power_curve_monotonic_in_ramp_region() -> None:
    ramp_speeds = np.linspace(CUT_IN_MPS, RATED_MPS, 20, endpoint=False)
    ws = _wind_speed_series(list(ramp_speeds))
    power = turbine_power_curve(ws, RATED_POWER_KW, POWER_CURVE)
    assert (power.diff().dropna() >= 0).all()


def test_power_curve_interpolates_between_table_breakpoints() -> None:
    # Hand-computed reference: linear interpolation halfway between the
    # table's (7.0, 0.2489) and (7.5, 0.3088) breakpoints, at v=7.25.
    # Expected fraction = 0.2489 + 0.5*(0.3088-0.2489) = 0.27885
    ws = _wind_speed_series([7.25])
    power = turbine_power_curve(ws, RATED_POWER_KW, POWER_CURVE)
    assert power.iloc[0] == pytest.approx(RATED_POWER_KW * 0.27885, abs=1e-4)


def test_power_curve_rejects_unsorted_speeds() -> None:
    ws = _wind_speed_series([5.0])
    with pytest.raises(ValueError, match="sorted ascending"):
        turbine_power_curve(ws, RATED_POWER_KW, [[5.0, 0.5], [2.0, 0.1]])


def test_power_curve_rejects_out_of_range_fraction() -> None:
    ws = _wind_speed_series([5.0])
    with pytest.raises(ValueError, match=r"power_fraction"):
        turbine_power_curve(ws, RATED_POWER_KW, [[0.0, 0.0], [10.0, 1.5]])


def test_hub_height_adjustment_increases_speed_for_taller_hub() -> None:
    ws = _wind_speed_series([5.0, 6.0])
    adjusted = adjust_wind_speed_to_hub_height(ws, reference_height_m=10.0, hub_height_m=50.0, shear_exponent=0.14)
    assert (adjusted > ws).all()


def test_hub_height_adjustment_identity_when_heights_equal() -> None:
    ws = _wind_speed_series([5.0, 6.0])
    adjusted = adjust_wind_speed_to_hub_height(ws, reference_height_m=10.0, hub_height_m=10.0, shear_exponent=0.14)
    assert np.allclose(adjusted.to_numpy(), ws.to_numpy())


def test_power_law_matches_hand_computed_reference() -> None:
    # v_hub = 6.0 * (60/10)**0.14 = 6.0 * 6**0.14
    ws = _wind_speed_series([6.0])
    adjusted = adjust_wind_speed_to_hub_height(ws, reference_height_m=10.0, hub_height_m=60.0, shear_exponent=0.14)
    expected = 6.0 * (60.0 / 10.0) ** 0.14
    assert adjusted.iloc[0] == pytest.approx(expected, abs=1e-9)


def test_log_law_matches_hand_computed_reference() -> None:
    # v(z2) = 6.0 * ln((60-0)/0.1) / ln((10-0)/0.1) = 6.0 * ln(600)/ln(100)
    ws = _wind_speed_series([6.0])
    adjusted = adjust_wind_speed_to_hub_height_log_law(
        ws, reference_height_m=10.0, hub_height_m=60.0, roughness_length_m=0.1,
    )
    expected = 6.0 * np.log(600.0) / np.log(100.0)
    assert adjusted.iloc[0] == pytest.approx(expected, abs=1e-9)


def test_log_law_rejects_non_positive_roughness() -> None:
    ws = _wind_speed_series([6.0])
    with pytest.raises(ValueError, match="roughness_length_m must be positive"):
        adjust_wind_speed_to_hub_height_log_law(ws, reference_height_m=10.0, hub_height_m=60.0, roughness_length_m=0.0)


def test_compute_wind_generation_end_to_end_bounds() -> None:
    index = pd.date_range("2023-01-01", periods=8760, freq="h", tz="Asia/Shanghai")
    weather = pd.DataFrame({"WS10M": (np.arange(8760) % 30).astype(float)}, index=index)
    wind_kw = compute_wind_generation(
        weather=weather,
        rated_power_kw=RATED_POWER_KW,
        reference_height_m=10.0,
        hub_height_m=15.0,
        shear_exponent=0.14,
        power_curve_speed_power_fraction=POWER_CURVE,
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
            power_curve_speed_power_fraction=POWER_CURVE,
        )
