"""Tests for `src.ai.features` (PROJECT_BRIEF.md §26)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ai.features import (
    LEAKAGE_COLUMNS,
    build_feature_matrix,
    compute_generation_features,
    compute_load_features,
    compute_net_load_features,
)


def _index(n: int = 8760) -> pd.DatetimeIndex:
    return pd.date_range("2023-01-01", periods=n, freq="h", tz="Asia/Shanghai")


def test_compute_load_features_basic_values() -> None:
    index = _index(24)
    load = pd.Series([1.0] * 24, index=index)
    features = compute_load_features(load)
    assert features["annual_load_kwh"] == pytest.approx(24.0)
    assert features["peak_load_kw"] == pytest.approx(1.0)
    assert features["load_factor"] == pytest.approx(1.0)  # flat load: mean == peak
    assert features["peak_to_average_ratio"] == pytest.approx(1.0)


def test_compute_load_features_daytime_nighttime_shares_sum_to_one() -> None:
    index = _index(24)
    load = pd.Series(np.random.default_rng(0).uniform(1, 5, 24), index=index)
    features = compute_load_features(load)
    assert features["daytime_load_share"] + features["nighttime_load_share"] == pytest.approx(1.0)


def test_compute_generation_features_capacity_factors() -> None:
    index = _index(24)
    pv = pd.Series([10.0] * 12 + [0.0] * 12, index=index)  # 50% "capacity factor" over the day
    wind = pd.Series([0.0] * 24, index=index)
    load = pd.Series([5.0] * 24, index=index)
    features = compute_generation_features(pv, wind, load, pv_capacity_kwp=10.0, wind_capacity_kw=5.0)
    assert features["pv_capacity_factor"] == pytest.approx(0.5)
    assert features["wind_capacity_factor"] == pytest.approx(0.0)
    assert features["renewable_to_load_ratio"] == pytest.approx(120.0 / 120.0)


def test_compute_net_load_features_all_surplus() -> None:
    index = _index(24)
    load = pd.Series([1.0] * 24, index=index)
    pv = pd.Series([10.0] * 24, index=index)
    wind = pd.Series([0.0] * 24, index=index)
    features = compute_net_load_features(load, pv, wind)
    assert features["n_deficit_hours"] == 0
    assert features["n_surplus_hours"] == 24
    assert features["total_deficit_energy_kwh"] == pytest.approx(0.0)
    assert features["max_consecutive_deficit_hours"] == 0


def test_compute_net_load_features_detects_consecutive_deficit_run() -> None:
    index = _index(10)
    load = pd.Series([5.0] * 10, index=index)
    pv = pd.Series([5.0, 5.0, 0.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 5.0], index=index)
    wind = pd.Series([0.0] * 10, index=index)
    features = compute_net_load_features(load, pv, wind)
    # Two deficit runs: hours 2-4 (length 3) and hours 7-8 (length 2) -> max run = 3.
    assert features["max_consecutive_deficit_hours"] == 3
    assert features["n_deficit_hours"] == 5


SITE_CONFIG = {
    "site": {"latitude": 36.65, "longitude": 117.12, "timezone": "Asia/Shanghai"},
    "pv": {"tilt_deg": 36.65, "azimuth_deg": 180.0, "system_losses_fraction": 0.14, "inverter_efficiency": 0.96},
    "wind": {"weather_reference_height_m": 10.0, "hub_height_m": 15.0, "wind_shear_exponent": 0.14,
             "cut_in_mps": 2.5, "rated_mps": 11.0, "cut_out_mps": 25.0},
}


def _synthetic_full_year_weather() -> pd.DataFrame:
    index = _index(8760)
    hour = index.hour.to_numpy()
    day_of_year = index.dayofyear.to_numpy()
    daylight = (hour >= 6) & (hour <= 18)
    seasonal = 1.0 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    ghi = np.where(daylight, 700 * np.sin(np.pi * (hour - 6) / 12) * seasonal, 0.0).clip(min=0)
    return pd.DataFrame(
        {
            "ALLSKY_SFC_SW_DWN": ghi,
            "ALLSKY_SFC_SW_DNI": ghi * 0.7,
            "ALLSKY_SFC_SW_DIFF": ghi * 0.3,
            "T2M": 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.where(daylight, 5, 0),
            "WS10M": 3 + 2 * np.abs(np.sin(2 * np.pi * day_of_year / 30)),
        },
        index=index,
    )


def _scenarios_df(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario_id": range(n),
            "timezone": ["Asia/Shanghai"] * n,
            "weather_year": [2023] * n,
            "random_seed": [42 + i for i in range(n)],
            "pv_capacity_kwp": [20.0 + i for i in range(n)],
            "wind_capacity_kw": [10.0 + i for i in range(n)],
            "annual_load_kwh": [30000.0 + i * 1000 for i in range(n)],
            "peak_load_kw": [15.0] * n,
            "reliability_target_load_served": [0.99] * n,
            "round_trip_efficiency": [0.95] * n,
            "usable_soc_window_fraction": [0.85] * n,
            # Labels that must never leak into the feature matrix:
            "optimal_capacity_kwh": [123.0] * n,
            "optimal_n_modules": [8] * n,
            "reference_lpsp": [0.005] * n,
            "reference_annualized_cost": [5000.0] * n,
        }
    )


def test_build_feature_matrix_excludes_leakage_columns() -> None:
    weather = _synthetic_full_year_weather()
    features = build_feature_matrix(_scenarios_df(), weather, SITE_CONFIG)
    assert not (LEAKAGE_COLUMNS & set(features.columns))


def test_build_feature_matrix_one_row_per_scenario_no_nans() -> None:
    weather = _synthetic_full_year_weather()
    scenarios = _scenarios_df(4)
    features = build_feature_matrix(scenarios, weather, SITE_CONFIG)
    assert len(features) == 4
    assert set(features["scenario_id"]) == set(scenarios["scenario_id"])
    assert features.isna().sum().sum() == 0
