"""Tests for `src.ai.physical_verification` (PROJECT_BRIEF.md §26).

Uses a small deterministic, synthetic (non-downloaded) full-year weather
fixture, matching the pattern in `tests/test_scenarios.py` -- no internet access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ai.physical_verification import (
    predicted_capacity_to_modules,
    summarize_verification,
    verify_predictions,
)

MODULE_CAPACITY_KWH = 15.36


def test_predicted_capacity_to_modules_rounds_up_never_down() -> None:
    # Exact multiple: still rounds to that exact count, not down.
    assert predicted_capacity_to_modules(30.72, MODULE_CAPACITY_KWH) == 2
    # Just over a multiple: must round UP to the next module, not truncate.
    assert predicted_capacity_to_modules(30.73, MODULE_CAPACITY_KWH) == 3
    # Just under a multiple: still rounds up to that count.
    assert predicted_capacity_to_modules(30.71, MODULE_CAPACITY_KWH) == 2
    # Zero prediction -> zero modules.
    assert predicted_capacity_to_modules(0.0, MODULE_CAPACITY_KWH) == 0


def test_predicted_capacity_to_modules_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        predicted_capacity_to_modules(-5.0, MODULE_CAPACITY_KWH)


def _synthetic_full_year_weather() -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=8760, freq="h", tz="Asia/Shanghai")
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


SITE_CONFIG = {
    "site": {"latitude": 36.65, "longitude": 117.12, "timezone": "Asia/Shanghai"},
    "pv": {"tilt_deg": 36.65, "azimuth_deg": 180.0, "system_losses_fraction": 0.14, "inverter_efficiency": 0.96},
    "wind": {"weather_reference_height_m": 10.0, "hub_height_m": 15.0, "wind_shear_exponent": 0.14,
             "cut_in_mps": 2.5, "rated_mps": 11.0, "cut_out_mps": 25.0},
    "battery": {"module_capacity_kwh": MODULE_CAPACITY_KWH, "module_rated_power_kw": 12.8, "initial_soc_fraction": 0.5,
                "installed_cost_eur_per_kwh": 550, "economic_lifetime_years": 10, "project_lifetime_years": 20},
    "economics": {"real_discount_rate": 0.05},
}


def _scenario_row(
    scenario_id: int = 0, optimal_n_modules: int = 15, reliability_target: float = 0.99,
) -> pd.Series:
    return pd.Series({
        "scenario_id": scenario_id,
        "timezone": "Asia/Shanghai",
        "weather_year": 2023,
        "random_seed": 42,
        "pv_capacity_kwp": 25.0,
        "wind_capacity_kw": 12.0,
        "annual_load_kwh": 28000.0,
        "peak_load_kw": 12.0,
        "reliability_target_load_served": reliability_target,
        "round_trip_efficiency": 0.95,
        "usable_soc_window_fraction": 0.85,
        "optimal_n_modules": optimal_n_modules,
        "optimal_capacity_kwh": optimal_n_modules * MODULE_CAPACITY_KWH,
    })


def test_verify_predictions_uses_predicted_capacity_not_reference() -> None:
    weather = _synthetic_full_year_weather()
    scenarios = pd.DataFrame([_scenario_row(scenario_id=0, optimal_n_modules=20)])
    # Predict far below the reference (5 modules vs. reference's 20).
    predicted = pd.Series([5 * MODULE_CAPACITY_KWH], index=scenarios.index)

    result = verify_predictions(scenarios, predicted, weather, SITE_CONFIG, MODULE_CAPACITY_KWH)

    assert result.loc[0, "installed_n_modules"] == 5  # uses the prediction...
    assert result.loc[0, "reference_n_modules"] == 20  # ...not the reference optimum
    assert result.loc[0, "installed_capacity_kwh"] == pytest.approx(5 * MODULE_CAPACITY_KWH)


def test_verify_predictions_flags_undersizing_and_oversizing_correctly() -> None:
    weather = _synthetic_full_year_weather()
    scenarios = pd.DataFrame([
        _scenario_row(scenario_id=0, optimal_n_modules=20),
        _scenario_row(scenario_id=1, optimal_n_modules=20),
    ])
    predicted = pd.Series([5 * MODULE_CAPACITY_KWH, 30 * MODULE_CAPACITY_KWH], index=scenarios.index)

    result = verify_predictions(scenarios, predicted, weather, SITE_CONFIG, MODULE_CAPACITY_KWH)

    assert result.loc[0, "undersized"] and not result.loc[0, "oversized"]
    assert result.loc[1, "oversized"] and not result.loc[1, "undersized"]


def test_verify_predictions_reliability_pass_reflects_own_target() -> None:
    weather = _synthetic_full_year_weather()
    # A tiny battery (1 module) against a demanding scenario should fail
    # reliability; a generous battery (30 modules, well above what 25kWp PV +
    # 12kW wind + 28 MWh load needs) should pass.
    scenarios = pd.DataFrame([
        _scenario_row(scenario_id=0, optimal_n_modules=20),
        _scenario_row(scenario_id=1, optimal_n_modules=20),
    ])
    predicted = pd.Series([1 * MODULE_CAPACITY_KWH, 30 * MODULE_CAPACITY_KWH], index=scenarios.index)

    result = verify_predictions(scenarios, predicted, weather, SITE_CONFIG, MODULE_CAPACITY_KWH)

    assert result.loc[0, "reliability_pass"] == False  # noqa: E712 (explicit bool check reads clearer here)
    assert result.loc[1, "reliability_pass"] == True  # noqa: E712


def test_summarize_verification_aggregates() -> None:
    verification = pd.DataFrame({
        "undersized": [True, False, True, False],
        "oversized": [False, True, False, False],
        "reliability_pass": [False, True, False, True],
        "excess_capacity_kwh": [-30.0, 20.0, -50.0, 0.0],
        "cost_difference_eur": [-1000.0, 500.0, -2000.0, 0.0],
        "curtailed_difference_kwh": [10.0, -5.0, 15.0, 0.0],
    })
    summary = summarize_verification(verification)

    assert summary["n_scenarios"] == 4
    assert summary["pct_satisfying_reliability"] == pytest.approx(0.5)
    assert summary["pct_undersized"] == pytest.approx(0.5)
    assert summary["pct_oversized"] == pytest.approx(0.25)
    assert summary["n_reliability_violations"] == 2
    assert summary["n_reliability_violations_from_undersizing"] == 2
    assert summary["max_underprediction_kwh"] == pytest.approx(50.0)
    assert summary["additional_cost_from_oversizing_eur"] == pytest.approx(500.0)
