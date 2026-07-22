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
from src.physics.optimization import run_battery_search
from src.physics.load_profile import generate_load_profile
from src.physics.pv_model import compute_pv_generation
from src.physics.wind_model import compute_wind_generation

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
    "system": {"backup": "diesel"},
    "pv": {"tilt_deg": 36.65, "azimuth_deg": 180.0, "system_losses_fraction": 0.14, "inverter_efficiency": 0.96,
           "installed_cost_eur_per_kwp": 700, "economic_lifetime_years": 25, "om_cost_fraction_per_year": 0.015},
    "wind": {"weather_reference_height_m": 10.0, "hub_height_m": 15.0, "wind_shear_exponent": 0.14,
             "power_curve_speed_power_fraction": [
                 [0.0, 0.0], [2.5, 0.0], [7.0, 0.2489], [11.0, 1.0], [25.0, 1.0], [25.01, 0.0], [50.0, 0.0],
             ],
             "installed_cost_eur_per_kw": 1200, "economic_lifetime_years": 20, "om_cost_fraction_per_year": 0.025},
    "battery": {"module_capacity_kwh": MODULE_CAPACITY_KWH, "module_rated_power_kw": 12.8, "initial_soc_fraction": 0.5,
                "self_discharge_rate_per_hour": 0.0,
                "installed_cost_eur_per_kwh": 550, "economic_lifetime_years": 10, "project_lifetime_years": 20},
    "diesel": {"sizing_factor": 1.25, "fuel_curve_intercept_l_per_kwh_rated": 0.08145,
               "fuel_curve_slope_l_per_kwh_output": 0.246, "fuel_price_eur_per_l": 0.9,
               "installed_cost_eur_per_kw": 650, "om_cost_fraction_per_year": 0.03,
               "economic_lifetime_years": 15, "project_lifetime_years": 20},
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


def test_verify_predictions_reliability_near_universal_with_diesel_backup() -> None:
    # PROJECT_BRIEF.md Addendum 3: diesel (default sizing_factor=1.25 * peak
    # load) covers any residual deficit regardless of battery size, so even a
    # tiny 1-module prediction should still satisfy reliability -- reliability
    # is no longer the discriminating question, extra system LCOE is.
    weather = _synthetic_full_year_weather()
    scenarios = pd.DataFrame([
        _scenario_row(scenario_id=0, optimal_n_modules=20),
        _scenario_row(scenario_id=1, optimal_n_modules=20),
    ])
    predicted = pd.Series([1 * MODULE_CAPACITY_KWH, 30 * MODULE_CAPACITY_KWH], index=scenarios.index)

    result = verify_predictions(scenarios, predicted, weather, SITE_CONFIG, MODULE_CAPACITY_KWH)

    assert result.loc[0, "reliability_pass"] == True  # noqa: E712
    assert result.loc[1, "reliability_pass"] == True  # noqa: E712


def test_verify_predictions_genuinely_infeasible_with_undersized_diesel() -> None:
    # The reliability guard is not dead code: an explicitly undersized
    # diesel (sizing_factor < 1.0) can still fail reliability for a tiny
    # battery prediction, same boundary case as
    # tests/test_optimization.py::test_undersized_diesel_can_be_genuinely_infeasible.
    undersized_diesel_config = {**SITE_CONFIG, "diesel": {**SITE_CONFIG["diesel"], "sizing_factor": 0.1}}
    weather = _synthetic_full_year_weather()
    scenarios = pd.DataFrame([_scenario_row(scenario_id=0, optimal_n_modules=20, reliability_target=0.999)])
    predicted = pd.Series([1 * MODULE_CAPACITY_KWH], index=scenarios.index)

    result = verify_predictions(scenarios, predicted, weather, undersized_diesel_config, MODULE_CAPACITY_KWH)

    assert result.loc[0, "reliability_pass"] == False  # noqa: E712


def test_verify_predictions_extra_lcoe_zero_when_prediction_matches_reference() -> None:
    # Identical module count -> identical dispatch -> identical system LCOE,
    # regardless of whether that count is actually the true LCOE optimum.
    weather = _synthetic_full_year_weather()
    scenarios = pd.DataFrame([_scenario_row(scenario_id=0, optimal_n_modules=20)])
    predicted = pd.Series([20 * MODULE_CAPACITY_KWH], index=scenarios.index)

    result = verify_predictions(scenarios, predicted, weather, SITE_CONFIG, MODULE_CAPACITY_KWH)

    assert result.loc[0, "extra_system_lcoe_eur_per_kwh"] == pytest.approx(0.0, abs=1e-9)
    assert result.loc[0, "verified_system_lcoe_eur_per_kwh"] == pytest.approx(
        result.loc[0, "reference_system_lcoe_eur_per_kwh"]
    )


def test_verify_predictions_extra_lcoe_nonzero_when_prediction_differs() -> None:
    weather = _synthetic_full_year_weather()
    scenarios = pd.DataFrame([_scenario_row(scenario_id=0, optimal_n_modules=20)])
    predicted = pd.Series([1 * MODULE_CAPACITY_KWH], index=scenarios.index)

    result = verify_predictions(scenarios, predicted, weather, SITE_CONFIG, MODULE_CAPACITY_KWH)

    assert result.loc[0, "extra_system_lcoe_eur_per_kwh"] != pytest.approx(0.0, abs=1e-9)


def test_verify_predictions_extra_lcoe_never_meaningfully_negative_vs_true_optimum() -> None:
    # Regression test for the diesel-sizing mismatch fixed above: diesel must
    # be sized from the load's ACHIEVED peak (load.max()), matching
    # run_battery_search's own convention exactly, not the scenario's
    # requested peak_load_kw -- otherwise "reference" is not guaranteed
    # optimal under this function's own re-evaluation, and predictions can
    # spuriously look cheaper than the true optimum by a small negative
    # extra_system_lcoe_eur_per_kwh.
    weather = _synthetic_full_year_weather()
    scenario_inputs = dict(
        pv_capacity_kwp=25.0, wind_capacity_kw=12.0, annual_load_kwh=28000.0, peak_load_kw=12.0,
        weather_year=2023, timezone="Asia/Shanghai", random_seed=42,
    )
    load = generate_load_profile(
        annual_consumption_kwh=scenario_inputs["annual_load_kwh"], peak_load_kw=scenario_inputs["peak_load_kw"],
        year=scenario_inputs["weather_year"], timezone=scenario_inputs["timezone"], random_seed=scenario_inputs["random_seed"],
    )
    pv = compute_pv_generation(
        weather=weather, capacity_kwp=scenario_inputs["pv_capacity_kwp"],
        latitude=SITE_CONFIG["site"]["latitude"], longitude=SITE_CONFIG["site"]["longitude"],
        tilt_deg=SITE_CONFIG["pv"]["tilt_deg"], azimuth_deg=SITE_CONFIG["pv"]["azimuth_deg"],
        system_losses_fraction=SITE_CONFIG["pv"]["system_losses_fraction"], inverter_efficiency=SITE_CONFIG["pv"]["inverter_efficiency"],
    )
    wind = compute_wind_generation(
        weather=weather, rated_power_kw=scenario_inputs["wind_capacity_kw"],
        reference_height_m=SITE_CONFIG["wind"]["weather_reference_height_m"], hub_height_m=SITE_CONFIG["wind"]["hub_height_m"],
        shear_exponent=SITE_CONFIG["wind"]["wind_shear_exponent"],
        power_curve_speed_power_fraction=SITE_CONFIG["wind"]["power_curve_speed_power_fraction"],
    )
    search_result = run_battery_search(
        pv, wind, load, n_max=25,
        module_capacity_kwh=MODULE_CAPACITY_KWH, module_rated_power_kw=SITE_CONFIG["battery"]["module_rated_power_kw"],
        round_trip_efficiency=0.95, min_soc_fraction=0.075, max_soc_fraction=0.925,
        initial_soc_fraction=SITE_CONFIG["battery"]["initial_soc_fraction"],
        self_discharge_rate_per_hour=SITE_CONFIG["battery"]["self_discharge_rate_per_hour"],
        installed_cost_eur_per_kwh=SITE_CONFIG["battery"]["installed_cost_eur_per_kwh"],
        economic_lifetime_years=SITE_CONFIG["battery"]["economic_lifetime_years"],
        project_lifetime_years=SITE_CONFIG["battery"]["project_lifetime_years"],
        real_discount_rate=SITE_CONFIG["economics"]["real_discount_rate"],
        pv_capacity_kwp=scenario_inputs["pv_capacity_kwp"], pv_cfg=SITE_CONFIG["pv"],
        wind_capacity_kw=scenario_inputs["wind_capacity_kw"], wind_cfg=SITE_CONFIG["wind"],
        diesel_cfg=SITE_CONFIG["diesel"], lpsp_target=0.01,
    )

    scenarios = pd.DataFrame([{
        **scenario_inputs,
        "scenario_id": 0,
        "reliability_target_load_served": 0.99,
        "round_trip_efficiency": 0.95,
        "usable_soc_window_fraction": 0.85,
        "optimal_n_modules": search_result.optimal_n_modules,
        "optimal_capacity_kwh": search_result.optimal_capacity_kwh,
    }])

    # Sweep predictions across the whole searched range; none should ever
    # look cheaper than the true optimum by more than floating-point noise.
    for n_modules in range(0, 26, 5):
        predicted = pd.Series([n_modules * MODULE_CAPACITY_KWH], index=scenarios.index)
        result = verify_predictions(scenarios, predicted, weather, SITE_CONFIG, MODULE_CAPACITY_KWH)
        assert result.loc[0, "extra_system_lcoe_eur_per_kwh"] >= -1e-9


def test_summarize_verification_aggregates() -> None:
    verification = pd.DataFrame({
        "undersized": [True, False, True, False],
        "oversized": [False, True, False, False],
        "reliability_pass": [False, True, False, True],
        "excess_capacity_kwh": [-30.0, 20.0, -50.0, 0.0],
        "cost_difference_eur": [-1000.0, 500.0, -2000.0, 0.0],
        "curtailed_difference_kwh": [10.0, -5.0, 15.0, 0.0],
        "extra_system_lcoe_eur_per_kwh": [0.01, 0.0, 0.05, 0.02],
        "reference_system_lcoe_eur_per_kwh": [0.20, 0.20, 0.20, 0.20],
    })
    summary = summarize_verification(verification)

    assert summary["n_scenarios"] == 4
    assert summary["mean_extra_system_lcoe_eur_per_kwh"] == pytest.approx(0.02)
    assert summary["median_extra_system_lcoe_eur_per_kwh"] == pytest.approx(0.015)
    assert summary["max_extra_system_lcoe_eur_per_kwh"] == pytest.approx(0.05)
    assert summary["pct_within_5pct_of_optimal_lcoe"] == pytest.approx(0.5)  # 0.0 and 0.01 are within 5% of 0.20
    assert summary["pct_satisfying_reliability"] == pytest.approx(0.5)
    assert summary["pct_undersized"] == pytest.approx(0.5)
    assert summary["pct_oversized"] == pytest.approx(0.25)
    assert summary["n_reliability_violations"] == 2
    assert summary["n_reliability_violations_from_undersizing"] == 2
    assert summary["max_underprediction_kwh"] == pytest.approx(50.0)
    assert summary["additional_cost_from_oversizing_eur"] == pytest.approx(500.0)
