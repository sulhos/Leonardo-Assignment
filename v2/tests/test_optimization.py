"""Tests for `src.physics.optimization` (PROJECT_BRIEF.md §26).

Uses small deterministic synthetic scenarios (no internet access). Rewritten
for the diesel/system-LCOE pivot (PROJECT_BRIEF.md Addendum 3): the objective
is now system-LCOE minimization with diesel backup, not a hard LPSP-feasible
constraint, so most of these tests check economic/reliability *metrics*
rather than a binary feasible/infeasible outcome.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.physics.optimization import run_battery_search

COMMON_BATTERY_KWARGS = dict(
    module_capacity_kwh=10.0,
    module_rated_power_kw=10.0,
    round_trip_efficiency=0.9,
    min_soc_fraction=0.1,
    max_soc_fraction=0.9,
    initial_soc_fraction=0.5,
    self_discharge_rate_per_hour=0.0,
    installed_cost_eur_per_kwh=500,
    economic_lifetime_years=10,
    project_lifetime_years=20,
    real_discount_rate=0.05,
)

PV_CFG = {"installed_cost_eur_per_kwp": 700, "economic_lifetime_years": 25, "om_cost_fraction_per_year": 0.015}
WIND_CFG = {"installed_cost_eur_per_kw": 1200, "economic_lifetime_years": 20, "om_cost_fraction_per_year": 0.025}
DIESEL_CFG = {
    "sizing_factor": 1.25,
    "fuel_curve_intercept_l_per_kwh_rated": 0.08145,
    "fuel_curve_slope_l_per_kwh_output": 0.246,
    "fuel_price_eur_per_l": 0.9,
    "installed_cost_eur_per_kw": 650,
    "om_cost_fraction_per_year": 0.03,
    "economic_lifetime_years": 15,
    "project_lifetime_years": 20,
}

COMMON_SYSTEM_KWARGS = dict(
    pv_capacity_kwp=20.0, pv_cfg=PV_CFG, wind_capacity_kw=0.0, wind_cfg=WIND_CFG, diesel_cfg=DIESEL_CFG,
)


def _index() -> pd.DatetimeIndex:
    return pd.date_range("2023-01-01", periods=24, freq="h", tz="Asia/Shanghai")


def _feasible_scenario() -> tuple[pd.Series, pd.Series, pd.Series]:
    # 6 daylight hours of strong PV surplus, small constant load: a modest
    # battery (5 modules, 40 kWh usable) fully covers the nightly deficit
    # from renewables+battery alone, before diesel is even needed.
    index = _index()
    load = pd.Series(2.0, index=index)
    pv = pd.Series([20.0 if 10 <= h <= 15 else 0.0 for h in range(24)], index=index)
    wind = pd.Series(0.0, index=index)
    return pv, wind, load


def _battery_range_limited_scenario() -> tuple[pd.Series, pd.Series, pd.Series]:
    # Annual renewable (9h * 20kW = 180) exceeds annual load (24h * 5kW =
    # 120), but the nightly deficit (75 kWh) is too large for a small
    # battery to fully cover from renewables+battery alone -- under the old
    # LPSP-constrained objective this was "infeasible"; under the new
    # diesel-backed objective, diesel simply covers the gap and the system
    # remains reliable, just with a lower renewable share.
    index = _index()
    load = pd.Series(5.0, index=index)
    pv = pd.Series([20.0 if 8 <= h <= 16 else 0.0 for h in range(24)], index=index)
    wind = pd.Series(0.0, index=index)
    return pv, wind, load


def _renewable_inadequate_scenario() -> tuple[pd.Series, pd.Series, pd.Series]:
    # Renewable is below load in every single hour: no battery, however
    # large, can close an annual energy deficit from renewables+battery
    # alone -- but diesel (sized on power, not energy) can still serve the
    # full load if needed, so the system stays reliable at renewable_share
    # near 0, not "infeasible."
    index = _index()
    load = pd.Series(10.0, index=index)
    pv = pd.Series(2.0, index=index)
    wind = pd.Series(0.0, index=index)
    return pv, wind, load


def test_min_lcoe_candidate_selected_correctly() -> None:
    pv, wind, load = _feasible_scenario()
    result = run_battery_search(pv, wind, load, n_max=6, **COMMON_BATTERY_KWARGS, **COMMON_SYSTEM_KWARGS)

    assert result.system_lcoe_eur_per_kwh == pytest.approx(result.candidates["system_lcoe_eur_per_kwh"].min())
    selected_row = result.candidates[result.candidates["n_modules"] == result.optimal_n_modules].iloc[0]
    assert selected_row["system_lcoe_eur_per_kwh"] == pytest.approx(result.system_lcoe_eur_per_kwh)


def test_diesel_backup_makes_battery_range_limited_scenario_reliable() -> None:
    # This exact scenario was "Battery-range inadequacy: infeasible" under
    # the pre-diesel objective (see git history) -- with diesel present, it
    # must now be reliable (LPSP after diesel ~= 0), just at a lower
    # renewable share than the fully-renewable-covered case.
    pv, wind, load = _battery_range_limited_scenario()
    result = run_battery_search(pv, wind, load, n_max=5, lpsp_target=0.01, **COMMON_BATTERY_KWARGS, **COMMON_SYSTEM_KWARGS)

    assert result.feasible is True
    assert result.infeasibility_reason is None
    assert result.lpsp == pytest.approx(0.0, abs=1e-9)
    assert 0.0 <= result.renewable_share <= 1.0


def test_renewable_inadequate_scenario_stays_reliable_via_diesel() -> None:
    # The scenario that used to be "Renewable-generation inadequacy:
    # infeasible" -- diesel is sized on power (1.25x peak), not energy, so
    # it can serve the whole load if renewables can't, regardless of the
    # annual energy shortfall.
    pv, wind, load = _renewable_inadequate_scenario()
    result = run_battery_search(pv, wind, load, n_max=5, lpsp_target=0.01, **COMMON_BATTERY_KWARGS, **COMMON_SYSTEM_KWARGS)

    assert result.feasible is True
    assert result.lpsp == pytest.approx(0.0, abs=1e-9)
    # Renewables (2 kW) are far below load (10 kW) every hour -- most energy
    # must come from diesel, so renewable share should be low.
    assert result.renewable_share < 0.5


def test_undersized_diesel_can_be_genuinely_infeasible() -> None:
    # A deliberately undersized diesel (sizing_factor < 1.0, i.e. rated
    # power below peak load) is the one way this design can still produce a
    # genuine reliability shortfall -- confirms the infeasibility guard
    # actually triggers rather than being dead code under the documented
    # default (sizing_factor=1.25, which is mathematically always feasible;
    # see src/physics/optimization.py's module docstring).
    pv, wind, load = _renewable_inadequate_scenario()
    undersized_diesel_cfg = {**DIESEL_CFG, "sizing_factor": 0.5}
    result = run_battery_search(
        pv, wind, load, n_max=0, lpsp_target=0.001,
        **COMMON_BATTERY_KWARGS,
        pv_capacity_kwp=20.0, pv_cfg=PV_CFG, wind_capacity_kw=0.0, wind_cfg=WIND_CFG, diesel_cfg=undersized_diesel_cfg,
    )
    assert result.feasible is False
    assert "Diesel-capacity inadequacy" in result.infeasibility_reason
    # Even when infeasible, the min-LCOE candidate is still reported (never
    # silently dropped) -- economic optimization doesn't stop just because
    # the reliability target isn't hit.
    assert result.optimal_n_modules is not None
    assert result.optimal_capacity_kwh is not None


def test_unserved_energy_non_increasing_with_capacity() -> None:
    pv, wind, load = _feasible_scenario()
    result = run_battery_search(pv, wind, load, n_max=6, **COMMON_BATTERY_KWARGS, **COMMON_SYSTEM_KWARGS)
    unserved = result.candidates.sort_values("n_modules")["unserved_energy_kwh"].to_numpy()
    diffs = unserved[1:] - unserved[:-1]
    assert (diffs <= 1e-6).all()  # never increases as capacity grows


def test_modular_capacity_calculated_correctly() -> None:
    pv, wind, load = _feasible_scenario()
    result = run_battery_search(pv, wind, load, n_max=6, **COMMON_BATTERY_KWARGS, **COMMON_SYSTEM_KWARGS)
    expected = result.candidates["n_modules"] * COMMON_BATTERY_KWARGS["module_capacity_kwh"]
    assert (result.candidates["nominal_battery_capacity_kwh"] == expected).all()


def test_candidates_include_system_lcoe_and_renewable_share() -> None:
    pv, wind, load = _feasible_scenario()
    result = run_battery_search(pv, wind, load, n_max=6, **COMMON_BATTERY_KWARGS, **COMMON_SYSTEM_KWARGS)
    assert "system_lcoe_eur_per_kwh" in result.candidates.columns
    assert "renewable_share" in result.candidates.columns
    assert (result.candidates["renewable_share"] >= 0.0).all()
    assert (result.candidates["renewable_share"] <= 1.0).all()


def test_diesel_rated_power_matches_sizing_factor() -> None:
    pv, wind, load = _feasible_scenario()
    result = run_battery_search(pv, wind, load, n_max=0, **COMMON_BATTERY_KWARGS, **COMMON_SYSTEM_KWARGS)
    expected_rated_power = DIESEL_CFG["sizing_factor"] * load.max()
    assert result.candidates["diesel_rated_power_kw"].iloc[0] == pytest.approx(expected_rated_power)


def test_reliability_sensitivity_targets_evaluated() -> None:
    pv, wind, load = _feasible_scenario()
    result = run_battery_search(
        pv, wind, load, n_max=6,
        reliability_sensitivity_targets=[0.990, 0.995, 0.999], **COMMON_BATTERY_KWARGS, **COMMON_SYSTEM_KWARGS,
    )
    assert list(result.reliability_sensitivity["load_served_target"]) == [0.990, 0.995, 0.999]
    assert "feasible" in result.reliability_sensitivity.columns


def test_rejects_negative_n_max() -> None:
    pv, wind, load = _feasible_scenario()
    with pytest.raises(ValueError, match="n_max"):
        run_battery_search(pv, wind, load, n_max=-1, **COMMON_BATTERY_KWARGS, **COMMON_SYSTEM_KWARGS)
