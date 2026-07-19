"""Tests for `src.physics.optimization` (PROJECT_BRIEF.md §26).

Uses small deterministic synthetic scenarios (no internet access) chosen so
the feasible/infeasible outcome and infeasibility category are known.
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
    installed_cost_eur_per_kwh=500,
    economic_lifetime_years=10,
    project_lifetime_years=20,
    real_discount_rate=0.05,
)


def _index() -> pd.DatetimeIndex:
    return pd.date_range("2023-01-01", periods=24, freq="h", tz="Asia/Shanghai")


def _feasible_scenario() -> tuple[pd.Series, pd.Series, pd.Series]:
    # 6 daylight hours of strong PV surplus, small constant load: a modest
    # battery (5 modules, 40 kWh usable) fully covers the nightly deficit.
    index = _index()
    load = pd.Series(2.0, index=index)
    pv = pd.Series([20.0 if 10 <= h <= 15 else 0.0 for h in range(24)], index=index)
    wind = pd.Series(0.0, index=index)
    return pv, wind, load


def _battery_range_infeasible_scenario() -> tuple[pd.Series, pd.Series, pd.Series]:
    # Annual renewable (9h * 20kW = 180) exceeds annual load (24h * 5kW =
    # 120), but the nightly deficit (75 kWh) is too large for a 5-module
    # (40 kWh usable) battery to fully cover -> feasible energy balance,
    # infeasible within the tested module-count range.
    index = _index()
    load = pd.Series(5.0, index=index)
    pv = pd.Series([20.0 if 8 <= h <= 16 else 0.0 for h in range(24)], index=index)
    wind = pd.Series(0.0, index=index)
    return pv, wind, load


def _renewable_inadequate_scenario() -> tuple[pd.Series, pd.Series, pd.Series]:
    # Renewable is below load in every single hour: no battery, however
    # large, can close an annual energy deficit.
    index = _index()
    load = pd.Series(10.0, index=index)
    pv = pd.Series(2.0, index=index)
    wind = pd.Series(0.0, index=index)
    return pv, wind, load


def test_feasible_minimum_candidate_selected_correctly() -> None:
    pv, wind, load = _feasible_scenario()
    result = run_battery_search(pv, wind, load, n_max=6, lpsp_target=0.02, **COMMON_BATTERY_KWARGS)

    assert result.feasible is True
    assert result.infeasibility_reason is None
    assert result.lpsp <= 0.02

    # n=6 is also feasible (larger capacity can't be worse), but n=5 is the
    # smallest capacity that meets the target, so it must be selected --
    # never a larger, more expensive candidate.
    feasible_candidates = result.candidates[result.candidates["lpsp"] <= 0.02]
    assert result.optimal_n_modules == feasible_candidates["n_modules"].min()


def test_infeasible_battery_range_reported_with_correct_reason() -> None:
    pv, wind, load = _battery_range_infeasible_scenario()
    result = run_battery_search(pv, wind, load, n_max=5, lpsp_target=0.05, **COMMON_BATTERY_KWARGS)

    assert result.feasible is False
    assert result.optimal_n_modules is None
    assert result.optimal_capacity_kwh is None
    assert result.lpsp is None
    assert "No feasible battery size was found" in result.infeasibility_reason
    assert "Battery-range inadequacy" in result.infeasibility_reason
    assert pv.sum() + wind.sum() >= load.sum()  # confirms this is NOT a renewable-inadequacy case


def test_infeasible_renewable_inadequacy_reported_with_correct_reason() -> None:
    pv, wind, load = _renewable_inadequate_scenario()
    result = run_battery_search(pv, wind, load, n_max=5, lpsp_target=0.05, **COMMON_BATTERY_KWARGS)

    assert result.feasible is False
    assert "Renewable-generation inadequacy" in result.infeasibility_reason
    assert pv.sum() + wind.sum() < load.sum()  # confirms the annual energy deficit is real


def test_infeasible_result_never_called_optimal() -> None:
    pv, wind, load = _renewable_inadequate_scenario()
    result = run_battery_search(pv, wind, load, n_max=3, lpsp_target=0.01, **COMMON_BATTERY_KWARGS)
    assert result.feasible is False
    # An infeasible result must never report a specific "optimal" selection.
    assert result.optimal_n_modules is None
    assert result.optimal_capacity_kwh is None


def test_unserved_energy_non_increasing_with_capacity() -> None:
    pv, wind, load = _feasible_scenario()
    result = run_battery_search(pv, wind, load, n_max=6, lpsp_target=0.02, **COMMON_BATTERY_KWARGS)
    unserved = result.candidates.sort_values("n_modules")["unserved_energy_kwh"].to_numpy()
    diffs = unserved[1:] - unserved[:-1]
    assert (diffs <= 1e-6).all()  # never increases as capacity grows


def test_modular_capacity_calculated_correctly() -> None:
    pv, wind, load = _feasible_scenario()
    result = run_battery_search(pv, wind, load, n_max=6, lpsp_target=0.02, **COMMON_BATTERY_KWARGS)
    expected = result.candidates["n_modules"] * COMMON_BATTERY_KWARGS["module_capacity_kwh"]
    assert (result.candidates["nominal_battery_capacity_kwh"] == expected).all()


def test_reliability_sensitivity_targets_evaluated() -> None:
    pv, wind, load = _feasible_scenario()
    result = run_battery_search(
        pv, wind, load, n_max=6, lpsp_target=0.02,
        reliability_sensitivity_targets=[0.990, 0.995, 0.999], **COMMON_BATTERY_KWARGS,
    )
    assert list(result.reliability_sensitivity["load_served_target"]) == [0.990, 0.995, 0.999]
    assert "feasible" in result.reliability_sensitivity.columns


def test_rejects_negative_n_max() -> None:
    pv, wind, load = _feasible_scenario()
    with pytest.raises(ValueError, match="n_max"):
        run_battery_search(pv, wind, load, n_max=-1, lpsp_target=0.02, **COMMON_BATTERY_KWARGS)
