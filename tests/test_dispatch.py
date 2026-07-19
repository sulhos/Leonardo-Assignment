"""Tests for `src.physics.dispatch` (PROJECT_BRIEF.md §26)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.physics.battery import BatterySpec
from src.physics.dispatch import resolve_initial_soc_bias, run_dispatch


def _battery(n_modules: int = 2) -> BatterySpec:
    return BatterySpec(
        module_capacity_kwh=15.36,
        module_rated_power_kw=12.8,
        round_trip_efficiency=0.95,
        min_soc_fraction=0.10,
        max_soc_fraction=0.95,
        initial_soc_fraction=0.50,
        n_modules=n_modules,
    )


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2023-06-15 00:00", periods=n, freq="h", tz="Asia/Shanghai")


def test_direct_supply_uses_renewable_before_battery() -> None:
    battery = _battery(n_modules=2)
    index = _index(3)
    pv = pd.Series([10.0, 10.0, 10.0], index=index)
    wind = pd.Series(0.0, index=index)
    load = pd.Series([4.0, 4.0, 4.0], index=index)

    result = run_dispatch(pv, wind, load, battery)
    # Renewable (10) exceeds load (4) every hour, so direct supply should
    # fully cover the load without ever touching unserved energy.
    assert (result["direct_supply_kwh"] == 4.0).all()
    assert (result["unserved_kwh"] == 0.0).all()


def test_surplus_charges_battery_before_curtailment() -> None:
    battery = _battery(n_modules=5)  # large headroom, won't hit max SOC
    index = _index(2)
    pv = pd.Series([50.0, 50.0], index=index)
    wind = pd.Series(0.0, index=index)
    load = pd.Series([10.0, 10.0], index=index)

    result = run_dispatch(pv, wind, load, battery, initial_soc_kwh=battery.min_soc_kwh)
    # With ample headroom and power capacity, surplus should go entirely to
    # charging (bounded by the rated power limit) before any curtailment.
    surplus = 40.0
    expected_charge = min(surplus, battery.total_rated_power_kw)
    assert result["battery_charge_kwh"].iloc[0] == pytest.approx(expected_charge)
    assert result["curtailed_kwh"].iloc[0] == pytest.approx(surplus - expected_charge)


def test_deficit_discharges_battery_before_unserved() -> None:
    battery = _battery(n_modules=5)
    index = _index(2)
    pv = pd.Series([0.0, 0.0], index=index)
    wind = pd.Series(0.0, index=index)
    load = pd.Series([5.0, 5.0], index=index)

    result = run_dispatch(pv, wind, load, battery, initial_soc_kwh=battery.max_soc_kwh)
    # With ample stored energy and power capacity, the deficit should be
    # fully covered by battery discharge, leaving zero unserved energy.
    assert result["battery_discharge_kwh"].iloc[0] == pytest.approx(5.0)
    assert result["unserved_kwh"].iloc[0] == pytest.approx(0.0)


def test_remaining_deficit_becomes_unserved_when_battery_exhausted() -> None:
    battery = _battery(n_modules=1)
    index = _index(1)
    pv = pd.Series([0.0], index=index)
    wind = pd.Series(0.0, index=index)
    load = pd.Series([1000.0], index=index)  # far beyond what the battery can supply

    result = run_dispatch(pv, wind, load, battery, initial_soc_kwh=battery.max_soc_kwh)
    assert result["unserved_kwh"].iloc[0] > 0
    assert result["unserved_kwh"].iloc[0] < 1000.0  # battery still contributed something


def test_soc_bias_resolution_converges_for_baseline_battery() -> None:
    battery = _battery(n_modules=3)
    n = 24 * 10  # 10 representative days
    index = _index(n)
    pv = pd.Series([20.0 if (t % 24) in range(7, 18) else 0.0 for t in range(n)], index=index)
    wind = pd.Series(2.0, index=index)
    load = pd.Series(6.0, index=index)

    result = resolve_initial_soc_bias(pv, wind, load, battery, max_iterations=20, tolerance_kwh=1e-2)
    assert result.attrs["soc_bias_converged"] is True
    assert result.attrs["soc_bias_iterations"] <= 20


def test_soc_bias_resolution_trivial_for_zero_capacity_battery() -> None:
    battery = _battery(n_modules=0)
    n = 24
    index = _index(n)
    pv = pd.Series(5.0, index=index)
    wind = pd.Series(0.0, index=index)
    load = pd.Series(5.0, index=index)

    result = resolve_initial_soc_bias(pv, wind, load, battery)
    assert result.attrs["soc_bias_converged"] is True
    assert result.attrs["soc_bias_iterations"] == 1


def test_run_dispatch_rejects_mismatched_indices() -> None:
    battery = _battery(n_modules=1)
    pv = pd.Series([1.0, 2.0], index=_index(2))
    wind = pd.Series([1.0, 2.0], index=_index(2))
    load = pd.Series([1.0, 2.0, 3.0], index=_index(3))
    with pytest.raises(ValueError, match="same index"):
        run_dispatch(pv, wind, load, battery)
