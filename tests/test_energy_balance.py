"""Tests for internal energy-balance consistency of `src.physics.dispatch.run_dispatch`
(PROJECT_BRIEF.md §26). Uses synthetic, seeded, non-internet fixtures spanning
several hundred hours with varied battery sizes (including zero capacity).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.physics.battery import BatterySpec, charge_efficiency, discharge_efficiency
from src.physics.dispatch import run_dispatch

TOLERANCE_KWH = 1e-6


def _random_scenario(n: int, seed: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-01", periods=n, freq="h", tz="Asia/Shanghai")
    pv = pd.Series(rng.uniform(0, 25, size=n), index=index)
    wind = pd.Series(rng.uniform(0, 10, size=n), index=index)
    load = pd.Series(rng.uniform(1, 15, size=n), index=index)
    return pv, wind, load


def _battery(n_modules: int) -> BatterySpec:
    return BatterySpec(
        module_capacity_kwh=15.36,
        module_rated_power_kw=12.8,
        round_trip_efficiency=0.95,
        min_soc_fraction=0.10,
        max_soc_fraction=0.95,
        initial_soc_fraction=0.50,
        n_modules=n_modules,
    )


@pytest.mark.parametrize("n_modules", [0, 1, 3, 8])
def test_renewable_side_balance_holds_every_hour(n_modules: int) -> None:
    pv, wind, load = _random_scenario(500, seed=100 + n_modules)
    battery = _battery(n_modules)
    result = run_dispatch(pv, wind, load, battery)

    renewable = result["pv_kw"] + result["wind_kw"]
    accounted = result["direct_supply_kwh"] + result["battery_charge_kwh"] + result["curtailed_kwh"]
    residual = (renewable - accounted).abs()
    assert (residual < TOLERANCE_KWH).all()


@pytest.mark.parametrize("n_modules", [0, 1, 3, 8])
def test_load_side_balance_holds_every_hour(n_modules: int) -> None:
    pv, wind, load = _random_scenario(500, seed=200 + n_modules)
    battery = _battery(n_modules)
    result = run_dispatch(pv, wind, load, battery)

    accounted = result["direct_supply_kwh"] + result["battery_discharge_kwh"] + result["unserved_kwh"]
    residual = (result["load_kw"] - accounted).abs()
    assert (residual < TOLERANCE_KWH).all()

    # load_served_kwh + unserved_kwh must equal the load exactly too.
    residual2 = (result["load_kw"] - (result["load_served_kwh"] + result["unserved_kwh"])).abs()
    assert (residual2 < TOLERANCE_KWH).all()


@pytest.mark.parametrize("n_modules", [1, 3, 8])
def test_soc_state_equation_holds_every_hour(n_modules: int) -> None:
    pv, wind, load = _random_scenario(300, seed=300 + n_modules)
    battery = _battery(n_modules)
    result = run_dispatch(pv, wind, load, battery)

    c = charge_efficiency(battery.round_trip_efficiency)
    d = discharge_efficiency(battery.round_trip_efficiency)

    soc_prev = battery.initial_soc_kwh
    for _, row in result.iterrows():
        expected_soc = soc_prev + c * row["battery_charge_kwh"] - row["battery_discharge_kwh"] / d
        assert expected_soc == pytest.approx(row["soc_kwh"], abs=1e-6)
        soc_prev = row["soc_kwh"]


@pytest.mark.parametrize("n_modules", [0, 1, 3, 8])
def test_annual_scale_residual_within_tolerance(n_modules: int) -> None:
    pv, wind, load = _random_scenario(8760, seed=400 + n_modules)
    battery = _battery(n_modules)
    result = run_dispatch(pv, wind, load, battery)

    renewable_total = (result["pv_kw"] + result["wind_kw"]).sum()
    accounted_total = (
        result["direct_supply_kwh"].sum() + result["battery_charge_kwh"].sum() + result["curtailed_kwh"].sum()
    )
    assert abs(renewable_total - accounted_total) < TOLERANCE_KWH * len(result)

    load_total = result["load_kw"].sum()
    load_accounted_total = result["load_served_kwh"].sum() + result["unserved_kwh"].sum()
    assert abs(load_total - load_accounted_total) < TOLERANCE_KWH * len(result)


def test_battery_losses_equal_charge_and_discharge_side_losses() -> None:
    pv, wind, load = _random_scenario(500, seed=999)
    battery = _battery(4)
    result = run_dispatch(pv, wind, load, battery)

    c = charge_efficiency(battery.round_trip_efficiency)
    d = discharge_efficiency(battery.round_trip_efficiency)

    charge_losses = result["battery_charge_kwh"] * (1 - c)
    discharge_losses = result["battery_discharge_kwh"] * (1 / d - 1)
    expected_losses = charge_losses + discharge_losses

    assert np.allclose(result["battery_losses_kwh"].to_numpy(), expected_losses.to_numpy(), atol=1e-6)
