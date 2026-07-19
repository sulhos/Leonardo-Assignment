"""Tests for `src.physics.battery` and battery-level behaviour of
`src.physics.dispatch.run_dispatch` (PROJECT_BRIEF.md §26)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.physics.battery import BatterySpec, charge_efficiency, discharge_efficiency, usable_capacity_kwh
from src.physics.dispatch import run_dispatch

ROUND_TRIP_EFFICIENCY = 0.95


def _battery(n_modules: int = 2) -> BatterySpec:
    return BatterySpec(
        module_capacity_kwh=15.36,
        module_rated_power_kw=12.8,
        round_trip_efficiency=ROUND_TRIP_EFFICIENCY,
        min_soc_fraction=0.10,
        max_soc_fraction=0.95,
        initial_soc_fraction=0.50,
        n_modules=n_modules,
    )


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2023-06-15 00:00", periods=n, freq="h", tz="Asia/Shanghai")


def test_charge_and_discharge_efficiency_are_sqrt_of_round_trip() -> None:
    assert charge_efficiency(ROUND_TRIP_EFFICIENCY) == pytest.approx(math.sqrt(ROUND_TRIP_EFFICIENCY))
    assert discharge_efficiency(ROUND_TRIP_EFFICIENCY) == pytest.approx(math.sqrt(ROUND_TRIP_EFFICIENCY))


def test_usable_capacity_respects_soc_bounds() -> None:
    battery = _battery(n_modules=2)
    expected = battery.total_capacity_kwh * (0.95 - 0.10)
    assert usable_capacity_kwh(battery) == pytest.approx(expected)


def test_battery_spec_rejects_invalid_soc_bounds() -> None:
    with pytest.raises(ValueError, match="min_soc_fraction < max_soc_fraction"):
        BatterySpec(
            module_capacity_kwh=15.36, module_rated_power_kw=12.8, round_trip_efficiency=0.95,
            min_soc_fraction=0.95, max_soc_fraction=0.10, initial_soc_fraction=0.5, n_modules=1,
        )


def test_battery_spec_rejects_negative_modules() -> None:
    with pytest.raises(ValueError, match="n_modules"):
        BatterySpec(
            module_capacity_kwh=15.36, module_rated_power_kw=12.8, round_trip_efficiency=0.95,
            min_soc_fraction=0.10, max_soc_fraction=0.95, initial_soc_fraction=0.5, n_modules=-1,
        )


def test_soc_remains_within_bounds() -> None:
    battery = _battery(n_modules=1)  # small capacity to force bound-hitting
    n = 72
    index = _index(n)
    # Alternate strong surplus / strong deficit hours to stress both bounds.
    pv = pd.Series([50.0 if t % 2 == 0 else 0.0 for t in range(n)], index=index)
    wind = pd.Series(0.0, index=index)
    load = pd.Series(20.0, index=index)

    result = run_dispatch(pv, wind, load, battery)
    assert (result["soc_kwh"] >= battery.min_soc_kwh - 1e-9).all()
    assert (result["soc_kwh"] <= battery.max_soc_kwh + 1e-9).all()


def test_charge_power_limit_respected() -> None:
    battery = _battery(n_modules=3)
    n = 5
    index = _index(n)
    pv = pd.Series(500.0, index=index)  # huge surplus every hour
    wind = pd.Series(0.0, index=index)
    load = pd.Series(1.0, index=index)

    result = run_dispatch(pv, wind, load, battery)
    assert (result["battery_charge_kwh"] <= battery.total_rated_power_kw + 1e-9).all()


def test_discharge_power_limit_respected() -> None:
    battery = _battery(n_modules=3)
    n = 5
    index = _index(n)
    pv = pd.Series(0.0, index=index)
    wind = pd.Series(0.0, index=index)
    load = pd.Series(500.0, index=index)  # huge deficit every hour

    result = run_dispatch(pv, wind, load, battery, initial_soc_kwh=battery.max_soc_kwh)
    max_discharge_kw = battery.total_rated_power_kw * discharge_efficiency(ROUND_TRIP_EFFICIENCY)
    assert (result["battery_discharge_kwh"] <= max_discharge_kw + 1e-9).all()


def test_no_simultaneous_charge_and_discharge() -> None:
    battery = _battery(n_modules=2)
    n = 99
    index = _index(n)
    pv = pd.Series([30.0 if t % 3 == 0 else 5.0 for t in range(n)], index=index)
    wind = pd.Series(([0.0, 10.0, 3.0] * (n // 3 + 1))[:n], index=index)
    load = pd.Series(([8.0, 12.0, 6.0] * (n // 3 + 1))[:n], index=index)

    result = run_dispatch(pv, wind, load, battery)
    both = (result["battery_charge_kwh"] > 0) & (result["battery_discharge_kwh"] > 0)
    assert not both.any()


def test_no_energy_created_losses_are_nonnegative() -> None:
    battery = _battery(n_modules=2)
    n = 99
    index = _index(n)
    pv = pd.Series([30.0 if t % 3 == 0 else 5.0 for t in range(n)], index=index)
    wind = pd.Series(2.0, index=index)
    load = pd.Series(10.0, index=index)

    result = run_dispatch(pv, wind, load, battery)
    assert (result["battery_losses_kwh"] >= -1e-9).all()


def test_zero_capacity_battery_never_charges_or_discharges() -> None:
    battery = _battery(n_modules=0)
    n = 48
    index = _index(n)
    pv = pd.Series([30.0 if t % 2 == 0 else 0.0 for t in range(n)], index=index)
    wind = pd.Series(0.0, index=index)
    load = pd.Series(10.0, index=index)

    result = run_dispatch(pv, wind, load, battery)
    assert (result["battery_charge_kwh"] == 0).all()
    assert (result["battery_discharge_kwh"] == 0).all()
    assert (result["soc_kwh"] == 0).all()
    # With no storage, every surplus hour is fully curtailed and every deficit hour fully unserved.
    surplus_hours = pv + wind > load
    deficit_hours = pv + wind < load
    assert (result.loc[surplus_hours, "curtailed_kwh"] == (pv + wind - load)[surplus_hours]).all()
    assert (result.loc[deficit_hours, "unserved_kwh"] == (load - pv - wind)[deficit_hours]).all()
