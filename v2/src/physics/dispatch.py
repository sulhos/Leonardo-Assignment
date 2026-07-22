"""Hourly mechanistic battery-dispatch simulation, audited against the
course lecture's exact energy-balance equation (PROJECT_BRIEF.md §11;
Task 1c, V2).

Dispatch priority per hour, matching PROJECT_BRIEF.md §11 exactly:
1. PV and wind generation directly supply the load.
2. Renewable surplus charges the battery.
3. During a renewable deficit, the battery discharges.
4. Remaining surplus becomes curtailed energy.
5. Remaining deficit becomes unserved energy.

Energy-flow convention (documented explicitly, per §11's requirement):
- `battery_charge_kwh[t]`: energy leaving the renewable surplus and entering
  the battery's input terminals during hour t (before charge losses).
- `battery_discharge_kwh[t]`: energy delivered to the load from the battery
  during hour t (after discharge losses) -- i.e. this is what the load
  actually receives, not the energy removed from the stored SOC.
- `battery_losses_kwh[t]`: charge-side losses (`battery_charge_kwh * (1 - charge_eff)`)
  plus discharge-side losses (`energy removed from SOC - battery_discharge_kwh`).
  Does NOT include self-discharge (see `self_discharge_kwh` below) -- these
  are two physically distinct loss mechanisms (conversion inefficiency vs.
  idle capacity decay), kept in separate columns.
- `self_discharge_kwh[t]`: SOC lost to idle self-discharge during hour t,
  applied to the *previous* hour's ending SOC before any charge/discharge
  activity that hour (lecture eq., Task 1c).
- State update (lecture eq., Task 1c -- generalizes PROJECT_BRIEF.md §11's
  original zero-self-discharge formula):
      SOC[t] = clip( SOC[t-1]*(1 - SDR) + charge_efficiency*battery_charge_kwh[t]
                       - battery_discharge_kwh[t]/discharge_efficiency,
                      E_min, E_max )
  `SDR` = `self_discharge_rate_per_hour` (`BatterySpec`, default 0.0 -- see
  `battery.py` module docstring).

No simultaneous charge+discharge: each hour is either a surplus hour
(charge only) or a deficit hour (discharge only), by construction of the
single net-renewable-vs-load comparison -- never both.

Performance: the per-hour SOC state dependency makes this loop a poor fit
for numpy vectorization (each hour depends on the previous hour's SOC) but
an excellent fit for JIT compilation -- `_dispatch_core` is numba-jitted
(~40x faster than the equivalent pure-Python loop, verified bit-for-bit
identical output before being adopted; added during the industrial-scale
pivot, PROJECT_BRIEF.md Addendum 2, to keep the wider 0-80 module search
range's runtime reasonable). This is an implementation-only change --
same algorithm, same numbers, not a model simplification.
"""

from __future__ import annotations

import logging

import numba
import numpy as np
import pandas as pd

from src.physics.battery import BatterySpec, charge_efficiency, discharge_efficiency

logger = logging.getLogger(__name__)


@numba.njit(cache=True)
def _dispatch_core(
    pv_kw: np.ndarray, wind_kw: np.ndarray, load_kw: np.ndarray,
    charge_eff: float, discharge_eff: float, min_soc_kwh: float, max_soc_kwh: float,
    rated_power_kwh_per_hour: float, initial_soc_kwh: float, self_discharge_rate_per_hour: float,
):
    """Numba-JIT'd hourly dispatch loop -- identical algorithm to the
    (formerly pure-Python) loop this replaces, verified bit-for-bit
    equivalent on real scenario data before being adopted (see
    tests/test_dispatch.py). This is purely an implementation-level
    speedup (~40x, since the loop's per-hour scalar state dependency
    makes it a poor fit for numpy vectorization but an excellent fit for
    JIT compilation) -- it changes nothing about the model itself.

    Self-discharge (Task 1c, V2) is applied to the previous hour's ending
    SOC at the start of each hour, before that hour's charge/discharge
    decision -- matching the lecture's `E_s,t-1*(1-SDR)` term. With
    `self_discharge_rate_per_hour=0.0` (the `BatterySpec` default), this is
    a no-op and dispatch behaves exactly as before Task 1c.
    """
    n = len(load_kw)
    direct_supply = np.zeros(n)
    battery_charge = np.zeros(n)
    battery_discharge = np.zeros(n)
    battery_losses = np.zeros(n)
    self_discharge = np.zeros(n)
    curtailed = np.zeros(n)
    unserved = np.zeros(n)
    soc_trace = np.zeros(n)

    soc = min(max(initial_soc_kwh, min_soc_kwh), max_soc_kwh)

    for t in range(n):
        soc_before_self_discharge = soc
        soc = soc * (1.0 - self_discharge_rate_per_hour)
        self_discharge[t] = soc_before_self_discharge - soc

        renewable = pv_kw[t] + wind_kw[t]
        net = renewable - load_kw[t]
        direct_supply[t] = min(renewable, load_kw[t])

        if net > 0:
            surplus = net
            headroom_kwh = max(max_soc_kwh - soc, 0.0)
            headroom_input_kwh = headroom_kwh / charge_eff if charge_eff > 0 else 0.0
            surplus_used = min(surplus, rated_power_kwh_per_hour, headroom_input_kwh)
            stored = surplus_used * charge_eff
            soc += stored
            battery_charge[t] = surplus_used
            battery_losses[t] = surplus_used - stored
            curtailed[t] = surplus - surplus_used
        elif net < 0:
            deficit = -net
            available_kwh = max(soc - min_soc_kwh, 0.0)
            deliverable_kwh = available_kwh * discharge_eff
            discharge_delivered = min(deficit, rated_power_kwh_per_hour * discharge_eff, deliverable_kwh)
            removed_from_soc = discharge_delivered / discharge_eff if discharge_eff > 0 else 0.0
            soc -= removed_from_soc
            battery_discharge[t] = discharge_delivered
            battery_losses[t] = removed_from_soc - discharge_delivered
            unserved[t] = deficit - discharge_delivered

        # Lecture eq. (Task 1c): SOC is clipped to [E_min, E_max] after the
        # full update. Charge/discharge already respect these bounds via
        # headroom/available-energy limits above; this only matters for the
        # (rare, and only with a non-trivial SDR) case of self-discharge
        # alone pushing SOC fractionally below E_min in a net == 0 hour.
        soc = min(max(soc, min_soc_kwh), max_soc_kwh)
        soc_trace[t] = soc

    return direct_supply, battery_charge, battery_discharge, battery_losses, self_discharge, curtailed, unserved, soc_trace, soc

DISPATCH_COLUMNS = [
    "pv_kw",
    "wind_kw",
    "renewable_kw",
    "load_kw",
    "direct_supply_kwh",
    "battery_charge_kwh",
    "battery_discharge_kwh",
    "battery_losses_kwh",
    "self_discharge_kwh",
    "curtailed_kwh",
    "unserved_kwh",
    "load_served_kwh",
    "soc_kwh",
]


def run_dispatch(
    pv_generation_kw: pd.Series,
    wind_generation_kw: pd.Series,
    load_kw: pd.Series,
    battery: BatterySpec,
    initial_soc_kwh: float | None = None,
) -> pd.DataFrame:
    """Run the full-year hourly battery-dispatch simulation.

    Args:
        pv_generation_kw: Hourly PV output, kW.
        wind_generation_kw: Hourly wind output, kW.
        load_kw: Hourly load, kW.
        battery: Battery specification (including module count).
        initial_soc_kwh: Starting SOC in kWh. Defaults to
            `battery.initial_soc_kwh` (from `initial_soc_fraction`) if not given
            -- used by `resolve_initial_soc_bias` to iterate toward a
            converged starting SOC.

    Returns:
        Hourly DataFrame (see `DISPATCH_COLUMNS`) with one row per input hour.
    """
    if not (pv_generation_kw.index.equals(wind_generation_kw.index) and pv_generation_kw.index.equals(load_kw.index)):
        raise ValueError("pv_generation_kw, wind_generation_kw, and load_kw must share the same index.")

    pv = pv_generation_kw.to_numpy()
    wind = wind_generation_kw.to_numpy()
    load = load_kw.to_numpy()
    renewable = pv + wind

    c = charge_efficiency(battery.round_trip_efficiency)
    d = discharge_efficiency(battery.round_trip_efficiency)
    min_soc = battery.min_soc_kwh
    max_soc = battery.max_soc_kwh
    rated_power_kwh_per_hour = battery.total_rated_power_kw  # kW for 1 hour = kWh

    soc0 = initial_soc_kwh if initial_soc_kwh is not None else battery.initial_soc_kwh

    direct_supply, battery_charge, battery_discharge, battery_losses, self_discharge, curtailed, unserved, soc_trace, soc = (
        _dispatch_core(
            pv, wind, load, c, d, min_soc, max_soc, rated_power_kwh_per_hour, soc0,
            battery.self_discharge_rate_per_hour,
        )
    )
    load_served = direct_supply + battery_discharge

    result = pd.DataFrame(
        {
            "pv_kw": pv,
            "wind_kw": wind,
            "renewable_kw": renewable,
            "load_kw": load,
            "direct_supply_kwh": direct_supply,
            "battery_charge_kwh": battery_charge,
            "battery_discharge_kwh": battery_discharge,
            "battery_losses_kwh": battery_losses,
            "self_discharge_kwh": self_discharge,
            "curtailed_kwh": curtailed,
            "unserved_kwh": unserved,
            "load_served_kwh": load_served,
            "soc_kwh": soc_trace,
        },
        index=load_kw.index,
    )
    result.attrs["battery"] = battery
    result.attrs["final_soc_kwh"] = soc
    return result


def resolve_initial_soc_bias(
    pv_generation_kw: pd.Series,
    wind_generation_kw: pd.Series,
    load_kw: pd.Series,
    battery: BatterySpec,
    method: str = "repeated_year",
    max_iterations: int = 10,
    tolerance_kwh: float = 1e-3,
) -> pd.DataFrame:
    """Run dispatch repeatedly until starting and ending SOC converge,
    avoiding an arbitrary initial-SOC bias (PROJECT_BRIEF.md §11, method A:
    repeated-year simulation).

    Method B (a closed-form cyclic SOC solution) is not implemented: the
    repeated-year method is simple, transparent, and converges in a handful
    of iterations for this system's efficiency/capacity ranges, which is
    sufficient for this project's scope.
    """
    if method != "repeated_year":
        raise NotImplementedError(f"method={method!r} not implemented; only 'repeated_year' is supported.")

    if battery.total_capacity_kwh == 0:
        # No storage: dispatch is identical regardless of "starting SOC", converges trivially.
        result = run_dispatch(pv_generation_kw, wind_generation_kw, load_kw, battery, initial_soc_kwh=0.0)
        result.attrs["soc_bias_converged"] = True
        result.attrs["soc_bias_iterations"] = 1
        return result

    starting_soc = battery.initial_soc_kwh
    result = None
    converged = False
    for iteration in range(1, max_iterations + 1):
        result = run_dispatch(pv_generation_kw, wind_generation_kw, load_kw, battery, initial_soc_kwh=starting_soc)
        ending_soc = result.attrs["final_soc_kwh"]
        delta = abs(ending_soc - starting_soc)
        logger.debug("SOC-bias iteration %d: start=%.4f end=%.4f delta=%.6f", iteration, starting_soc, ending_soc, delta)
        if delta <= tolerance_kwh:
            converged = True
            break
        starting_soc = ending_soc

    result.attrs["soc_bias_converged"] = converged
    result.attrs["soc_bias_iterations"] = iteration
    if not converged:
        logger.warning(
            "Initial-SOC bias did not converge within %d iterations (tolerance=%.4f kWh).",
            max_iterations, tolerance_kwh,
        )
    return result
