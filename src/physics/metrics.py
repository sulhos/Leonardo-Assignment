"""Mechanistic performance metrics per battery candidate (PROJECT_BRIEF.md §13).

Because the system is fully off-grid with no fossil generator, all served
energy is renewable by construction -- metrics here emphasize load-served
fraction, renewable utilization, curtailment, and reliability (LPSP) rather
than a meaningless "renewable share of served energy" figure.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.physics.battery import BatterySpec, usable_capacity_kwh
from src.physics.economics import cost_per_kwh_served, equivalent_annual_cost, present_value_cost

logger = logging.getLogger(__name__)


def lpsp(dispatch_result: pd.DataFrame) -> float:
    """Loss of Power Supply Probability: unserved energy / total load energy."""
    total_load = dispatch_result["load_kw"].sum()
    if total_load == 0:
        return 0.0
    return dispatch_result["unserved_kwh"].sum() / total_load


def compute_candidate_metrics(
    dispatch_result: pd.DataFrame,
    battery: BatterySpec,
    installed_cost_eur_per_kwh: float,
    economic_lifetime_years: int,
    project_lifetime_years: int,
    real_discount_rate: float,
    candidate_runtime_seconds: float,
) -> dict:
    """Compute the full metric set (PROJECT_BRIEF.md §13) for one dispatch
    simulation result (one battery candidate for one scenario)."""
    annual_load_kwh = dispatch_result["load_kw"].sum()
    annual_pv_kwh = dispatch_result["pv_kw"].sum()
    annual_wind_kwh = dispatch_result["wind_kw"].sum()
    total_renewable_kwh = annual_pv_kwh + annual_wind_kwh
    curtailed_kwh = dispatch_result["curtailed_kwh"].sum()
    unserved_kwh = dispatch_result["unserved_kwh"].sum()
    load_served_kwh = dispatch_result["load_served_kwh"].sum()
    battery_charge_kwh = dispatch_result["battery_charge_kwh"].sum()
    battery_discharge_kwh = dispatch_result["battery_discharge_kwh"].sum()

    usable_kwh = usable_capacity_kwh(battery)
    pv_cost = present_value_cost(
        installed_cost_eur_per_kwh, battery.total_capacity_kwh,
        economic_lifetime_years, project_lifetime_years, real_discount_rate,
    )
    eac = equivalent_annual_cost(pv_cost, project_lifetime_years, real_discount_rate)

    return {
        "n_modules": battery.n_modules,
        "nominal_battery_capacity_kwh": battery.total_capacity_kwh,
        "usable_battery_capacity_kwh": usable_kwh,
        "rated_battery_power_kw": battery.total_rated_power_kw,
        "annual_load_kwh": annual_load_kwh,
        "peak_load_kw": dispatch_result["load_kw"].max(),
        "annual_pv_kwh": annual_pv_kwh,
        "annual_wind_kwh": annual_wind_kwh,
        "total_renewable_kwh": total_renewable_kwh,
        "direct_renewable_consumption_kwh": dispatch_result["direct_supply_kwh"].sum(),
        "load_served_kwh": load_served_kwh,
        "load_served_fraction": load_served_kwh / annual_load_kwh if annual_load_kwh > 0 else 1.0,
        "lpsp": lpsp(dispatch_result),
        "unserved_energy_kwh": unserved_kwh,
        "hours_with_unmet_load": int((dispatch_result["unserved_kwh"] > 0).sum()),
        "curtailed_energy_kwh": curtailed_kwh,
        "renewable_utilization": (
            (total_renewable_kwh - curtailed_kwh) / total_renewable_kwh if total_renewable_kwh > 0 else 0.0
        ),
        "battery_charge_kwh": battery_charge_kwh,
        "battery_discharge_kwh": battery_discharge_kwh,
        "battery_losses_kwh": dispatch_result["battery_losses_kwh"].sum(),
        "min_soc_kwh": dispatch_result["soc_kwh"].min(),
        "max_soc_kwh": dispatch_result["soc_kwh"].max(),
        "mean_soc_kwh": dispatch_result["soc_kwh"].mean(),
        "battery_throughput_kwh": battery_charge_kwh + battery_discharge_kwh,
        "equivalent_full_cycles": battery_discharge_kwh / usable_kwh if usable_kwh > 0 else 0.0,
        "initial_cost_eur": installed_cost_eur_per_kwh * battery.total_capacity_kwh,
        "replacement_cost_eur": (
            installed_cost_eur_per_kwh * battery.total_capacity_kwh
            if economic_lifetime_years < project_lifetime_years else 0.0
        ),
        "present_value_cost_eur": pv_cost,
        "equivalent_annual_cost_eur": eac,
        "cost_per_kwh_served_eur": cost_per_kwh_served(eac, load_served_kwh),
        "candidate_runtime_seconds": candidate_runtime_seconds,
    }
