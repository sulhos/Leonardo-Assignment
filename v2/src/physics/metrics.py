"""Mechanistic performance metrics per battery candidate (PROJECT_BRIEF.md §13).

`compute_candidate_metrics` covers the renewables+battery-only dispatch
(unchanged from the original off-grid, no-backup-generator design) --
load-served fraction, renewable utilization, curtailment, LPSP, and
battery-only economics.

`compute_system_metrics` (added for the diesel/system-LCOE pivot,
PROJECT_BRIEF.md Addendum 3) layers the diesel generator and PV/wind capital
costs on top of a `compute_candidate_metrics` result, producing the
system-level `renewable_share` and `system_lcoe_eur_per_kwh` that the new
optimization objective (`src/physics/optimization.py`) minimizes. It expects
a dispatch result that has already been through
`src.physics.diesel.apply_diesel_backup`.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.physics.battery import BatterySpec, usable_capacity_kwh
from src.physics.diesel import DieselSpec, diesel_annual_fuel_cost_eur, diesel_annualized_cost_eur
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
        "battery_self_discharge_kwh": dispatch_result["self_discharge_kwh"].sum(),
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


def compute_system_metrics(
    dispatch_result_with_diesel: pd.DataFrame,
    battery_equivalent_annual_cost_eur: float,
    pv_capacity_kwp: float,
    pv_cfg: dict,
    wind_capacity_kw: float,
    wind_cfg: dict,
    diesel: DieselSpec,
    project_lifetime_years: int,
    real_discount_rate: float,
) -> dict:
    """System-level metrics (PROJECT_BRIEF.md Addendum 3): renewable share and
    system LCOE, combining PV + wind + battery + diesel capital/O&M costs with
    diesel fuel cost, over total energy served. `dispatch_result_with_diesel`
    must already have been through `src.physics.diesel.apply_diesel_backup`
    (i.e. carries `diesel_output_kwh`, `diesel_fuel_l`, `still_unserved_kwh`).

    `pv_cfg`/`wind_cfg` are the `config/*.yaml` `pv:`/`wind:` sub-dicts,
    read directly for `installed_cost_eur_per_kwp`/`installed_cost_eur_per_kw`,
    `economic_lifetime_years`, and `om_cost_fraction_per_year`.
    """
    total_load_kwh = dispatch_result_with_diesel["load_kw"].sum()
    load_served_kwh = dispatch_result_with_diesel["load_served_kwh"].sum()
    diesel_energy_kwh = dispatch_result_with_diesel["diesel_output_kwh"].sum()
    still_unserved_kwh = dispatch_result_with_diesel["still_unserved_kwh"].sum()

    renewable_share = (
        1.0 - (diesel_energy_kwh + still_unserved_kwh) / total_load_kwh if total_load_kwh > 0 else 1.0
    )
    lpsp_after_diesel = still_unserved_kwh / total_load_kwh if total_load_kwh > 0 else 0.0

    pv_present_value = present_value_cost(
        pv_cfg["installed_cost_eur_per_kwp"], pv_capacity_kwp,
        pv_cfg["economic_lifetime_years"], project_lifetime_years, real_discount_rate,
    )
    pv_eac = equivalent_annual_cost(pv_present_value, project_lifetime_years, real_discount_rate)
    pv_om_eac = pv_cfg["om_cost_fraction_per_year"] * pv_cfg["installed_cost_eur_per_kwp"] * pv_capacity_kwp

    wind_present_value = present_value_cost(
        wind_cfg["installed_cost_eur_per_kw"], wind_capacity_kw,
        wind_cfg["economic_lifetime_years"], project_lifetime_years, real_discount_rate,
    )
    wind_eac = equivalent_annual_cost(wind_present_value, project_lifetime_years, real_discount_rate)
    wind_om_eac = wind_cfg["om_cost_fraction_per_year"] * wind_cfg["installed_cost_eur_per_kw"] * wind_capacity_kw

    diesel_capital_eac = diesel_annualized_cost_eur(diesel, real_discount_rate)
    diesel_fuel_cost_eur = diesel_annual_fuel_cost_eur(dispatch_result_with_diesel, diesel)

    total_annualized_cost_eur = (
        pv_eac + pv_om_eac + wind_eac + wind_om_eac
        + battery_equivalent_annual_cost_eur + diesel_capital_eac + diesel_fuel_cost_eur
    )
    system_lcoe = cost_per_kwh_served(total_annualized_cost_eur, load_served_kwh)

    return {
        "diesel_rated_power_kw": diesel.rated_power_kw,
        "diesel_annual_energy_kwh": diesel_energy_kwh,
        "diesel_annual_fuel_l": dispatch_result_with_diesel["diesel_fuel_l"].sum(),
        "diesel_annual_fuel_cost_eur": diesel_fuel_cost_eur,
        "diesel_annualized_capital_cost_eur": diesel_capital_eac,
        "still_unserved_energy_kwh": still_unserved_kwh,
        "lpsp_after_diesel": lpsp_after_diesel,
        "renewable_share": renewable_share,
        "pv_annualized_cost_eur": pv_eac + pv_om_eac,
        "wind_annualized_cost_eur": wind_eac + wind_om_eac,
        "battery_annualized_cost_eur": battery_equivalent_annual_cost_eur,
        "total_annualized_cost_eur": total_annualized_cost_eur,
        "system_lcoe_eur_per_kwh": system_lcoe,
    }
