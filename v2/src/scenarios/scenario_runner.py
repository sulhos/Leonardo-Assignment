"""Run the mechanistic pipeline (weather -> load -> PV -> wind -> dispatch ->
battery optimization) for one scenario (PROJECT_BRIEF.md §14).

Weather is loaded once per (location, weather_year) outside this function
and passed in, since it doesn't vary across scenarios sharing the same
location/year -- avoids redundant API/cache reads across hundreds of
scenario runs. Only PV/wind capacity, load parameters, reliability target,
battery efficiency, and usable SOC window vary per scenario (matching the
system boundary in PROJECT_BRIEF.md §3: PV/wind capacities are scenario
inputs, not decision variables within a single optimization).

Label field names (`optimal_capacity_kwh`, `optimal_n_modules`,
`reference_lpsp`, `reference_annualized_cost`) intentionally match
`src.ai.features.LEAKAGE_COLUMNS` exactly, so the leakage guard there
applies directly without a name-mapping step.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.physics.load_profile import generate_load_profile
from src.physics.optimization import run_battery_search
from src.physics.pv_model import compute_pv_generation
from src.physics.wind_model import compute_wind_generation

logger = logging.getLogger(__name__)


def run_scenario(scenario: dict, site_config: dict, weather: pd.DataFrame, n_max: int) -> dict:
    """Run the full mechanistic pipeline for one scenario and return its
    labels (optimal capacity/modules, feasibility, LPSP, cost, runtime).

    Args:
        scenario: One scenario dict from `src.scenarios.sampling.sample_scenarios`.
        site_config: The fixed site configuration (siting assumptions for PV
            tilt/azimuth/losses, wind hub height/shear/curve, battery module
            spec, and economics) -- everything that does NOT vary per scenario.
        weather: Pre-loaded local-year hourly weather for the scenario's site/year.
        n_max: Maximum battery module count to search (fixed per dataset run,
            from `config/scenario_generation.yaml` -> `candidate_module_counts.max`).

    Returns:
        A dict combining the input scenario with its mechanistic labels.
    """
    load = generate_load_profile(
        annual_consumption_kwh=scenario["annual_load_kwh"],
        peak_load_kw=scenario["peak_load_kw"],
        year=scenario["weather_year"],
        timezone=scenario["timezone"],
        random_seed=scenario["random_seed"],
    )

    pv_cfg = site_config["pv"]
    pv = compute_pv_generation(
        weather=weather,
        capacity_kwp=scenario["pv_capacity_kwp"],
        latitude=site_config["site"]["latitude"],
        longitude=site_config["site"]["longitude"],
        tilt_deg=pv_cfg["tilt_deg"],
        azimuth_deg=pv_cfg["azimuth_deg"],
        system_losses_fraction=pv_cfg["system_losses_fraction"],
        inverter_efficiency=pv_cfg["inverter_efficiency"],
    )

    wind_cfg = site_config["wind"]
    wind = compute_wind_generation(
        weather=weather,
        rated_power_kw=scenario["wind_capacity_kw"],
        reference_height_m=wind_cfg["weather_reference_height_m"],
        hub_height_m=wind_cfg["hub_height_m"],
        shear_exponent=wind_cfg["wind_shear_exponent"],
        power_curve_speed_power_fraction=wind_cfg["power_curve_speed_power_fraction"],
    )

    battery_cfg = site_config["battery"]
    usable_window = scenario["usable_soc_window_fraction"]
    min_soc_fraction = (1.0 - usable_window) / 2.0
    max_soc_fraction = min_soc_fraction + usable_window

    lpsp_target = 1.0 - scenario["reliability_target_load_served"]
    search_result = run_battery_search(
        pv, wind, load,
        n_max=n_max,
        module_capacity_kwh=battery_cfg["module_capacity_kwh"],
        module_rated_power_kw=battery_cfg["module_rated_power_kw"],
        round_trip_efficiency=scenario["round_trip_efficiency"],
        min_soc_fraction=min_soc_fraction,
        max_soc_fraction=max_soc_fraction,
        initial_soc_fraction=battery_cfg["initial_soc_fraction"],
        self_discharge_rate_per_hour=battery_cfg["self_discharge_rate_per_hour"],
        installed_cost_eur_per_kwh=battery_cfg["installed_cost_eur_per_kwh"],
        economic_lifetime_years=battery_cfg["economic_lifetime_years"],
        project_lifetime_years=battery_cfg["project_lifetime_years"],
        real_discount_rate=site_config["economics"]["real_discount_rate"],
        pv_capacity_kwp=scenario["pv_capacity_kwp"],
        pv_cfg=pv_cfg,
        wind_capacity_kw=scenario["wind_capacity_kw"],
        wind_cfg=wind_cfg,
        diesel_cfg=site_config["diesel"],
        lpsp_target=lpsp_target,
        system_backup=site_config["system"]["backup"],
    )

    labels = {
        "feasible": search_result.feasible,
        "optimal_n_modules": search_result.optimal_n_modules,
        "optimal_capacity_kwh": search_result.optimal_capacity_kwh,
        "reference_lpsp": search_result.lpsp,
        "reference_system_lcoe_eur_per_kwh": search_result.system_lcoe_eur_per_kwh,
        "reference_renewable_share": search_result.renewable_share,
        "reference_annualized_cost": (
            search_result.candidates.loc[
                search_result.candidates["n_modules"] == search_result.optimal_n_modules,
                "equivalent_annual_cost_eur",
            ].iloc[0]
            if search_result.optimal_n_modules is not None else None
        ),
        "infeasibility_reason": search_result.infeasibility_reason,
        "mechanistic_optimization_runtime_seconds": search_result.search_runtime_seconds,
        "annual_pv_kwh": pv.sum(),
        "annual_wind_kwh": wind.sum(),
        "annual_renewable_kwh": pv.sum() + wind.sum(),
    }

    logger.debug(
        "Scenario %d: feasible=%s optimal_n=%s runtime=%.3fs",
        scenario["scenario_id"], labels["feasible"], labels["optimal_n_modules"],
        labels["mechanistic_optimization_runtime_seconds"],
    )
    return {**scenario, **labels}
