"""Mandatory physical (mechanistic) verification of AI predictions
(PROJECT_BRIEF.md §19-§20; refinement addendum §1.6 -- do not skip this stage,
even under time pressure; it is the most important scientific check in the
whole project).

Accuracy metrics (Stage 5/6) are NOT sufficient on their own. For every
model's prediction on the test split, this module:

1. Converts the continuous prediction to an installable module count via
   ceiling division (never rounds down).
2. Re-runs the full hourly mechanistic dispatch model (`src.physics.dispatch`)
   with that installed battery size -- using the scenario's OWN
   round_trip_efficiency and usable_soc_window_fraction, not site defaults.
3. Also re-runs dispatch for the mechanistic-optimal battery size
   (`optimal_n_modules`) for the same scenario, so the comparison uses two
   freshly-computed, directly comparable metric sets rather than diffing
   against possibly-stale stored summary stats.
4. Reports reliability pass/fail (against the scenario's OWN reliability
   target, not a fixed threshold), over/undersizing, and the cost/curtailment
   consequences of each.
"""

from __future__ import annotations

import logging
import math

import pandas as pd

from src.physics.battery import BatterySpec
from src.physics.dispatch import resolve_initial_soc_bias
from src.physics.load_profile import generate_load_profile
from src.physics.metrics import compute_candidate_metrics
from src.physics.pv_model import compute_pv_generation
from src.physics.wind_model import compute_wind_generation

logger = logging.getLogger(__name__)


def predicted_capacity_to_modules(predicted_capacity_kwh: float, module_capacity_kwh: float) -> int:
    """Ceiling conversion from a continuous prediction to an installable
    module count. Never rounds down (PROJECT_BRIEF.md §19)."""
    if predicted_capacity_kwh < 0:
        raise ValueError(f"predicted_capacity_kwh must be non-negative, got {predicted_capacity_kwh}.")
    return math.ceil(predicted_capacity_kwh / module_capacity_kwh)


def _battery_soc_fractions(usable_soc_window_fraction: float) -> tuple[float, float]:
    """Same convention as src.scenarios.scenario_runner: a symmetric usable
    window around 50% SOC."""
    min_soc = (1.0 - usable_soc_window_fraction) / 2.0
    return min_soc, min_soc + usable_soc_window_fraction


def _run_and_measure(
    n_modules: int, scenario: pd.Series, pv: pd.Series, wind: pd.Series, load: pd.Series,
    site_config: dict, module_capacity_kwh: float, module_rated_power_kw: float,
) -> dict:
    min_soc, max_soc = _battery_soc_fractions(scenario["usable_soc_window_fraction"])
    battery = BatterySpec(
        module_capacity_kwh=module_capacity_kwh,
        module_rated_power_kw=module_rated_power_kw,
        round_trip_efficiency=scenario["round_trip_efficiency"],
        min_soc_fraction=min_soc,
        max_soc_fraction=max_soc,
        initial_soc_fraction=site_config["battery"]["initial_soc_fraction"],
        n_modules=n_modules,
    )
    dispatch_result = resolve_initial_soc_bias(pv, wind, load, battery)
    return compute_candidate_metrics(
        dispatch_result, battery,
        installed_cost_eur_per_kwh=site_config["battery"]["installed_cost_eur_per_kwh"],
        economic_lifetime_years=site_config["battery"]["economic_lifetime_years"],
        project_lifetime_years=site_config["battery"]["project_lifetime_years"],
        real_discount_rate=site_config["economics"]["real_discount_rate"],
        candidate_runtime_seconds=0.0,
    )


def verify_predictions(
    scenarios: pd.DataFrame,
    predicted_capacities_kwh: pd.Series,
    weather: pd.DataFrame,
    site_config: dict,
    module_capacity_kwh: float,
) -> pd.DataFrame:
    """Run mechanistic dispatch for every AI-predicted (rounded) battery size
    AND the mechanistic-optimal battery size, for each scenario in `scenarios`,
    and report reliability/cost/curtailment outcomes.

    Args:
        scenarios: Test-split scenario rows (must include `optimal_n_modules`,
            `optimal_capacity_kwh`, `reliability_target_load_served`,
            `round_trip_efficiency`, `usable_soc_window_fraction`, and the
            raw generation inputs) -- i.e. feasible scenarios only, since
            infeasible ones have no reference to compare against.
        predicted_capacities_kwh: Model predictions, aligned by `scenarios.index`.
        weather: Shared local-year hourly weather for these scenarios' site/year.
        site_config: Fixed site configuration (siting assumptions, battery
            module spec, economics).
        module_capacity_kwh: Battery module capacity used for ceiling rounding.

    Returns:
        One row per scenario with the verified AI-selected and reference
        metrics side by side, plus derived comparison columns.
    """
    module_rated_power_kw = site_config["battery"]["module_rated_power_kw"]
    pv_cfg, wind_cfg = site_config["pv"], site_config["wind"]
    rows = []

    for idx, scenario in scenarios.iterrows():
        load = generate_load_profile(
            annual_consumption_kwh=scenario["annual_load_kwh"], peak_load_kw=scenario["peak_load_kw"],
            year=scenario["weather_year"], timezone=scenario["timezone"], random_seed=scenario["random_seed"],
        )
        pv = compute_pv_generation(
            weather=weather, capacity_kwp=scenario["pv_capacity_kwp"],
            latitude=site_config["site"]["latitude"], longitude=site_config["site"]["longitude"],
            tilt_deg=pv_cfg["tilt_deg"], azimuth_deg=pv_cfg["azimuth_deg"],
            system_losses_fraction=pv_cfg["system_losses_fraction"], inverter_efficiency=pv_cfg["inverter_efficiency"],
        )
        wind = compute_wind_generation(
            weather=weather, rated_power_kw=scenario["wind_capacity_kw"],
            reference_height_m=wind_cfg["weather_reference_height_m"], hub_height_m=wind_cfg["hub_height_m"],
            shear_exponent=wind_cfg["wind_shear_exponent"], cut_in_mps=wind_cfg["cut_in_mps"],
            rated_mps=wind_cfg["rated_mps"], cut_out_mps=wind_cfg["cut_out_mps"],
        )

        predicted_capacity = predicted_capacities_kwh.loc[idx]
        n_ai = predicted_capacity_to_modules(predicted_capacity, module_capacity_kwh)
        n_reference = int(scenario["optimal_n_modules"])

        ai_metrics = _run_and_measure(n_ai, scenario, pv, wind, load, site_config, module_capacity_kwh, module_rated_power_kw)
        ref_metrics = _run_and_measure(n_reference, scenario, pv, wind, load, site_config, module_capacity_kwh, module_rated_power_kw)

        lpsp_target = 1.0 - scenario["reliability_target_load_served"]
        installed_capacity_kwh = ai_metrics["nominal_battery_capacity_kwh"]
        reference_capacity_kwh = ref_metrics["nominal_battery_capacity_kwh"]

        rows.append({
            "scenario_id": scenario["scenario_id"],
            "predicted_capacity_kwh": predicted_capacity,
            "installed_n_modules": n_ai,
            "installed_capacity_kwh": installed_capacity_kwh,
            "reference_n_modules": n_reference,
            "reference_capacity_kwh": reference_capacity_kwh,
            "excess_capacity_kwh": installed_capacity_kwh - reference_capacity_kwh,
            "undersized": installed_capacity_kwh < reference_capacity_kwh,
            "oversized": installed_capacity_kwh > reference_capacity_kwh,
            "lpsp_target": lpsp_target,
            "verified_lpsp": ai_metrics["lpsp"],
            "reliability_pass": ai_metrics["lpsp"] <= lpsp_target,
            "reference_lpsp": ref_metrics["lpsp"],
            "verified_cost_eur": ai_metrics["equivalent_annual_cost_eur"],
            "reference_cost_eur": ref_metrics["equivalent_annual_cost_eur"],
            "cost_difference_eur": ai_metrics["equivalent_annual_cost_eur"] - ref_metrics["equivalent_annual_cost_eur"],
            "verified_curtailed_kwh": ai_metrics["curtailed_energy_kwh"],
            "reference_curtailed_kwh": ref_metrics["curtailed_energy_kwh"],
            "curtailed_difference_kwh": ai_metrics["curtailed_energy_kwh"] - ref_metrics["curtailed_energy_kwh"],
        })

    result = pd.DataFrame(rows)
    logger.info(
        "Physical verification: %d/%d predictions pass reliability (%.0f%%).",
        result["reliability_pass"].sum(), len(result), 100 * result["reliability_pass"].mean(),
    )
    return result


def summarize_verification(verification: pd.DataFrame) -> dict:
    """Aggregate verification results into the summary metrics required by
    PROJECT_BRIEF.md §20."""
    undersized = verification[verification["undersized"]]
    oversized = verification[verification["oversized"]]
    reliability_violations = verification[~verification["reliability_pass"]]

    return {
        "n_scenarios": len(verification),
        "pct_satisfying_reliability": float(verification["reliability_pass"].mean()),
        "pct_undersized": float(verification["undersized"].mean()),
        "pct_oversized": float(verification["oversized"].mean()),
        "mean_excess_capacity_kwh": float(verification["excess_capacity_kwh"].mean()),
        "max_underprediction_kwh": float(-undersized["excess_capacity_kwh"].min()) if len(undersized) else 0.0,
        "additional_cost_from_oversizing_eur": float(oversized["cost_difference_eur"].sum()),
        "n_reliability_violations": int((~verification["reliability_pass"]).sum()),
        "n_reliability_violations_from_undersizing": int((reliability_violations["undersized"]).sum()),
        "mean_curtailed_difference_kwh": float(verification["curtailed_difference_kwh"].mean()),
        "mean_cost_difference_eur": float(verification["cost_difference_eur"].mean()),
    }
