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
   round_trip_efficiency and usable_soc_window_fraction, not site defaults --
   then applies diesel backup (`src.physics.diesel.apply_diesel_backup`) at
   the scenario's own peak-load-derived diesel rating.
3. Also re-runs dispatch + diesel for the mechanistic-optimal battery size
   (`optimal_n_modules`) for the same scenario, so the comparison uses two
   freshly-computed, directly comparable metric sets rather than diffing
   against possibly-stale stored summary stats.
4. **Reframed by PROJECT_BRIEF.md Addendum 3:** since diesel makes
   reliability near-universal by construction (see
   `src.physics.optimization` module docstring), a binary reliability
   pass/fail is no longer the primary verification question. Instead this
   module reports `extra_system_lcoe_eur_per_kwh` -- how much more expensive
   the AI's predicted battery capacity makes the system, per kWh served,
   than installing the true LCOE-minimizing capacity would have. This is a
   continuous, always-computable measure of "how economically costly is
   trusting the AI's answer," directly matching this project's stated
   overarching goal (optimize battery capacity for an economically feasible
   system). Reliability/over-under-sizing metrics are still reported
   alongside it for continuity and diagnostic value.
"""

from __future__ import annotations

import logging
import math

import pandas as pd

from src.physics.battery import BatterySpec
from src.physics.diesel import DieselSpec, apply_diesel_backup, diesel_rated_power_kw
from src.physics.dispatch import resolve_initial_soc_bias
from src.physics.load_profile import generate_load_profile
from src.physics.metrics import compute_candidate_metrics, compute_system_metrics
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
    site_config: dict, module_capacity_kwh: float, module_rated_power_kw: float, diesel: DieselSpec,
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
    battery_metrics = compute_candidate_metrics(
        dispatch_result, battery,
        installed_cost_eur_per_kwh=site_config["battery"]["installed_cost_eur_per_kwh"],
        economic_lifetime_years=site_config["battery"]["economic_lifetime_years"],
        project_lifetime_years=site_config["battery"]["project_lifetime_years"],
        real_discount_rate=site_config["economics"]["real_discount_rate"],
        candidate_runtime_seconds=0.0,
    )
    dispatch_with_diesel = apply_diesel_backup(dispatch_result, diesel)
    system_metrics = compute_system_metrics(
        dispatch_with_diesel,
        battery_equivalent_annual_cost_eur=battery_metrics["equivalent_annual_cost_eur"],
        pv_capacity_kwp=scenario["pv_capacity_kwp"], pv_cfg=site_config["pv"],
        wind_capacity_kw=scenario["wind_capacity_kw"], wind_cfg=site_config["wind"],
        diesel=diesel, project_lifetime_years=site_config["battery"]["project_lifetime_years"],
        real_discount_rate=site_config["economics"]["real_discount_rate"],
    )
    return {**battery_metrics, **system_metrics}


def verify_predictions(
    scenarios: pd.DataFrame,
    predicted_capacities_kwh: pd.Series,
    weather: pd.DataFrame,
    site_config: dict,
    module_capacity_kwh: float,
) -> pd.DataFrame:
    """Run mechanistic dispatch + diesel backup for every AI-predicted
    (rounded) battery size AND the mechanistic-optimal battery size, for
    each scenario in `scenarios`, and report system-LCOE/reliability/cost/
    curtailment outcomes (PROJECT_BRIEF.md Addendum 3).

    Args:
        scenarios: Test-split scenario rows (must include `optimal_n_modules`,
            `optimal_capacity_kwh`, `reliability_target_load_served`,
            `round_trip_efficiency`, `usable_soc_window_fraction`, and the
            raw generation inputs) -- i.e. feasible scenarios only, since
            infeasible ones have no reference to compare against.
            **`annual_load_kwh` and `peak_load_kw` here must be the ORIGINAL
            scenario inputs (e.g. from `data/scenarios/*.csv`), never
            `src.ai.features`'s engineered/achieved versions of those same
            column names.** `compute_load_features` legitimately returns the
            *achieved* peak/annual load of the regenerated series as an ML
            feature, which can differ from the original request by a tiny
            floating-point rescaling -- enough that feeding it back into
            `generate_load_profile` here can occasionally trip that
            function's own internal consistency check. If joining scenario
            rows against a `features.py` output, explicitly overwrite these
            two columns from the raw scenario table before calling this
            function (see `notebooks/05_model_comparison.ipynb`'s Stage 8/9
            section for the pattern).
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

        diesel_cfg = site_config["diesel"]
        diesel = DieselSpec(
            rated_power_kw=diesel_rated_power_kw(float(scenario["peak_load_kw"]), diesel_cfg["sizing_factor"]),
            fuel_curve_intercept_l_per_kwh_rated=diesel_cfg["fuel_curve_intercept_l_per_kwh_rated"],
            fuel_curve_slope_l_per_kwh_output=diesel_cfg["fuel_curve_slope_l_per_kwh_output"],
            fuel_price_eur_per_l=diesel_cfg["fuel_price_eur_per_l"],
            installed_cost_eur_per_kw=diesel_cfg["installed_cost_eur_per_kw"],
            om_cost_fraction_per_year=diesel_cfg["om_cost_fraction_per_year"],
            economic_lifetime_years=diesel_cfg["economic_lifetime_years"],
            project_lifetime_years=diesel_cfg["project_lifetime_years"],
        )

        predicted_capacity = predicted_capacities_kwh.loc[idx]
        n_ai = predicted_capacity_to_modules(predicted_capacity, module_capacity_kwh)
        n_reference = int(scenario["optimal_n_modules"])

        ai_metrics = _run_and_measure(n_ai, scenario, pv, wind, load, site_config, module_capacity_kwh, module_rated_power_kw, diesel)
        ref_metrics = _run_and_measure(n_reference, scenario, pv, wind, load, site_config, module_capacity_kwh, module_rated_power_kw, diesel)

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
            "verified_lpsp": ai_metrics["lpsp_after_diesel"],
            "reliability_pass": ai_metrics["lpsp_after_diesel"] <= lpsp_target,
            "reference_lpsp": ref_metrics["lpsp_after_diesel"],
            "verified_renewable_share": ai_metrics["renewable_share"],
            "reference_renewable_share": ref_metrics["renewable_share"],
            "renewable_share_difference": ai_metrics["renewable_share"] - ref_metrics["renewable_share"],
            "verified_system_lcoe_eur_per_kwh": ai_metrics["system_lcoe_eur_per_kwh"],
            "reference_system_lcoe_eur_per_kwh": ref_metrics["system_lcoe_eur_per_kwh"],
            "extra_system_lcoe_eur_per_kwh": (
                ai_metrics["system_lcoe_eur_per_kwh"] - ref_metrics["system_lcoe_eur_per_kwh"]
            ),
            "verified_cost_eur": ai_metrics["total_annualized_cost_eur"],
            "reference_cost_eur": ref_metrics["total_annualized_cost_eur"],
            "cost_difference_eur": ai_metrics["total_annualized_cost_eur"] - ref_metrics["total_annualized_cost_eur"],
            "verified_curtailed_kwh": ai_metrics["curtailed_energy_kwh"],
            "reference_curtailed_kwh": ref_metrics["curtailed_energy_kwh"],
            "curtailed_difference_kwh": ai_metrics["curtailed_energy_kwh"] - ref_metrics["curtailed_energy_kwh"],
        })

    result = pd.DataFrame(rows)
    logger.info(
        "Physical verification: mean extra system LCOE from trusting the AI's prediction: "
        "%.4f EUR/kWh (%d/%d predictions pass reliability, %.0f%%).",
        result["extra_system_lcoe_eur_per_kwh"].mean(),
        result["reliability_pass"].sum(), len(result), 100 * result["reliability_pass"].mean(),
    )
    return result


def summarize_verification(verification: pd.DataFrame) -> dict:
    """Aggregate verification results into the summary metrics required by
    PROJECT_BRIEF.md §20, headlined (per Addendum 3) by the extra system LCOE
    incurred from trusting the AI's predicted battery capacity instead of the
    true LCOE-minimizing one -- a continuous measure that remains meaningful
    even though `pct_satisfying_reliability` is now near-universal by
    construction (diesel backup)."""
    undersized = verification[verification["undersized"]]
    oversized = verification[verification["oversized"]]
    reliability_violations = verification[~verification["reliability_pass"]]

    return {
        "n_scenarios": len(verification),
        "mean_extra_system_lcoe_eur_per_kwh": float(verification["extra_system_lcoe_eur_per_kwh"].mean()),
        "median_extra_system_lcoe_eur_per_kwh": float(verification["extra_system_lcoe_eur_per_kwh"].median()),
        "max_extra_system_lcoe_eur_per_kwh": float(verification["extra_system_lcoe_eur_per_kwh"].max()),
        "pct_within_5pct_of_optimal_lcoe": float(
            (verification["extra_system_lcoe_eur_per_kwh"]
             <= 0.05 * verification["reference_system_lcoe_eur_per_kwh"]).mean()
        ),
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
