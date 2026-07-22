"""V2 Task 10: generate the lean figure set for the technical report.

Deliberately curated, not exhaustive (roughly half of V1's figure count):
one baseline-physics showcase figure, one dataset-composition figure, and
one figure per Task 5-8 analysis, plus the standard predicted-vs-actual
scatter with R^2 annotated. Writes to outputs/figures/ as
NN_description.png, numbered for report cross-referencing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.ai.evaluation import compute_accuracy_metrics
from src.config import load_site_config
from src.data.weather import get_processed_weather
from src.physics.load_profile import generate_load_profile
from src.physics.optimization import run_battery_search
from src.physics.pv_model import compute_pv_generation
from src.physics.wind_model import compute_wind_generation
from src.visualization.plotting import (
    plot_annual_profile,
    plot_cost_sensitivity,
    plot_cross_validation_error_bars,
    plot_energy_flow_balance,
    plot_feasibility_counts,
    plot_learning_curve,
    plot_optimal_capacity_distribution,
    plot_permutation_importance,
    plot_predicted_vs_reference_with_r2,
    plot_training_history,
    save_figure,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SITE = "jinan"


def run() -> None:
    out_dir = REPO_ROOT / "outputs/figures"
    cfg = load_site_config(SITE)
    weather = get_processed_weather(SITE, cfg["simulation"]["year"])

    # 1. Baseline off-grid physics showcase (annual profile + energy flow balance).
    load_cfg, pv_cfg, wind_cfg, battery_cfg = cfg["load"], cfg["pv"], cfg["wind"], cfg["battery"]
    load = generate_load_profile(
        annual_consumption_kwh=load_cfg["target_annual_kwh"], peak_load_kw=load_cfg["max_hourly_kw"],
        year=cfg["simulation"]["year"], timezone=cfg["site"]["timezone"], random_seed=load_cfg["random_seed"],
    )
    pv = compute_pv_generation(
        weather=weather, capacity_kwp=pv_cfg["capacity_kwp"], latitude=cfg["site"]["latitude"],
        longitude=cfg["site"]["longitude"], tilt_deg=pv_cfg["tilt_deg"], azimuth_deg=pv_cfg["azimuth_deg"],
        system_losses_fraction=pv_cfg["system_losses_fraction"], inverter_efficiency=pv_cfg["inverter_efficiency"],
    )
    wind = compute_wind_generation(
        weather=weather, rated_power_kw=wind_cfg["rated_power_kw"], reference_height_m=wind_cfg["weather_reference_height_m"],
        hub_height_m=wind_cfg["hub_height_m"], shear_exponent=wind_cfg["wind_shear_exponent"],
        power_curve_speed_power_fraction=wind_cfg["power_curve_speed_power_fraction"],
    )
    save_figure(plot_annual_profile(load, pv, wind, "Jinan Off-Grid Baseline: Annual Load, PV, Wind"), out_dir / "01_annual_profile.png")

    result = run_battery_search(
        pv, wind, load, n_max=battery_cfg["candidate_module_counts"]["max"],
        module_capacity_kwh=battery_cfg["module_capacity_kwh"], module_rated_power_kw=battery_cfg["module_rated_power_kw"],
        round_trip_efficiency=battery_cfg["round_trip_efficiency"], min_soc_fraction=battery_cfg["min_soc_fraction"],
        max_soc_fraction=battery_cfg["max_soc_fraction"], initial_soc_fraction=battery_cfg["initial_soc_fraction"],
        self_discharge_rate_per_hour=battery_cfg["self_discharge_rate_per_hour"],
        installed_cost_eur_per_kwh=battery_cfg["installed_cost_eur_per_kwh"], economic_lifetime_years=battery_cfg["economic_lifetime_years"],
        project_lifetime_years=battery_cfg["project_lifetime_years"], real_discount_rate=cfg["economics"]["real_discount_rate"],
        pv_capacity_kwp=pv_cfg["capacity_kwp"], pv_cfg=pv_cfg, wind_capacity_kw=wind_cfg["rated_power_kw"], wind_cfg=wind_cfg,
        diesel_cfg=cfg["diesel"], lpsp_target=cfg["reliability"]["primary_lpsp_target"], system_backup=cfg["system"]["backup"],
    )
    from src.physics.battery import BatterySpec
    from src.physics.diesel import DieselSpec, apply_diesel_backup
    from src.physics.dispatch import resolve_initial_soc_bias
    battery = BatterySpec(
        module_capacity_kwh=battery_cfg["module_capacity_kwh"], module_rated_power_kw=battery_cfg["module_rated_power_kw"],
        round_trip_efficiency=battery_cfg["round_trip_efficiency"], min_soc_fraction=battery_cfg["min_soc_fraction"],
        max_soc_fraction=battery_cfg["max_soc_fraction"], initial_soc_fraction=battery_cfg["initial_soc_fraction"],
        self_discharge_rate_per_hour=battery_cfg["self_discharge_rate_per_hour"], n_modules=result.optimal_n_modules,
    )
    dispatch = resolve_initial_soc_bias(pv, wind, load, battery)
    diesel = DieselSpec(
        rated_power_kw=0.0, fuel_curve_intercept_l_per_kwh_rated=cfg["diesel"]["fuel_curve_intercept_l_per_kwh_rated"],
        fuel_curve_slope_l_per_kwh_output=cfg["diesel"]["fuel_curve_slope_l_per_kwh_output"],
        fuel_price_eur_per_l=cfg["diesel"]["fuel_price_eur_per_l"], installed_cost_eur_per_kw=cfg["diesel"]["installed_cost_eur_per_kw"],
        om_cost_fraction_per_year=cfg["diesel"]["om_cost_fraction_per_year"], economic_lifetime_years=cfg["diesel"]["economic_lifetime_years"],
        project_lifetime_years=cfg["diesel"]["project_lifetime_years"],
    )
    dispatch_with_diesel = apply_diesel_backup(dispatch, diesel)
    save_figure(plot_energy_flow_balance(dispatch_with_diesel), out_dir / "02_energy_flow_balance.png")
    print(f"baseline: optimal_n_modules={result.optimal_n_modules} feasible={result.feasible} lpsp={result.lpsp:.4f}")

    # 2. Dataset composition.
    scenarios = pd.read_csv(REPO_ROOT / "data/scenarios/full_scenarios.csv")
    save_figure(plot_feasibility_counts(scenarios), out_dir / "03_feasibility_counts.png")
    save_figure(plot_optimal_capacity_distribution(scenarios), out_dir / "04_optimal_capacity_distribution.png")

    # 3. NN training + accuracy.
    with open(REPO_ROOT / "outputs/tables/neural_network_training_history_full.json") as f:
        history = json.load(f)
    save_figure(plot_training_history(history), out_dir / "05_training_history.png")

    verification = pd.read_csv(REPO_ROOT / "outputs/tables/physical_verification_neural_network_full.csv")
    accuracy = compute_accuracy_metrics(verification["reference_capacity_kwh"], verification["predicted_capacity_kwh"])
    save_figure(
        plot_predicted_vs_reference_with_r2(
            verification["reference_capacity_kwh"], verification["predicted_capacity_kwh"], accuracy["r2"], "Neural Network",
        ),
        out_dir / "06_predicted_vs_reference.png",
    )

    # 4. Task 5-8 analysis figures.
    cv_runs = pd.read_csv(REPO_ROOT / "outputs/tables/cross_validation_runs_neural_network_full.csv")
    save_figure(plot_cross_validation_error_bars(cv_runs), out_dir / "07_cross_validation.png")

    learning_curve = pd.read_csv(REPO_ROOT / "outputs/tables/learning_curve_neural_network_full.csv")
    save_figure(plot_learning_curve(learning_curve), out_dir / "08_learning_curve.png")

    importance = pd.read_csv(REPO_ROOT / "outputs/tables/permutation_importance_neural_network_full.csv")
    save_figure(plot_permutation_importance(importance), out_dir / "09_permutation_importance.png")

    cost_sensitivity = pd.read_csv(REPO_ROOT / "outputs/tables/cost_sensitivity_battery_full.csv")
    save_figure(plot_cost_sensitivity(cost_sensitivity), out_dir / "10_cost_sensitivity.png")

    print(f"wrote figures to {out_dir}")


if __name__ == "__main__":
    run()
