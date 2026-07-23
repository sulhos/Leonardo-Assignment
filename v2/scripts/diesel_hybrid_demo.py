"""Diesel-hybrid demonstration (post-Task-11 addendum): run the mechanistic
pipeline at the current diesel-backed baseline (config/jinan.yaml,
system.backup: diesel, load.target_annual_kwh: 4,500,000), find the
LCOE-optimal battery, and produce the evidence requested for the
demonstration: the "diesel engaging" week figure, the annual energy-flow
breakdown figure, a metrics summary, and an energy-balance verification.

Mechanistic-only: does not touch the scenario dataset, the trained neural
network, or the existing off-grid study's reports -- see
v2/DIESEL_HYBRID_DEMO.md for how this relates to that prior work.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import load_site_config
from src.data.weather import get_processed_weather
from src.physics.battery import BatterySpec
from src.physics.diesel import DieselSpec, apply_diesel_backup, diesel_rated_power_kw
from src.physics.dispatch import resolve_initial_soc_bias
from src.physics.load_profile import generate_load_profile
from src.physics.optimization import run_battery_search
from src.physics.pv_model import compute_pv_generation
from src.physics.wind_model import adjust_wind_speed_to_hub_height, compute_wind_generation
from src.visualization.plotting import (
    plot_annual_profile,
    plot_diesel_engagement_week,
    plot_energy_flow_balance_with_diesel,
    plot_metric_vs_battery_capacity,
    plot_renewable_share_vs_lcoe,
    plot_wind_speed_distribution_and_power_curve,
    save_figure,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SITE = "jinan"
DIESEL_WEEK_START = "2023-12-25"  # highest-diesel-output week of the year, found by inspection


def run() -> dict:
    cfg = load_site_config(SITE)
    if cfg["system"]["backup"] != "diesel":
        raise RuntimeError(
            f"Expected config/jinan.yaml system.backup == 'diesel' for this demo, got {cfg['system']['backup']!r}."
        )

    weather = get_processed_weather(SITE, cfg["simulation"]["year"])
    load_cfg, pv_cfg, wind_cfg, battery_cfg, diesel_cfg = (
        cfg["load"], cfg["pv"], cfg["wind"], cfg["battery"], cfg["diesel"]
    )

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

    out_dir = REPO_ROOT / "outputs/figures"
    save_figure(
        plot_annual_profile(load, pv, wind, "Jinan Diesel-Hybrid Baseline: Annual Load, PV, Wind"),
        out_dir / "01_annual_profile.png",
    )

    wind_speed_hub = adjust_wind_speed_to_hub_height(
        weather["WS10M"], wind_cfg["weather_reference_height_m"], wind_cfg["hub_height_m"], wind_cfg["wind_shear_exponent"],
    )
    save_figure(
        plot_wind_speed_distribution_and_power_curve(
            wind_speed_hub, wind_cfg["rated_power_kw"], wind_cfg["power_curve_speed_power_fraction"],
            wind_cfg["cut_in_mps"], wind_cfg["rated_mps"], wind_cfg["cut_out_mps"],
        ),
        out_dir / "06_wind_speed_power_curve.png",
    )

    result = run_battery_search(
        pv, wind, load, n_max=battery_cfg["candidate_module_counts"]["max"],
        module_capacity_kwh=battery_cfg["module_capacity_kwh"], module_rated_power_kw=battery_cfg["module_rated_power_kw"],
        round_trip_efficiency=battery_cfg["round_trip_efficiency"], min_soc_fraction=battery_cfg["min_soc_fraction"],
        max_soc_fraction=battery_cfg["max_soc_fraction"], initial_soc_fraction=battery_cfg["initial_soc_fraction"],
        self_discharge_rate_per_hour=battery_cfg["self_discharge_rate_per_hour"],
        installed_cost_eur_per_kwh=battery_cfg["installed_cost_eur_per_kwh"], economic_lifetime_years=battery_cfg["economic_lifetime_years"],
        project_lifetime_years=battery_cfg["project_lifetime_years"], real_discount_rate=cfg["economics"]["real_discount_rate"],
        pv_capacity_kwp=pv_cfg["capacity_kwp"], pv_cfg=pv_cfg, wind_capacity_kw=wind_cfg["rated_power_kw"], wind_cfg=wind_cfg,
        diesel_cfg=diesel_cfg, lpsp_target=cfg["reliability"]["primary_lpsp_target"], system_backup=cfg["system"]["backup"],
    )

    battery = BatterySpec(
        module_capacity_kwh=battery_cfg["module_capacity_kwh"], module_rated_power_kw=battery_cfg["module_rated_power_kw"],
        round_trip_efficiency=battery_cfg["round_trip_efficiency"], min_soc_fraction=battery_cfg["min_soc_fraction"],
        max_soc_fraction=battery_cfg["max_soc_fraction"], initial_soc_fraction=battery_cfg["initial_soc_fraction"],
        self_discharge_rate_per_hour=battery_cfg["self_discharge_rate_per_hour"], n_modules=result.optimal_n_modules,
    )
    dispatch = resolve_initial_soc_bias(pv, wind, load, battery)
    diesel = DieselSpec(
        rated_power_kw=diesel_rated_power_kw(float(load.max()), diesel_cfg["sizing_factor"]),
        fuel_curve_intercept_l_per_kwh_rated=diesel_cfg["fuel_curve_intercept_l_per_kwh_rated"],
        fuel_curve_slope_l_per_kwh_output=diesel_cfg["fuel_curve_slope_l_per_kwh_output"],
        fuel_price_eur_per_l=diesel_cfg["fuel_price_eur_per_l"], installed_cost_eur_per_kw=diesel_cfg["installed_cost_eur_per_kw"],
        om_cost_fraction_per_year=diesel_cfg["om_cost_fraction_per_year"], economic_lifetime_years=diesel_cfg["economic_lifetime_years"],
        project_lifetime_years=diesel_cfg["project_lifetime_years"],
    )
    dwd = apply_diesel_backup(dispatch, diesel)

    save_figure(
        plot_diesel_engagement_week(
            dwd, DIESEL_WEEK_START, battery.min_soc_kwh, battery.max_soc_kwh,
            title=f"Diesel Engaging During a Low-Renewable Week ({DIESEL_WEEK_START} to +7d)",
        ),
        out_dir / "11_diesel_engagement_week.png",
    )
    save_figure(plot_energy_flow_balance_with_diesel(dwd), out_dir / "12_energy_flow_balance_diesel_hybrid.png")

    # Where does more battery/renewable share stop paying for itself? The
    # search's own objective already answers "where is the minimum" (the
    # selected candidate); this identifies the SHAPE around that minimum --
    # the diesel-only floor, and how much system LCOE rises again once
    # renewable share is pushed toward its ceiling.
    candidates_sorted = result.candidates.sort_values("nominal_battery_capacity_kwh").reset_index(drop=True)
    diesel_only = candidates_sorted.iloc[0]  # n_modules = 0
    max_battery = candidates_sorted.iloc[-1]  # n_modules = n_max
    optimal_row = candidates_sorted.loc[candidates_sorted["system_lcoe_eur_per_kwh"].idxmin()]

    fig13 = plot_renewable_share_vs_lcoe(
        result.candidates, title="Renewable Share vs. System LCOE (Diesel-Hybrid Demonstration)",
    )
    ax13 = fig13.axes[0]
    ax13.scatter(
        [optimal_row["system_lcoe_eur_per_kwh"]], [optimal_row["renewable_share"] * 100],
        color="firebrick", s=90, zorder=5, marker="*",
        label=f"Sweet spot: {int(optimal_row['n_modules'])} modules, {optimal_row['renewable_share']*100:.1f}% renewable",
    )
    ax13.annotate(
        f"minimum LCOE\n{optimal_row['system_lcoe_eur_per_kwh']:.3f} EUR/kWh\n@ {optimal_row['renewable_share']*100:.1f}% renewable",
        xy=(optimal_row["system_lcoe_eur_per_kwh"], optimal_row["renewable_share"] * 100),
        xytext=(55, -55), textcoords="offset points", fontsize=9, color="firebrick", ha="left",
        arrowprops=dict(arrowstyle="->", color="firebrick", lw=1),
    )
    ax13.legend(loc="lower right", fontsize=8)
    save_figure(fig13, out_dir / "13_renewable_share_vs_lcoe.png")

    fig14 = plot_metric_vs_battery_capacity(
        result.candidates, metric_column="system_lcoe_eur_per_kwh", ylabel="System LCOE (EUR/kWh)",
    )
    ax14 = fig14.axes[0]
    ax14.axvline(optimal_row["nominal_battery_capacity_kwh"], color="firebrick", linestyle="--", linewidth=1, alpha=0.7)
    ax14.scatter(
        [optimal_row["nominal_battery_capacity_kwh"]], [optimal_row["system_lcoe_eur_per_kwh"]],
        color="firebrick", s=90, zorder=5, marker="*",
        label=f"Minimum LCOE: {int(optimal_row['n_modules'])} modules, {optimal_row['nominal_battery_capacity_kwh']:.0f} kWh",
    )
    ax14.annotate(
        "cheaper to add\nbattery here", xy=(optimal_row["nominal_battery_capacity_kwh"] * 0.35, diesel_only["system_lcoe_eur_per_kwh"] * 0.985),
        fontsize=9, color="seagreen", ha="center",
    )
    ax14.annotate(
        "battery capex now\noutweighs fuel savings", xy=(candidates_sorted["nominal_battery_capacity_kwh"].iloc[-15], candidates_sorted["system_lcoe_eur_per_kwh"].iloc[-15]),
        xytext=(-110, 15), textcoords="offset points", fontsize=9, color="darkorange",
        arrowprops=dict(arrowstyle="->", color="darkorange", lw=1),
    )
    ax14.legend(loc="upper center", fontsize=8)
    save_figure(fig14, out_dir / "14_battery_capacity_vs_lcoe.png")
    lcoe_curve_analysis = {
        "diesel_only_n_modules": int(diesel_only["n_modules"]),
        "diesel_only_lcoe_eur_per_kwh": float(diesel_only["system_lcoe_eur_per_kwh"]),
        "diesel_only_renewable_share_pct": float(diesel_only["renewable_share"] * 100),
        "optimal_n_modules": int(optimal_row["n_modules"]),
        "optimal_lcoe_eur_per_kwh": float(optimal_row["system_lcoe_eur_per_kwh"]),
        "optimal_renewable_share_pct": float(optimal_row["renewable_share"] * 100),
        "max_battery_n_modules": int(max_battery["n_modules"]),
        "max_battery_lcoe_eur_per_kwh": float(max_battery["system_lcoe_eur_per_kwh"]),
        "max_battery_renewable_share_pct": float(max_battery["renewable_share"] * 100),
        "lcoe_saving_diesel_only_to_optimal_eur_per_kwh": float(diesel_only["system_lcoe_eur_per_kwh"] - optimal_row["system_lcoe_eur_per_kwh"]),
        "lcoe_penalty_optimal_to_max_battery_eur_per_kwh": float(max_battery["system_lcoe_eur_per_kwh"] - optimal_row["system_lcoe_eur_per_kwh"]),
    }
    print("LCOE curve shape:", json.dumps(lcoe_curve_analysis, indent=2))

    total_load = float(dwd["load_kw"].sum())
    direct = float(dwd["direct_supply_kwh"].sum())
    batt_discharge = float(dwd["battery_discharge_kwh"].sum())
    diesel_out = float(dwd["diesel_output_kwh"].sum())
    still_unserved = float(dwd["still_unserved_kwh"].sum())
    curtailed = float(dwd["curtailed_kwh"].sum())
    renewable_gen = float(dwd["pv_kw"].sum() + dwd["wind_kw"].sum())

    lhs = dwd["load_kw"]
    rhs = dwd["direct_supply_kwh"] + dwd["battery_discharge_kwh"] + dwd["diesel_output_kwh"] + dwd["still_unserved_kwh"]
    max_hourly_residual = float((lhs - rhs).abs().max())
    annual_residual = float(lhs.sum() - rhs.sum())

    summary = {
        "annual_load_kwh": total_load,
        "renewable_generated_kwh": renewable_gen,
        "renewable_to_load_ratio": renewable_gen / total_load,
        "direct_renewable_to_load_kwh": direct,
        "battery_discharge_to_load_kwh": batt_discharge,
        "diesel_to_load_kwh": diesel_out,
        "still_unserved_kwh": still_unserved,
        "curtailed_kwh": curtailed,
        "curtailment_pct_of_generated_renewable": 100 * curtailed / renewable_gen,
        "renewable_share_pct": 100 * (1.0 - (diesel_out + still_unserved) / total_load),
        "diesel_share_pct": 100 * diesel_out / total_load,
        "lpsp": still_unserved / total_load,
        "diesel_operating_hours": int((dwd["diesel_output_kwh"] > 0).sum()),
        "diesel_fuel_l_per_year": float(dwd["diesel_fuel_l"].sum()),
        "diesel_rated_power_kw": diesel.rated_power_kw,
        "diesel_capacity_factor_pct": 100 * diesel_out / (diesel.rated_power_kw * len(dwd)),
        "optimal_n_modules": result.optimal_n_modules,
        "optimal_capacity_kwh": result.optimal_capacity_kwh,
        "system_lcoe_eur_per_kwh": result.system_lcoe_eur_per_kwh,
        "max_hourly_energy_balance_residual_kwh": max_hourly_residual,
        "annual_energy_balance_residual_kwh": annual_residual,
        "lcoe_curve_analysis": lcoe_curve_analysis,
    }
    print(json.dumps(summary, indent=2))

    out_dir_tables = REPO_ROOT / "outputs/tables"
    out_dir_tables.mkdir(parents=True, exist_ok=True)
    with open(out_dir_tables / "diesel_hybrid_demo_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    dwd.to_csv(out_dir_tables / "diesel_hybrid_demo_hourly_dispatch.csv")
    candidates_sorted.to_csv(out_dir_tables / "diesel_hybrid_demo_battery_candidates.csv", index=False)

    return summary


if __name__ == "__main__":
    run()
