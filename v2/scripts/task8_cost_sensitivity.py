"""V2 Task 8: cost-sensitivity analysis.

Battery installed cost swept across config/jinan.yaml's own
`battery.cost_sensitivity_eur_per_kwh: [220, 300, 400]` -- documented in
economics.py as "alternative values for a future battery-price sensitivity
sweep (not yet run)" before this script -- rather than the brief's generic
suggested 400/550/700 EUR/kWh, since the project's own config already
carries site-appropriate industrial-scale figures bracketing its baseline
(300 EUR/kWh).

Diesel price sensitivity (the brief's other suggested leg) does not apply:
V2's sole study configuration is off-grid (config/jinan.yaml system.backup:
none, V2 Task 3 checkpoint) -- diesel_rated_power_kw is 0 for every
scenario, so diesel cost/fuel price cannot affect anything. Diesel-backed
remains V1's documented prior work, not re-run in V2.

Closed-form, not a re-simulation: PV/wind/battery capital+O&M cost are pure
functions of capacity and config (src.physics.economics), independent of
the hourly dispatch. Off-grid means total_annualized_cost = pv_eac + pv_om
+ wind_eac + wind_om + battery_eac (diesel terms are exactly 0), so the
already-labelled reference_system_lcoe_eur_per_kwh and
reference_annualized_cost (== the battery-only EAC at the baseline
300 EUR/kWh -- see src.scenarios.scenario_runner's label construction) are
enough to back out load_served_kwh exactly, without needing the stored
hourly dispatch series per scenario:

    load_served_kwh = (pv_eac+pv_om + wind_eac+wind_om + battery_eac_300)
                       / reference_system_lcoe_eur_per_kwh

then re-evaluating battery_eac at each swept cost and recombining gives the
exact system LCOE at that cost, for the SAME (already mechanistically
optimal) capacity -- correct because under off-grid, capacity selection is
LPSP-driven and does not depend on battery cost at all (see
src.physics.optimization's off-grid branch: `_select_smallest_feasible`
takes no cost argument), so re-optimizing candidates per cost point would
select the identical capacity anyway.
"""
from __future__ import annotations

import pandas as pd

from src.config import load_site_config
from src.physics.economics import equivalent_annual_cost, present_value_cost

SITE = "jinan"
REPO_ROOT_MARKER = "config"


def _pv_wind_annualized_cost(cfg: dict, pv_capacity_kwp: pd.Series, wind_capacity_kw: pd.Series) -> pd.Series:
    pv_cfg, wind_cfg, econ = cfg["pv"], cfg["wind"], cfg["economics"]
    project_life = cfg["battery"]["project_lifetime_years"]

    pv_pv_cost = pv_capacity_kwp.apply(
        lambda kwp: present_value_cost(
            pv_cfg["installed_cost_eur_per_kwp"], kwp, pv_cfg["economic_lifetime_years"],
            project_life, econ["real_discount_rate"],
        )
    )
    pv_eac = pv_pv_cost.apply(lambda pv: equivalent_annual_cost(pv, project_life, econ["real_discount_rate"]))
    pv_om = pv_cfg["om_cost_fraction_per_year"] * pv_cfg["installed_cost_eur_per_kwp"] * pv_capacity_kwp

    wind_pv_cost = wind_capacity_kw.apply(
        lambda kw: present_value_cost(
            wind_cfg["installed_cost_eur_per_kw"], kw, wind_cfg["economic_lifetime_years"],
            project_life, econ["real_discount_rate"],
        )
    )
    wind_eac = wind_pv_cost.apply(lambda pv: equivalent_annual_cost(pv, project_life, econ["real_discount_rate"]))
    wind_om = wind_cfg["om_cost_fraction_per_year"] * wind_cfg["installed_cost_eur_per_kw"] * wind_capacity_kw

    return pv_eac + pv_om + wind_eac + wind_om


def _battery_eac(cfg: dict, cost_eur_per_kwh: float, capacity_kwh: pd.Series) -> pd.Series:
    battery_cfg, econ = cfg["battery"], cfg["economics"]
    pv_cost = capacity_kwh.apply(
        lambda kwh: present_value_cost(
            cost_eur_per_kwh, kwh, battery_cfg["economic_lifetime_years"],
            battery_cfg["project_lifetime_years"], econ["real_discount_rate"],
        )
    )
    return pv_cost.apply(
        lambda pv: equivalent_annual_cost(pv, battery_cfg["project_lifetime_years"], econ["real_discount_rate"])
    )


def run(repo_root) -> pd.DataFrame:
    cfg = load_site_config(SITE)
    baseline_cost = cfg["battery"]["installed_cost_eur_per_kwh"]
    sweep_costs = cfg["battery"]["cost_sensitivity_eur_per_kwh"]
    assert baseline_cost in sweep_costs, "baseline installed cost must be one of the sweep points"

    scenarios = pd.read_csv(repo_root / "data/scenarios/full_scenarios.csv")
    feasible = scenarios[scenarios["feasible"]].reset_index(drop=True)

    pv_wind_annualized = _pv_wind_annualized_cost(cfg, feasible["pv_capacity_kwp"], feasible["wind_capacity_kw"])
    battery_eac_baseline = feasible["reference_annualized_cost"]
    total_cost_baseline = pv_wind_annualized + battery_eac_baseline
    load_served_kwh = total_cost_baseline / feasible["reference_system_lcoe_eur_per_kwh"]

    rows = []
    for cost in sweep_costs:
        battery_eac = _battery_eac(cfg, cost, feasible["optimal_capacity_kwh"])
        total_cost = pv_wind_annualized + battery_eac
        system_lcoe = total_cost / load_served_kwh
        rows.append({
            "battery_cost_eur_per_kwh": cost,
            "mean_system_lcoe_eur_per_kwh": float(system_lcoe.mean()),
            "median_system_lcoe_eur_per_kwh": float(system_lcoe.median()),
            "p10_system_lcoe_eur_per_kwh": float(system_lcoe.quantile(0.10)),
            "p90_system_lcoe_eur_per_kwh": float(system_lcoe.quantile(0.90)),
            "mean_battery_share_of_total_cost": float((battery_eac / total_cost).mean()),
        })

    # Sanity check: recovers the exact stored baseline system LCOE at the baseline cost point.
    baseline_row = [r for r in rows if r["battery_cost_eur_per_kwh"] == baseline_cost][0]
    stored_mean = float(feasible["reference_system_lcoe_eur_per_kwh"].mean())
    assert abs(baseline_row["mean_system_lcoe_eur_per_kwh"] - stored_mean) < 1e-6, (
        f"baseline reconstruction mismatch: {baseline_row['mean_system_lcoe_eur_per_kwh']} vs {stored_mean}"
    )

    results = pd.DataFrame(rows)
    print(results.to_string())
    print(f"n_scenarios={len(feasible)}, baseline_cost={baseline_cost}, sweep={sweep_costs}")
    print("Diesel price sensitivity: N/A (off-grid, diesel_rated_power_kw=0 for all V2 scenarios).")

    out_dir = repo_root / "outputs/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "cost_sensitivity_battery_full.csv", index=False)
    return results


if __name__ == "__main__":
    from pathlib import Path
    run(Path(__file__).resolve().parents[1])
