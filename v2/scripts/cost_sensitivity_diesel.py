"""V2 Part F: cost sensitivity for the diesel-hybrid study.

Unlike the completed off-grid study (where battery cost changes the
REPORTED system LCOE but never which candidate gets selected, since
off-grid sizing is driven purely by the reliability constraint), under
diesel min-LCOE the optimal battery genuinely DEPENDS on battery cost --
cheaper battery pulls the optimum larger, more expensive battery pulls it
smaller/toward zero. This means the ML target itself shifts with the cost
assumption: the trained model (scripts/train_nn_diesel.py) is conditioned
on the 300 EUR/kWh baseline cost in config/jinan.yaml, not cost-agnostic.

Re-runs the full mechanistic search (not a closed-form shortcut -- the
selected candidate itself changes, not just its reported cost) on a
1,000-scenario random subsample at battery costs 220/400 EUR/kWh,
reusing the already-computed 300 EUR/kWh labels from
data/scenarios/diesel_full_scenarios.csv for the same subsample rather
than re-running that point too.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_site_config
from src.data.weather import get_processed_weather
from src.scenarios.scenario_runner import run_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]
SITE = "jinan"
SUBSAMPLE_N = 1000
SUBSAMPLE_SEED = 42
SWEEP_COSTS = [220, 400]  # 300 (baseline) already computed in the full dataset


def run() -> pd.DataFrame:
    cfg = load_site_config(SITE)
    weather = get_processed_weather(SITE, cfg["simulation"]["year"])
    n_max = 80

    all_scenarios = pd.read_csv(REPO_ROOT / "data/scenarios/diesel_full_scenarios.csv")
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    subsample_idx = rng.choice(len(all_scenarios), size=SUBSAMPLE_N, replace=False)
    subsample = all_scenarios.iloc[subsample_idx].reset_index(drop=True)

    rows = []
    # Baseline (300 EUR/kWh) reused directly, no re-run needed.
    for _, row in subsample.iterrows():
        rows.append({
            "scenario_id": row["scenario_id"], "battery_cost_eur_per_kwh": cfg["battery"]["installed_cost_eur_per_kwh"],
            "optimal_n_modules": row["optimal_n_modules"], "optimal_capacity_kwh": row["optimal_capacity_kwh"],
            "system_lcoe_eur_per_kwh": row["reference_system_lcoe_eur_per_kwh"],
            "renewable_share": row["reference_renewable_share"],
        })

    for cost in SWEEP_COSTS:
        t0 = time.time()
        swept_cfg = copy.deepcopy(cfg)
        swept_cfg["battery"]["installed_cost_eur_per_kwh"] = cost
        for _, row in subsample.iterrows():
            scenario = row.to_dict()
            result = run_scenario(scenario, swept_cfg, weather, n_max)
            rows.append({
                "scenario_id": row["scenario_id"], "battery_cost_eur_per_kwh": cost,
                "optimal_n_modules": result["optimal_n_modules"], "optimal_capacity_kwh": result["optimal_capacity_kwh"],
                "system_lcoe_eur_per_kwh": result["reference_system_lcoe_eur_per_kwh"],
                "renewable_share": result["reference_renewable_share"],
            })
        print(f"cost={cost} EUR/kWh done in {time.time() - t0:.1f}s")

    results = pd.DataFrame(rows)
    summary = results.groupby("battery_cost_eur_per_kwh")["optimal_n_modules"].agg(
        ["mean", "std", "median", "min", "max"]
    )
    print(summary)

    out_dir = REPO_ROOT / "outputs/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "cost_sensitivity_diesel_full.csv", index=False)
    summary.to_csv(out_dir / "cost_sensitivity_diesel_full_summary.csv")
    return results


if __name__ == "__main__":
    run()
