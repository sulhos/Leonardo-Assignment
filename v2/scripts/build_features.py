"""Build and persist the engineered ML feature matrix for the full off-grid
scenario dataset (V2 Task 5-8 prerequisite).

Feasible scenarios only (infeasible ones have no optimal_capacity_kwh target
to regress on -- src.ai.features.build_feature_matrix's docstring / the
existing V1 pattern). Writes to data/ml_dataset/ (gitignored -- cheaply
regenerable, ~100ms/scenario per features.py's own docstring).
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.ai.features import build_feature_matrix
from src.config import load_site_config
from src.data.weather import get_processed_weather

SITE = "jinan"
REPO_ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    cfg = load_site_config(SITE)
    weather = get_processed_weather(SITE, cfg["simulation"]["year"])
    scenarios = pd.read_csv(REPO_ROOT / "data/scenarios/full_scenarios.csv")

    feasible = scenarios[scenarios["feasible"]].reset_index(drop=True)
    print(f"{len(feasible)}/{len(scenarios)} feasible scenarios -> building feature matrix")

    t0 = time.time()
    features = build_feature_matrix(feasible, weather, cfg)
    print(f"built in {time.time() - t0:.1f}s, shape={features.shape}")

    labels = feasible[[
        "scenario_id", "optimal_capacity_kwh", "optimal_n_modules",
        "reference_lpsp", "reference_system_lcoe_eur_per_kwh", "reference_renewable_share",
        "reference_annualized_cost", "annual_load_kwh", "peak_load_kw",
    ]].rename(columns={"annual_load_kwh": "raw_annual_load_kwh", "peak_load_kw": "raw_peak_load_kw"})

    out_dir = REPO_ROOT / "data/ml_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_dir / "full_features.csv", index=False)
    labels.to_csv(out_dir / "full_labels.csv", index=False)
    print(f"wrote {out_dir / 'full_features.csv'} and full_labels.csv")
