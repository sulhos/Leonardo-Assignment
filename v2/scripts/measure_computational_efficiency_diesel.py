"""Measure real per-scenario compute time for the diesel-hybrid study: the
full mechanistic 0-80-module search vs. neural-network inference (single-
call and batched), and plot the comparison (report SS15).

Mechanistic timing reuses the per-scenario runtimes already recorded while
generating the 5,000-scenario dataset (data/scenarios/diesel_full_scenarios.csv,
mechanistic_optimization_runtime_seconds). Neural-network timing is measured
fresh here, on the same trained model and test split used throughout the
report, rather than reused from an earlier ad-hoc measurement.
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from scripts.common import load_merged_dataset, split_dataset
from src.config import load_yaml_config
from src.visualization.plotting import plot_computational_efficiency, save_figure

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "diesel_full"
N_BATCHED_REPEATS = 10
N_SINGLE_CALLS = 50


def run() -> dict:
    scenarios = pd.read_csv(REPO_ROOT / f"data/scenarios/{DATASET_NAME}_scenarios.csv")
    mechanistic_seconds = float(scenarios["mechanistic_optimization_runtime_seconds"].mean())

    df, _ = load_merged_dataset(dataset_name=DATASET_NAME)
    ml_cfg = load_yaml_config("ml_training")
    splits = split_dataset(df, random_seed=42, ml_cfg=ml_cfg)
    test_df = df.loc[splits["test"]]

    scaler = joblib.load(REPO_ROOT / f"models/preprocessing_{DATASET_NAME}/scaler.pkl")
    feature_columns = joblib.load(REPO_ROOT / f"models/preprocessing_{DATASET_NAME}/feature_columns.pkl")
    model = keras.models.load_model(REPO_ROOT / f"models/neural_network_{DATASET_NAME}/best_model.keras")

    x_test_scaled = scaler.transform(test_df[feature_columns].values)
    model.predict(x_test_scaled, verbose=0)  # warm-up, excluded from timing

    t0 = time.perf_counter()
    for _ in range(N_BATCHED_REPEATS):
        model.predict(x_test_scaled, verbose=0)
    batched_seconds_per_scenario = (time.perf_counter() - t0) / N_BATCHED_REPEATS / len(x_test_scaled)

    single_call_seconds = []
    for i in range(N_SINGLE_CALLS):
        row = x_test_scaled[i % len(x_test_scaled)].reshape(1, -1)
        t0 = time.perf_counter()
        model.predict(row, verbose=0)
        single_call_seconds.append(time.perf_counter() - t0)
    single_seconds_per_scenario = float(np.mean(single_call_seconds))

    results = {
        "mechanistic_seconds_per_scenario": mechanistic_seconds,
        "nn_single_seconds_per_scenario": single_seconds_per_scenario,
        "nn_batched_seconds_per_scenario": batched_seconds_per_scenario,
        "n_test_scenarios": len(x_test_scaled),
        "speedup_single": mechanistic_seconds / single_seconds_per_scenario,
        "speedup_batched": mechanistic_seconds / batched_seconds_per_scenario,
    }
    print(results)

    out_dir = REPO_ROOT / "outputs/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([results]).to_csv(out_dir / f"computational_efficiency_{DATASET_NAME}.csv", index=False)

    fig = plot_computational_efficiency(mechanistic_seconds, single_seconds_per_scenario, batched_seconds_per_scenario)
    save_figure(fig, REPO_ROOT / "outputs/figures/28_computational_efficiency.png")
    return results


if __name__ == "__main__":
    run()
