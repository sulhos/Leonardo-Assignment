"""V2 Task 5: cross-validation error bars on NN accuracy.

Repeated-random-split cross-validation (10 independent 70/15/15 grouped
splits, seeds 0-9) rather than k-fold: split_experiment_a already
implements exactly this grouped-split logic parameterized by random_seed,
so reusing it directly (instead of implementing separate k-fold machinery)
gives 10 independent train/val/test partitions of the same feasible-scenario
pool, each producing one MAE/RMSE/R2 on a freshly held-out test split.
Reports mean +/- std across the 10 runs -- the headline single-seed=42
number reported elsewhere (accuracy_metrics_neural_network_full.csv) is one
draw from this same distribution, not a separately-optimized best case.
"""
from __future__ import annotations

import time

import pandas as pd
from common import REPO_ROOT, load_merged_dataset, split_dataset, train_and_evaluate

from src.config import load_yaml_config

N_REPEATS = 10
SEEDS = list(range(N_REPEATS))


def run(dataset_name: str = "full") -> pd.DataFrame:
    ml_cfg = load_yaml_config("ml_training")
    merged, feature_cols = load_merged_dataset(dataset_name)

    rows = []
    for seed in SEEDS:
        t0 = time.time()
        splits = split_dataset(merged, seed, ml_cfg)
        result = train_and_evaluate(merged, feature_cols, splits, seed, ml_cfg, checkpoint_dir=None)
        acc = result["accuracy"]
        row = {"seed": seed, "n_train": result["n_train"], "n_test": acc["n_samples"],
               "mae": acc["mae"], "rmse": acc["rmse"], "r2": acc["r2"], "runtime_s": time.time() - t0}
        print(row)
        rows.append(row)

    results = pd.DataFrame(rows)
    summary = results[["mae", "rmse", "r2"]].agg(["mean", "std", "min", "max"])
    print(summary)

    out_dir = REPO_ROOT / "outputs/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / f"cross_validation_runs_neural_network_{dataset_name}.csv", index=False)
    summary.to_csv(out_dir / f"cross_validation_summary_neural_network_{dataset_name}.csv")
    return results


if __name__ == "__main__":
    import sys
    run(dataset_name=sys.argv[1] if len(sys.argv) > 1 else "full")
