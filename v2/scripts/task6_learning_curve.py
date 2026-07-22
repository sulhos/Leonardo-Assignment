"""V2 Task 6: learning curve across training-set sizes.

Fixed val/test split (seed=42, matching the headline model in train_nn.py)
so every point is evaluated on the identical held-out test set; only the
training subset size varies (subsampled from the fixed train split via a
fixed RNG, so each larger size's subset is a superset of the smaller ones'
-- a proper nested learning-curve, not independent re-splits).

Target sizes are 100/250/500/1000/2000/3500 (the brief's suggested points),
but the fixed 70/15/15 split of this dataset's 4,526 feasible scenarios
yields only 3,168 training rows -- fewer than 5,000 * 0.70 = 3,500. The last
point is capped at the actual available training size (3,168, i.e. the full
training split) rather than forcing an unavailable 3,500; this is reported
honestly rather than silently rounding the dataset up.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from common import REPO_ROOT, load_merged_dataset, split_dataset, train_and_evaluate

from src.config import load_yaml_config

FIXED_SEED = 42
REQUESTED_SIZES = [100, 250, 500, 1000, 2000, 3500]


def run() -> pd.DataFrame:
    ml_cfg = load_yaml_config("ml_training")
    merged, feature_cols = load_merged_dataset()
    splits = split_dataset(merged, FIXED_SEED, ml_cfg)
    full_train_idx = splits["train"]
    n_train_available = len(full_train_idx)

    sizes = sorted({min(s, n_train_available) for s in REQUESTED_SIZES})
    print(f"training pool has {n_train_available} rows; using sizes {sizes}")

    rng = np.random.default_rng(FIXED_SEED)
    shuffled_train_idx = full_train_idx[rng.permutation(len(full_train_idx))]

    rows = []
    for size in sizes:
        t0 = time.time()
        subset_idx = shuffled_train_idx[:size]
        result = train_and_evaluate(
            merged, feature_cols, splits, FIXED_SEED, ml_cfg,
            checkpoint_dir=None, train_subset=pd.Index(subset_idx),
        )
        acc = result["accuracy"]
        row = {"n_train": result["n_train"], "mae": acc["mae"], "rmse": acc["rmse"], "r2": acc["r2"],
               "n_test": acc["n_samples"], "runtime_s": time.time() - t0}
        print(row)
        rows.append(row)

    results = pd.DataFrame(rows)
    out_dir = REPO_ROOT / "outputs/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "learning_curve_neural_network_full.csv", index=False)
    return results


if __name__ == "__main__":
    run()
