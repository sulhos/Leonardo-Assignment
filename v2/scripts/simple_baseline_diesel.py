"""V2 Part D: one simple sanity-check baseline for the diesel-hybrid study.

Plain linear regression on the top-3 features by permutation importance
(scripts/task7_permutation_importance.py's diesel_full output -- run this
AFTER that script, not before) -- the simplest possible "does the NN beat
the obvious approach" reference. This matters more for the diesel target
than it did off-grid: many diesel scenarios have a near-zero optimal
battery and the target is dominated by a handful of features
(renewable-to-load ratio, load size), so a trivial model may do
surprisingly well -- reported honestly either way, not assumed to lose.
"""
from __future__ import annotations

import pandas as pd
from common import REPO_ROOT, TARGET_COL, load_merged_dataset, split_dataset
from sklearn.linear_model import LinearRegression

from src.ai.evaluation import compute_accuracy_metrics
from src.config import load_yaml_config

FIXED_SEED = 42
DATASET_NAME = "diesel_full"
TOP_N_FEATURES = 3


def run() -> dict:
    importance = pd.read_csv(REPO_ROOT / f"outputs/tables/permutation_importance_neural_network_{DATASET_NAME}.csv")
    top_features = importance.sort_values("mean_mae_increase_kwh", ascending=False)["feature"].head(TOP_N_FEATURES).tolist()
    print(f"top {TOP_N_FEATURES} features by NN permutation importance: {top_features}")

    ml_cfg = load_yaml_config("ml_training")
    merged, _ = load_merged_dataset(DATASET_NAME)
    splits = split_dataset(merged, FIXED_SEED, ml_cfg)
    train, test = merged.loc[splits["train"]], merged.loc[splits["test"]]

    model = LinearRegression()
    model.fit(train[top_features], train[TARGET_COL])
    pred = model.predict(test[top_features])
    pred = pred.clip(min=0.0)  # battery capacity can't be negative

    accuracy = compute_accuracy_metrics(test[TARGET_COL], pred)
    print("simple baseline (linear regression on top-3 features) accuracy:", accuracy)

    nn_accuracy = pd.read_csv(REPO_ROOT / f"outputs/tables/accuracy_metrics_neural_network_{DATASET_NAME}.csv").iloc[0]
    print("for comparison, NN accuracy:", dict(nn_accuracy))

    result = {
        "model": "linear_regression_top3", "features_used": top_features,
        **{f"{k}": v for k, v in accuracy.items()},
    }
    out_dir = REPO_ROOT / "outputs/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(out_dir / f"accuracy_metrics_simple_baseline_{DATASET_NAME}.csv", index=False)
    return result


if __name__ == "__main__":
    run()
