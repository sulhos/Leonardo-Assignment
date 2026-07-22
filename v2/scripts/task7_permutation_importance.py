"""V2 Task 7: permutation feature importance for the trained NN.

Loads the already-trained headline model (models/neural_network_full,
models/preprocessing_full -- from train_nn.py's seed=42 run) rather than
retraining: permutation importance measures a fixed model's sensitivity to
each input feature, not a training-time property, so reusing the headline
checkpoint keeps this consistent with the accuracy numbers reported
elsewhere for that same model.

For each feature, independently shuffles that column's values across the
test split (breaking its relationship with the target while preserving its
marginal distribution and every other feature's values), re-predicts, and
records the resulting increase in MAE over the unshuffled baseline -- larger
increase = more important. Repeated 10x per feature (different shuffles) and
averaged, since a single shuffle is a noisy estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from common import REPO_ROOT, SITE, TARGET_COL, load_merged_dataset, split_dataset

from src.ai.evaluation import compute_accuracy_metrics
from src.ai.neural_network import load_trained_model, predict
from src.ai.preprocessing import load_preprocessing_objects
from src.config import load_yaml_config

FIXED_SEED = 42
N_REPEATS = 10


def run(dataset_name: str = "full") -> pd.DataFrame:
    ml_cfg = load_yaml_config("ml_training")
    merged, feature_cols = load_merged_dataset(dataset_name)
    splits = split_dataset(merged, FIXED_SEED, ml_cfg)
    test = merged.loc[splits["test"]]

    preprocessing = load_preprocessing_objects(REPO_ROOT / f"models/preprocessing_{dataset_name}")
    scaler = preprocessing["scaler"]
    saved_feature_cols = preprocessing["feature_columns"]
    assert saved_feature_cols == feature_cols, "feature column order mismatch vs. saved preprocessing"

    model = load_trained_model(
        n_features=len(feature_cols), config=ml_cfg["neural_network"],
        weights_path=REPO_ROOT / f"models/neural_network_{dataset_name}/best_model.weights.h5",
    )

    test_x_scaled = pd.DataFrame(
        scaler.transform(test[feature_cols]), columns=feature_cols, index=test.index
    )
    baseline_pred = predict(model, test_x_scaled)
    baseline_mae = compute_accuracy_metrics(test[TARGET_COL], baseline_pred)["mae"]
    print(f"baseline test MAE: {baseline_mae:.2f} kWh")

    rng = np.random.default_rng(FIXED_SEED)
    rows = []
    for col in feature_cols:
        increases = []
        for _ in range(N_REPEATS):
            shuffled = test_x_scaled.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            pred = predict(model, shuffled)
            mae = compute_accuracy_metrics(test[TARGET_COL], pred)["mae"]
            increases.append(mae - baseline_mae)
        rows.append({
            "feature": col,
            "mean_mae_increase_kwh": float(np.mean(increases)),
            "std_mae_increase_kwh": float(np.std(increases)),
            "mean_mae_increase_pct_of_baseline": float(np.mean(increases) / baseline_mae * 100),
        })

    results = pd.DataFrame(rows).sort_values("mean_mae_increase_kwh", ascending=False).reset_index(drop=True)
    print(results.head(15).to_string())

    out_dir = REPO_ROOT / "outputs/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / f"permutation_importance_neural_network_{dataset_name}.csv", index=False)
    return results


if __name__ == "__main__":
    run()
