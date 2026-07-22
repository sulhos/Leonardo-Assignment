"""Shared plumbing for the V2 Task 5-8 analysis scripts: load the engineered
feature matrix, split, scale, and train/evaluate the NN. Extracted from
train_nn.py so cross-validation (Task 5), the learning curve (Task 6), and
permutation importance (Task 7) can reuse one training path instead of three
divergent copies.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ai.evaluation import compute_accuracy_metrics
from src.ai.features import LEAKAGE_COLUMNS
from src.ai.neural_network import build_model, predict, set_global_seed, train_model
from src.ai.preprocessing import fit_scaler
from src.ai.splitting import split_experiment_a
from src.config import load_yaml_config

REPO_ROOT = Path(__file__).resolve().parents[1]
SITE = "jinan"
TARGET_COL = "optimal_capacity_kwh"


def load_merged_dataset(dataset_name: str = "full") -> tuple[pd.DataFrame, list[str]]:
    """`dataset_name`: "full" (the completed off-grid study) or "diesel_full"
    (the diesel-hybrid study that replaced it as the current ML target) --
    selects which of data/ml_dataset/{name}_features.csv/{name}_labels.csv
    to load, so this one function serves both without duplication."""
    features = pd.read_csv(REPO_ROOT / f"data/ml_dataset/{dataset_name}_features.csv")
    labels = pd.read_csv(REPO_ROOT / f"data/ml_dataset/{dataset_name}_labels.csv")
    merged = features.merge(labels, on="scenario_id", how="inner", validate="one_to_one")
    feature_cols = [c for c in features.columns if c != "scenario_id"]
    assert not (LEAKAGE_COLUMNS & set(feature_cols)), "leakage columns present in feature_cols"
    return merged, feature_cols


def split_dataset(merged: pd.DataFrame, random_seed: int, ml_cfg: dict) -> dict[str, pd.Index]:
    split_cfg = ml_cfg["splitting"]["experiment_a"]
    return split_experiment_a(
        merged, group_column="scenario_id",
        train_fraction=split_cfg["train_fraction"], val_fraction=split_cfg["val_fraction"],
        random_seed=random_seed,
    )


def train_and_evaluate(
    merged: pd.DataFrame,
    feature_cols: list[str],
    splits: dict[str, pd.Index],
    random_seed: int,
    ml_cfg: dict,
    checkpoint_dir: Path | None = None,
    train_subset: pd.Index | None = None,
) -> dict:
    """Fit a scaler on train (or `train_subset` of it), train the NN, and
    return test-split accuracy metrics plus the fitted model/scaler/history.

    `train_subset` (a subset of `splits["train"]`) is for the Task 6 learning
    curve: shrinks the training set while val/test stay fixed, so points are
    directly comparable.
    """
    set_global_seed(random_seed)
    train_idx = train_subset if train_subset is not None else splits["train"]
    train, val, test = merged.loc[train_idx], merged.loc[splits["val"]], merged.loc[splits["test"]]

    scaler = fit_scaler(train[feature_cols])
    train_x = pd.DataFrame(scaler.transform(train[feature_cols]), columns=feature_cols, index=train.index)
    val_x = pd.DataFrame(scaler.transform(val[feature_cols]), columns=feature_cols, index=val.index)
    test_x = pd.DataFrame(scaler.transform(test[feature_cols]), columns=feature_cols, index=test.index)

    nn_cfg = ml_cfg["neural_network"]
    model = build_model(n_features=len(feature_cols), config=nn_cfg)
    ckpt_dir = checkpoint_dir if checkpoint_dir is not None else (REPO_ROOT / "outputs/.tmp_checkpoint")
    model, history = train_model(model, train_x, train[TARGET_COL], val_x, val[TARGET_COL], nn_cfg, checkpoint_dir=ckpt_dir)

    test_pred = predict(model, test_x)
    accuracy = compute_accuracy_metrics(test[TARGET_COL], test_pred)

    return {
        "model": model, "scaler": scaler, "history": history,
        "accuracy": accuracy, "n_train": len(train), "test_pred": test_pred,
        "test": test, "test_x": test_x,
    }
