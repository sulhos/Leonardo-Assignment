"""Tests for `src.ai.neural_network` (Stage 6). Not part of the original
Stage-1 test-file list, added because this module has real behaviour worth
covering. Uses a small deterministic synthetic dataset and a small
architecture override to keep training fast (~seconds, not the full
128-64-32 config) -- no internet access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ai.neural_network import build_model, predict, set_global_seed, train_model

SMALL_CONFIG = {
    "architecture": {
        "hidden_layers": [8, 4],
        "activation": "relu",
        "dropout_after_first_layer": 0.10,
        "output_activation": "relu",
    },
    "training": {
        "loss": "huber",
        "learning_rate": 0.01,
        "batch_size": 8,
        "max_epochs": 30,
        "early_stopping": {"monitor": "val_loss", "patience": 5, "restore_best_weights": True},
        "checkpoint": {"monitor": "val_loss", "save_best_only": True},
    },
}


def _linear_dataset(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 5, n)
    y = 3.0 * x1 - 2.0 * x2 + 20 + rng.normal(0, 0.1, n)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    y = pd.Series(np.maximum(y, 0))  # keep targets non-negative, like real capacities
    return X[:40], y[:40], X[40:50], y[40:50], X[50:], y[50:]


def test_build_model_output_shape() -> None:
    model = build_model(n_features=5, config=SMALL_CONFIG)
    assert model.input_shape == (None, 5)
    assert model.output_shape == (None, 1)


def test_build_model_param_count_matches_architecture() -> None:
    model = build_model(n_features=3, config=SMALL_CONFIG)
    # 3->8 (32 params incl. bias) + 8->4 (36) + 4->1 (5) = 73
    expected = (3 * 8 + 8) + (8 * 4 + 4) + (4 * 1 + 1)
    assert model.count_params() == expected


def test_predict_never_negative() -> None:
    set_global_seed(42)
    model = build_model(n_features=2, config=SMALL_CONFIG)
    # Untrained model with random weights could in principle output small
    # values; ReLU output activation + defensive clip guarantee non-negativity.
    features = pd.DataFrame({"x1": [-100.0, 0.0, 100.0], "x2": [-50.0, 0.0, 50.0]})
    preds = predict(model, features)
    assert (preds >= 0).all()


def test_train_model_creates_checkpoint_and_history(tmp_path) -> None:
    X_train, y_train, X_val, y_val, _, _ = _linear_dataset()
    set_global_seed(42)
    model = build_model(n_features=2, config=SMALL_CONFIG)
    model, history = train_model(model, X_train, y_train, X_val, y_val, SMALL_CONFIG, tmp_path)

    assert (tmp_path / "best_model.keras").is_file()
    assert "loss" in history.history
    assert "val_loss" in history.history
    assert len(history.history["loss"]) > 0


def test_train_model_fits_linear_signal_reasonably() -> None:
    X_train, y_train, X_val, y_val, X_test, y_test = _linear_dataset()
    set_global_seed(42)
    model = build_model(n_features=2, config=SMALL_CONFIG)
    config = {**SMALL_CONFIG, "training": {**SMALL_CONFIG["training"], "max_epochs": 200,
                                            "early_stopping": {"monitor": "val_loss", "patience": 20, "restore_best_weights": True}}}
    model, _ = train_model(model, X_train, y_train, X_val, y_val, config, __import__("pathlib").Path("/tmp/nn_pytest_checkpoint"))
    preds = predict(model, X_test)
    mae = np.mean(np.abs(preds - y_test.to_numpy()))
    assert mae < 10.0  # should learn a clear linear signal reasonably well given enough epochs


def test_early_stopping_can_stop_before_max_epochs(tmp_path) -> None:
    X_train, y_train, X_val, y_val, _, _ = _linear_dataset()
    set_global_seed(42)
    model = build_model(n_features=2, config=SMALL_CONFIG)
    config = {**SMALL_CONFIG, "training": {**SMALL_CONFIG["training"], "max_epochs": 500,
                                            "early_stopping": {"monitor": "val_loss", "patience": 3, "restore_best_weights": True}}}
    _, history = train_model(model, X_train, y_train, X_val, y_val, config, tmp_path)
    assert len(history.history["loss"]) < 500
