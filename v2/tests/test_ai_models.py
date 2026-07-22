"""Tests for `src.ai.baselines` and `src.ai.evaluation` (Stage 5). Not part
of the original Stage-1 test-file list, added because these modules have
real behaviour worth covering. Small deterministic synthetic data, no
internet access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ai.baselines import fit_linear_baseline, fit_naive_baseline, fit_tree_baseline
from src.ai.evaluation import compute_accuracy_metrics, compute_operational_metrics


def _linear_dataset(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 5, n)
    y = 3.0 * x1 - 2.0 * x2 + 10 + rng.normal(0, 0.1, n)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    y = pd.Series(y)
    return X[:40], y[:40], X[40:50], y[40:50], X[50:], y[50:]


def test_naive_baseline_predicts_training_median_constant() -> None:
    X_train, y_train, X_test, _, _, _ = _linear_dataset()
    model = fit_naive_baseline(X_train, y_train)
    preds = model.predict(X_test)
    assert np.allclose(preds, y_train.median())


def test_linear_baseline_fits_near_perfect_linear_signal() -> None:
    X_train, y_train, X_val, y_val, X_test, y_test = _linear_dataset()
    model = fit_linear_baseline(X_train, y_train, X_val, y_val)
    preds = model.predict(X_test)
    mae = np.mean(np.abs(preds - y_test))
    assert mae < 1.0  # near-linear synthetic data should fit tightly


def test_tree_baseline_outperforms_naive_on_clear_signal() -> None:
    X_train, y_train, X_val, y_val, X_test, y_test = _linear_dataset()
    naive = fit_naive_baseline(X_train, y_train)
    tree = fit_tree_baseline(X_train, y_train, X_val, y_val)

    naive_mae = np.mean(np.abs(naive.predict(X_test) - y_test))
    tree_mae = np.mean(np.abs(tree.predict(X_test) - y_test))
    assert tree_mae < naive_mae


def test_accuracy_metrics_perfect_predictions() -> None:
    y_true = pd.Series([10.0, 20.0, 30.0])
    metrics = compute_accuracy_metrics(y_true, y_true.to_numpy())
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["pct_within_5pct"] == pytest.approx(1.0)


def test_accuracy_metrics_known_bias() -> None:
    y_true = pd.Series([100.0, 100.0, 100.0])
    y_pred = np.array([90.0, 90.0, 90.0])  # consistent 10% underprediction
    metrics = compute_accuracy_metrics(y_true, y_pred)
    assert metrics["mean_signed_error_bias"] == pytest.approx(-10.0)
    assert metrics["mape"] == pytest.approx(0.10)


def test_operational_metrics_all_underprediction() -> None:
    y_true = pd.Series([100.0, 200.0, 300.0])
    y_pred = np.array([90.0, 180.0, 250.0])
    metrics = compute_operational_metrics(y_true, y_pred)
    assert metrics["underprediction_rate"] == pytest.approx(1.0)
    assert metrics["overprediction_rate"] == pytest.approx(0.0)
    assert metrics["max_underprediction_kwh"] == pytest.approx(50.0)


def test_operational_metrics_mixed() -> None:
    y_true = pd.Series([100.0, 100.0])
    y_pred = np.array([90.0, 110.0])  # one under, one over
    metrics = compute_operational_metrics(y_true, y_pred)
    assert metrics["underprediction_rate"] == pytest.approx(0.5)
    assert metrics["overprediction_rate"] == pytest.approx(0.5)
    assert metrics["mean_underprediction_kwh"] == pytest.approx(10.0)
    assert metrics["mean_overprediction_kwh"] == pytest.approx(10.0)
