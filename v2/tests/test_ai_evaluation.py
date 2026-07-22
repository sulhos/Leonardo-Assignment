"""Tests for `src.ai.evaluation` (accuracy/operational metrics shared by the
mechanistic-vs-neural-network comparison). Small deterministic synthetic
data, no internet access.

(V2 Task 2: the naive/ridge/random-forest baseline models previously tested
alongside these metrics in this file were removed with `src/ai/baselines.py`
-- V2's scope is mechanistic vs. neural network only.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ai.evaluation import compute_accuracy_metrics, compute_operational_metrics


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
