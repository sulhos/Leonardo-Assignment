"""Accuracy and operational evaluation metrics (PROJECT_BRIEF.md §21).

Reliability pass rate "after mechanistic verification" is NOT computed here
-- it requires re-running the hourly dispatch model with each predicted
(rounded-up) battery capacity, which is `src.ai.physical_verification`'s job
(Stage 7), not a property of the prediction alone. Reported separately from
these metrics once that mandatory step exists.

Underprediction is reported separately from averaged accuracy metrics
throughout (never folded into a single blended number), since it is a
reliability risk, not just an error statistic (PROJECT_BRIEF.md §21).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def compute_accuracy_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    errors = y_pred_arr - y_true_arr
    abs_errors = np.abs(errors)

    nonzero_mask = y_true_arr != 0
    mape = float(np.mean(abs_errors[nonzero_mask] / y_true_arr[nonzero_mask])) if nonzero_mask.any() else float("nan")

    return {
        "mae": mean_absolute_error(y_true_arr, y_pred_arr),
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "r2": r2_score(y_true_arr, y_pred_arr) if len(y_true_arr) > 1 else float("nan"),
        "median_absolute_error": float(np.median(abs_errors)),
        "mape": mape,
        "max_absolute_error": float(np.max(abs_errors)),
        "mean_signed_error_bias": float(np.mean(errors)),
        "pct_within_5pct": float(np.mean(abs_errors <= 0.05 * np.abs(y_true_arr))),
        "pct_within_10pct": float(np.mean(abs_errors <= 0.10 * np.abs(y_true_arr))),
        "pct_within_20pct": float(np.mean(abs_errors <= 0.20 * np.abs(y_true_arr))),
        "n_samples": len(y_true_arr),
    }


def compute_operational_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Underprediction/overprediction rates and magnitudes, reported
    separately from averaged accuracy metrics. Underprediction here means
    the model predicted LESS battery capacity than the mechanistic
    reference -- a reliability risk if deployed as-is (before physical
    verification, Stage 7)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    errors = y_pred_arr - y_true_arr  # negative = underprediction, positive = overprediction

    underpredictions = -errors[errors < 0]
    overpredictions = errors[errors > 0]

    return {
        "underprediction_rate": float(np.mean(errors < 0)),
        "overprediction_rate": float(np.mean(errors > 0)),
        "exact_match_rate": float(np.mean(errors == 0)),
        "mean_underprediction_kwh": float(underpredictions.mean()) if len(underpredictions) else 0.0,
        "max_underprediction_kwh": float(underpredictions.max()) if len(underpredictions) else 0.0,
        "mean_overprediction_kwh": float(overpredictions.mean()) if len(overpredictions) else 0.0,
        "max_overprediction_kwh": float(overpredictions.max()) if len(overpredictions) else 0.0,
    }
