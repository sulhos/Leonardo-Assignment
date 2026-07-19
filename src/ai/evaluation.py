"""Accuracy and operational evaluation metrics (PROJECT_BRIEF.md §21).

Responsibilities (implemented in Stage 5/6):

- Continuous-prediction metrics: MAE, RMSE, R^2, median absolute error, MAPE
  (careful zero-target handling), max absolute error, mean signed error/bias,
  percentage within +-5%/+-10%/+-20%.
- Asymmetric operational metrics: underprediction rate, overprediction rate,
  mean/maximum underprediction, mean overprediction, reliability pass rate
  after mechanistic (physical) verification.
- Underprediction is reported separately, never hidden inside averaged error
  metrics (this is a reliability risk, not just an accuracy statistic).

This module intentionally contains no implementation yet -- it is a Stage 1
placeholder. Functions raise NotImplementedError until Stage 5/6.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_accuracy_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    raise NotImplementedError("Implemented in Stage 5/6.")


def compute_operational_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Underprediction/overprediction rates and magnitudes, reported
    separately from averaged accuracy metrics."""
    raise NotImplementedError("Implemented in Stage 5/6.")
