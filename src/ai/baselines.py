"""Statistical / conventional ML baseline models (PROJECT_BRIEF.md §17).

Responsibilities (implemented in Stage 5):

- Naive/rule-based baseline.
- Linear or ridge regression.
- Random forest or gradient-boosting regression.
- Hyperparameter selection using only training + validation data (never the
  test set), using the same features and splits as the neural network so the
  comparison in Stage 9 is fair.

This module intentionally contains no implementation yet -- it is a Stage 1
placeholder. Functions raise NotImplementedError until Stage 5.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def fit_naive_baseline(train_features: pd.DataFrame, train_targets: pd.Series):
    raise NotImplementedError("Implemented in Stage 5.")


def fit_linear_baseline(train_features: pd.DataFrame, train_targets: pd.Series, val_features: pd.DataFrame, val_targets: pd.Series):
    raise NotImplementedError("Implemented in Stage 5.")


def fit_tree_baseline(train_features: pd.DataFrame, train_targets: pd.Series, val_features: pd.DataFrame, val_targets: pd.Series):
    raise NotImplementedError("Implemented in Stage 5.")
