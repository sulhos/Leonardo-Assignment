"""Statistical / conventional ML baseline models (PROJECT_BRIEF.md §17).

Hyperparameter selection uses only training + validation data (never the
test set): each candidate hyperparameter combination is fit on the training
split and scored on the validation split by MAE; the best-scoring model is
then refit on the training split only (not train+val) so its parameters
never see validation data either, matching a strict train/val/test
separation.

The naive baseline predicts the training-set median capacity for every
scenario, ignoring features entirely (`sklearn.dummy.DummyRegressor`) --
`config/ml_training.yaml`'s `naive.strategy: median_by_group` is simplified
to a single global median here, since the pilot dataset has no natural
scenario groups to condition on (see `src/ai/splitting.py`'s docstring for
the same reasoning applied to splitting).
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)


def fit_naive_baseline(train_features: pd.DataFrame, train_targets: pd.Series) -> DummyRegressor:
    model = DummyRegressor(strategy="median")
    model.fit(train_features, train_targets)
    return model


def fit_linear_baseline(
    train_features: pd.DataFrame, train_targets: pd.Series,
    val_features: pd.DataFrame, val_targets: pd.Series,
    alpha_grid: list[float] | None = None,
) -> Ridge:
    grid = alpha_grid or [0.01, 0.1, 1.0, 10.0]
    best_alpha, best_mae = None, float("inf")
    for alpha in grid:
        candidate = Ridge(alpha=alpha)
        candidate.fit(train_features, train_targets)
        mae = mean_absolute_error(val_targets, candidate.predict(val_features))
        logger.debug("Ridge alpha=%s val_MAE=%.3f", alpha, mae)
        if mae < best_mae:
            best_alpha, best_mae = alpha, mae

    logger.info("Selected Ridge alpha=%s (val MAE=%.3f)", best_alpha, best_mae)
    model = Ridge(alpha=best_alpha)
    model.fit(train_features, train_targets)
    return model


def fit_tree_baseline(
    train_features: pd.DataFrame, train_targets: pd.Series,
    val_features: pd.DataFrame, val_targets: pd.Series,
    n_estimators_grid: list[int] | None = None,
    max_depth_grid: list[int | None] | None = None,
    random_state: int = 42,
) -> RandomForestRegressor:
    n_estimators_options = n_estimators_grid or [100, 300]
    max_depth_options = max_depth_grid or [3, 5, None]

    best_params, best_mae, best_model = None, float("inf"), None
    for n_estimators in n_estimators_options:
        for max_depth in max_depth_options:
            candidate = RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth, random_state=random_state,
            )
            candidate.fit(train_features, train_targets)
            mae = mean_absolute_error(val_targets, candidate.predict(val_features))
            logger.debug("RandomForest n_estimators=%d max_depth=%s val_MAE=%.3f", n_estimators, max_depth, mae)
            if mae < best_mae:
                best_params, best_mae, best_model = (n_estimators, max_depth), mae, candidate

    logger.info("Selected RandomForest n_estimators=%d max_depth=%s (val MAE=%.3f)", *best_params, best_mae)
    return best_model
