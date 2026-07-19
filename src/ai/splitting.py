"""Dataset splitting strategies (PROJECT_BRIEF.md §16).

Responsibilities (implemented in Stage 5):

- Experiment A (core scope): grouped in-domain split (70/15/15) that keeps
  near-identical scenario variants together, never a naive random row split.
- Experiment B (stretch): unseen-weather-year holdout.
- Experiment C (optional stretch): geographic transfer, Jinan -> Vasteras.
- Persist split identifiers so every model (baselines and neural network) is
  evaluated on exactly the same train/val/test observations.

This module intentionally contains no implementation yet -- it is a Stage 1
placeholder. Functions raise NotImplementedError until Stage 5.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def split_experiment_a(
    scenarios: pd.DataFrame,
    group_column: str,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    random_seed: int = 42,
) -> dict[str, pd.Index]:
    """Grouped in-domain split; returns {"train": ..., "val": ..., "test": ...}
    index objects."""
    raise NotImplementedError("Implemented in Stage 5.")


def split_experiment_b(scenarios: pd.DataFrame, holdout_weather_year: int) -> dict[str, pd.Index]:
    """Unseen-weather-year holdout split (stretch goal)."""
    raise NotImplementedError("Stretch goal — not part of core scope (addendum §1.1).")


def split_experiment_c(scenarios: pd.DataFrame, train_locations: list[str], test_locations: list[str]) -> dict[str, pd.Index]:
    """Geographic-transfer split, Jinan -> Vasteras (optional stretch goal)."""
    raise NotImplementedError("Optional stretch goal — not part of core scope (addendum §1.1).")
