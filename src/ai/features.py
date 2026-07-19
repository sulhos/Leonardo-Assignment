"""Feature engineering for the ML dataset (PROJECT_BRIEF.md §15).

Responsibilities (implemented in Stage 4/5):

- Compute load, generation, net-load, and system features per scenario, as
  listed in PROJECT_BRIEF.md §15.
- Never include the mechanistically optimized battery capacity, reference
  optimal module count, or reference optimal cost as an input feature (target
  leakage) -- these are labels/reference outputs only.
- Fit all feature scalers on the training split only (enforced together with
  src/ai/splitting.py and src/ai/preprocessing.py).

This module intentionally contains no implementation yet -- it is a Stage 1
placeholder. Functions raise NotImplementedError until Stage 4/5.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that must never appear as ML input features (target leakage guard).
# Enforced by src/ai/features.py and checked in tests/test_features.py.
LEAKAGE_COLUMNS = frozenset(
    {
        "optimal_capacity_kwh",
        "optimal_n_modules",
        "reference_lpsp",
        "reference_annualized_cost",
    }
)


def compute_load_features(load_kw: pd.Series) -> dict:
    raise NotImplementedError("Implemented in Stage 4/5.")


def compute_generation_features(pv_kw: pd.Series, wind_kw: pd.Series) -> dict:
    raise NotImplementedError("Implemented in Stage 4/5.")


def compute_net_load_features(load_kw: pd.Series, pv_kw: pd.Series, wind_kw: pd.Series) -> dict:
    raise NotImplementedError("Implemented in Stage 4/5.")


def build_feature_matrix(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Assemble the full feature matrix for all scenarios, guaranteed free of
    LEAKAGE_COLUMNS."""
    raise NotImplementedError("Implemented in Stage 4/5.")
