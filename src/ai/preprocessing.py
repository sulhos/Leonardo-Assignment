"""Preprocessing: scaling and persisted preprocessing objects (PROJECT_BRIEF.md §17-§18).

Responsibilities (implemented in Stage 5):

- Fit numeric feature scalers on the training split only, and apply the same
  fitted transform to validation/test splits.
- Save fitted preprocessing objects under models/preprocessing/ so inference
  (Stage 6/7) uses the exact training-time transform.

This module intentionally contains no implementation yet -- it is a Stage 1
placeholder. Functions raise NotImplementedError until Stage 5.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def fit_scaler(train_features: pd.DataFrame):
    """Fit a scaler on training features only."""
    raise NotImplementedError("Implemented in Stage 5.")


def save_preprocessing_objects(objects: dict, output_dir: Path) -> None:
    raise NotImplementedError("Implemented in Stage 5.")


def load_preprocessing_objects(input_dir: Path) -> dict:
    raise NotImplementedError("Implemented in Stage 5.")
