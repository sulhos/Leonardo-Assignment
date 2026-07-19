"""Feedforward MLP battery-sizing neural network (PROJECT_BRIEF.md §18).

**Framework: Keras/TensorFlow** (pinned per refinement addendum §1.2). This
project's early-stopping, best-model-checkpointing, and non-negative-output
requirements map directly onto Keras's built-in `EarlyStopping` and
`ModelCheckpoint` callbacks, plus a `ReLU` output activation for the
non-negativity constraint. Do not mix in PyTorch (or any other framework)
elsewhere in `src/ai/` -- if this ever changes, update this docstring and
implement early stopping/checkpointing manually, since PyTorch has no
built-in equivalent.

Responsibilities (implemented in Stage 6):

- Architecture: input layer sized to the feature count, Dense(128, relu),
  Dropout(0.10), Dense(64, relu), Dense(32, relu), Dense(1) output.
- Standardize numeric inputs using scalers fit on training data only
  (src/ai/preprocessing.py).
- Regression loss: MAE/MSE/Huber (configurable; see config/ml_training.yaml).
- Monitor validation loss, use early stopping, save the best model to
  models/neural_network/.
- Prevent negative capacity predictions (ReLU output activation and/or
  post-hoc np.maximum(pred, 0) as a defensive second layer).
- Record architecture, random seeds, and training/validation loss curves.
- Do not implement an LSTM; this is tabular regression over engineered
  annual features, not a sequence model.

This module intentionally contains no implementation yet -- it is a Stage 1
placeholder. Functions raise NotImplementedError until Stage 6.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_model(n_features: int, config: dict):
    """Build the Keras MLP model per config/ml_training.yaml `neural_network.architecture`."""
    raise NotImplementedError("Implemented in Stage 6.")


def train_model(
    model,
    train_features: pd.DataFrame,
    train_targets: pd.Series,
    val_features: pd.DataFrame,
    val_targets: pd.Series,
    config: dict,
    checkpoint_dir: Path,
):
    """Train with early stopping and best-model checkpointing per
    config/ml_training.yaml `neural_network.training`."""
    raise NotImplementedError("Implemented in Stage 6.")


def predict(model, features: pd.DataFrame) -> np.ndarray:
    """Predict battery capacity, clamped to be non-negative."""
    raise NotImplementedError("Implemented in Stage 6.")
