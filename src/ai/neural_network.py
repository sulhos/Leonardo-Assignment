"""Feedforward MLP battery-sizing neural network (PROJECT_BRIEF.md §18).

**Framework: Keras/TensorFlow** (pinned per refinement addendum §1.2). This
project's early-stopping, best-model-checkpointing, and non-negative-output
requirements map directly onto Keras's built-in `EarlyStopping` and
`ModelCheckpoint` callbacks, plus a `ReLU` output activation for the
non-negativity constraint. Do not mix in PyTorch (or any other framework)
elsewhere in `src/ai/` -- if this ever changes, update this docstring and
implement early stopping/checkpointing manually, since PyTorch has no
built-in equivalent.

Architecture and training hyperparameters come from
`config/ml_training.yaml` -> `neural_network`, not hard-coded here.

Do not implement an LSTM; this is tabular regression over engineered annual
features, not a sequence model.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Set Python/NumPy/TensorFlow seeds for as-reproducible-as-reasonably-
    possible training. Full bitwise determinism is not guaranteed on every
    backend/hardware combination (see config/ml_training.yaml's
    `neural_network.reproducibility.note`)."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(n_features: int, config: dict) -> keras.Model:
    """Build the Keras MLP model per config/ml_training.yaml `neural_network.architecture`."""
    arch = config["architecture"]
    hidden_layers = arch["hidden_layers"]
    activation = arch["activation"]
    dropout_rate = arch["dropout_after_first_layer"]
    output_activation = arch["output_activation"]

    inputs = keras.Input(shape=(n_features,), name="features")
    x = keras.layers.Dense(hidden_layers[0], activation=activation)(inputs)
    x = keras.layers.Dropout(dropout_rate)(x)
    for units in hidden_layers[1:]:
        x = keras.layers.Dense(units, activation=activation)(x)
    outputs = keras.layers.Dense(1, activation=output_activation, name="capacity_kwh")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="battery_capacity_mlp")

    training_cfg = config["training"]
    optimizer = keras.optimizers.Adam(learning_rate=training_cfg["learning_rate"])
    model.compile(optimizer=optimizer, loss=training_cfg["loss"], metrics=["mae"])
    return model


def train_model(
    model: keras.Model,
    train_features: pd.DataFrame,
    train_targets: pd.Series,
    val_features: pd.DataFrame,
    val_targets: pd.Series,
    config: dict,
    checkpoint_dir: Path,
) -> tuple[keras.Model, keras.callbacks.History]:
    """Train with early stopping and best-model checkpointing per
    config/ml_training.yaml `neural_network.training`."""
    training_cfg = config["training"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.keras"

    early_stopping_cfg = training_cfg["early_stopping"]
    checkpoint_cfg = training_cfg["checkpoint"]

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=early_stopping_cfg["monitor"],
            patience=early_stopping_cfg["patience"],
            restore_best_weights=early_stopping_cfg["restore_best_weights"],
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=checkpoint_cfg["monitor"],
            save_best_only=checkpoint_cfg["save_best_only"],
        ),
    ]

    history = model.fit(
        train_features.to_numpy(dtype="float32"),
        train_targets.to_numpy(dtype="float32"),
        validation_data=(val_features.to_numpy(dtype="float32"), val_targets.to_numpy(dtype="float32")),
        epochs=training_cfg["max_epochs"],
        batch_size=training_cfg["batch_size"],
        callbacks=callbacks,
        verbose=0,
    )

    logger.info(
        "Training finished after %d epochs (best val_loss=%.4f).",
        len(history.history["loss"]), min(history.history["val_loss"]),
    )
    return model, history


def predict(model: keras.Model, features: pd.DataFrame) -> np.ndarray:
    """Predict battery capacity, clamped to be non-negative.

    The output layer already uses a ReLU activation (non-negative by
    construction, per config), but `np.maximum(..., 0.0)` is applied as a
    defensive second layer per PROJECT_BRIEF.md §18, in case the output
    activation is ever changed.
    """
    raw = model.predict(features.to_numpy(dtype="float32"), verbose=0).flatten()
    return np.maximum(raw, 0.0)
