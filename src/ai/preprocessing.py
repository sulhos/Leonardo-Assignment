"""Preprocessing: scaling and persisted preprocessing objects (PROJECT_BRIEF.md §17-§18).

Scalers are fit on the training split only (never validation/test), then
applied unchanged to validation/test -- this is what prevents the
validation/test metrics from being contaminated by information about their
own distribution.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def fit_scaler(train_features: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on training features only."""
    scaler = StandardScaler()
    scaler.fit(train_features)
    return scaler


def save_preprocessing_objects(objects: dict, output_dir: Path) -> None:
    """Persist a dict of preprocessing objects (e.g. {"scaler": scaler,
    "feature_columns": [...]}) as one pickle file per key under `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in objects.items():
        path = output_dir / f"{name}.pkl"
        with path.open("wb") as f:
            pickle.dump(obj, f)
        logger.info("Saved preprocessing object %r to %s", name, path)


def load_preprocessing_objects(input_dir: Path) -> dict:
    """Load all `*.pkl` preprocessing objects from `input_dir`, keyed by filename stem."""
    objects = {}
    for path in sorted(input_dir.glob("*.pkl")):
        with path.open("rb") as f:
            objects[path.stem] = pickle.load(f)
    return objects
