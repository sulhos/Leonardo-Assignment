"""Dataset splitting strategies (PROJECT_BRIEF.md §16).

**Experiment A (in-domain, implemented):** grouped 70/15/15 train/val/test
split. `group_column` keeps all rows sharing a group value in the same
split, so near-identical scenario variants never leak across train/test.

For the pilot dataset specifically, each of the 100 scenarios is an
independent draw from continuous uniform distributions (`src.scenarios.sampling`)
-- there is no scenario-family structure (no near-duplicate variants of a
common template) to group by, unlike a hypothetical dataset built by
perturbing a smaller set of base scenarios. Grouping by `scenario_id` in
this case is therefore equivalent to a plain random split, which is the
correct behaviour for i.i.d. samples -- not a shortcut around the "no naive
random split across near-identical scenarios" requirement, since there are
no near-identical scenarios to leak between splits here. If a future larger
dataset (Stage 8) is built with real structured families, pass the actual
family column instead.

**Experiment B (unseen-weather-year holdout)** and **Experiment C
(geographic transfer)** remain stretch goals (refinement addendum §1.1) and
are not implemented -- the pilot uses a single location and single weather
year.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)


def split_experiment_a(
    scenarios: pd.DataFrame,
    group_column: str,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    random_seed: int = 42,
) -> dict[str, pd.Index]:
    """Grouped in-domain split; returns {"train": ..., "val": ..., "test": ...}
    index objects (subsets of `scenarios.index`)."""
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and train_fraction + val_fraction < 1):
        raise ValueError("train_fraction and val_fraction must be in (0, 1) and sum to < 1.")

    groups = scenarios[group_column]

    splitter_1 = GroupShuffleSplit(n_splits=1, train_size=train_fraction, random_state=random_seed)
    train_pos, rest_pos = next(splitter_1.split(scenarios, groups=groups))

    rest = scenarios.iloc[rest_pos]
    rest_groups = rest[group_column]
    val_fraction_of_rest = val_fraction / (1 - train_fraction)
    splitter_2 = GroupShuffleSplit(n_splits=1, train_size=val_fraction_of_rest, random_state=random_seed)
    val_pos_in_rest, test_pos_in_rest = next(splitter_2.split(rest, groups=rest_groups))

    result = {
        "train": scenarios.index[train_pos],
        "val": rest.index[val_pos_in_rest],
        "test": rest.index[test_pos_in_rest],
    }
    logger.info(
        "Experiment A split: train=%d val=%d test=%d (of %d total)",
        len(result["train"]), len(result["val"]), len(result["test"]), len(scenarios),
    )
    return result


def split_experiment_b(scenarios: pd.DataFrame, holdout_weather_year: int) -> dict[str, pd.Index]:
    """Unseen-weather-year holdout split (stretch goal)."""
    raise NotImplementedError("Stretch goal — not part of core scope (addendum §1.1).")


def split_experiment_c(scenarios: pd.DataFrame, train_locations: list[str], test_locations: list[str]) -> dict[str, pd.Index]:
    """Geographic-transfer split, Jinan -> Vasteras (optional stretch goal)."""
    raise NotImplementedError("Optional stretch goal — not part of core scope (addendum §1.1).")
