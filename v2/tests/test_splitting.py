"""Tests for `src.ai.splitting` (PROJECT_BRIEF.md §26)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ai.splitting import split_experiment_a


def _scenarios(n: int = 40) -> pd.DataFrame:
    return pd.DataFrame({"scenario_id": range(n), "value": range(n)})


def test_split_reproducible_with_same_seed() -> None:
    scenarios = _scenarios()
    a = split_experiment_a(scenarios, group_column="scenario_id", random_seed=42)
    b = split_experiment_a(scenarios, group_column="scenario_id", random_seed=42)
    assert list(a["train"]) == list(b["train"])
    assert list(a["val"]) == list(b["val"])
    assert list(a["test"]) == list(b["test"])


def test_split_different_seed_can_differ() -> None:
    scenarios = _scenarios()
    a = split_experiment_a(scenarios, group_column="scenario_id", random_seed=42)
    b = split_experiment_a(scenarios, group_column="scenario_id", random_seed=7)
    assert list(a["train"]) != list(b["train"])


def test_split_no_overlap_between_splits() -> None:
    scenarios = _scenarios()
    splits = split_experiment_a(scenarios, group_column="scenario_id", random_seed=42)
    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)


def test_split_covers_all_rows() -> None:
    scenarios = _scenarios()
    splits = split_experiment_a(scenarios, group_column="scenario_id", random_seed=42)
    covered = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    assert covered == set(scenarios.index)


def test_split_approximate_proportions() -> None:
    scenarios = _scenarios(100)
    splits = split_experiment_a(scenarios, group_column="scenario_id", train_fraction=0.70, val_fraction=0.15, random_seed=42)
    assert 60 <= len(splits["train"]) <= 80
    assert 5 <= len(splits["val"]) <= 25
    assert 5 <= len(splits["test"]) <= 25


def test_split_keeps_groups_together() -> None:
    # Two rows sharing group "A" must always land in the same split.
    scenarios = pd.DataFrame({
        "scenario_id": range(20),
        "group": ["A", "A"] + [f"g{i}" for i in range(18)],
    })
    splits = split_experiment_a(scenarios, group_column="group", random_seed=42)
    for split_index in splits.values():
        group_values = scenarios.loc[split_index, "group"]
        if "A" in group_values.values:
            assert (group_values == "A").sum() == (scenarios["group"] == "A").sum()


def test_split_rejects_invalid_fractions() -> None:
    scenarios = _scenarios()
    with pytest.raises(ValueError, match="train_fraction"):
        split_experiment_a(scenarios, group_column="scenario_id", train_fraction=0.8, val_fraction=0.3)
