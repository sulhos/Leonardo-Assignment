"""Tests for `src.physics.load_profile` (PROJECT_BRIEF.md §26)."""

from __future__ import annotations

import pytest

from src.physics.load_profile import generate_load_profile, load_duration_curve

BASELINE_KWARGS = dict(
    annual_consumption_kwh=30000,
    peak_load_kw=15,
    year=2023,
    timezone="Asia/Shanghai",
    random_seed=42,
)


def test_no_negative_demand() -> None:
    load = generate_load_profile(**BASELINE_KWARGS)
    assert (load > 0).all()


def test_annual_consumption_within_tolerance() -> None:
    load = generate_load_profile(**BASELINE_KWARGS)
    assert load.sum() == pytest.approx(BASELINE_KWARGS["annual_consumption_kwh"], rel=1e-6)


def test_peak_demand_respected() -> None:
    load = generate_load_profile(**BASELINE_KWARGS)
    assert load.max() <= BASELINE_KWARGS["peak_load_kw"]


def test_identical_seed_produces_identical_profile() -> None:
    load_a = generate_load_profile(**BASELINE_KWARGS)
    load_b = generate_load_profile(**BASELINE_KWARGS)
    assert load_a.equals(load_b)


def test_different_seed_produces_different_profile() -> None:
    load_a = generate_load_profile(**BASELINE_KWARGS)
    other_kwargs = {**BASELINE_KWARGS, "random_seed": 7}
    load_b = generate_load_profile(**other_kwargs)
    assert not load_a.equals(load_b)


def test_correct_length_non_leap_year() -> None:
    load = generate_load_profile(**BASELINE_KWARGS)
    assert len(load) == 8760


def test_unachievable_peak_raises() -> None:
    # A tiny peak_load_kw relative to annual consumption cannot be satisfied
    # by this generator's shape.
    with pytest.raises(ValueError, match="exceeds the requested"):
        generate_load_profile(
            annual_consumption_kwh=30000,
            peak_load_kw=1.0,
            year=2023,
            timezone="Asia/Shanghai",
            random_seed=42,
        )


def test_load_duration_curve_is_sorted_descending() -> None:
    load = generate_load_profile(**BASELINE_KWARGS)
    ldc = load_duration_curve(load)
    assert (ldc.diff().dropna() <= 0).all()
    assert ldc.index.min() > 0
    assert ldc.index.max() == pytest.approx(1.0)
