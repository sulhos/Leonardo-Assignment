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


def test_unknown_profile_family_raises() -> None:
    with pytest.raises(NotImplementedError, match="not implemented"):
        generate_load_profile(**{**BASELINE_KWARGS, "profile_family": "commercial_baseline"})


INDUSTRIAL_KWARGS = dict(
    annual_consumption_kwh=5_000_000,
    peak_load_kw=1_000,
    year=2023,
    timezone="Asia/Shanghai",
    random_seed=42,
    profile_family="industrial_baseline",
)


def test_industrial_profile_basic_properties() -> None:
    load = generate_load_profile(**INDUSTRIAL_KWARGS)
    assert (load > 0).all()
    assert load.sum() == pytest.approx(INDUSTRIAL_KWARGS["annual_consumption_kwh"], rel=1e-6)
    assert load.max() <= INDUSTRIAL_KWARGS["peak_load_kw"]
    assert len(load) == 8760


def test_industrial_profile_has_higher_load_factor_than_residential() -> None:
    # The industrial shape is deliberately flatter (shift-based operation,
    # equipment running overnight) than the residential double-peak shape --
    # this should show up as a materially higher load factor (mean/peak).
    industrial = generate_load_profile(**INDUSTRIAL_KWARGS)
    residential = generate_load_profile(**BASELINE_KWARGS)
    industrial_load_factor = industrial.mean() / industrial.max()
    residential_load_factor = residential.mean() / residential.max()
    assert industrial_load_factor > residential_load_factor


def test_industrial_profile_weekend_reduction_is_substantial() -> None:
    # Per PROJECT_BRIEF.md Addendum 2: shift-based facility, weekend load
    # should drop to roughly 40-50% of the weekday level.
    load = generate_load_profile(**INDUSTRIAL_KWARGS)
    weekday_mean = load[load.index.dayofweek < 5].mean()
    weekend_mean = load[load.index.dayofweek >= 5].mean()
    ratio = weekend_mean / weekday_mean
    assert 0.35 < ratio < 0.55


def test_industrial_profile_seasonal_swing_is_smaller_than_residential() -> None:
    # Industrial process loads are much less weather-sensitive than
    # residential heating/cooling.
    industrial = generate_load_profile(**INDUSTRIAL_KWARGS)
    residential = generate_load_profile(**BASELINE_KWARGS)
    industrial_monthly = industrial.groupby(industrial.index.month).mean()
    residential_monthly = residential.groupby(residential.index.month).mean()
    industrial_swing = (industrial_monthly.max() - industrial_monthly.min()) / industrial_monthly.mean()
    residential_swing = (residential_monthly.max() - residential_monthly.min()) / residential_monthly.mean()
    assert industrial_swing < residential_swing
