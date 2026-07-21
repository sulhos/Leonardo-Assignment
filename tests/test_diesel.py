"""Tests for `src.physics.diesel` (added for the diesel/system-LCOE pivot,
PROJECT_BRIEF.md Addendum 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.physics.diesel import (
    DieselSpec,
    apply_diesel_backup,
    diesel_annual_fuel_cost_eur,
    diesel_annualized_cost_eur,
    diesel_rated_power_kw,
    fuel_consumption_l_per_hour,
)

DIESEL = DieselSpec(
    rated_power_kw=100.0,
    fuel_curve_intercept_l_per_kwh_rated=0.08145,
    fuel_curve_slope_l_per_kwh_output=0.246,
    fuel_price_eur_per_l=0.9,
    installed_cost_eur_per_kw=650,
    om_cost_fraction_per_year=0.03,
    economic_lifetime_years=15,
    project_lifetime_years=20,
)


def test_diesel_rated_power_matches_sizing_factor() -> None:
    assert diesel_rated_power_kw(peak_load_kw=80.0, sizing_factor=1.25) == pytest.approx(100.0)


def test_diesel_spec_rejects_negative_rated_power() -> None:
    with pytest.raises(ValueError, match="rated_power_kw"):
        DieselSpec(
            rated_power_kw=-1.0, fuel_curve_intercept_l_per_kwh_rated=0.08145,
            fuel_curve_slope_l_per_kwh_output=0.246, fuel_price_eur_per_l=0.9,
            installed_cost_eur_per_kw=650, om_cost_fraction_per_year=0.03,
            economic_lifetime_years=15, project_lifetime_years=20,
        )


def test_fuel_consumption_zero_at_zero_output() -> None:
    fuel = fuel_consumption_l_per_hour(np.array([0.0, 0.0]), DIESEL)
    assert (fuel == 0.0).all()


def test_fuel_consumption_matches_homer_linear_curve() -> None:
    output = np.array([50.0])
    fuel = fuel_consumption_l_per_hour(output, DIESEL)
    expected = DIESEL.fuel_curve_intercept_l_per_kwh_rated * DIESEL.rated_power_kw + DIESEL.fuel_curve_slope_l_per_kwh_output * 50.0
    assert fuel[0] == pytest.approx(expected)


def test_fuel_consumption_increasing_in_output() -> None:
    fuel = fuel_consumption_l_per_hour(np.array([10.0, 50.0, 90.0]), DIESEL)
    assert fuel[0] < fuel[1] < fuel[2]


def _dispatch_result(unserved: list[float]) -> pd.DataFrame:
    n = len(unserved)
    index = pd.date_range("2023-01-01", periods=n, freq="h", tz="Asia/Shanghai")
    return pd.DataFrame(
        {
            "load_kw": [10.0] * n,
            "unserved_kwh": unserved,
            "load_served_kwh": [10.0 - u for u in unserved],
        },
        index=index,
    )


def test_apply_diesel_backup_fully_covers_small_deficit() -> None:
    # Deficit (5 kWh) well under diesel's 100 kW rated power -> fully covered.
    result = apply_diesel_backup(_dispatch_result([5.0, 0.0, 5.0]), DIESEL)
    assert result["diesel_output_kwh"].tolist() == [5.0, 0.0, 5.0]
    assert (result["still_unserved_kwh"] == 0.0).all()
    assert result["load_served_kwh"].tolist() == [10.0, 10.0, 10.0]


def test_apply_diesel_backup_caps_at_rated_power() -> None:
    # Deficit (150 kWh) exceeds diesel's 100 kW rated power -> residual unserved.
    result = apply_diesel_backup(_dispatch_result([150.0]), DIESEL)
    assert result["diesel_output_kwh"].iloc[0] == pytest.approx(100.0)
    assert result["still_unserved_kwh"].iloc[0] == pytest.approx(50.0)


def test_apply_diesel_backup_zero_deficit_no_fuel_burned() -> None:
    result = apply_diesel_backup(_dispatch_result([0.0, 0.0]), DIESEL)
    assert (result["diesel_output_kwh"] == 0.0).all()
    assert (result["diesel_fuel_l"] == 0.0).all()


def test_diesel_annual_fuel_cost_matches_fuel_times_price() -> None:
    result = apply_diesel_backup(_dispatch_result([5.0]), DIESEL)
    expected_fuel_l = result["diesel_fuel_l"].sum()
    assert diesel_annual_fuel_cost_eur(result, DIESEL) == pytest.approx(expected_fuel_l * DIESEL.fuel_price_eur_per_l)


def test_diesel_annualized_cost_is_positive() -> None:
    assert diesel_annualized_cost_eur(DIESEL, real_discount_rate=0.05) > 0
