"""Tests for `src.physics.economics.economic_assumptions_table` (added
alongside the diesel/system-LCOE pivot's cost documentation, PROJECT_BRIEF.md
Addendum 3)."""

from __future__ import annotations

from src.physics.economics import economic_assumptions_table

SITE_CONFIG = {
    "pv": {"installed_cost_eur_per_kwp": 700, "om_cost_fraction_per_year": 0.015, "economic_lifetime_years": 25},
    "wind": {"installed_cost_eur_per_kw": 1200, "om_cost_fraction_per_year": 0.025, "economic_lifetime_years": 20},
    "battery": {"installed_cost_eur_per_kwh": 300, "cost_sensitivity_eur_per_kwh": [220, 300, 400],
                "economic_lifetime_years": 10, "project_lifetime_years": 20},
    "diesel": {"fuel_price_eur_per_l": 0.9, "installed_cost_eur_per_kw": 650, "om_cost_fraction_per_year": 0.03,
               "economic_lifetime_years": 15, "project_lifetime_years": 20, "sizing_factor": 1.25,
               "fuel_curve_intercept_l_per_kwh_rated": 0.08145, "fuel_curve_slope_l_per_kwh_output": 0.246},
    "economics": {"real_discount_rate": 0.05},
}


def test_economic_assumptions_table_covers_all_four_technologies() -> None:
    table = economic_assumptions_table(SITE_CONFIG)
    assert set(table["technology"]) == {"PV", "Wind", "Battery", "Diesel", "System-wide"}


def test_economic_assumptions_table_reads_values_from_config_not_hardcoded() -> None:
    table = economic_assumptions_table(SITE_CONFIG)
    pv_cost_row = table[(table["technology"] == "PV") & (table["parameter"] == "installed_cost_eur_per_kwp")]
    assert pv_cost_row["value"].iloc[0] == 700

    modified_config = {**SITE_CONFIG, "pv": {**SITE_CONFIG["pv"], "installed_cost_eur_per_kwp": 999}}
    modified_table = economic_assumptions_table(modified_config)
    modified_row = modified_table[
        (modified_table["technology"] == "PV") & (modified_table["parameter"] == "installed_cost_eur_per_kwp")
    ]
    assert modified_row["value"].iloc[0] == 999


def test_economic_assumptions_table_every_row_has_a_source_basis() -> None:
    table = economic_assumptions_table(SITE_CONFIG)
    assert (table["source_basis"].str.len() > 0).all()
    assert (table["status"].str.contains("not an official price quote")).all()
