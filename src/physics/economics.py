"""Generic capital-asset economic analysis (PROJECT_BRIEF.md §23).

Despite the historical module name, these functions are technology-agnostic:
`present_value_cost`/`equivalent_annual_cost` are reused for PV, wind,
battery, and the diesel generator's *capital* cost alike (all four now have
an `installed_cost_eur_per_*` economic assumption, added for the diesel/
system-LCOE pivot, PROJECT_BRIEF.md Addendum 3 -- previously only the
battery had a cost model, since only battery capacity was ever being
optimized).

**Economic assumptions**: every `installed_cost_eur_per_*` value is a
configurable modelling input, not an official manufacturer/vendor retail
price (PROJECT_BRIEF.md §5). `real_discount_rate` and each technology's
`economic_lifetime_years`/`project_lifetime_years` are likewise configurable
assumptions, not measured quantities.

Cost model: one initial investment at year 0, plus exactly one replacement
at `economic_lifetime_years` (if that falls strictly within
`project_lifetime_years`), both discounted to present value at
`real_discount_rate`. This covers *capital* cost only -- diesel's fuel cost
is a separate, purely annual operating cost with no present-value/
replacement structure (`src/physics/diesel.py::diesel_annual_fuel_cost_eur`),
since fuel is consumed continuously rather than replaced periodically.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# (technology, parameter, config_path, unit, source_basis) -- the documented
# basis for every cost/economic assumption used across PV, wind, battery, and
# diesel (PROJECT_BRIEF.md Addendum 3). None of these are official price
# quotes or manufacturer specs; see `economic_assumptions_table`'s docstring.
_ECONOMIC_ASSUMPTIONS = [
    ("PV", "installed_cost_eur_per_kwp", "pv.installed_cost_eur_per_kwp", "EUR/kWp",
     "2024 global utility-scale weighted-average installed cost (IRENA-style figures, ~$691-779/kWp)"),
    ("PV", "om_cost_fraction_per_year", "pv.om_cost_fraction_per_year", "fraction of capital/year",
     "Standard O&M assumption (~1.5%/year)"),
    ("PV", "economic_lifetime_years", "pv.economic_lifetime_years", "years",
     "Typical PV module warranty/economic-life assumption"),

    ("Wind", "installed_cost_eur_per_kw", "wind.installed_cost_eur_per_kw", "EUR/kW",
     "Between utility-scale onshore wind's global weighted average (~1,000 EUR/kW, IRENA) and smaller "
     "commercial-scale systems' higher per-kW cost, since this project's turbine range (200-2,000 kW) "
     "sits below utility wind-farm scale"),
    ("Wind", "om_cost_fraction_per_year", "wind.om_cost_fraction_per_year", "fraction of capital/year",
     "Standard O&M assumption (~2.5%/year)"),
    ("Wind", "economic_lifetime_years", "wind.economic_lifetime_years", "years",
     "Typical turbine design-life assumption"),

    ("Battery", "installed_cost_eur_per_kwh", "battery.installed_cost_eur_per_kwh", "EUR/kWh",
     "C&I/utility-scale LFP economies of scale vs. the residential-build figure (550 EUR/kWh) used "
     "earlier in this project -- not an official price quote"),
    ("Battery", "cost_sensitivity_eur_per_kwh", "battery.cost_sensitivity_eur_per_kwh", "EUR/kWh (list)",
     "Documented alternative values for a future battery-price sensitivity sweep (not yet run, "
     "PROJECT_BRIEF.md limitations)"),
    ("Battery", "economic_lifetime_years", "battery.economic_lifetime_years", "years",
     "Economic assumption -- replaced once within the 20-year project horizon"),
    ("Battery", "project_lifetime_years", "battery.project_lifetime_years", "years",
     "Economic assumption, shared project horizon across all four technologies"),

    ("Diesel", "fuel_price_eur_per_l", "diesel.fuel_price_eur_per_l", "EUR/L",
     "~7 CNY/L, China 2023-2024 average retail diesel price"),
    ("Diesel", "installed_cost_eur_per_kw", "diesel.installed_cost_eur_per_kw", "EUR/kW",
     "Industrial-scale genset installed-cost range (~$675-1,000/kW)"),
    ("Diesel", "om_cost_fraction_per_year", "diesel.om_cost_fraction_per_year", "fraction of capital/year",
     "Standard genset O&M assumption (~3%/year)"),
    ("Diesel", "economic_lifetime_years", "diesel.economic_lifetime_years", "years",
     "Typical diesel genset design-life assumption"),
    ("Diesel", "project_lifetime_years", "diesel.project_lifetime_years", "years",
     "Matches battery/PV/wind project lifetime"),
    ("Diesel", "sizing_factor", "diesel.sizing_factor", "x peak load kW",
     "Matches the OptiCE course lecture material's own Diesel_rated_power convention exactly "
     "(rated_power_kw = sizing_factor * peak_load_kw)"),
    ("Diesel", "fuel_curve_intercept_l_per_kwh_rated", "diesel.fuel_curve_intercept_l_per_kwh_rated",
     "L/hr per kW rated (F0)", "HOMER Energy's standard linear diesel generator fuel curve, widely-cited default"),
    ("Diesel", "fuel_curve_slope_l_per_kwh_output", "diesel.fuel_curve_slope_l_per_kwh_output",
     "L/hr per kW output (F1)", "HOMER Energy's standard linear diesel generator fuel curve, widely-cited default"),

    ("System-wide", "real_discount_rate", "economics.real_discount_rate", "fraction",
     "Standard economic assumption, applied uniformly across all four technologies"),
]


def economic_assumptions_table(site_config: dict) -> pd.DataFrame:
    """Tabulate every cost/economic assumption used by the diesel-backed
    system-LCOE pipeline (PV, wind, battery, diesel, discount rate), read
    directly from `site_config` (never hardcoded here) so the table always
    reflects whatever config actually drove a given run.

    Every value here is a documented modelling/economic assumption, not an
    official price quote or manufacturer spec (PROJECT_BRIEF.md §5); the
    `source_basis` column records where each figure came from.
    """
    rows = []
    for technology, parameter, config_path, unit, source in _ECONOMIC_ASSUMPTIONS:
        section, key = config_path.split(".")
        rows.append({
            "technology": technology,
            "parameter": parameter,
            "value": site_config[section][key],
            "unit": unit,
            "source_basis": source,
            "status": "Documented modelling/economic assumption -- not an official price quote or manufacturer spec",
            "config_path": f"config/jinan.yaml -> {config_path}",
        })
    return pd.DataFrame(rows)


def present_value_cost(
    installed_cost_eur_per_kwh: float,
    capacity_kwh: float,
    economic_lifetime_years: int,
    project_lifetime_years: int,
    real_discount_rate: float,
) -> float:
    """Present value of initial investment plus one replacement at
    `economic_lifetime_years`, over `project_lifetime_years`.

    A replacement is included only if the battery's economic lifetime ends
    strictly before the project lifetime (i.e. it wears out partway through
    the project and must be replaced once); if `economic_lifetime_years >=
    project_lifetime_years`, no replacement is needed.
    """
    initial_cost = installed_cost_eur_per_kwh * capacity_kwh
    pv = initial_cost

    if economic_lifetime_years < project_lifetime_years:
        replacement_cost = installed_cost_eur_per_kwh * capacity_kwh
        pv += replacement_cost / (1 + real_discount_rate) ** economic_lifetime_years

    return pv


def equivalent_annual_cost(
    present_value: float, project_lifetime_years: int, real_discount_rate: float
) -> float:
    """Convert a present-value cost to an equivalent annual cost (annuity
    formula, i.e. present value multiplied by the capital recovery factor)."""
    if real_discount_rate == 0:
        return present_value / project_lifetime_years

    r = real_discount_rate
    n = project_lifetime_years
    capital_recovery_factor = r / (1 - (1 + r) ** (-n))
    return present_value * capital_recovery_factor


def cost_per_kwh_served(equivalent_annual_cost_value: float, annual_load_served_kwh: float) -> float:
    """Equivalent annual cost divided by annual load energy actually served.

    Returns 0.0 if no load was served and the cost is also zero (e.g. a
    zero-capacity battery with a fully-served load from renewables alone);
    otherwise division by (near-)zero served energy is allowed to propagate
    as inf/nan so it is visible rather than silently hidden.
    """
    if annual_load_served_kwh == 0 and equivalent_annual_cost_value == 0:
        return 0.0
    return equivalent_annual_cost_value / annual_load_served_kwh
