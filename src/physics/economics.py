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

logger = logging.getLogger(__name__)


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
