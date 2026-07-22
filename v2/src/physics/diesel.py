"""Diesel-generator backup model (PROJECT_BRIEF.md Addendum 3 -- Path B pivot
from a hard-reliability-constrained off-grid system to a diesel-backstopped
hybrid system optimized for system LCOE, following the OptiCE-style
"renewable share vs. LCOE" framing from this project's course lecture
material).

Distinguishes, same as `src/physics/battery.py`:
1. **A structural convention** -- diesel rated power is sized at
   `1.25 * peak_load_kw`, matching OptiCE's own `Diesel_rated_power` sizing
   rule exactly (`Power_lcc_diesel_generator.m` in the course's OptiCE
   codebase). This guarantees the generator alone can always cover the
   system's single highest-demand hour, which is what makes every battery
   candidate "reliable" once diesel is present (see module-level note below).
2. **Modelling assumptions, cited** -- the linear fuel-consumption curve
   (HOMER's standard `F(P) = F0 * P_rated + F1 * P_output` form; see
   https://homerenergy.com/products/pro/docs/latest/fuel_curve.html) with the
   commonly-cited default coefficients F0=0.08145, F1=0.246 L/hr/kW, and a
   fuel price of 0.90 EUR/L (~7 CNY/L, China 2023-2024 average retail
   diesel price). None of these are official prices or manufacturer specs --
   they are documented assumptions, same status as this project's other
   economic inputs (`config/jinan.yaml`).
3. **Economic assumptions** -- installed capital cost (~650 EUR/kW,
   industrial-scale genset range), O&M as a fraction of capital per year,
   and lifetime/replacement handled via the same present-value/annuity
   machinery as `src/physics/economics.py` (diesel capital behaves like any
   other capital asset; fuel is a separate, purely annual operating cost
   with no present-value/replacement structure).

**Why this makes every battery candidate "reliable":** diesel is sized by
*power*, not *energy* -- it has no capacity limit, only a per-hour output
cap. Because that cap (1.25x peak load) exceeds the load in every single
hour by construction, diesel alone (with zero battery) can already serve
100% of the load if required to. This is precisely why the mechanistic
optimization objective changes in this pivot: LPSP is no longer a binding
constraint that makes some battery sizes "infeasible" -- it is a reported
metric that stays near zero for every candidate, and the real trade-off,
matching the course lecture's "Typical results (1)" chart, is renewable
share (%) vs. system LCOE ($/kWh) as battery capacity varies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.physics.economics import equivalent_annual_cost, present_value_cost

logger = logging.getLogger(__name__)

# HOMER's standard linear diesel fuel-consumption curve coefficients,
# widely cited as defaults in the hybrid-microgrid literature (see module
# docstring). L/hr/kW. Note (Task 1e, V2): this fuel curve and its default
# coefficients are from HOMER Energy, not the course lecture -- the lecture
# material does not specify a diesel fuel-consumption model of its own.
DEFAULT_FUEL_CURVE_INTERCEPT_L_PER_KWH_RATED = 0.08145
DEFAULT_FUEL_CURVE_SLOPE_L_PER_KWH_OUTPUT = 0.246


@dataclass(frozen=True)
class DieselSpec:
    """Diesel generator specification, loaded from YAML config."""

    rated_power_kw: float
    fuel_curve_intercept_l_per_kwh_rated: float
    fuel_curve_slope_l_per_kwh_output: float
    fuel_price_eur_per_l: float
    installed_cost_eur_per_kw: float
    om_cost_fraction_per_year: float
    economic_lifetime_years: int
    project_lifetime_years: int

    def __post_init__(self) -> None:
        if self.rated_power_kw < 0:
            raise ValueError("rated_power_kw must be >= 0.")
        if self.fuel_curve_intercept_l_per_kwh_rated < 0 or self.fuel_curve_slope_l_per_kwh_output < 0:
            raise ValueError("Fuel curve coefficients must be non-negative.")
        if self.fuel_price_eur_per_l < 0:
            raise ValueError("fuel_price_eur_per_l must be >= 0.")


def diesel_rated_power_kw(peak_load_kw: float, sizing_factor: float = 1.25) -> float:
    """Diesel rated power = `sizing_factor * peak_load_kw`, matching OptiCE's
    own `Diesel_rated_power = 1.25 * max(Power_consumption)` convention."""
    return sizing_factor * peak_load_kw


def fuel_consumption_l_per_hour(output_kw: np.ndarray | pd.Series, diesel: DieselSpec) -> np.ndarray:
    """HOMER's linear fuel curve: F(P) = F0 * P_rated + F1 * P_output,
    evaluated only for hours with non-zero output (a generator producing
    zero output is off, not idling, in this model)."""
    output = np.asarray(output_kw, dtype=float)
    fuel = np.where(
        output > 0,
        diesel.fuel_curve_intercept_l_per_kwh_rated * diesel.rated_power_kw
        + diesel.fuel_curve_slope_l_per_kwh_output * output,
        0.0,
    )
    return fuel


def apply_diesel_backup(dispatch_result: pd.DataFrame, diesel: DieselSpec) -> pd.DataFrame:
    """Post-process a `run_dispatch` result: split `unserved_kwh` into
    diesel-served energy (capped at `diesel.rated_power_kw` per hour) and any
    true residual still-unserved energy. Does not modify `run_dispatch`
    itself -- this is a separate step, mirroring how OptiCE's own code keeps
    `Operational_strategy_battery.m`/`Battery.m` (renewables+battery) and
    `Power_lcc_diesel_generator.m` (diesel) as distinct stages.

    Adds columns: `diesel_output_kwh`, `diesel_fuel_l`, `still_unserved_kwh`,
    and updates `load_served_kwh` to include the diesel contribution.
    """
    result = dispatch_result.copy()
    unserved = result["unserved_kwh"].to_numpy()

    diesel_output = np.minimum(unserved, diesel.rated_power_kw)
    still_unserved = unserved - diesel_output

    result["diesel_output_kwh"] = diesel_output
    result["diesel_fuel_l"] = fuel_consumption_l_per_hour(diesel_output, diesel)
    result["still_unserved_kwh"] = still_unserved
    result["load_served_kwh"] = result["load_served_kwh"] + diesel_output

    return result


def diesel_annual_fuel_cost_eur(dispatch_result_with_diesel: pd.DataFrame, diesel: DieselSpec) -> float:
    """Annual fuel cost = total annual fuel consumption (L) * fuel price (EUR/L)."""
    return float(dispatch_result_with_diesel["diesel_fuel_l"].sum() * diesel.fuel_price_eur_per_l)


def diesel_annualized_cost_eur(diesel: DieselSpec, real_discount_rate: float) -> float:
    """Equivalent annual capital + O&M cost of the diesel generator itself
    (excludes fuel, which is a separate annual operating cost with no
    present-value/replacement structure -- see `diesel_annual_fuel_cost_eur`)."""
    pv = present_value_cost(
        diesel.installed_cost_eur_per_kw, diesel.rated_power_kw,
        diesel.economic_lifetime_years, diesel.project_lifetime_years, real_discount_rate,
    )
    eac = equivalent_annual_cost(pv, diesel.project_lifetime_years, real_discount_rate)
    om_cost = diesel.om_cost_fraction_per_year * diesel.installed_cost_eur_per_kw * diesel.rated_power_kw
    return eac + om_cost
