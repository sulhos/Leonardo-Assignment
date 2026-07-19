"""Battery-system economic analysis (PROJECT_BRIEF.md §23).

Responsibilities (implemented in Stage 3):

- Initial battery investment, replacement in year 10 (economic lifetime),
  configurable real discount rate, present value, and equivalent annual cost
  over the project lifetime (20 years).
- Cost per kWh of load served.
- Battery-cost sensitivity (400 / 550 / 700 EUR/kWh) -- stretch goal beyond
  the single baseline price, per refinement addendum §1.1.
- Cost penalty from AI oversizing and reliability risk from AI undersizing
  (used by src/ai/physical_verification.py in Stage 7).
- No grid-import, grid-export, or fossil-fuel revenue terms (off-grid, no
  backup generator system boundary).

This module intentionally contains no implementation yet -- it is a Stage 1
placeholder. Functions raise NotImplementedError until Stage 3.
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
    `economic_lifetime_years`, over `project_lifetime_years`."""
    raise NotImplementedError("Implemented in Stage 3.")


def equivalent_annual_cost(present_value: float, project_lifetime_years: int, real_discount_rate: float) -> float:
    """Convert a present-value cost to an equivalent annual cost."""
    raise NotImplementedError("Implemented in Stage 3.")


def cost_per_kwh_served(equivalent_annual_cost_value: float, annual_load_served_kwh: float) -> float:
    """Equivalent annual cost divided by annual load energy actually served."""
    raise NotImplementedError("Implemented in Stage 3.")
