"""Discrete exhaustive mechanistic battery-size optimization (PROJECT_BRIEF.md §12).

Responsibilities (implemented in Stage 3):

- Enumerate candidate battery module counts n = 0, 1, ..., N_max
  (E_battery = n * module_capacity_kwh).
- For each candidate: run the full hourly dispatch, compute LPSP, curtailment,
  utilization, cost, and record computational time.
- Select the least-cost candidate satisfying LPSP <= the configured target
  (default 0.01). Also evaluate the 99.0% / 99.5% / 99.9% load-served
  sensitivity targets.
- Return an explicit infeasibility result (never a silently "best-effort"
  answer) if no candidate in the tested range satisfies the target, and
  distinguish "insufficient renewable generation" from "battery too small"
  when zero unserved energy is unreachable.
- Record the mechanistic runtime for the full search (used as the
  denominator basis for later AI break-even comparisons per addendum §1.5).

This module intentionally contains no implementation yet -- it is a Stage 1
placeholder. Functions raise NotImplementedError until Stage 3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizationResult:
    """Result of an exhaustive battery-size search for one scenario."""

    feasible: bool
    optimal_n_modules: int | None
    optimal_capacity_kwh: float | None
    lpsp: float | None
    candidates: pd.DataFrame  # one row per tested module count
    search_runtime_seconds: float
    infeasibility_reason: str | None = None


def run_battery_search(
    pv_generation_kw: pd.Series,
    wind_generation_kw: pd.Series,
    load_kw: pd.Series,
    n_max: int,
    lpsp_target: float = 0.01,
    **battery_kwargs,
) -> OptimizationResult:
    """Exhaustively search battery module counts 0..n_max and select the
    least-cost feasible candidate. See PROJECT_BRIEF.md §12 for the full
    per-candidate metric list and infeasibility-reporting requirements."""
    raise NotImplementedError("Implemented in Stage 3.")
