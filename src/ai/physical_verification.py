"""Mandatory physical (mechanistic) verification of AI predictions
(PROJECT_BRIEF.md §19-§20; refinement addendum §1.6 — do not skip this stage).

Responsibilities (implemented in Stage 7):

- Convert each continuous AI prediction to an installable module count via
  ceiling division (never round down): N_AI = ceil(E_AI_predicted / module_capacity_kwh).
- Re-insert the rounded, AI-selected battery size into the hourly mechanistic
  dispatch model and run the full annual simulation using the PREDICTED
  capacity, never the reference/optimal capacity.
- Compute achieved LPSP, reliability pass/fail, cost, and curtailment for the
  AI-selected battery, and compare against the mechanistic optimum.
- Report: % of AI selections satisfying reliability, % undersized, %
  oversized, mean excess capacity, maximum underprediction, additional cost
  from overprediction, reliability violations from underprediction,
  difference in curtailed energy, difference in annualized cost.
- This step is mandatory and must run before drawing any conclusion about AI
  performance -- error metrics alone are not sufficient.

This module intentionally contains no implementation yet -- it is a Stage 1
placeholder. Functions raise NotImplementedError until Stage 7.
"""

from __future__ import annotations

import logging
import math

import pandas as pd

logger = logging.getLogger(__name__)


def predicted_capacity_to_modules(predicted_capacity_kwh: float, module_capacity_kwh: float) -> int:
    """Ceiling conversion from a continuous prediction to an installable
    module count. Never rounds down."""
    raise NotImplementedError("Implemented in Stage 7.")


def verify_predictions(predictions: pd.DataFrame, scenarios: pd.DataFrame) -> pd.DataFrame:
    """Run mechanistic dispatch for every AI-predicted (rounded) battery size
    in the test set and report reliability/cost/curtailment outcomes."""
    raise NotImplementedError("Implemented in Stage 7.")
