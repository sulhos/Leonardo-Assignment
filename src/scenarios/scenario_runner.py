"""Run the mechanistic pipeline (weather -> load -> PV -> wind -> dispatch ->
battery optimization) for one scenario (PROJECT_BRIEF.md §14).

Responsibilities (implemented in Stage 4):

- Given one sampled scenario dict, run the full mechanistic chain and return
  the mechanistic label set: optimal battery capacity (kWh), optimal module
  count, feasibility flag, reference LPSP, reference annualized cost, and
  mechanistic optimization runtime.
- Support caching of completed scenarios so interrupted generation can resume
  without recomputation.

This module intentionally contains no implementation yet -- it is a Stage 1
placeholder. Functions raise NotImplementedError until Stage 4.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def run_scenario(scenario: dict) -> dict:
    """Run the full mechanistic pipeline for one scenario and return its
    labels (optimal capacity/modules, feasibility, LPSP, cost, runtime)."""
    raise NotImplementedError("Implemented in Stage 4.")
