"""Mechanistic performance metrics per battery candidate (PROJECT_BRIEF.md §13).

Responsibilities (implemented in Stage 3):

Compute, per candidate: module count, nominal/usable capacity, rated power,
annual/peak load, annual PV/wind/total renewable production, direct renewable
consumption, load served / load-served fraction, LPSP, unserved energy, hours
with unmet load, curtailed energy, renewable utilization, battery charge/
discharge energy and losses, min/max/mean SOC, throughput, equivalent full
cycles, initial/replacement/present-value/equivalent-annual cost, cost per kWh
served, and optimization runtime.

Because the system is fully off-grid with no fossil generator, all served
energy is renewable by construction -- metrics must emphasize load-served
fraction, renewable utilization, curtailment, and reliability rather than a
meaningless "renewable share of served energy" figure.

This module intentionally contains no implementation yet -- it is a Stage 1
placeholder. Functions raise NotImplementedError until Stage 3.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_candidate_metrics(dispatch_result: pd.DataFrame, **context) -> dict:
    """Compute the full metric set (PROJECT_BRIEF.md §13) for one dispatch
    simulation result (one battery candidate for one scenario)."""
    raise NotImplementedError("Implemented in Stage 3.")


def lpsp(dispatch_result: pd.DataFrame) -> float:
    """Loss of Power Supply Probability: unserved energy / total load energy."""
    raise NotImplementedError("Implemented in Stage 3.")
