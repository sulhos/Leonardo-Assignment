"""Hourly mechanistic battery-dispatch simulation (PROJECT_BRIEF.md §11).

Responsibilities (implemented in Stage 2):

For every hour, in order: (1) PV+wind directly supply load, (2) renewable
surplus charges the battery, (3) renewable deficit discharges the battery,
(4) remaining surplus is curtailed, (5) remaining deficit is unserved energy.

Enforces: min/max SOC, charge/discharge efficiency, battery-capacity limit,
charge/discharge power limits, no simultaneous charge+discharge, no grid
exchange, no backup generator, no negative physical energy flows.

Records: direct renewable supply, battery charging, battery discharging,
curtailed energy, unserved energy, battery losses, SOC, renewable generation,
load served -- per hour, for the full local year.

Implements one of the two documented initial-SOC-bias mitigation methods
(repeated-year convergence, or a cyclic SOC solution) and tests it explicitly.

This module intentionally contains no implementation yet -- it is a Stage 1
placeholder. Functions raise NotImplementedError until Stage 2.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.physics.battery import BatterySpec

logger = logging.getLogger(__name__)


def run_dispatch(
    pv_generation_kw: pd.Series,
    wind_generation_kw: pd.Series,
    load_kw: pd.Series,
    battery: BatterySpec,
) -> pd.DataFrame:
    """Run the full-year hourly battery-dispatch simulation.

    Args:
        pv_generation_kw: Hourly PV output, kW.
        wind_generation_kw: Hourly wind output, kW.
        load_kw: Hourly load, kW.
        battery: Battery specification (including module count).

    Returns:
        Hourly DataFrame with columns for direct renewable supply, charging,
        discharging, curtailment, unserved energy, battery losses, and SOC.
    """
    raise NotImplementedError("Implemented in Stage 2.")


def resolve_initial_soc_bias(
    pv_generation_kw: pd.Series,
    wind_generation_kw: pd.Series,
    load_kw: pd.Series,
    battery: BatterySpec,
    method: str = "repeated_year",
    max_iterations: int = 10,
    tolerance_kwh: float = 1e-3,
) -> pd.DataFrame:
    """Run dispatch repeatedly (or via a cyclic SOC solution) until starting
    and ending SOC converge, avoiding an arbitrary initial-SOC bias."""
    raise NotImplementedError("Implemented in Stage 2.")
