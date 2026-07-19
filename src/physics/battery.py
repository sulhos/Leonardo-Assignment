"""Battery model: state, efficiency, and operational limits (PROJECT_BRIEF.md §5, §6).

Responsibilities (implemented in Stage 2):

- Represent a configurable LFP battery (BYD Battery-Box Premium LVL-style):
  module capacity/power, round-trip efficiency split into charge/discharge
  efficiency, min/max/initial SOC, module count -> total capacity.
- Enforce SOC bounds, charge/discharge power limits, and no simultaneous
  charge+discharge.
- Implement the battery state-update equation:
      E[t+1] = E[t] + charge_efficiency * charging_energy[t]
                     - discharging_energy_delivered[t] / discharge_efficiency
- Support the optional temperature-derating feature (charge/discharge power
  derating, optional thermal-management auxiliary load) as a clearly
  documented, opt-in extension -- disabled by default (controlled enclosure
  temperature assumed for the initial comparison).
- Clearly distinguish manufacturer specifications, modelling assumptions, and
  economic assumptions in configuration and documentation.

This module intentionally contains no implementation yet -- it is a Stage 1
placeholder. Functions raise NotImplementedError until Stage 2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatterySpec:
    """Configurable battery specification, loaded from YAML config.

    Fields map directly onto config/*.yaml `battery:` section keys. See
    PROJECT_BRIEF.md §5 for the manufacturer-spec vs. modelling-assumption vs.
    economic-assumption distinction.
    """

    module_capacity_kwh: float
    module_rated_power_kw: float
    round_trip_efficiency: float
    min_soc_fraction: float
    max_soc_fraction: float
    initial_soc_fraction: float
    n_modules: int


def charge_efficiency(round_trip_efficiency: float) -> float:
    """sqrt(round_trip_efficiency), per PROJECT_BRIEF.md §5."""
    raise NotImplementedError("Implemented in Stage 2.")


def discharge_efficiency(round_trip_efficiency: float) -> float:
    """sqrt(round_trip_efficiency), per PROJECT_BRIEF.md §5."""
    raise NotImplementedError("Implemented in Stage 2.")


def usable_capacity_kwh(spec: BatterySpec) -> float:
    """Usable capacity given min/max SOC bounds and module count."""
    raise NotImplementedError("Implemented in Stage 2.")
