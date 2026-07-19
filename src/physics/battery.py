"""Battery model: state, efficiency, and operational limits (PROJECT_BRIEF.md §5, §6).

Distinguishes:
1. **Manufacturer specifications** -- `module_capacity_kwh`, `module_rated_power_kw`,
   `round_trip_efficiency` (BYD Battery-Box Premium LVL-style reference values).
2. **Modelling assumptions** -- `min_soc_fraction`, `max_soc_fraction`,
   `initial_soc_fraction`, the charge/discharge efficiency split
   (`sqrt(round_trip_efficiency)` each way), and the temperature-derating
   feature (opt-in, disabled by default -- see `config/*.yaml` `battery.temperature_derating`).
3. **Economic assumptions** -- handled separately in `src/physics/economics.py`
   (installed cost, lifetimes, discount rate).

Temperature derating is intentionally NOT implemented here yet: for the
initial comparison a controlled/enclosure battery temperature is assumed
(PROJECT_BRIEF.md §6), and `config/*.yaml` has `temperature_derating.enabled: false`.
If it is enabled later, it must derate `rated_power_kwh_per_hour` in
`dispatch.py` explicitly and be documented per-experiment -- never applied silently.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatterySpec:
    """Configurable battery specification, loaded from YAML config.

    Fields map directly onto config/*.yaml `battery:` section keys. See
    module docstring for the manufacturer-spec vs. modelling-assumption vs.
    economic-assumption distinction.
    """

    module_capacity_kwh: float
    module_rated_power_kw: float
    round_trip_efficiency: float
    min_soc_fraction: float
    max_soc_fraction: float
    initial_soc_fraction: float
    n_modules: int

    def __post_init__(self) -> None:
        if self.n_modules < 0:
            raise ValueError("n_modules must be >= 0.")
        if not (0.0 <= self.min_soc_fraction < self.max_soc_fraction <= 1.0):
            raise ValueError(
                f"Expected 0 <= min_soc_fraction < max_soc_fraction <= 1, got "
                f"min={self.min_soc_fraction}, max={self.max_soc_fraction}."
            )
        if not (0.0 < self.round_trip_efficiency <= 1.0):
            raise ValueError("round_trip_efficiency must be in (0, 1].")

    @property
    def total_capacity_kwh(self) -> float:
        """Nominal (nameplate) capacity across all modules."""
        return self.n_modules * self.module_capacity_kwh

    @property
    def total_rated_power_kw(self) -> float:
        """Combined charge/discharge power limit across all modules."""
        return self.n_modules * self.module_rated_power_kw

    @property
    def min_soc_kwh(self) -> float:
        return self.total_capacity_kwh * self.min_soc_fraction

    @property
    def max_soc_kwh(self) -> float:
        return self.total_capacity_kwh * self.max_soc_fraction

    @property
    def initial_soc_kwh(self) -> float:
        return self.total_capacity_kwh * self.initial_soc_fraction


def charge_efficiency(round_trip_efficiency: float) -> float:
    """sqrt(round_trip_efficiency), per PROJECT_BRIEF.md §5."""
    return math.sqrt(round_trip_efficiency)


def discharge_efficiency(round_trip_efficiency: float) -> float:
    """sqrt(round_trip_efficiency), per PROJECT_BRIEF.md §5."""
    return math.sqrt(round_trip_efficiency)


def usable_capacity_kwh(spec: BatterySpec) -> float:
    """Usable capacity given min/max SOC bounds and module count."""
    return spec.max_soc_kwh - spec.min_soc_kwh
