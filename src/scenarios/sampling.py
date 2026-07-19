"""Scenario input sampling with physical-consistency checks (PROJECT_BRIEF.md §14).

Responsibilities (implemented in Stage 4):

- Sample scenario inputs (location, weather year, PV/wind capacity, load
  parameters, reliability target, efficiency, usable SOC window, etc.) within
  the ranges defined in config/scenario_generation.yaml.
- Reject or flag physically inconsistent combinations (e.g. annual
  consumption incompatible with peak demand, candidate renewable production
  implausible relative to demand) rather than generating them blindly.
- Never silently replace an infeasible scenario with an arbitrary maximum
  battery size -- infeasible scenarios are labelled as such, not discarded
  or faked.
- Use a fixed random seed and record configuration hashes for reproducibility.

This module intentionally contains no implementation yet -- it is a Stage 1
placeholder. Functions raise NotImplementedError until Stage 4.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def sample_scenarios(n_scenarios: int, ranges: dict[str, Any], random_seed: int) -> list[dict]:
    """Sample `n_scenarios` physically-consistent scenario input dictionaries."""
    raise NotImplementedError("Implemented in Stage 4.")


def check_scenario_consistency(scenario: dict) -> tuple[bool, str | None]:
    """Return (is_consistent, reason_if_not) for one sampled scenario."""
    raise NotImplementedError("Implemented in Stage 4.")
