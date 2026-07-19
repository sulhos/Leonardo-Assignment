"""Scenario input sampling with physical-consistency checks (PROJECT_BRIEF.md §14).

Samples scenario inputs uniformly within the ranges configured in
`config/scenario_generation.yaml`, then checks each sampled combination for
physical consistency before accepting it -- resampling (not silently
discarding or replacing with an arbitrary value) any inconsistent draw.

The peak-load/annual-consumption compatibility check reuses
`src.physics.load_profile.generate_load_profile`'s own internal constraint
(it raises `ValueError` if the requested peak cannot support the requested
annual consumption for this load shape) rather than a separately-derived
approximate heuristic, so the check is exactly as strict as the actual
generator that will be used downstream -- no double standard between
"sampling looks fine" and "generation actually works."

Renewable-vs-demand plausibility is deliberately NOT used to reject
scenarios here: an unfavourable renewable-to-load ratio is a legitimate
scenario outcome that the mechanistic optimizer labels as infeasible
(PROJECT_BRIEF.md §12), not something that gets silently avoided at the
sampling stage.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.physics.load_profile import generate_load_profile

logger = logging.getLogger(__name__)

MAX_RESAMPLE_ATTEMPTS = 50


def check_scenario_consistency(scenario: dict) -> tuple[bool, str | None]:
    """Return (is_consistent, reason_if_not) for one sampled scenario.

    Checks:
    - All sampled quantities are positive / within valid unit ranges.
    - The requested peak_load_kw can actually support annual_load_kwh for
      the implemented load-profile shape (verified by actually attempting
      generation, not an approximate bound).
    """
    if scenario["pv_capacity_kwp"] <= 0:
        return False, "pv_capacity_kwp must be positive."
    if scenario["wind_capacity_kw"] < 0:
        return False, "wind_capacity_kw must be non-negative."
    if scenario["annual_load_kwh"] <= 0:
        return False, "annual_load_kwh must be positive."
    if scenario["peak_load_kw"] <= 0:
        return False, "peak_load_kw must be positive."
    if not (0.0 < scenario["reliability_target_load_served"] < 1.0):
        return False, "reliability_target_load_served must be in (0, 1)."
    if not (0.0 < scenario["round_trip_efficiency"] <= 1.0):
        return False, "round_trip_efficiency must be in (0, 1]."
    if not (0.0 < scenario["usable_soc_window_fraction"] < 1.0):
        return False, "usable_soc_window_fraction must be in (0, 1)."

    try:
        generate_load_profile(
            annual_consumption_kwh=scenario["annual_load_kwh"],
            peak_load_kw=scenario["peak_load_kw"],
            year=scenario["weather_year"],
            timezone=scenario["timezone"],
            random_seed=scenario["random_seed"],
        )
    except ValueError as exc:
        return False, f"Incompatible peak_load_kw/annual_load_kwh combination: {exc}"

    return True, None


def sample_scenarios(
    n_scenarios: int,
    ranges: dict[str, Any],
    random_seed: int,
    location: str,
    timezone: str,
    weather_year: int,
) -> list[dict]:
    """Sample `n_scenarios` physically-consistent scenario input dictionaries.

    Args:
        n_scenarios: Number of scenarios to sample.
        ranges: Dict of `[low, high]` pairs, keyed by: pv_capacity_kwp,
            wind_capacity_kw, annual_load_kwh, peak_load_kw,
            reliability_target_load_served, round_trip_efficiency,
            usable_soc_window_fraction (matches `config/scenario_generation.yaml`).
        random_seed: Base seed; each scenario gets a derived, reproducible
            per-scenario seed (`random_seed * 100_003 + scenario_id`).
        location: Fixed site name for this dataset (single-location pilot).
        timezone: Site IANA timezone (needed for the consistency check).
        weather_year: Fixed weather year for this dataset (single-year pilot).

    Returns:
        List of scenario dicts, one per accepted sample, each carrying a
        unique `scenario_id` and derived `random_seed`.
    """
    rng = np.random.default_rng(random_seed)
    scenarios = []

    for scenario_id in range(n_scenarios):
        scenario_seed = random_seed * 100_003 + scenario_id  # reproducible per-scenario seed
        accepted = None
        for attempt in range(MAX_RESAMPLE_ATTEMPTS):
            candidate = {
                "scenario_id": scenario_id,
                "location": location,
                "timezone": timezone,
                "weather_year": weather_year,
                "random_seed": scenario_seed,
                "pv_capacity_kwp": float(rng.uniform(*ranges["pv_capacity_kwp"])),
                "wind_capacity_kw": float(rng.uniform(*ranges["wind_capacity_kw"])),
                "annual_load_kwh": float(rng.uniform(*ranges["annual_load_kwh"])),
                "peak_load_kw": float(rng.uniform(*ranges["peak_load_kw"])),
                "reliability_target_load_served": float(rng.uniform(*ranges["reliability_target_load_served"])),
                "round_trip_efficiency": float(rng.uniform(*ranges["round_trip_efficiency"])),
                "usable_soc_window_fraction": float(rng.uniform(*ranges["usable_soc_window_fraction"])),
            }
            is_consistent, reason = check_scenario_consistency(candidate)
            if is_consistent:
                accepted = candidate
                break
            logger.debug("Scenario %d attempt %d rejected: %s", scenario_id, attempt, reason)

        if accepted is None:
            raise RuntimeError(
                f"Could not sample a physically-consistent scenario {scenario_id} "
                f"within {MAX_RESAMPLE_ATTEMPTS} attempts. Ranges may need review: {ranges}"
            )
        scenarios.append(accepted)

    logger.info("Sampled %d physically-consistent scenarios (seed=%d).", len(scenarios), random_seed)
    return scenarios
