"""Discrete exhaustive mechanistic battery-size optimization (PROJECT_BRIEF.md §12).

Enumerates candidate module counts n = 0, 1, ..., n_max. For each: runs the
complete hourly dispatch (with initial-SOC-bias resolution), computes the
full metric set (`src.physics.metrics`), and records per-candidate runtime.

Primary objective: the least-cost battery configuration satisfying
`LPSP <= lpsp_target`. Because battery price increases monotonically with
capacity here, this is always the smallest feasible module count -- so
selection is a simple minimum over feasible candidates, not a general cost
search.

Infeasibility is never silently accepted. If no candidate in the tested
range satisfies the target, the result is explicitly marked infeasible with
a reason that distinguishes two structurally different causes:
- **Renewable-generation inadequacy**: annual PV + wind production is less
  than annual load, so no finite battery can close the gap (a battery only
  shifts energy in time, it cannot create it).
- **Battery-range inadequacy**: renewable production is sufficient overall,
  but the tested module-count range (0..n_max) was not large enough.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import pandas as pd

from src.physics.battery import BatterySpec
from src.physics.dispatch import resolve_initial_soc_bias
from src.physics.metrics import compute_candidate_metrics

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
    reliability_sensitivity: pd.DataFrame = field(default_factory=pd.DataFrame)


def _infeasibility_reason(pv_generation_kw: pd.Series, wind_generation_kw: pd.Series, load_kw: pd.Series) -> str:
    renewable_total = pv_generation_kw.sum() + wind_generation_kw.sum()
    load_total = load_kw.sum()
    if renewable_total < load_total:
        return (
            "Renewable-generation inadequacy: annual PV+wind production "
            f"({renewable_total:.0f} kWh) is less than annual load ({load_total:.0f} kWh). "
            "No finite battery can close this gap -- storage shifts energy in time, it "
            "does not create it. Increasing PV/wind capacity (a scenario input, not a "
            "battery decision variable in this optimization) would be required."
        )
    return (
        "Battery-range inadequacy: annual renewable production is sufficient overall, "
        "but no module count within the tested range (0..n_max) achieved the reliability "
        "target. Consider increasing candidate_module_counts.max and re-running."
    )


def _select_smallest_feasible(candidates: pd.DataFrame, target_lpsp: float) -> pd.Series | None:
    feasible = candidates[candidates["lpsp"] <= target_lpsp]
    if feasible.empty:
        return None
    return feasible.loc[feasible["n_modules"].idxmin()]


def evaluate_reliability_sensitivity(
    candidates: pd.DataFrame, load_served_targets: list[float]
) -> pd.DataFrame:
    """For each target load-served fraction (e.g. 0.990, 0.995, 0.999), find
    the smallest feasible module count. PROJECT_BRIEF.md §12: "Also test:
    99.0%, 99.5%, 99.9% load served."
    """
    rows = []
    for target in load_served_targets:
        target_lpsp = 1.0 - target
        selected = _select_smallest_feasible(candidates, target_lpsp)
        rows.append(
            {
                "load_served_target": target,
                "feasible": selected is not None,
                "n_modules": selected["n_modules"] if selected is not None else None,
                "capacity_kwh": selected["nominal_battery_capacity_kwh"] if selected is not None else None,
                "achieved_lpsp": selected["lpsp"] if selected is not None else None,
            }
        )
    return pd.DataFrame(rows)


def run_battery_search(
    pv_generation_kw: pd.Series,
    wind_generation_kw: pd.Series,
    load_kw: pd.Series,
    n_max: int,
    module_capacity_kwh: float,
    module_rated_power_kw: float,
    round_trip_efficiency: float,
    min_soc_fraction: float,
    max_soc_fraction: float,
    initial_soc_fraction: float,
    installed_cost_eur_per_kwh: float,
    economic_lifetime_years: int,
    project_lifetime_years: int,
    real_discount_rate: float,
    lpsp_target: float = 0.01,
    reliability_sensitivity_targets: list[float] | None = None,
) -> OptimizationResult:
    """Exhaustively search battery module counts 0..n_max and select the
    least-cost feasible candidate. See PROJECT_BRIEF.md §12 for the full
    per-candidate metric list and infeasibility-reporting requirements."""
    if n_max < 0:
        raise ValueError("n_max must be >= 0.")

    search_start = time.perf_counter()
    rows = []
    for n in range(n_max + 1):
        candidate_start = time.perf_counter()
        battery = BatterySpec(
            module_capacity_kwh=module_capacity_kwh,
            module_rated_power_kw=module_rated_power_kw,
            round_trip_efficiency=round_trip_efficiency,
            min_soc_fraction=min_soc_fraction,
            max_soc_fraction=max_soc_fraction,
            initial_soc_fraction=initial_soc_fraction,
            n_modules=n,
        )
        dispatch_result = resolve_initial_soc_bias(pv_generation_kw, wind_generation_kw, load_kw, battery)
        candidate_runtime = time.perf_counter() - candidate_start

        metrics = compute_candidate_metrics(
            dispatch_result, battery, installed_cost_eur_per_kwh,
            economic_lifetime_years, project_lifetime_years, real_discount_rate,
            candidate_runtime,
        )
        rows.append(metrics)
        logger.debug("n_modules=%d LPSP=%.5f runtime=%.4fs", n, metrics["lpsp"], candidate_runtime)

    candidates = pd.DataFrame(rows)
    search_runtime = time.perf_counter() - search_start

    selected = _select_smallest_feasible(candidates, lpsp_target)
    sensitivity = evaluate_reliability_sensitivity(
        candidates, reliability_sensitivity_targets or [0.990, 0.995, 0.999]
    )

    if selected is None:
        reason = _infeasibility_reason(pv_generation_kw, wind_generation_kw, load_kw)
        logger.warning("No feasible battery size found within the tested range: %s", reason)
        return OptimizationResult(
            feasible=False,
            optimal_n_modules=None,
            optimal_capacity_kwh=None,
            lpsp=None,
            candidates=candidates,
            search_runtime_seconds=search_runtime,
            infeasibility_reason=(
                "No feasible battery size was found within the tested range for the "
                f"selected fixed PV and wind capacities. {reason}"
            ),
            reliability_sensitivity=sensitivity,
        )

    return OptimizationResult(
        feasible=True,
        optimal_n_modules=int(selected["n_modules"]),
        optimal_capacity_kwh=float(selected["nominal_battery_capacity_kwh"]),
        lpsp=float(selected["lpsp"]),
        candidates=candidates,
        search_runtime_seconds=search_runtime,
        infeasibility_reason=None,
        reliability_sensitivity=sensitivity,
    )
