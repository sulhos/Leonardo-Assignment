"""Discrete exhaustive mechanistic battery-size optimization (PROJECT_BRIEF.md §12).

Enumerates candidate module counts n = 0, 1, ..., n_max. For each: runs the
complete hourly dispatch (with initial-SOC-bias resolution), applies diesel
backup (`src.physics.diesel.apply_diesel_backup`), computes the full metric
set (`src.physics.metrics`), and records per-candidate runtime.

**Two system configurations (V2 Task 3), selected via `system_backup`:**

1. `system_backup="diesel"` (the diesel/system-LCOE pivot, PROJECT_BRIEF.md
   Addendum 3): the least-*system-LCOE* battery configuration, not the
   smallest battery satisfying a hard `LPSP <= lpsp_target` constraint.
   Diesel is sized by power (`1.25 * peak_load_kw`, always >= the load in
   every hour by construction), not energy, so it can serve 100% of the
   load at any battery capacity if required -- this makes LPSP-after-diesel
   near-zero for every candidate, and the real trade-off (matching the
   course lecture's OptiCE "renewable share vs LCOE" chart) is economic,
   not a feasibility gate. Physical verification under this configuration
   tests cost-optimality, not reliability (reliability is guaranteed by
   construction) -- reported honestly as such.
2. `system_backup="none"` (off-grid, matching the original spec and
   research question 3 word-for-word): no diesel at all -- implemented as
   a `DieselSpec` with `rated_power_kw=0.0`, which flows through the exact
   same `apply_diesel_backup` code path and naturally yields zero diesel
   output/fuel/capital cost at every hour, so `still_unserved_kwh` becomes
   the true renewables+battery-only unserved energy. The optimum is the
   *smallest* battery module count meeting `lpsp_target`
   (`_select_smallest_feasible`); a scenario is genuinely infeasible (not
   silently reported as met) if no candidate within `0..n_max` achieves it
   -- reliability is a hard constraint here, so physical verification
   pass/fail is a real, informative check again.

Infeasibility under `system_backup="diesel"` is a rare, near-degenerate case
(kept, not removed, per this project's "never silently accept infeasibility"
principle): if diesel's fixed power cap cannot cover some residual demand
even at the min-LCOE candidate (e.g. an hour's load exceeding
`1.25 * peak_load_kw` due to floating-point/profile-generation edge cases),
that candidate is still returned, but flagged as not meeting the reliability
target. Under `system_backup="none"`, infeasibility is expected to be a real,
non-trivial fraction of scenarios (renewable/load timing mismatches a
battery alone cannot bridge) -- reported honestly, never replaced with a
maximum-battery guess.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import pandas as pd

from src.physics.battery import BatterySpec
from src.physics.diesel import DieselSpec, apply_diesel_backup, diesel_rated_power_kw
from src.physics.dispatch import resolve_initial_soc_bias
from src.physics.metrics import compute_candidate_metrics, compute_system_metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizationResult:
    """Result of an exhaustive battery-size search for one scenario."""

    feasible: bool
    optimal_n_modules: int | None
    optimal_capacity_kwh: float | None
    lpsp: float | None
    system_lcoe_eur_per_kwh: float | None
    renewable_share: float | None
    candidates: pd.DataFrame  # one row per tested module count
    search_runtime_seconds: float
    infeasibility_reason: str | None = None
    reliability_sensitivity: pd.DataFrame = field(default_factory=pd.DataFrame)


def _select_min_lcoe(candidates: pd.DataFrame) -> pd.Series:
    """Every candidate is economically valid once diesel backstops
    reliability -- select the one minimizing system LCOE, full stop."""
    return candidates.loc[candidates["system_lcoe_eur_per_kwh"].idxmin()]


def _select_smallest_feasible(candidates: pd.DataFrame, target_lpsp: float, lpsp_column: str = "lpsp") -> pd.Series | None:
    feasible = candidates[candidates[lpsp_column] <= target_lpsp]
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
        selected = _select_smallest_feasible(candidates, target_lpsp, lpsp_column="lpsp_after_diesel")
        rows.append(
            {
                "load_served_target": target,
                "feasible": selected is not None,
                "n_modules": selected["n_modules"] if selected is not None else None,
                "capacity_kwh": selected["nominal_battery_capacity_kwh"] if selected is not None else None,
                "achieved_lpsp": selected["lpsp_after_diesel"] if selected is not None else None,
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
    self_discharge_rate_per_hour: float,
    installed_cost_eur_per_kwh: float,
    economic_lifetime_years: int,
    project_lifetime_years: int,
    real_discount_rate: float,
    pv_capacity_kwp: float,
    pv_cfg: dict,
    wind_capacity_kw: float,
    wind_cfg: dict,
    diesel_cfg: dict,
    lpsp_target: float = 0.01,
    reliability_sensitivity_targets: list[float] | None = None,
    system_backup: str = "diesel",
) -> OptimizationResult:
    """Exhaustively search battery module counts 0..n_max and select the
    optimal candidate under `system_backup` (V2 Task 3 -- see module
    docstring for the two configurations and how the objective/feasibility
    semantics differ between them). See PROJECT_BRIEF.md §12 for the full
    per-candidate metric list.

    Args:
        system_backup: `"diesel"` (default, PROJECT_BRIEF.md Addendum 3) or
            `"none"` (off-grid, V2 Task 3).
        pv_cfg/wind_cfg/diesel_cfg: the `config/*.yaml` `pv:`/`wind:`/
            `diesel:` sub-dicts. Under `system_backup="diesel"`, diesel's
            rated power is computed here from
            `diesel_cfg["sizing_factor"] * peak_load_kw` (`peak_load_kw`
            read from `load_kw.max()`), not passed in directly, since it is
            always derived from the scenario's own peak load. Under
            `system_backup="none"`, `diesel_cfg` is still required (its
            fuel-curve/cost fields are used to construct a zero-rated-power
            `DieselSpec` so the rest of the pipeline needs no branching) but
            its `sizing_factor` is ignored -- rated power is fixed at 0.
    """
    if n_max < 0:
        raise ValueError("n_max must be >= 0.")
    if system_backup not in ("diesel", "none"):
        raise ValueError(f"system_backup must be 'diesel' or 'none', got {system_backup!r}.")

    peak_load_kw = float(load_kw.max())
    diesel_rated_power = (
        diesel_rated_power_kw(peak_load_kw, diesel_cfg["sizing_factor"]) if system_backup == "diesel" else 0.0
    )
    diesel = DieselSpec(
        rated_power_kw=diesel_rated_power,
        fuel_curve_intercept_l_per_kwh_rated=diesel_cfg["fuel_curve_intercept_l_per_kwh_rated"],
        fuel_curve_slope_l_per_kwh_output=diesel_cfg["fuel_curve_slope_l_per_kwh_output"],
        fuel_price_eur_per_l=diesel_cfg["fuel_price_eur_per_l"],
        installed_cost_eur_per_kw=diesel_cfg["installed_cost_eur_per_kw"],
        om_cost_fraction_per_year=diesel_cfg["om_cost_fraction_per_year"],
        economic_lifetime_years=diesel_cfg["economic_lifetime_years"],
        project_lifetime_years=diesel_cfg["project_lifetime_years"],
    )

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
            self_discharge_rate_per_hour=self_discharge_rate_per_hour,
            initial_soc_fraction=initial_soc_fraction,
            n_modules=n,
        )
        dispatch_result = resolve_initial_soc_bias(pv_generation_kw, wind_generation_kw, load_kw, battery)
        dispatch_result_with_diesel = apply_diesel_backup(dispatch_result, diesel)
        candidate_runtime = time.perf_counter() - candidate_start

        battery_metrics = compute_candidate_metrics(
            dispatch_result, battery, installed_cost_eur_per_kwh,
            economic_lifetime_years, project_lifetime_years, real_discount_rate,
            candidate_runtime,
        )
        system_metrics = compute_system_metrics(
            dispatch_result_with_diesel, battery_metrics["equivalent_annual_cost_eur"],
            pv_capacity_kwp, pv_cfg, wind_capacity_kw, wind_cfg, diesel,
            project_lifetime_years, real_discount_rate,
        )
        metrics = {**battery_metrics, **system_metrics}
        rows.append(metrics)
        logger.debug(
            "n_modules=%d LPSP=%.5f LCOE=%.4f renewable_share=%.3f runtime=%.4fs",
            n, metrics["lpsp"], metrics["system_lcoe_eur_per_kwh"], metrics["renewable_share"], candidate_runtime,
        )

    candidates = pd.DataFrame(rows)
    search_runtime = time.perf_counter() - search_start

    sensitivity = evaluate_reliability_sensitivity(
        candidates, reliability_sensitivity_targets or [0.990, 0.995, 0.999]
    )

    if system_backup == "diesel":
        selected = _select_min_lcoe(candidates)
        is_feasible = bool(selected["lpsp_after_diesel"] <= lpsp_target)
        infeasibility_reason = None
        if not is_feasible:
            infeasibility_reason = (
                "Diesel-capacity inadequacy: even the system-LCOE-minimizing candidate "
                f"({int(selected['n_modules'])} modules) has residual unserved energy "
                f"({selected['still_unserved_energy_kwh']:.1f} kWh/yr) exceeding the "
                f"{lpsp_target:.4f} LPSP target after diesel backup. This means at least one "
                "hour's demand exceeded diesel's rated power "
                f"({diesel.rated_power_kw:.1f} kW = {diesel_cfg['sizing_factor']} x peak load) -- "
                "an edge case the fixed sizing_factor convention does not fully cover for this "
                "scenario's load shape."
            )
            logger.warning("System-LCOE-optimal candidate does not meet the reliability target: %s", infeasibility_reason)

        return OptimizationResult(
            feasible=is_feasible,
            optimal_n_modules=int(selected["n_modules"]),
            optimal_capacity_kwh=float(selected["nominal_battery_capacity_kwh"]),
            lpsp=float(selected["lpsp_after_diesel"]),
            system_lcoe_eur_per_kwh=float(selected["system_lcoe_eur_per_kwh"]),
            renewable_share=float(selected["renewable_share"]),
            candidates=candidates,
            search_runtime_seconds=search_runtime,
            infeasibility_reason=infeasibility_reason,
            reliability_sensitivity=sensitivity,
        )

    # system_backup == "none" (off-grid, V2 Task 3): smallest battery meeting
    # the hard reliability constraint, genuinely infeasible (not silently
    # replaced by a maximum-battery guess) if no candidate within 0..n_max
    # achieves it -- diesel_rated_power_kw is 0 here, so "lpsp_after_diesel"
    # is exactly the renewables+battery-only LPSP.
    selected = _select_smallest_feasible(candidates, lpsp_target, lpsp_column="lpsp_after_diesel")
    if selected is None:
        return OptimizationResult(
            feasible=False,
            optimal_n_modules=None,
            optimal_capacity_kwh=None,
            lpsp=None,
            system_lcoe_eur_per_kwh=None,
            renewable_share=None,
            candidates=candidates,
            search_runtime_seconds=search_runtime,
            infeasibility_reason=(
                f"Battery-range inadequacy: no candidate within 0-{n_max} modules achieves "
                f"LPSP <= {lpsp_target:.4f} without diesel backup. This is a seasonal or "
                "sustained renewable/load timing mismatch a battery of any size within the "
                "tested range cannot bridge -- not a fixable sizing choice."
            ),
            reliability_sensitivity=sensitivity,
        )

    return OptimizationResult(
        feasible=True,
        optimal_n_modules=int(selected["n_modules"]),
        optimal_capacity_kwh=float(selected["nominal_battery_capacity_kwh"]),
        lpsp=float(selected["lpsp_after_diesel"]),
        system_lcoe_eur_per_kwh=float(selected["system_lcoe_eur_per_kwh"]),
        renewable_share=float(selected["renewable_share"]),
        candidates=candidates,
        search_runtime_seconds=search_runtime,
        infeasibility_reason=None,
        reliability_sensitivity=sensitivity,
    )
