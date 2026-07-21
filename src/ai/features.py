"""Feature engineering for the ML dataset (PROJECT_BRIEF.md §15).

The scenario dataset (`data/scenarios/pilot_scenarios.csv`) stores scenario
inputs and mechanistic labels, but not the full 8,760-hour PV/wind/load
series per scenario (100 scenarios x 8,760 hours x 3 series would be a lot
of redundant storage). Instead, `build_feature_matrix` deterministically
regenerates each scenario's hourly series from its stored inputs (capacity,
annual load, peak load, random seed) -- identical to what
`src.scenarios.scenario_runner.run_scenario` did originally -- and computes
engineered features from them. This is fast (~100ms/scenario for
generation, no battery search needed) and exactly reproducible.

Never includes the mechanistically optimized battery capacity, reference
optimal module count, or reference optimal cost as an input feature (target
leakage) -- these are labels/reference outputs only, guarded by
`LEAKAGE_COLUMNS`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.physics.load_profile import generate_load_profile
from src.physics.pv_model import compute_pv_generation
from src.physics.wind_model import compute_wind_generation

logger = logging.getLogger(__name__)

# Columns that must never appear as ML input features (target leakage guard).
# Names match src.scenarios.scenario_runner.run_scenario's label output exactly.
LEAKAGE_COLUMNS = frozenset(
    {
        "optimal_capacity_kwh",
        "optimal_n_modules",
        "reference_lpsp",
        "reference_annualized_cost",
        "reference_system_lcoe_eur_per_kwh",  # added for the diesel/system-LCOE pivot, PROJECT_BRIEF.md Addendum 3
        "reference_renewable_share",
    }
)

DAYTIME_HOURS = range(6, 19)  # 06:00-18:59 local, used for daytime/nighttime load share


def compute_load_features(load_kw: pd.Series) -> dict:
    annual_load_kwh = load_kw.sum()
    peak_load_kw = load_kw.max()
    mean_load_kw = load_kw.mean()
    monthly = load_kw.resample("MS").sum()
    is_daytime = load_kw.index.hour.isin(DAYTIME_HOURS)

    return {
        "annual_load_kwh": annual_load_kwh,
        "peak_load_kw": peak_load_kw,
        "mean_load_kw": mean_load_kw,
        "std_load_kw": load_kw.std(),
        "load_factor": mean_load_kw / peak_load_kw if peak_load_kw > 0 else 0.0,
        "peak_to_average_ratio": peak_load_kw / mean_load_kw if mean_load_kw > 0 else 0.0,
        "daytime_load_share": load_kw[is_daytime].sum() / annual_load_kwh if annual_load_kwh > 0 else 0.0,
        "nighttime_load_share": load_kw[~is_daytime].sum() / annual_load_kwh if annual_load_kwh > 0 else 0.0,
        "seasonal_load_variability": monthly.std() / monthly.mean() if monthly.mean() > 0 else 0.0,
        "monthly_max_load_kwh": monthly.max(),
        "monthly_min_load_kwh": monthly.min(),
    }


def compute_generation_features(pv_kw: pd.Series, wind_kw: pd.Series, load_kw: pd.Series, pv_capacity_kwp: float, wind_capacity_kw: float) -> dict:
    annual_pv_kwh = pv_kw.sum()
    annual_wind_kwh = wind_kw.sum()
    total_renewable_kwh = annual_pv_kwh + annual_wind_kwh
    annual_load_kwh = load_kw.sum()
    renewable = pv_kw + wind_kw
    monthly_renewable = renewable.resample("MS").sum()
    daily_renewable = renewable.resample("D").sum()

    return {
        "pv_capacity_kwp": pv_capacity_kwp,
        "wind_capacity_kw": wind_capacity_kw,
        "annual_pv_kwh": annual_pv_kwh,
        "annual_wind_kwh": annual_wind_kwh,
        "total_renewable_kwh": total_renewable_kwh,
        "pv_capacity_factor": annual_pv_kwh / (pv_capacity_kwp * len(pv_kw)) if pv_capacity_kwp > 0 else 0.0,
        "wind_capacity_factor": annual_wind_kwh / (wind_capacity_kw * len(wind_kw)) if wind_capacity_kw > 0 else 0.0,
        "renewable_to_load_ratio": total_renewable_kwh / annual_load_kwh if annual_load_kwh > 0 else 0.0,
        "monthly_renewable_variability": monthly_renewable.std() / monthly_renewable.mean() if monthly_renewable.mean() > 0 else 0.0,
        "daily_renewable_variability": daily_renewable.std() / daily_renewable.mean() if daily_renewable.mean() > 0 else 0.0,
    }


def compute_net_load_features(load_kw: pd.Series, pv_kw: pd.Series, wind_kw: pd.Series) -> dict:
    net_load = load_kw - pv_kw - wind_kw  # positive = deficit hour, negative = surplus hour
    deficit = net_load.clip(lower=0.0)
    surplus = (-net_load).clip(lower=0.0)
    deficit_hours_mask = net_load > 0

    # Longest consecutive run of deficit hours, and the largest single
    # consecutive-deficit *energy* event (sum of net_load over that run).
    run_id = (deficit_hours_mask != deficit_hours_mask.shift(fill_value=False)).cumsum()
    deficit_runs = net_load[deficit_hours_mask].groupby(run_id[deficit_hours_mask])
    max_consecutive_deficit_hours = int(deficit_runs.size().max()) if deficit_hours_mask.any() else 0
    max_cumulative_deficit_event_kwh = float(deficit_runs.sum().max()) if deficit_hours_mask.any() else 0.0

    monthly_net = pd.DataFrame({"load": load_kw, "renewable": pv_kw + wind_kw}).resample("MS").sum()
    monthly_ratio = monthly_net["renewable"] / monthly_net["load"]

    return {
        "max_positive_net_load_kw": float(deficit.max()),
        "mean_positive_net_load_kw": float(deficit[deficit > 0].mean()) if (deficit > 0).any() else 0.0,
        "std_net_load_kw": float(net_load.std()),
        "total_deficit_energy_kwh": float(deficit.sum()),
        "total_surplus_energy_kwh": float(surplus.sum()),
        "max_consecutive_deficit_hours": max_consecutive_deficit_hours,
        "max_cumulative_deficit_event_kwh": max_cumulative_deficit_event_kwh,
        "n_deficit_hours": int(deficit_hours_mask.sum()),
        "n_surplus_hours": int((net_load < 0).sum()),
        "net_load_q10_kw": float(net_load.quantile(0.10)),
        "net_load_q50_kw": float(net_load.quantile(0.50)),
        "net_load_q90_kw": float(net_load.quantile(0.90)),
        "worst_monthly_renewable_to_load_ratio": float(monthly_ratio.min()),
    }


def build_feature_matrix(scenarios: pd.DataFrame, weather: pd.DataFrame, site_config: dict) -> pd.DataFrame:
    """Regenerate hourly series and assemble the full feature matrix for all
    scenarios, guaranteed free of LEAKAGE_COLUMNS.

    Args:
        scenarios: The raw scenario inputs+labels dataset
            (`data/scenarios/pilot_scenarios.csv`).
        weather: Local-year hourly weather for the scenarios' shared
            location/year (all pilot scenarios share one location/year).
        site_config: Fixed site configuration (PV/wind siting assumptions).

    Returns:
        One row per scenario: engineered features plus `scenario_id` (for
        joining back to labels) -- never any `LEAKAGE_COLUMNS`.
    """
    pv_cfg = site_config["pv"]
    wind_cfg = site_config["wind"]
    rows = []

    for _, scenario in scenarios.iterrows():
        load = generate_load_profile(
            annual_consumption_kwh=scenario["annual_load_kwh"],
            peak_load_kw=scenario["peak_load_kw"],
            year=scenario["weather_year"],
            timezone=scenario["timezone"],
            random_seed=scenario["random_seed"],
        )
        pv = compute_pv_generation(
            weather=weather, capacity_kwp=scenario["pv_capacity_kwp"],
            latitude=site_config["site"]["latitude"], longitude=site_config["site"]["longitude"],
            tilt_deg=pv_cfg["tilt_deg"], azimuth_deg=pv_cfg["azimuth_deg"],
            system_losses_fraction=pv_cfg["system_losses_fraction"], inverter_efficiency=pv_cfg["inverter_efficiency"],
        )
        wind = compute_wind_generation(
            weather=weather, rated_power_kw=scenario["wind_capacity_kw"],
            reference_height_m=wind_cfg["weather_reference_height_m"], hub_height_m=wind_cfg["hub_height_m"],
            shear_exponent=wind_cfg["wind_shear_exponent"], cut_in_mps=wind_cfg["cut_in_mps"],
            rated_mps=wind_cfg["rated_mps"], cut_out_mps=wind_cfg["cut_out_mps"],
        )

        features = {"scenario_id": scenario["scenario_id"]}
        features.update(compute_load_features(load))
        features.update(compute_generation_features(pv, wind, load, scenario["pv_capacity_kwp"], scenario["wind_capacity_kw"]))
        features.update(compute_net_load_features(load, pv, wind))
        features["reliability_target_load_served"] = scenario["reliability_target_load_served"]
        features["round_trip_efficiency"] = scenario["round_trip_efficiency"]
        features["usable_soc_window_fraction"] = scenario["usable_soc_window_fraction"]
        rows.append(features)

    feature_matrix = pd.DataFrame(rows)
    leaked = LEAKAGE_COLUMNS & set(feature_matrix.columns)
    if leaked:
        raise RuntimeError(f"Leakage columns present in feature matrix: {leaked}")

    logger.info("Built feature matrix: %d scenarios, %d features.", len(feature_matrix), feature_matrix.shape[1] - 1)
    return feature_matrix
