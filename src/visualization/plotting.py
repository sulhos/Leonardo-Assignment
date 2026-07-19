"""Shared plotting utilities for mechanistic and AI-comparison figures
(PROJECT_BRIEF.md §27).

Stage 2 figures (weather/load/PV/wind, before any battery dispatch/optimization
exists) are implemented here. Figures that require dispatch or optimization
results (LPSP/curtailment/cost vs. battery capacity, SOC profiles, energy-flow
balance) are Stage 3; AI-comparison figures are Stage 6+.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DPI = 300


def save_figure(fig: plt.Figure, output_path: Path, dpi: int = DEFAULT_DPI) -> None:
    """Save a figure at publication resolution, creating parent dirs as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info("Saved figure to %s", output_path)


def plot_annual_profile(load_kw: pd.Series, pv_kw: pd.Series, wind_kw: pd.Series, title: str) -> plt.Figure:
    """Full-year hourly load, PV, and wind profiles on one axis."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(load_kw.index, load_kw.to_numpy(), label="Load", color="black", linewidth=0.6)
    ax.plot(pv_kw.index, pv_kw.to_numpy(), label="PV", color="orange", linewidth=0.5, alpha=0.8)
    ax.plot(wind_kw.index, wind_kw.to_numpy(), label="Wind", color="steelblue", linewidth=0.5, alpha=0.8)
    ax.set_ylabel("Power (kW)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_representative_week(load_kw: pd.Series, pv_kw: pd.Series, wind_kw: pd.Series, week_start: str, title: str) -> plt.Figure:
    """Hourly load/PV/wind for the 7 days starting at `week_start` (any
    timestamp parseable by pandas, matching the series' timezone)."""
    start = pd.Timestamp(week_start, tz=load_kw.index.tz)
    end = start + pd.Timedelta(days=7)
    mask = (load_kw.index >= start) & (load_kw.index < end)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(load_kw.index[mask], load_kw.to_numpy()[mask], label="Load", color="black", linewidth=1.2)
    ax.plot(pv_kw.index[mask], pv_kw.to_numpy()[mask], label="PV", color="orange", linewidth=1.2)
    ax.plot(wind_kw.index[mask], wind_kw.to_numpy()[mask], label="Wind", color="steelblue", linewidth=1.2)
    ax.set_ylabel("Power (kW)")
    ax.set_xlabel("Local time")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_load_duration_curve(load_kw: pd.Series) -> plt.Figure:
    """Load-duration curve: load sorted descending vs. duration fraction."""
    from src.physics.load_profile import load_duration_curve

    ldc = load_duration_curve(load_kw)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ldc.index * 100, ldc.to_numpy(), color="black", linewidth=1.5)
    ax.set_xlabel("Percentage of year (%)")
    ax.set_ylabel("Load (kW)")
    ax.set_title("Annual Load-Duration Curve")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    return fig


def plot_monthly_energy(load_kw: pd.Series, pv_kw: pd.Series, wind_kw: pd.Series) -> plt.Figure:
    """Monthly energy totals for load, PV, and wind (kWh, hourly kW summed per month)."""
    monthly_load = load_kw.resample("MS").sum()
    monthly_pv = pv_kw.resample("MS").sum()
    monthly_wind = wind_kw.resample("MS").sum()

    months = monthly_load.index.strftime("%b")
    x = np.arange(len(months))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, monthly_load.to_numpy(), width, label="Load", color="black")
    ax.bar(x, monthly_pv.to_numpy(), width, label="PV", color="orange")
    ax.bar(x + width, monthly_wind.to_numpy(), width, label="Wind", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.set_ylabel("Energy (kWh)")
    ax.set_title("Monthly Load and Renewable Production")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_wind_speed_distribution_and_power_curve(
    wind_speed_hub_mps: pd.Series,
    rated_power_kw: float,
    cut_in_mps: float,
    rated_mps: float,
    cut_out_mps: float,
) -> plt.Figure:
    """Side-by-side wind-speed histogram (at hub height) and the turbine's
    normalized power curve."""
    from src.physics.wind_model import turbine_power_curve

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.hist(wind_speed_hub_mps.to_numpy(), bins=40, color="steelblue", edgecolor="white")
    ax1.set_xlabel("Wind speed at hub height (m/s)")
    ax1.set_ylabel("Hours")
    ax1.set_title("Wind-Speed Distribution")

    speeds = pd.Series(np.linspace(0, cut_out_mps + 5, 200))
    power = turbine_power_curve(speeds, rated_power_kw, cut_in_mps, rated_mps, cut_out_mps)
    ax2.plot(speeds, power, color="black", linewidth=1.5)
    ax2.axvline(cut_in_mps, color="grey", linestyle="--", linewidth=0.8, label="Cut-in")
    ax2.axvline(rated_mps, color="grey", linestyle=":", linewidth=0.8, label="Rated")
    ax2.axvline(cut_out_mps, color="grey", linestyle="-.", linewidth=0.8, label="Cut-out")
    ax2.set_xlabel("Wind speed (m/s)")
    ax2.set_ylabel("Power output (kW)")
    ax2.set_title("Turbine Power Curve")
    ax2.legend()

    fig.tight_layout()
    return fig


def plot_metric_vs_battery_capacity(
    candidates: pd.DataFrame,
    metric_column: str,
    ylabel: str,
    target_line: float | None = None,
    target_label: str = "Target",
) -> plt.Figure:
    """Generic candidate-sweep plot: a metric column vs. nominal battery
    capacity, e.g. LPSP, curtailed_energy_kwh, or equivalent_annual_cost_eur."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        candidates["nominal_battery_capacity_kwh"], candidates[metric_column],
        marker="o", markersize=3, color="black", linewidth=1.2,
    )
    if target_line is not None:
        ax.axhline(target_line, color="firebrick", linestyle="--", linewidth=1, label=target_label)
        ax.legend()
    ax.set_xlabel("Battery capacity (kWh)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs. Battery Capacity")
    fig.tight_layout()
    return fig


def plot_soc_profile(dispatch_result: pd.DataFrame, week_start: str, battery_min_soc_kwh: float, battery_max_soc_kwh: float, title: str) -> plt.Figure:
    """Representative SOC trajectory for the 7 days starting at `week_start`."""
    start = pd.Timestamp(week_start, tz=dispatch_result.index.tz)
    end = start + pd.Timedelta(days=7)
    mask = (dispatch_result.index >= start) & (dispatch_result.index < end)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dispatch_result.index[mask], dispatch_result["soc_kwh"].to_numpy()[mask], color="black", linewidth=1.2)
    ax.axhline(battery_min_soc_kwh, color="firebrick", linestyle="--", linewidth=0.8, label="Min SOC")
    ax.axhline(battery_max_soc_kwh, color="seagreen", linestyle="--", linewidth=0.8, label="Max SOC")
    ax.set_ylabel("State of charge (kWh)")
    ax.set_xlabel("Local time")
    ax.set_title(title)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_energy_flow_balance(dispatch_result: pd.DataFrame) -> plt.Figure:
    """Annual energy-flow balance: renewable-side (direct supply, battery
    charge, curtailed) and load-side (direct supply, battery discharge,
    unserved) totals, as two stacked bars."""
    renewable_flows = {
        "Direct supply": dispatch_result["direct_supply_kwh"].sum(),
        "Battery charge": dispatch_result["battery_charge_kwh"].sum(),
        "Curtailed": dispatch_result["curtailed_kwh"].sum(),
    }
    load_flows = {
        "Direct supply": dispatch_result["direct_supply_kwh"].sum(),
        "Battery discharge": dispatch_result["battery_discharge_kwh"].sum(),
        "Unserved": dispatch_result["unserved_kwh"].sum(),
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {"Direct supply": "seagreen", "Battery charge": "steelblue", "Curtailed": "lightgrey",
              "Battery discharge": "orange", "Unserved": "firebrick"}

    bottom = 0.0
    for label, value in renewable_flows.items():
        ax.bar("Renewable\nproduction", value, bottom=bottom, color=colors[label], label=label)
        bottom += value

    bottom = 0.0
    for label, value in load_flows.items():
        already_labeled = label in renewable_flows
        ax.bar("Load", value, bottom=bottom, color=colors[label], label=None if already_labeled else label)
        bottom += value

    ax.set_ylabel("Annual energy (kWh)")
    ax.set_title("Annual Energy-Flow Balance")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0))
    fig.tight_layout()
    return fig


def plot_scenario_input_distributions(scenarios: pd.DataFrame, columns: list[str]) -> plt.Figure:
    """Histogram grid of scenario input distributions (PROJECT_BRIEF.md §27, figure 12)."""
    n = len(columns)
    ncols = 3
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, col in zip(axes, columns):
        ax.hist(scenarios[col].dropna(), bins=20, color="steelblue", edgecolor="white")
        ax.set_title(col)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Scenario Input Distributions")
    fig.tight_layout()
    return fig


def plot_optimal_capacity_distribution(scenarios: pd.DataFrame) -> plt.Figure:
    """Distribution of mechanistically optimal battery capacities among
    feasible scenarios (PROJECT_BRIEF.md §27, figure 13)."""
    feasible = scenarios.loc[scenarios["feasible"], "optimal_capacity_kwh"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(feasible, bins=20, color="seagreen", edgecolor="white")
    ax.set_xlabel("Optimal battery capacity (kWh)")
    ax.set_ylabel("Number of scenarios")
    ax.set_title(f"Distribution of Optimal Battery Capacity (n={len(feasible)} feasible scenarios)")
    fig.tight_layout()
    return fig


def plot_feasibility_counts(scenarios: pd.DataFrame) -> plt.Figure:
    """Feasible vs. infeasible scenario counts, infeasible split by reason
    category (PROJECT_BRIEF.md §27, figure 14)."""
    n_feasible = int(scenarios["feasible"].sum())
    infeasible = scenarios.loc[~scenarios["feasible"]]
    n_renewable_inadequate = int(infeasible["infeasibility_reason"].str.contains("Renewable-generation", na=False).sum())
    n_battery_range = int(infeasible["infeasibility_reason"].str.contains("Battery-range", na=False).sum())

    labels = ["Feasible", "Infeasible:\nrenewable inadequacy", "Infeasible:\nbattery-range"]
    counts = [n_feasible, n_renewable_inadequate, n_battery_range]
    colors = ["seagreen", "firebrick", "orange"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, counts, color=colors)
    for i, c in enumerate(counts):
        ax.text(i, c + 0.5, str(c), ha="center")
    ax.set_ylabel("Number of scenarios")
    ax.set_title("Feasible vs. Infeasible Scenarios")
    fig.tight_layout()
    return fig


def plot_feature_correlation_matrix(scenarios: pd.DataFrame, columns: list[str]) -> plt.Figure:
    """Correlation matrix heatmap for selected scenario columns (PROJECT_BRIEF.md §27, figure 15)."""
    corr = scenarios[columns].corr()
    fig, ax = plt.subplots(figsize=(0.7 * len(columns) + 3, 0.7 * len(columns) + 2))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels(columns)
    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    return fig


def plot_predicted_vs_reference(y_true: pd.Series, y_pred: pd.Series, model_name: str) -> plt.Figure:
    raise NotImplementedError("Implemented in Stage 6.")
