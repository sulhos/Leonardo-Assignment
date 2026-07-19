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


def plot_metric_vs_battery_capacity(candidates: pd.DataFrame, metric_column: str, ylabel: str) -> plt.Figure:
    raise NotImplementedError("Implemented in Stage 3.")


def plot_predicted_vs_reference(y_true: pd.Series, y_pred: pd.Series, model_name: str) -> plt.Figure:
    raise NotImplementedError("Implemented in Stage 6.")
