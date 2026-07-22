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


def plot_renewable_share_vs_lcoe(candidates: pd.DataFrame, title: str = "Renewable Share vs. System LCOE") -> plt.Figure:
    """Renewable share (%) vs. system LCOE (EUR/kWh), swept across battery
    candidates (PROJECT_BRIEF.md Addendum 3). Reproduces the shape of the
    course lecture's OptiCE "Typical results (1)" chart directly from this
    project's own exhaustive battery search, using one fixed PV/wind
    combination swept over battery capacity -- not OptiCE's own joint
    PV/wind/battery genetic-algorithm sweep, a documented scope difference
    (see `reports/technical_report.md`'s Related Work / Limitations)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sorted_candidates = candidates.sort_values("nominal_battery_capacity_kwh")
    ax.plot(
        sorted_candidates["system_lcoe_eur_per_kwh"], sorted_candidates["renewable_share"] * 100,
        marker="o", markersize=4, color="steelblue", linewidth=1.2,
    )
    ax.set_xlabel("System LCOE (EUR/kWh)")
    ax.set_ylabel("Renewable share (%)")
    ax.set_ylim(-2, 105)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_energy_flow_balance_with_diesel(dispatch_result_with_diesel: pd.DataFrame) -> plt.Figure:
    """Annual energy-flow balance including diesel backup (PROJECT_BRIEF.md
    Addendum 3): renewable-side (direct supply, battery charge, curtailed)
    and load-side (direct supply, battery discharge, diesel, still-unserved)
    totals, as two stacked bars. `dispatch_result_with_diesel` must already
    have been through `src.physics.diesel.apply_diesel_backup`."""
    renewable_flows = {
        "Direct supply": dispatch_result_with_diesel["direct_supply_kwh"].sum(),
        "Battery charge": dispatch_result_with_diesel["battery_charge_kwh"].sum(),
        "Curtailed": dispatch_result_with_diesel["curtailed_kwh"].sum(),
    }
    load_flows = {
        "Direct supply": dispatch_result_with_diesel["direct_supply_kwh"].sum(),
        "Battery discharge": dispatch_result_with_diesel["battery_discharge_kwh"].sum(),
        "Diesel": dispatch_result_with_diesel["diesel_output_kwh"].sum(),
        "Still unserved": dispatch_result_with_diesel["still_unserved_kwh"].sum(),
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {"Direct supply": "seagreen", "Battery charge": "steelblue", "Curtailed": "lightgrey",
              "Battery discharge": "orange", "Diesel": "dimgrey", "Still unserved": "firebrick"}

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
    ax.set_title("Annual Energy-Flow Balance (with Diesel Backup)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.0))
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
    reasons = infeasible["infeasibility_reason"].astype("string")
    n_renewable_inadequate = int(reasons.str.contains("Renewable-generation", na=False).sum())
    n_battery_range = int(reasons.str.contains("Battery-range", na=False).sum())

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


def plot_training_history(history: dict, title: str = "Neural Network Training") -> plt.Figure:
    """Training and validation loss curves (PROJECT_BRIEF.md §27, figure 16)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["loss"], label="Training loss", color="steelblue")
    ax.plot(history["val_loss"], label="Validation loss", color="firebrick")
    best_epoch = int(np.argmin(history["val_loss"]))
    ax.axvline(best_epoch, color="grey", linestyle="--", linewidth=0.8, label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_predicted_vs_reference(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> plt.Figure:
    """Scatter of predicted vs. mechanistic-reference battery capacity, with
    a y=x reference line (PROJECT_BRIEF.md §27, figure 17). Generic across
    any model (baselines or the neural network) -- not NN-specific."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    lims = [0, max(y_true_arr.max(), y_pred_arr.max()) * 1.05]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(lims, lims, color="grey", linestyle="--", linewidth=1, label="y = x")
    ax.scatter(y_true_arr, y_pred_arr, color="steelblue", edgecolor="white", s=50)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Mechanistic reference capacity (kWh)")
    ax.set_ylabel("Predicted capacity (kWh)")
    ax.set_title(f"Predicted vs. Reference: {model_name}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residual_distribution(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> plt.Figure:
    """Histogram of prediction residuals (predicted - reference), figure 18."""
    residuals = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=15, color="steelblue", edgecolor="white")
    ax.axvline(0, color="firebrick", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual: predicted - reference (kWh)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Distribution: {model_name}")
    fig.tight_layout()
    return fig


def plot_cross_model_metric_comparison(metrics_by_model: dict[str, dict], metric_keys: list[str]) -> plt.Figure:
    """Grouped bar chart comparing several accuracy metrics across models
    (PROJECT_BRIEF.md §27, figure 21)."""
    model_names = list(metrics_by_model.keys())
    x = np.arange(len(metric_keys))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(2 + 2 * len(metric_keys), 5))
    for i, model_name in enumerate(model_names):
        values = [metrics_by_model[model_name][k] for k in metric_keys]
        ax.bar(x + i * width, values, width, label=model_name)
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(metric_keys)
    ax.set_ylabel("Value")
    ax.set_title("Accuracy Metric Comparison Across Models")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_over_under_prediction_rates(operational_metrics_by_model: dict[str, dict]) -> plt.Figure:
    """Under-/over-prediction rate comparison across models (PROJECT_BRIEF.md
    §27, figure 23)."""
    model_names = list(operational_metrics_by_model.keys())
    under = [operational_metrics_by_model[m]["underprediction_rate"] for m in model_names]
    over = [operational_metrics_by_model[m]["overprediction_rate"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(2 + 1.5 * len(model_names), 5))
    ax.bar(x - width / 2, under, width, label="Underprediction rate", color="firebrick")
    ax.bar(x + width / 2, over, width, label="Overprediction rate", color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Rate")
    ax.set_title("Under-/Over-prediction Rate by Model")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_reliability_pass_rate_by_model(summary_by_model: dict[str, dict]) -> plt.Figure:
    """Physical-verification reliability pass rate across models
    (PROJECT_BRIEF.md §27, figure 22) -- the key Stage 7 result."""
    model_names = list(summary_by_model.keys())
    pass_rates = [summary_by_model[m]["pct_satisfying_reliability"] for m in model_names]

    fig, ax = plt.subplots(figsize=(2 + 1.5 * len(model_names), 5))
    bars = ax.bar(model_names, pass_rates, color="seagreen")
    for bar, rate in zip(bars, pass_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.02, f"{rate:.0%}", ha="center")
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Reliability pass rate (verified)")
    ax.set_title("Physical Verification: Reliability Pass Rate by Model")
    fig.tight_layout()
    return fig


def plot_runtime_comparison(runtime_by_stage: dict[str, float]) -> plt.Figure:
    """Runtime comparison across mechanistic search, ML training, and ML
    inference stages (PROJECT_BRIEF.md §27, figure 24). Log scale, since
    these span many orders of magnitude."""
    labels = list(runtime_by_stage.keys())
    values = list(runtime_by_stage.values())

    fig, ax = plt.subplots(figsize=(2 + 1.2 * len(labels), 5))
    ax.bar(labels, values, color="steelblue")
    ax.set_yscale("log")
    ax.set_ylabel("Seconds (log scale)")
    ax.set_title("Runtime Comparison")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def plot_accuracy_runtime_tradeoff(accuracy_by_model: dict[str, dict], inference_seconds_by_model: dict[str, float]) -> plt.Figure:
    """Accuracy (MAE) vs. inference time per model (PROJECT_BRIEF.md §27,
    figure 25)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in accuracy_by_model:
        ax.scatter(inference_seconds_by_model[name], accuracy_by_model[name]["mae"], s=80, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Inference time per prediction (s, log scale)")
    ax.set_ylabel("MAE (kWh)")
    ax.set_title("Accuracy vs. Inference-Time Tradeoff")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_cost_penalty_from_oversizing(summary_by_model: dict[str, dict]) -> plt.Figure:
    """Additional annualized cost caused by AI oversizing, by model
    (PROJECT_BRIEF.md §27, figure 26)."""
    model_names = list(summary_by_model.keys())
    penalties = [summary_by_model[m]["additional_cost_from_oversizing_eur"] for m in model_names]

    fig, ax = plt.subplots(figsize=(2 + 1.5 * len(model_names), 5))
    ax.bar(model_names, penalties, color="orange")
    ax.set_ylabel("Additional annualized cost from oversizing (EUR)")
    ax.set_title("Cost Penalty from AI Oversizing")
    fig.tight_layout()
    return fig


def plot_extra_lcoe_by_model(summary_by_model: dict[str, dict]) -> plt.Figure:
    """Mean extra system LCOE incurred from installing each model's predicted
    battery capacity instead of the true LCOE-minimizing one (PROJECT_BRIEF.md
    Addendum 3's headline Stage 7 result, figure 29) -- the diesel-backed
    reframing of "how costly is trusting the AI's answer" as a continuous
    economic measure rather than a binary reliability pass/fail."""
    model_names = list(summary_by_model.keys())
    extra_lcoe = [summary_by_model[m]["mean_extra_system_lcoe_eur_per_kwh"] for m in model_names]

    fig, ax = plt.subplots(figsize=(2 + 1.5 * len(model_names), 5))
    bars = ax.bar(model_names, extra_lcoe, color="firebrick")
    for bar, value in zip(bars, extra_lcoe):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom")
    ax.set_ylabel("Mean extra system LCOE vs. optimal (EUR/kWh)")
    ax.set_title("Physical Verification: Extra System LCOE from Trusting the AI")
    fig.tight_layout()
    return fig


def plot_system_lcoe_curve_with_ai_prediction(
    candidates: pd.DataFrame,
    reference_capacity_kwh: float,
    reference_lcoe_eur_per_kwh: float,
    predicted_capacity_kwh: float,
    predicted_lcoe_eur_per_kwh: float,
    model_name: str,
) -> plt.Figure:
    """Same System LCOE vs. battery capacity curve as figure 27
    (`plot_metric_vs_battery_capacity`), for one specific scenario's full
    exhaustive candidate sweep, with the mechanistic optimum and the AI
    model's predicted capacity marked directly on it -- the AI-comparison
    counterpart to figure 27, which shows the mechanistic curve alone.

    Unlike `plot_capacity_comparison_line`/`plot_renewable_share_comparison_line`
    (which compare many scenarios' single reference/predicted values against
    each other), this reuses figure 27's own per-scenario curve shape so the
    AI's answer can be read directly against the U-shaped cost trade-off it
    is implicitly trying to approximate, for one scenario at a time -- the
    neural network itself has no such curve (it predicts one capacity
    number, not a cost function), so its "curve" here is a single marked
    point on the mechanistic curve."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        candidates["nominal_battery_capacity_kwh"], candidates["system_lcoe_eur_per_kwh"],
        marker="o", markersize=3, color="black", linewidth=1.2, label="Mechanistic (all candidates)",
    )
    ax.scatter(
        [reference_capacity_kwh], [reference_lcoe_eur_per_kwh],
        color="seagreen", marker="*", s=220, zorder=5, label="Mechanistic optimum",
    )
    ax.scatter(
        [predicted_capacity_kwh], [predicted_lcoe_eur_per_kwh],
        color="firebrick", marker="*", s=220, zorder=5, label=f"{model_name} prediction",
    )
    ax.set_xlabel("Battery capacity (kWh)")
    ax.set_ylabel("System LCOE (EUR/kWh)")
    ax.set_title(f"System LCOE vs. Battery Capacity: Mechanistic vs. {model_name}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_renewable_share_vs_lcoe_with_ai_prediction(
    candidates: pd.DataFrame,
    reference_lcoe_eur_per_kwh: float,
    reference_renewable_share: float,
    predicted_lcoe_eur_per_kwh: float,
    predicted_renewable_share: float,
    model_name: str,
) -> plt.Figure:
    """Same Renewable Share vs. System LCOE curve as figure 28
    (`plot_renewable_share_vs_lcoe`), for one specific scenario's full
    exhaustive candidate sweep, with the mechanistic optimum and the AI
    model's predicted point both marked directly on it -- the
    renewable-share counterpart to
    `plot_system_lcoe_curve_with_ai_prediction`, which does the same for
    the battery-capacity curve (figure 27)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sorted_candidates = candidates.sort_values("nominal_battery_capacity_kwh")
    ax.plot(
        sorted_candidates["system_lcoe_eur_per_kwh"], sorted_candidates["renewable_share"] * 100,
        marker="o", markersize=4, color="steelblue", linewidth=1.2, label="Mechanistic (all candidates)",
    )
    ax.scatter(
        [reference_lcoe_eur_per_kwh], [reference_renewable_share * 100],
        color="seagreen", marker="*", s=220, zorder=5, label="Mechanistic optimum",
    )
    ax.scatter(
        [predicted_lcoe_eur_per_kwh], [predicted_renewable_share * 100],
        color="firebrick", marker="*", s=220, zorder=5, label=f"{model_name} prediction",
    )
    ax.set_xlabel("System LCOE (EUR/kWh)")
    ax.set_ylabel("Renewable share (%)")
    ax.set_ylim(-2, 105)
    ax.set_title(f"Renewable Share vs. System LCOE: Mechanistic vs. {model_name}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_capacity_comparison_line(verification: pd.DataFrame, model_name: str) -> plt.Figure:
    """Line comparison of mechanistic-reference vs. AI-predicted battery
    capacity, one point per test scenario, sorted by the mechanistic
    reference value (PROJECT_BRIEF.md Addendum 3). Complements the single-
    number accuracy metrics (MAE, R^2) with a direct scenario-by-scenario
    view of how closely the AI's installed capacity tracks the mechanistic
    optimum across the full range of test scenarios, rather than the
    mechanistic-only search curve (figure 27)."""
    ordered = verification.sort_values("reference_capacity_kwh").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ordered.index, ordered["reference_capacity_kwh"], color="black", linewidth=1.5,
            label="Mechanistic (reference)")
    ax.plot(ordered.index, ordered["installed_capacity_kwh"], color="steelblue", linewidth=0.8, alpha=0.8,
            label=f"{model_name} (predicted)")
    ax.set_xlabel("Test scenarios, sorted by mechanistic reference capacity")
    ax.set_ylabel("Battery capacity (kWh)")
    ax.set_title(f"Battery Capacity: Mechanistic vs. {model_name}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_renewable_share_comparison_line(verification: pd.DataFrame, model_name: str) -> plt.Figure:
    """Line comparison of mechanistic-reference vs. AI-predicted renewable
    share, one point per test scenario, sorted by the mechanistic reference
    value -- the AI-side counterpart to figure 28 (which is mechanistic-only),
    computed by re-running dispatch at each model's own predicted battery
    capacity (`src.ai.physical_verification.verify_predictions`)."""
    ordered = verification.sort_values("reference_renewable_share").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ordered.index, 100 * ordered["reference_renewable_share"], color="black", linewidth=1.5,
            label="Mechanistic (reference)")
    ax.plot(ordered.index, 100 * ordered["verified_renewable_share"], color="darkorange", linewidth=0.8, alpha=0.8,
            label=f"{model_name} (predicted)")
    ax.set_xlabel("Test scenarios, sorted by mechanistic reference renewable share")
    ax.set_ylabel("Renewable share (%)")
    ax.set_title(f"Renewable Share: Mechanistic vs. {model_name}")
    ax.legend()
    fig.tight_layout()
    return fig
