"""Shared plotting utilities for mechanistic and AI-comparison figures
(PROJECT_BRIEF.md §27).

Responsibilities (implemented starting Stage 3, extended through Stage 9):

- Mechanistic figures: annual/weekly/seasonal load-PV-wind profiles,
  load-duration curve, wind-speed distribution and turbine power curve,
  LPSP/curtailment/cost vs. battery capacity, SOC profiles, annual
  energy-flow balance.
- Dataset figures: scenario-input distributions, optimal-capacity
  distribution, feasible/infeasible counts, feature correlation matrix.
- AI-comparison figures: training/validation loss, predicted vs. reference
  capacity, residuals, error vs. battery size / renewable-to-load ratio,
  cross-model MAE/RMSE/R2, reliability pass rate, under/over-prediction
  rates, runtime comparison, accuracy-runtime tradeoff, cost-penalty,
  reliability-target sensitivity, geographic-transfer results (if
  implemented).
- All figures use readable labels/units and are saved at publication
  resolution under outputs/figures/.

This module intentionally contains no implementation yet -- it is a Stage 1
placeholder. Functions raise NotImplementedError until Stage 3+.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DPI = 300


def save_figure(fig: plt.Figure, output_path: Path, dpi: int = DEFAULT_DPI) -> None:
    """Save a figure at publication resolution, creating parent dirs as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info("Saved figure to %s", output_path)


def plot_representative_week(load_kw: pd.Series, pv_kw: pd.Series, wind_kw: pd.Series, week_start: str, title: str) -> plt.Figure:
    raise NotImplementedError("Implemented in Stage 3.")


def plot_load_duration_curve(load_kw: pd.Series) -> plt.Figure:
    raise NotImplementedError("Implemented in Stage 3.")


def plot_metric_vs_battery_capacity(candidates: pd.DataFrame, metric_column: str, ylabel: str) -> plt.Figure:
    raise NotImplementedError("Implemented in Stage 3.")


def plot_predicted_vs_reference(y_true: pd.Series, y_pred: pd.Series, model_name: str) -> plt.Figure:
    raise NotImplementedError("Implemented in Stage 6.")
