"""V2 Part G: generate the diesel-hybrid figure set for the rewritten report.

Reuses figures 01/02 (baseline annual profile, energy-flow balance) and
11-14 (diesel engagement week, energy-flow-with-diesel, renewable-share-
vs-LCOE, battery-vs-LCOE) from the earlier diesel-hybrid demonstration
work unchanged -- same baseline case, nothing to regenerate. New figures
cover the retrained NN (Part C), the simple baseline comparison (Part D),
and the cost-sensitivity target shift (Part F).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.ai.evaluation import compute_accuracy_metrics
from src.visualization.plotting import (
    plot_cross_validation_error_bars,
    plot_lean_on_diesel_diagnostic,
    plot_learning_curve,
    plot_optimal_capacity_distribution,
    plot_optimal_modules_by_cost,
    plot_permutation_importance,
    plot_predicted_vs_reference_with_r2,
    plot_training_history,
    save_figure,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "diesel_full"


def run() -> None:
    out_dir = REPO_ROOT / "outputs/figures"

    scenarios = pd.read_csv(REPO_ROOT / f"data/scenarios/{DATASET_NAME}_scenarios.csv")
    save_figure(plot_optimal_capacity_distribution(scenarios), out_dir / "20_optimal_capacity_distribution_diesel.png")

    with open(REPO_ROOT / f"outputs/tables/neural_network_training_history_{DATASET_NAME}.json") as f:
        history = json.load(f)
    save_figure(plot_training_history(history, title="Neural Network Training (Diesel-Hybrid)"), out_dir / "21_training_history_diesel.png")

    verification = pd.read_csv(REPO_ROOT / f"outputs/tables/physical_verification_neural_network_{DATASET_NAME}.csv")
    accuracy = compute_accuracy_metrics(verification["reference_capacity_kwh"], verification["predicted_capacity_kwh"])
    save_figure(
        plot_predicted_vs_reference_with_r2(
            verification["reference_capacity_kwh"], verification["predicted_capacity_kwh"], accuracy["r2"], "Neural Network (Diesel-Hybrid)",
        ),
        out_dir / "22_predicted_vs_reference_diesel.png",
    )

    cv_runs = pd.read_csv(REPO_ROOT / f"outputs/tables/cross_validation_runs_neural_network_{DATASET_NAME}.csv")
    save_figure(plot_cross_validation_error_bars(cv_runs), out_dir / "23_cross_validation_diesel.png")

    learning_curve = pd.read_csv(REPO_ROOT / f"outputs/tables/learning_curve_neural_network_{DATASET_NAME}.csv")
    save_figure(plot_learning_curve(learning_curve), out_dir / "24_learning_curve_diesel.png")

    importance = pd.read_csv(REPO_ROOT / f"outputs/tables/permutation_importance_neural_network_{DATASET_NAME}.csv")
    save_figure(plot_permutation_importance(importance), out_dir / "25_permutation_importance_diesel.png")

    save_figure(plot_lean_on_diesel_diagnostic(verification), out_dir / "26_lean_on_diesel_diagnostic.png")

    cost_sensitivity = pd.read_csv(REPO_ROOT / "outputs/tables/cost_sensitivity_diesel_full.csv")
    save_figure(plot_optimal_modules_by_cost(cost_sensitivity), out_dir / "27_optimal_modules_by_cost.png")

    print(f"wrote figures to {out_dir}")


if __name__ == "__main__":
    run()
