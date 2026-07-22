"""Train the neural network on the diesel-hybrid dataset (Part C): the
current ML study, replacing the completed off-grid one. Same architecture/
config as the off-grid run (train_nn.py) -- only the dataset and the
verification's reframed diagnostics differ.

Saves: models/neural_network_diesel_full/{best_model.keras,best_model.weights.h5},
models/preprocessing_diesel_full/{scaler.pkl,feature_columns.pkl},
outputs/tables/{accuracy,operational,physical_verification,
physical_verification_summary}_neural_network_diesel_full.csv,
dataset_split_assignments_diesel_full.csv.
"""
from __future__ import annotations

import json

import pandas as pd
from common import REPO_ROOT, SITE, TARGET_COL, load_merged_dataset, split_dataset, train_and_evaluate

from src.ai.evaluation import compute_operational_metrics
from src.ai.physical_verification import summarize_verification, verify_predictions
from src.ai.preprocessing import save_preprocessing_objects
from src.config import load_site_config, load_yaml_config
from src.data.weather import get_processed_weather

DATASET_NAME = "diesel_full"


def run(random_seed: int = 42, save_artifacts: bool = True) -> dict:
    cfg = load_site_config(SITE)
    ml_cfg = load_yaml_config("ml_training")

    merged, feature_cols = load_merged_dataset(DATASET_NAME)
    splits = split_dataset(merged, random_seed, ml_cfg)

    result = train_and_evaluate(
        merged, feature_cols, splits, random_seed, ml_cfg,
        checkpoint_dir=REPO_ROOT / f"models/neural_network_{DATASET_NAME}",
    )
    test, test_pred, model, scaler, history = (
        result["test"], result["test_pred"], result["model"], result["scaler"], result["history"]
    )
    accuracy = result["accuracy"]
    operational = compute_operational_metrics(test[TARGET_COL], test_pred)
    print("accuracy:", accuracy)
    print("operational:", operational)

    weather = get_processed_weather(SITE, cfg["simulation"]["year"])
    verification_scenarios = test.copy()
    verification_scenarios["annual_load_kwh"] = verification_scenarios["raw_annual_load_kwh"]
    verification_scenarios["peak_load_kw"] = verification_scenarios["raw_peak_load_kw"]
    verification_scenarios["location"] = SITE
    verification_scenarios["timezone"] = cfg["site"]["timezone"]
    verification_scenarios["weather_year"] = cfg["simulation"]["year"]
    verification_scenarios["random_seed"] = verification_scenarios["scenario_id"].apply(
        lambda sid: 42 * 100_003 + int(sid)
    )
    verification = verify_predictions(
        verification_scenarios, pd.Series(test_pred, index=test.index), weather, cfg,
        module_capacity_kwh=cfg["battery"]["module_capacity_kwh"],
    )
    verification_summary = summarize_verification(verification)
    print("verification summary:", verification_summary)

    if save_artifacts:
        save_preprocessing_objects(
            {"scaler": scaler, "feature_columns": list(feature_cols)},
            REPO_ROOT / f"models/preprocessing_{DATASET_NAME}",
        )
        out_dir = REPO_ROOT / "outputs/tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([accuracy]).to_csv(out_dir / f"accuracy_metrics_neural_network_{DATASET_NAME}.csv", index=False)
        pd.DataFrame([operational]).to_csv(out_dir / f"operational_metrics_neural_network_{DATASET_NAME}.csv", index=False)
        verification.to_csv(out_dir / f"physical_verification_neural_network_{DATASET_NAME}.csv", index=False)
        pd.DataFrame([verification_summary]).to_csv(out_dir / f"physical_verification_summary_neural_network_{DATASET_NAME}.csv", index=False)

        split_assignments = pd.concat([
            pd.DataFrame({"scenario_id": merged.loc[idx, "scenario_id"], "split": name})
            for name, idx in splits.items()
        ])
        split_assignments.to_csv(out_dir / f"dataset_split_assignments_{DATASET_NAME}.csv", index=False)

        with open(out_dir / f"neural_network_training_history_{DATASET_NAME}.json", "w") as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    return {"accuracy": accuracy, "operational": operational, "verification_summary": verification_summary}


if __name__ == "__main__":
    run()
