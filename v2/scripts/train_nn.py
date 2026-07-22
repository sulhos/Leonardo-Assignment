"""Train the neural network on the full off-grid dataset, evaluate accuracy +
operational metrics, and run mandatory physical verification against the
mechanistic reference (V2 Task 5-8 prerequisite; mirrors V1's Stage 6-7).

Saves: models/neural_network_full/{best_model.keras,best_model.weights.h5},
models/preprocessing_full/{scaler.pkl,feature_columns.pkl},
outputs/tables/{accuracy_metrics_full.csv,operational_metrics_full.csv,
physical_verification_full.csv,physical_verification_summary_full.csv,
dataset_split_assignments_full.csv}.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.ai.evaluation import compute_accuracy_metrics, compute_operational_metrics
from src.ai.features import LEAKAGE_COLUMNS
from src.ai.neural_network import build_model, load_trained_model, predict, set_global_seed, train_model
from src.ai.physical_verification import summarize_verification, verify_predictions
from src.ai.preprocessing import fit_scaler, save_preprocessing_objects
from src.ai.splitting import split_experiment_a
from src.config import load_site_config, load_yaml_config
from src.data.weather import get_processed_weather

REPO_ROOT = Path(__file__).resolve().parents[1]
SITE = "jinan"


def load_and_split(random_seed: int, ml_cfg: dict) -> tuple[pd.DataFrame, pd.Series, dict]:
    features = pd.read_csv(REPO_ROOT / "data/ml_dataset/full_features.csv")
    labels = pd.read_csv(REPO_ROOT / "data/ml_dataset/full_labels.csv")
    merged = features.merge(labels, on="scenario_id", how="inner", validate="one_to_one")

    split_cfg = ml_cfg["splitting"]["experiment_a"]
    splits = split_experiment_a(
        merged, group_column="scenario_id",
        train_fraction=split_cfg["train_fraction"], val_fraction=split_cfg["val_fraction"],
        random_seed=random_seed,
    )
    feature_cols = [c for c in features.columns if c != "scenario_id"]
    assert not (LEAKAGE_COLUMNS & set(feature_cols)), "leakage columns present in feature_cols"
    return merged, pd.Index(feature_cols), splits


def run(random_seed: int = 42, save_artifacts: bool = True) -> dict:
    cfg = load_site_config(SITE)
    ml_cfg = load_yaml_config("ml_training")
    set_global_seed(random_seed)

    merged, feature_cols, splits = load_and_split(random_seed, ml_cfg)
    target_col = "optimal_capacity_kwh"

    train, val, test = (merged.loc[splits[s]] for s in ("train", "val", "test"))
    scaler = fit_scaler(train[feature_cols])
    train_x = pd.DataFrame(scaler.transform(train[feature_cols]), columns=feature_cols, index=train.index)
    val_x = pd.DataFrame(scaler.transform(val[feature_cols]), columns=feature_cols, index=val.index)
    test_x = pd.DataFrame(scaler.transform(test[feature_cols]), columns=feature_cols, index=test.index)

    nn_cfg = ml_cfg["neural_network"]
    model = build_model(n_features=len(feature_cols), config=nn_cfg)
    checkpoint_dir = REPO_ROOT / "models/neural_network_full"
    model, history = train_model(
        model, train_x, train[target_col], val_x, val[target_col], nn_cfg,
        checkpoint_dir=checkpoint_dir,
    )

    test_pred = predict(model, test_x)
    accuracy = compute_accuracy_metrics(test[target_col], test_pred)
    operational = compute_operational_metrics(test[target_col], test_pred)
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
            REPO_ROOT / "models/preprocessing_full",
        )
        out_dir = REPO_ROOT / "outputs/tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([accuracy]).to_csv(out_dir / "accuracy_metrics_neural_network_full.csv", index=False)
        pd.DataFrame([operational]).to_csv(out_dir / "operational_metrics_neural_network_full.csv", index=False)
        verification.to_csv(out_dir / "physical_verification_neural_network_full.csv", index=False)
        pd.DataFrame([verification_summary]).to_csv(out_dir / "physical_verification_summary_neural_network_full.csv", index=False)

        split_assignments = pd.concat([
            pd.DataFrame({"scenario_id": merged.loc[idx, "scenario_id"], "split": name})
            for name, idx in splits.items()
        ])
        split_assignments.to_csv(out_dir / "dataset_split_assignments_full.csv", index=False)

        with open(out_dir / "neural_network_training_history_full.json", "w") as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    return {"accuracy": accuracy, "operational": operational, "verification_summary": verification_summary}


if __name__ == "__main__":
    run()
