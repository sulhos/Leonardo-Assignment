"""Build and persist the ML-ready scenario dataset (PROJECT_BRIEF.md §14).

Orchestrates sampling (`sampling.py`) and per-scenario execution
(`scenario_runner.py`) into a single labelled dataset, writing one CSV row
at a time so an interrupted run can resume without recomputing already-
completed scenarios. A sidecar JSON file records the random seed and a
configuration hash, per PROJECT_BRIEF.md §14's reproducibility requirement.

Parallel execution is intentionally NOT implemented: at pilot scale (~100
scenarios, ~1-1.5s each) sequential generation finishes in well under two
minutes, so the added complexity and reproducibility risk of parallelizing
isn't justified yet. If the full 5,000-scenario dataset (Stage 8, stretch
goal) becomes the target, revisit this with a design that preserves
deterministic per-scenario seeding regardless of execution order.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.weather import get_processed_weather
from src.scenarios.sampling import sample_scenarios
from src.scenarios.scenario_runner import run_scenario

logger = logging.getLogger(__name__)


def _config_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def build_dataset(
    n_scenarios: int,
    output_dir: Path,
    random_seed: int,
    site_config: dict,
    ranges: dict[str, Any],
    n_max: int,
    location: str = "jinan",
    weather_year: int = 2023,
    dataset_name: str = "pilot",
    resume: bool = True,
) -> pd.DataFrame:
    """Generate (or resume generating) a labelled scenario dataset of the
    requested size and persist it under `output_dir`.

    Args:
        n_scenarios: Target number of scenarios.
        output_dir: Directory to write `{dataset_name}_scenarios.csv` and
            `{dataset_name}_metadata.json` into.
        random_seed: Base seed for scenario sampling (see `sampling.py` for
            the per-scenario seed derivation).
        site_config: Fixed site configuration (siting assumptions).
        ranges: Scenario input ranges (`config/scenario_generation.yaml` -> `ranges`).
        n_max: Maximum battery module count for each scenario's search.
        location: Fixed site name for this dataset.
        weather_year: Fixed weather year for this dataset.
        dataset_name: Used to name the output files (e.g. "pilot", "dev", "full").
        resume: If True and a partial CSV already exists, skip scenario_ids
            already present in it.

    Returns:
        The complete labelled dataset as a DataFrame.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{dataset_name}_scenarios.csv"
    metadata_path = output_dir / f"{dataset_name}_metadata.json"

    config_hash = _config_hash(
        {"ranges": ranges, "random_seed": random_seed, "n_scenarios": n_scenarios,
         "location": location, "weather_year": weather_year, "n_max": n_max}
    )

    completed_ids: set[int] = set()
    if resume and csv_path.is_file():
        existing = pd.read_csv(csv_path)
        completed_ids = set(existing["scenario_id"].tolist())
        logger.info("Resuming dataset build: %d scenarios already completed.", len(completed_ids))

    scenarios = sample_scenarios(
        n_scenarios, ranges, random_seed, location, site_config["site"]["timezone"], weather_year
    )
    weather = get_processed_weather(location, weather_year)

    write_header = not (resume and csv_path.is_file())
    start_time = time.perf_counter()
    n_run = 0

    for scenario in scenarios:
        if scenario["scenario_id"] in completed_ids:
            continue

        row = run_scenario(scenario, site_config, weather, n_max)
        row_df = pd.DataFrame([row])
        row_df.to_csv(csv_path, mode="a", header=write_header, index=False)
        write_header = False
        n_run += 1

        if n_run % 10 == 0:
            logger.info("Completed %d/%d new scenarios.", n_run, n_scenarios - len(completed_ids))

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Dataset build finished: %d new scenarios run in %.1fs (%.2fs/scenario).",
        n_run, elapsed, elapsed / n_run if n_run > 0 else 0.0,
    )

    metadata = {
        "dataset_name": dataset_name,
        "n_scenarios_requested": n_scenarios,
        "random_seed": random_seed,
        "config_hash": config_hash,
        "location": location,
        "weather_year": weather_year,
        "n_max": n_max,
        "ranges": ranges,
        "last_run_new_scenarios": n_run,
        "last_run_seconds": elapsed,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return pd.read_csv(csv_path)
