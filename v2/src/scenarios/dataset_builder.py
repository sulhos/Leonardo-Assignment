"""Build and persist the ML-ready scenario dataset (PROJECT_BRIEF.md §14).

Orchestrates sampling (`sampling.py`) and per-scenario execution
(`scenario_runner.py`) into a single labelled dataset, writing rows in
periodic batches so an interrupted run can resume without recomputing
already-completed scenarios. A sidecar JSON file records the random seed and
a configuration hash, per PROJECT_BRIEF.md §14's reproducibility requirement.

**Parallel execution** (`parallel=True`) uses a `ProcessPoolExecutor`. This
is safe and reproducibility-preserving because each scenario's outcome
depends only on its own inputs (sampled deterministically from
`scenario_id` before any parallel dispatch, per `sampling.py`) -- never on
execution order or on any other scenario's result. Worker processes are
initialized once with the shared weather DataFrame and site config (via
`initializer=`) rather than re-pickling them per task, since those are the
same for every scenario in one dataset.

Not parallelized at pilot scale (~100 scenarios, ~1.2s each, well under two
minutes sequential) -- enabled starting with the Stage 8 scale-up, where
sequential generation would take well over an hour for 5,000 scenarios.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.weather import get_processed_weather
from src.scenarios.sampling import sample_scenarios
from src.scenarios.scenario_runner import run_scenario

logger = logging.getLogger(__name__)

FLUSH_EVERY = 50  # scenarios per CSV append batch, balances resumability vs. I/O overhead

# Populated once per worker process by _init_worker; avoids re-pickling the
# (shared, identical-per-dataset) weather/site_config/n_max on every task.
_worker_weather: pd.DataFrame | None = None
_worker_site_config: dict | None = None
_worker_n_max: int | None = None


def _init_worker(weather: pd.DataFrame, site_config: dict, n_max: int) -> None:
    global _worker_weather, _worker_site_config, _worker_n_max
    _worker_weather = weather
    _worker_site_config = site_config
    _worker_n_max = n_max


def _run_scenario_in_worker(scenario: dict) -> dict:
    return run_scenario(scenario, _worker_site_config, _worker_weather, _worker_n_max)


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
    parallel: bool = False,
    max_workers: int | None = None,
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
        parallel: If True, run scenarios across a process pool instead of
            sequentially. Reproducible either way (see module docstring).
        max_workers: Worker process count when `parallel=True`. Defaults to
            `os.cpu_count()`.

    Returns:
        The complete labelled dataset as a DataFrame.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{dataset_name}_scenarios.csv"
    metadata_path = output_dir / f"{dataset_name}_metadata.json"

    # n_scenarios is deliberately excluded from the hash: growing a dataset
    # across resumed calls with a larger n_scenarios (e.g. the incremental
    # 100 -> 500 -> 1000 -> 5000 scale-up, PROJECT_BRIEF.md Addendum §1.1)
    # is the canonical legitimate resume pattern, not a stale-config
    # mismatch. Everything else must match for existing rows to stay valid.
    config_hash = _config_hash(
        {"ranges": ranges, "random_seed": random_seed,
         "location": location, "weather_year": weather_year, "n_max": n_max}
    )

    completed_ids: set[int] = set()
    if resume and csv_path.is_file():
        if metadata_path.is_file():
            existing_config_hash = json.loads(metadata_path.read_text()).get("config_hash")
            if existing_config_hash is not None and existing_config_hash != config_hash:
                raise RuntimeError(
                    f"Refusing to resume {csv_path}: its config_hash ({existing_config_hash}) "
                    f"does not match the current request's config_hash ({config_hash}). This "
                    "means ranges/n_scenarios/location/weather_year/n_max changed since that "
                    "file was generated (e.g. after a config edit) -- resuming would silently "
                    "mix stale rows generated under the old config with the new request instead "
                    "of regenerating, and (if n_scenarios/n_scenarios's scenario_ids fully "
                    "overlap) could return the *entirely* stale dataset with no new rows at all. "
                    f"Delete {csv_path} and {metadata_path} (or pass resume=False) to force a "
                    "clean regeneration under the new config."
                )
        existing = pd.read_csv(csv_path)
        completed_ids = set(existing["scenario_id"].tolist())
        logger.info("Resuming dataset build: %d scenarios already completed.", len(completed_ids))

    scenarios = sample_scenarios(
        n_scenarios, ranges, random_seed, location, site_config["site"]["timezone"], weather_year
    )
    weather = get_processed_weather(location, weather_year)
    scenarios_to_run = [s for s in scenarios if s["scenario_id"] not in completed_ids]

    write_header = not (resume and csv_path.is_file())
    start_time = time.perf_counter()
    n_run = 0
    buffer: list[dict] = []

    def _flush() -> None:
        nonlocal write_header, buffer
        if not buffer:
            return
        pd.DataFrame(buffer).to_csv(csv_path, mode="a", header=write_header, index=False)
        write_header = False
        buffer = []

    if parallel and scenarios_to_run:
        with ProcessPoolExecutor(
            max_workers=max_workers, initializer=_init_worker, initargs=(weather, site_config, n_max)
        ) as executor:
            for row in executor.map(_run_scenario_in_worker, scenarios_to_run):
                buffer.append(row)
                n_run += 1
                if len(buffer) >= FLUSH_EVERY:
                    _flush()
                if n_run % 100 == 0:
                    logger.info("Completed %d/%d new scenarios.", n_run, len(scenarios_to_run))
    else:
        for scenario in scenarios_to_run:
            buffer.append(run_scenario(scenario, site_config, weather, n_max))
            n_run += 1
            if len(buffer) >= FLUSH_EVERY:
                _flush()
            if n_run % 50 == 0:
                logger.info("Completed %d/%d new scenarios.", n_run, len(scenarios_to_run))

    _flush()

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Dataset build finished: %d new scenarios run in %.1fs (%.2fs/scenario, parallel=%s).",
        n_run, elapsed, elapsed / n_run if n_run > 0 else 0.0, parallel,
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
        "last_run_parallel": parallel,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return pd.read_csv(csv_path)
