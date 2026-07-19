"""Build and persist the ML-ready scenario dataset (PROJECT_BRIEF.md §14).

Responsibilities (implemented in Stage 4):

- Orchestrate sampling (sampling.py) and per-scenario execution
  (scenario_runner.py) into a single labelled dataset.
- Start with a pilot size (100), then scale incrementally as a stretch goal
  (100 -> 500 -> 1,000 -> full 5,000), re-validating at each step per
  refinement addendum §1.1 -- never jump straight to the full size.
- Support parallel scenario generation only once reproducibility under
  parallelism is verified; preserve deterministic seeding regardless.
- Persist results under data/scenarios/, resumable via a completed-scenario
  cache.

This module intentionally contains no implementation yet -- it is a Stage 1
placeholder. Functions raise NotImplementedError until Stage 4.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def build_dataset(
    n_scenarios: int,
    output_dir: Path,
    random_seed: int,
    resume: bool = True,
) -> pd.DataFrame:
    """Generate (or resume generating) a labelled scenario dataset of the
    requested size and persist it under `output_dir`."""
    raise NotImplementedError("Implemented in Stage 4.")
