# Battery Sizing: Mechanistic vs Neural Network Comparison

Bachelor thesis project comparing a physics-based (mechanistic) battery-sizing method
against a neural-network-based method for an off-grid PV–wind energy system.

**Title:** Comparison of Neural-Network and Mechanistic Models for Battery Sizing in an
Off-Grid PV–Wind Energy System

## Status

Stage 1 (project scaffold) is complete. See `docs/` history / commit log for progress
through later stages. This project is built incrementally, stage by stage — see
`PROJECT_BRIEF.md` (implementation stages) for the full plan and current scope.

## Research objective

Compare a mechanistic (physics-based, exhaustive-search) battery-sizing method with an
AI-based neural-network method for a fixed-PV, fixed-wind, off-grid system, and test
(not assume) whether the neural network can reproduce mechanistic battery-capacity
decisions accurately, quickly, and reliably enough to be useful in practice. See
`PROJECT_BRIEF.md` §1 and §2 for the full objective and hypothesis.

## Scope

This build follows a reduced "core scope" (MVP) before attempting stretch goals, per the
refinement addendum in `PROJECT_BRIEF.md` §1.1:

- **Core:** project scaffold → mechanistic components → baseline battery optimization
  (Jinan 2023) → pilot scenario dataset (~100 scenarios) → baseline ML models → neural
  network (MLP) → physical AI verification.
- **Stretch (only after core is complete, tested, documented):** full 5,000-scenario
  dataset, unseen-weather-year holdout, Västerås geographic transfer, full economic
  sensitivity sweeps, full report/presentation polish.

## System boundary

- Fixed PV capacity + fixed wind capacity + battery storage + electrical load.
- No grid connection, no diesel/backup generator, no import/export.
- Hourly simulation, one complete local calendar year per scenario.
- The mechanistic optimization varies **only** battery capacity (in discrete modules).

## Baseline case

| Parameter | Value |
|---|---|
| Location | Jinan, China (36.65 N, 117.12 E), Asia/Shanghai |
| Simulation year | 2023 (8,760 hourly steps) |
| Annual load | ~30,000 kWh/year, peak ≤ 15 kW |
| PV capacity | 20 kWp (fixed) |
| Wind capacity | 10 kW (fixed) |
| Reliability requirement | LPSP ≤ 1% (≥ 99% load served) |
| Battery module | BYD Battery-Box Premium LVL-style, 15.36 kWh / 12.8 kW per module |

A secondary location (Västerås, Sweden) is configured for later geographic-transfer /
sensitivity work but is **not** the baseline.

## Project layout

See `docs/architecture.md`-equivalent in `PROJECT_BRIEF.md` §24 for the full target
tree. Summary:

- `config/` — YAML configuration (site, scenario generation, ML training).
- `data/` — raw/processed weather, generated scenarios, ML-ready datasets. See
  `data/README.md`.
- `notebooks/` — thin notebooks that call into `src/`; no logic lives in notebooks.
- `src/` — all reusable logic (`data`, `physics`, `scenarios`, `ai`, `visualization`).
- `models/` — saved trained models and preprocessing objects.
- `tests/` — automated tests (deterministic, no internet access required).
- `outputs/` — generated figures, tables, logs, comparison results.
- `reports/` — technical report, decision-maker report, LLM prompt appendix.
- `presentation/` — presentation outline.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate battery-sizing-ai-comparison
```

## Running

All commands are intended to be run from the project root. As of Stage 1, no
executable pipelines exist yet — modules are documented placeholders. Once Stage 2+
land, this section will be updated with concrete commands (e.g. weather download,
baseline optimization run, dataset generation, model training).

Run tests with:

```bash
pytest
```

## Key modelling caveats (see `PROJECT_BRIEF.md` for full detail)

- The mechanistic model is a **reference method**, not physical ground truth. AI
  agreement with it is reported as agreement-with-reference, never as "validation
  against reality."
- Battery capacity underprediction is a reliability risk and is always reported
  separately from averaged error metrics.
- All ML feature scalers are fit on training data only; no target leakage (mechanistic
  optimal capacity/module count/cost are never used as ML inputs); dataset splits are
  grouped/temporal, never a naive random split across near-identical scenarios.

## License

TBD.
