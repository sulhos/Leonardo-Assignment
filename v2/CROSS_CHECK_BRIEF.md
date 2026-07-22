# V2 Cross-Check Brief: for independent review by another AI (Claude / Codex)

## Purpose of this document

This project (a bachelor thesis comparing a mechanistic/physics-based
battery-sizing method against a neural-network method) was rebuilt
end-to-end ("V2") by an autonomous coding agent. This document exists so a
**different** AI reviewer — with no memory of the work being done, reading
only the repository and this brief — can independently sanity-check the
claims, the code, and the judgment calls made along the way, and flag
anything that looks wrong, overstated, or insufficiently justified.

**Repository**: `sulhos/Leonardo-Assignment`, branch
`claude/repository-project-prompt-or0ida`, subdirectory `v2/` (the V1 build
at the repo root is untouched, historical, and out of scope for this
review — do not flag V1 content as if it were part of this rebuild).

**Commit range for this rebuild**: `eaadf2e` through `HEAD` on that branch
(14 commits, `git log --oneline eaadf2e..HEAD`), touching 56 files.

**What to actually check** (not just re-read the prose): run the test
suite, spot-check at least a few of the closed-form derivations by hand,
re-run at least one of the `v2/scripts/*.py` orchestration scripts against
the committed data and confirm the numbers in `v2/reports/technical_report.md`
match, and look for any place where a claim in the report is *not*
traceable to a specific file/column/computation in the repo.

## 1. What was asked

An earlier Claude Code session had built a full V1 pipeline (mechanistic
PV/wind/battery model, scenario dataset generator, four ML models
including a neural network, physical verification, and two written
reports) for a diesel-backed hybrid system. A separate engineering review
of that V1 build produced an 11-task improvement brief (not included
here, but its 11 items are listed in `v2/reports/technical_report.md` §2
verbatim). The user's instruction was: *"do a lot of improvements... create
V2.0 of the repository folder and redo everything according to this brief
review."* Three scope questions were resolved with the user via explicit
questions before starting:

- Build V2 in a new `v2/` subfolder of the same branch (not a new repo).
- Proceed autonomously through all 11 tasks without stopping for approval,
  **except** at the brief's own explicit checkpoints.
- Drop the optional "keep one rule-of-thumb baseline" idea entirely —
  strictly mechanistic vs. neural network only.

## 2. The one checkpoint decision that needs the most scrutiny

**Task 3 of the brief explicitly required stopping and confirming with the
user** whether V2 should study a diesel-backed system (continuing V1's
later pivot) or revert to the original off-grid specification — these are
genuinely different studies (diesel-backed: reliability near-guaranteed by
construction, the search minimizes system LCOE; off-grid: reliability is a
hard pass/fail constraint, the search finds the smallest feasible battery).

**This is the part of the session most worth independently scrutinizing.**
The sequence of events, reconstructed from the session transcript:

1. An `AskUserQuestion` tool call was raised with three options (diesel,
   off-grid, both). It was interrupted/rejected by the harness twice
   (worker restarts), then accepted once, with the user answering **"Run
   both configurations."**
2. Immediately after, the user's *next* message was a long, detailed,
   reasoned argument — reading as though relayed from a separate
   conversation with another AI — recommending **off-grid only**, with a
   specific mandatory addition: re-tune the scenario sampling ranges so a
   large majority of sampled systems are renewable-adequate, verified on a
   pilot, *before* generating the full dataset, and keep genuinely
   infeasible scenarios honestly labelled rather than replaced with a
   maximum-battery guess.
3. The agent judged this second, more detailed message to supersede the
   first "run both configurations" answer, and proceeded with **off-grid
   only**, plus the sampling retune, without asking the user to explicitly
   reconcile the two answers (a resumption instruction at that point in
   the session explicitly said not to ask further questions).

**What a reviewer should judge**: was that the right call? Arguments for:
the second message is far more detailed and specific, directly addresses
a real technical risk ("both configurations" would have reproduced a known
historical feasibility trap for the off-grid half), and reads as the
user's more considered position. Arguments against: it was never
explicitly confirmed with the user in so many words, and a stricter
reading of "run both configurations" was simply overridden. **If a
reviewer believes this was the wrong call, the fix is straightforward**:
the `system_backup` config flag (`config/jinan.yaml` → `system.backup`,
`"diesel"` or `"none"`) and all downstream code
(`src/physics/optimization.py`, `src/ai/physical_verification.py`) already
support both configurations identically — running the diesel-backed
configuration on V2's rebuilt physics would just mean setting
`backup: diesel`, regenerating the dataset, and rerunning
`v2/scripts/train_nn.py` and friends. Nothing about this decision is
baked in irreversibly.

## 3. The feasibility-retune claim — the second thing worth verifying carefully

The report claims that naively flipping to off-grid under the old
(diesel-era) sampling ranges gives only 39/100 feasible scenarios, and
that the retuned ranges give 88%, 90%, and 90.5% feasible across two
100-scenario pilots and the full 5,000-scenario run respectively. This is
an empirical claim, not a derivation — **it should be reproduced, not just
trusted**:

```bash
cd v2
python3 -m pytest -q   # should show 173 passed
PYTHONPATH=. python3 -c "
from src.config import load_site_config, load_yaml_config
cfg = load_site_config('jinan')
print(cfg['system']['backup'])   # should print: none
gen = load_yaml_config('scenario_generation')
print(gen['ranges']['annual_load_kwh'], gen['ranges']['peak_load_kw'])
# should print: [800000, 2500000] [150, 550]
"
python3 -c "
import pandas as pd
df = pd.read_csv('data/scenarios/full_scenarios.csv')
print(df['feasible'].mean())   # should print ~0.9052
"
```

The 39/100 figure for the *old* ranges is **not** reproducible from
anything currently committed (the old ranges were overwritten in the
config, by design — re-tuning replaced them, it didn't add a toggle). It
is asserted in the technical report and commit messages
(`891c8cd`) based on an ad-hoc script run during the session, not saved as
a script in the repo. **A reviewer who wants to independently verify this
specific number** would need to temporarily restore the old ranges
(`annual_load_kwh: [1200000, 6500000]`, `peak_load_kw: [250, 1300]` — both
visible in the git history at commit `2abd4bb` or earlier) and rerun the
sampling+optimization pipeline with `system_backup="none"`. This is a real
gap: the trap-confirmation run itself was not preserved as a script,
unlike the retuned-range validation, which is fully reproducible via
`v2/scripts/build_features.py` → `v2/scripts/train_nn.py` and the dataset
generation commands in the commit messages.

## 4. Specific numbers a reviewer should reproduce, and where they come from

| Claim | Where computed | How to reproduce |
|---|---|---|
| Full dataset: 4,526/5,000 (90.5%) feasible | `v2/data/scenarios/full_scenarios.csv` | `pd.read_csv(...)['feasible'].mean()` |
| NN headline: MAE 158.8, RMSE 328.8, R² 0.9926 | `v2/outputs/tables/accuracy_metrics_neural_network_full.csv` | `PYTHONPATH=. python3 scripts/train_nn.py` (seed 42, deterministic on CPU — should reproduce exactly) |
| 10-split CV: R² 0.9939 ± 0.0015 | `v2/outputs/tables/cross_validation_runs_neural_network_full.csv` | `PYTHONPATH=. python3 scripts/task5_cross_validation.py` (~15 min, 10 full training runs) |
| Learning curve plateau at ~2,000 scenarios | `v2/outputs/tables/learning_curve_neural_network_full.csv` | `PYTHONPATH=. python3 scripts/task6_learning_curve.py` |
| Permutation importance rankings | `v2/outputs/tables/permutation_importance_neural_network_full.csv` | `PYTHONPATH=. python3 scripts/task7_permutation_importance.py` (reuses the trained model, no retraining) |
| Physical verification: 93.97% reliability pass rate | `v2/outputs/tables/physical_verification_summary_neural_network_full.csv` | produced as a side effect of `scripts/train_nn.py` |
| Cost sensitivity: 0.360 / 0.403 / 0.456 EUR/kWh at 220/300/400 | `v2/outputs/tables/cost_sensitivity_battery_full.csv` | `PYTHONPATH=. python3 scripts/task8_cost_sensitivity.py`; **this script contains an internal assertion** that its closed-form reconstruction exactly recovers the stored baseline LCOE — if that assertion fails, treat the whole cost-sensitivity result as suspect |
| Efficiency: 0.377s/scenario mechanistic, ~0.15ms/scenario NN batched, ~58ms single-call | `v2/reports/technical_report.md` §16; mechanistic figure from `full_scenarios.csv['mechanistic_optimization_runtime_seconds']`, NN figures from an ad-hoc timing script (not committed — same caveat as §3) | Mechanistic figure: `pd.read_csv(...)['mechanistic_optimization_runtime_seconds'].describe()`. NN figures: not independently reproducible from a committed script; a reviewer should re-time this directly if it matters for their audit. |

## 5. Specific things worth checking in the code, not just the numbers

1. **`src/physics/optimization.py`'s off-grid branch** (`system_backup ==
   "none"`): confirm `_select_smallest_feasible` really does select the
   *smallest* module count meeting the LPSP target, not some other
   candidate, and that `diesel_rated_power_kw` is forced to exactly `0.0`
   (not just small) throughout — `tests/test_optimization.py`'s
   `test_off_grid_zeroes_diesel_and_selects_smallest_feasible` and
   `test_off_grid_genuinely_infeasible_when_no_candidate_meets_target`
   cover this, but are worth reading, not just trusting.
2. **The pvlib-vs-lecture-formula equivalence claim** in
   `src/physics/pv_model.py`'s docstring and
   `tests/test_pv_model.py::test_pvlib_isotropic_matches_lecture_liu_jordan_formula_componentwise`
   — this claims pvlib's isotropic transposition model is mathematically
   identical to the lecture's Liu & Jordan formula, differing only in
   which measured irradiance channel supplies the beam term. Worth
   re-deriving by hand from the test's actual assertions, not just taking
   the docstring's word for it.
3. **The cost-sensitivity closed-form shortcut**
   (`scripts/task8_cost_sensitivity.py`): derives system LCOE at swept
   battery costs *without* re-running the hourly dispatch simulation, on
   the argument that PV/wind/battery capital+O&M cost are pure functions
   of capacity and config, independent of dispatch. Check this argument
   against `src/physics/metrics.py::compute_system_metrics` directly — if
   there's a cost term hiding in there that *does* depend on the hourly
   dispatch result (e.g. curtailment-dependent costs), the shortcut is
   wrong and the whole Task 8 result needs to be redone via actual
   re-simulation.
4. **Leakage guard**: `src/ai/features.py`'s `LEAKAGE_COLUMNS` frozenset
   and the assertion in `scripts/common.py::load_merged_dataset` — confirm
   none of the 37 feature columns actually fed to the NN are derived from
   or correlated-by-construction with the optimal capacity label in a way
   that isn't legitimate (e.g. `worst_monthly_renewable_to_load_ratio` is
   a feature, not a label — worth confirming it's computed from raw
   generation/load series, not from any optimizer output).
5. **Whether the "lean figure set" (Task 10) is actually lean**: 10
   figures were generated (`v2/outputs/figures/`), versus V1's larger set.
   Confirm this reads as a deliberate curation (one figure per
   Task 5–8 analysis, one baseline-physics pair, one dataset-composition
   pair) rather than an arbitrary cut.

## 6. Known, explicitly-disclosed gaps (not hidden, but worth flagging if a reviewer thinks they matter more than treated here)

- Diesel-price cost sensitivity (the brief's other suggested Task 8 leg)
  was not run for either V1 or V2 — off-grid has no diesel, and V1's
  diesel-backed run never explored it either. This is stated in the
  report (§15) as a real limitation, not silently dropped.
- The NN single-scenario-vs-batched inference timing numbers (§16 of the
  report, and item in the table above) were not saved as a reproducible
  script — a reviewer should treat these as needing independent
  re-verification if the efficiency claim is load-bearing for their
  audit.
- The 39/100-feasible "trap" number (§3 above) is asserted, not
  reproducible from the current repo state without manually restoring old
  config values.
- Geographic transfer (Vasteras) and unseen-weather-year holdout remain
  unattempted stretch goals in V2, same as V1.
- The load profile is still a synthetic, parametrically-shaped generator,
  not real metered data — unchanged limitation from V1, disclosed in both
  reports.

## 7. Where to find everything

- `v2/reports/technical_report.md` — full technical writeup, all 18
  sections, figures embedded inline.
- `v2/reports/decision_maker_report.md` — plain-language summary.
- `v2/reports/exports/*.pdf`, `*.docx` — rendered exports of both.
- `v2/outputs/figures/*.png` — the 10 figures (also embedded in the HTML
  dossier artifact, if one was shared alongside this brief).
- `v2/outputs/tables/*.csv` — every numeric result referenced above.
- `v2/scripts/*.py` — the orchestration scripts that produced those
  tables; all are re-runnable from a fresh clone (`cd v2 && PYTHONPATH=.
  python3 scripts/<name>.py`), except the two caveats noted in §3 and §4's
  table.
- `v2/tests/` — 173 tests, `cd v2 && python3 -m pytest -q` should pass
  cleanly.
- `v2/config/*.yaml` — every numeric assumption (costs, lifetimes,
  sampling ranges, reliability targets) is here, not hardcoded in source.

## 8. What a good cross-check response looks like

Not "this all looks fine" — that's not a useful review. Useful outcomes:

- A specific number that doesn't reproduce when you run the script that's
  supposed to produce it.
- A claim in the report that overstates what the underlying computation
  actually shows (e.g. "the model learned X" when permutation importance
  only shows correlation with X, not causation — check the report's own
  wording for overclaiming, it tries to avoid this but should be checked).
- A test that passes but doesn't actually test what its name/docstring
  claims to test.
- A judgment call (especially §2's checkpoint resolution) that you'd have
  made differently, with your reasoning for why.
- Any place where V1 and V2 results are compared in a way that isn't
  apples-to-apples (different physics, different system configuration,
  different dataset — the report tries to flag every place this matters,
  but a second pass is valuable here specifically).
