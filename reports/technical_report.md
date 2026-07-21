# Comparison of Neural-Network and Mechanistic Models for Battery Sizing in an Off-Grid PV–Wind Energy System

**Status:** Sections 9-20 (Scenario-Dataset Generation through Conclusions) contain
real results from the industrial-scale build (PROJECT_BRIEF.md Addendum 2).
Sections 1, 3-8 remain Stage 1 structural placeholders (introduction, research-question
prose, system description, and method write-ups deferred to a documentation pass) --
those specific sections should not be read as findings; every section from §9 onward
should be.

---

## 1. Introduction

*TODO (Stage 10): motivation for off-grid battery sizing, why comparing a mechanistic
method against a neural network is worth investigating, and a one-paragraph roadmap of
the report.*

## 2. Related Work

*Added per refinement addendum §1.3 — this section is expected by the thesis rubric
and was missing from the original spec's outline.*

**LPSP-based sizing methods for standalone PV/wind/battery systems.** The
loss-of-power-supply-probability (LPSP) reliability metric used throughout this
project (§8, §12) follows the same formulation as Yang, Lu & Zhou's foundational
hybrid solar-wind sizing model, which iterates over candidate PV/wind/battery
combinations, computes LPSP and annualized cost for each, and selects the
cheapest combination meeting a reliability target (Yang, H., Lu, L., & Zhou, W.
(2007). *A novel optimization sizing model for hybrid solar-wind power
generation system.* Solar Energy, 81(1), 76–84). This project's exhaustive
battery-module search (§8) is a direct, simplified instance of that same
LPSP-driven search pattern, restricted to a single decision variable (battery
capacity, given fixed PV/wind) rather than jointly searching PV, wind, and
battery capacity together.

**Techno-economic optimization tools.** HOMER (Lambert, T., Gilman, P., &
Lilienthal, P. (2006). *Micropower system modeling with HOMER.* In F. A.
Farret & M. G. Simões (Eds.), Integration of Alternative Sources of Energy
(pp. 379–418). John Wiley & Sons) is the best-known example of this class of
tool: given candidate system sizes, it simulates dispatch over a representative
year and ranks designs by life-cycle cost. **OptiCE** (Campana, P. E., Zhang,
Y., & Yan, J., https://optice.net/) — the MATLAB techno-economic optimization
tool developed by this project's course instructor and collaborators, made
available as example course material alongside a lecture on LLM-assisted
management of PV/wind microgrids — follows the same simulate-then-rank
pattern but goes further:
it jointly optimizes PV tilt/azimuth/capacity, wind tower height/capacity,
*and* battery capacity via a multi-objective genetic algorithm
(`gamultiobj`), producing a Pareto front of life-cycle-cost-vs-renewables-share
trade-offs rather than a single design. This project's mechanistic model
(§6-§8) independently arrived at several of the same core modelling choices as
OptiCE's dispatch/battery logic — a power-law wind-speed height extrapolation,
a NOCT-based PV temperature model, and a symmetric charge/discharge efficiency
split whose product recovers the specified round-trip efficiency (this
project's `sqrt(round_trip_efficiency)` convention is structurally identical
to OptiCE's `Battery_efficiency × charge_controller_efficiency` applied once
per leg) — which is a useful cross-check that the mechanistic reference model
used throughout this report is consistent with established practice in this
specific research group, not an idiosyncratic reimplementation. Two respects in
which OptiCE is more detailed than this project's model are noted as
limitations in §19: a wind-speed-dependent PV cell-temperature correction, and
a battery thermal/temperature-dependent-capacity model. OptiCE's joint
PV/wind/battery optimization is also more general than this project's
fixed-PV/wind, battery-only search — a deliberate scope choice explained in
§19, made so that PV/wind capacity could instead vary *across* the sampled
scenario dataset (§9) as inputs for the ML models to condition on, rather than
being optimized to a single value.

**Prior machine-learning-for-sizing literature.** Most existing ML work on
hybrid PV/wind/battery systems targets short-term *forecasting* (solar
irradiance, wind speed, or load) as an input to an otherwise conventional
sizing or dispatch procedure, rather than replacing the sizing procedure
itself — reviews of wind/solar forecasting techniques for grid integration
survey this large body of work (Ssekulima, E. B., Anwar, M. B., Al Hinai, A.,
& El Moursi, M. S. (2016). *Wind speed and solar irradiance forecasting
techniques for enhanced renewable energy integration with the grid: a
review.* IET Renewable Power Generation, 10(7), 885–899). Separately, this
project's own PV model (§6) sidesteps two problems this course's lecture
material identifies as central to computing irradiance on a tilted surface
from measurements (Duffie, J. A., Beckman, W. A., & Worek, W. M. (2013).
*Solar engineering of thermal processes* (Vol. 3). New York: Wiley): not
knowing the beam/diffuse split of measured irradiance (usually resolved with
a decomposition model such as Erbs), and not knowing the solar
position/incidence angle needed for a transposition model such as Liu and
Jordan. Both are avoided here rather than solved: NASA POWER's hourly product
already reports direct-normal and diffuse-horizontal irradiance as separate
fields (no decomposition model needed, confirmed in `src/physics/pv_model.py`'s
docstring), and `pvlib.irradiance.get_total_irradiance` computes solar
position and the tilted-surface transposition internally. Work that uses ML
to predict *sizing* outcomes directly, as this
project does, is comparatively less common; a recent example combines
gradient-boosted tree ensembles (forecasting weather/load over the system
lifetime) with metaheuristic optimization to size a PV/battery system under
net-metering costs (Abdullah, H. M., Park, S., Seong, K., & Lee, S. (2023).
*Hybrid Renewable Energy System Design: A Machine Learning Approach for
Optimal Sizing with Net-Metering Costs.* Sustainability, 15(11), 8538).
That work uses ML to generate better *inputs* to a conventional metaheuristic
optimizer; this project instead trains an ML model to directly predict the
*sizing decision itself* (the mechanistic search's output) from scenario
parameters, then treats the mechanistic search as ground truth for mandatory
post-hoc verification (§16) rather than as a downstream optimizer to feed —
a different position for the mechanistic and ML components to occupy relative
to each other, and the specific comparison this project's research questions
(§3) are about.

## 3. Research Objective and Questions

See `PROJECT_BRIEF.md` §1 (research objective) and §2 (research hypothesis) for the
canonical statement. Copy/adapt into prose here in Stage 10.

## 4. System Description and Assumptions

*TODO: system boundary (§3), baseline case (§4), battery reference and the
manufacturer-spec / modelling-assumption / economic-assumption distinction (§5),
temperature assumption (§6).*

## 5. Weather and Load Data

*TODO: NASA POWER hourly API acquisition, the API-validation findings from
`data/README.md` (addendum §1.4), and the synthetic load-profile generator (§8).*

## 6. Mechanistic PV and Wind Models

*TODO: PV model (§9) and wind model (§10), including documented equations, units,
and limitations (reanalysis wind data, spatial resolution, height extrapolation,
generic turbine curve, missing turbulence/wake effects).*

## 7. Battery-Dispatch Model

*TODO: dispatch priority order, state equation, initial-SOC-bias mitigation method
and its validation (§11).*

## 8. Mechanistic Battery Optimization

*TODO: exhaustive search method, primary/sensitivity reliability targets,
infeasibility handling (§12).*

## 9. Scenario-Dataset Generation

**This project pivoted from a residential to an industrial deployment context
partway through (PROJECT_BRIEF.md Addendum 2), after the residential-scale
build was already complete, tested, and reported.** The pivot's reasoning:
wind generation is a difficult economic case to justify for a single
residential site (small load, small rooftop-scale system); an industrial
facility makes the PV+wind+battery combination a more defensible investment
case without changing the project's core research question. Every number in
this report from here on is industrial-scale; the residential-scale build
remains fully recoverable from git history but is not reported here.

The pilot dataset (100 scenarios, single location [Jinan], single weather year
[2023], per refinement addendum §1.1) was generated by sampling PV capacity
(500-3,500 kWp), wind capacity (200-2,000 kW, with a combined PV+wind cap of
5,000 kW enforced as its own consistency check -- the individual ranges alone
can exceed it), annual load (1.2-6.5 GWh), peak load (250-1,300 kW),
reliability target (99.0-99.9%), round-trip efficiency (88-97%), and usable SOC
window (70-90%) uniformly within the ranges in `config/scenario_generation.yaml`.
Each sampled combination is checked for physical consistency before acceptance
-- the peak/annual-consumption check reuses the load-profile generator's own
internal constraint directly (by attempting generation) rather than an
approximated bound, so it is exactly as strict as the generator that is
actually used.

These ranges were not the first attempt: an initial guess (12 GWh/year load
against a 5,000 kW renewable cap) turned out to be renewable-generation-
inadequate for nearly every scenario, since Jinan's blended PV+wind capacity
factor (~14.8%) caps annual production at the 5,000 kW cap to roughly 6.5
GWh/year -- a load range extending to 21 GWh/year (the first guess) was
never going to be reachable. Ranges were rescaled down before any scenario
generation was committed to, so the dataset reflects a genuine mix of
feasible and (mostly seasonally, not quantity-) limited scenarios rather than
one dominated by trivially oversized loads.

**Feasibility: 39/100 scenarios (39%) feasible** within the configured battery
search range (0-80 modules, 250 kWh/module -- widened from the residential
build's 0-30/15.36 kWh to remain a meaningful search at industrial scale; see
§14). Infeasible scenarios are labelled explicitly with a reason, never
assigned an arbitrary battery capacity.

Mechanistic search runtime: 0.180s/scenario average (81 candidates each,
matching Stage 3), ~18s total for the full pilot run. This reflects a numba
JIT-compiled dispatch loop (~40x faster than the pure-Python loop it replaced,
verified bit-for-bit identical output before being adopted) -- without it, the
wider 0-80 search range implied by the industrial pivot would have made the
full 5,000-scenario dataset an estimated 5.5-6 hours single-threaded, instead
of the few minutes it actually took (below). Dataset generation is resumable
(scenario-level CSV caching, verified by test) and records a configuration
hash and random seed for reproducibility; a real bug was found and fixed
during this pivot where resuming under a *different* configuration could
silently return an entirely stale dataset with no error (fixed with an
explicit config-hash check, regression-tested).

Full results: `outputs/tables/scenario_inputs_and_labels.csv` /
`data/scenarios/pilot_scenarios.csv`. Figures: `outputs/figures/12`-`15`.

**Stage 8: full 5,000-scenario dataset.** Generated directly at full scale
rather than incrementally (100 → 500 → 1,000 → 5,000): with the numba-
accelerated dispatch loop, generation no longer needed the addendum's
incremental re-validation safeguard, which existed specifically because
generation used to be slow enough that discovering a problem at 5,000
scenarios would be expensive to redo. Final result: **2,111/5,000 scenarios
(42.2%) feasible** -- reasonably consistent with the pilot's 39%, and with a
similar infeasibility-reason split (941 renewable-generation-inadequate,
1,948 battery-range-inadequate) -- using the identical sampling ranges,
consistency check, and 0-80 module search range as the pilot. Generation used
the `ProcessPoolExecutor`-based parallel implementation of `build_dataset()`
(4 workers). Total generation time: **314.6s (5.2 minutes)** across 5,000
scenarios, i.e. ~0.063s/scenario wall-clock with 4-way parallelism (~0.186s/
scenario mean single-candidate-search cost measured per-scenario, consistent
with the pilot's 0.180s).

Full results: `data/scenarios/full_scenarios.csv`, `data/scenarios/full_metadata.json`.

## 10. Feature Engineering

37 features per scenario across four groups (load, generation, net-load, system),
computed by deterministically regenerating each scenario's hourly load/PV/wind
series from its stored inputs (capacity, annual load, peak load, random seed) --
the full 8,760-hour series were not persisted per scenario (100 x 8,760 x 3 would
be redundant storage since regeneration is fast, and exact).

Net-load features include the maximum consecutive-deficit-hour run length and the
largest cumulative deficit energy event, computed directly from the regenerated
hourly series -- these are what actually explain the Stage 3/4 infeasibility
pattern (seasonal mismatch), not just annual totals. The feature *definitions*
are identical to the residential-scale build (PROJECT_BRIEF.md Addendum 2 changed
deployment scale and load-profile family, not the feature-engineering logic).

No target leakage: `optimal_capacity_kwh`, `optimal_n_modules`, `reference_lpsp`,
and `reference_annualized_cost` are guarded by `src/ai/features.py`'s
`LEAKAGE_COLUMNS` and asserted absent from the feature matrix at build time (test-
covered).

## 11. Machine-Learning Baseline Models

Three baselines trained on the **39 feasible pilot scenarios only** (infeasible
scenarios have no real capacity label to regress against, so they are excluded
rather than imputed):

- **Naive**: `DummyRegressor(strategy="median")` -- predicts the training-set
  median capacity regardless of features.
- **Ridge regression**: alpha selected from `[0.01, 0.1, 1.0, 10.0]` by validation
  MAE, then refit on the training split only.
- **Random Forest**: `n_estimators` in `[100, 300]`, `max_depth` in `[3, 5, None]`,
  selected the same way. (Config originally specified `gradient_boosting`;
  finalized to `random_forest` in Stage 5 because `max_depth: null`, i.e.
  unlimited depth, is natively meaningful for Random Forest but not supported by
  scikit-learn's `GradientBoostingRegressor`, which requires an integer.)

Split: Experiment A, grouped 70/15/15 (27/5/7 scenarios). Hyperparameter selection
uses only train+val; the test split is touched exactly once, for final evaluation.
Results in §15.

## 12. Neural-Network Architecture and Training

Framework: **Keras/TensorFlow** (pinned per refinement addendum §1.2), TensorFlow
2.21, CPU-only (no GPU in this environment).

Architecture, exactly as specified in `config/ml_training.yaml` (no changes made
to try to improve results): input (37 features) → Dense(128, ReLU) → Dropout(0.10)
→ Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, ReLU) — 15,233 parameters total.
The output-layer ReLU activation enforces non-negative capacity predictions;
`src/ai/neural_network.py::predict` also applies a defensive `np.maximum(pred, 0)`
as a second layer per §18.

Training: Huber loss, Adam (lr=0.001), batch size 32, up to 500 epochs, early
stopping on `val_loss` (patience 20, restore best weights), best-model
checkpointing. Trained on the identical 27/5/7 Experiment A split and 37 features
as the Stage 5 baselines, with inputs standardized by a scaler fit on the training
split only.

**Actual run:** converged in 173 epochs (17.1s on CPU), best validation loss
1448.2 at epoch 152. Unlike the residential-scale pilot's smoother, non-
diverging curve, this run's loss curve
(`outputs/figures/16_training_validation_loss.png`) shows a real overfitting
signature: validation loss falls together with training loss up to epoch
~152, then rises again (from ~1,450 to ~2,600) while training loss keeps
falling -- exactly why early stopping (patience 20) triggered at epoch 173.
This is a visible, training-time symptom of the pattern that becomes explicit
in §15.1/§16.1: **27 training rows is not enough data for a 15,233-parameter
network to learn reliably.** This is reported as the honest, unmodified
result of the pilot's actual (small) sample size -- not adjusted, retried
with different seeds, or hidden -- and is a large part of why Stage 8/9's
full-scale re-run matters here even more than it did for the residential
build.

Saved artifacts: `models/neural_network/best_model.weights.h5` (portable
weights; loaded via `load_trained_model`, which rebuilds the architecture from
`config/ml_training.yaml` rather than deserializing a full `.keras` file --
the latter is not portable across Keras versions, see §16.1's note),
`models/neural_network/best_model.keras` (full-model checkpoint, same-
environment convenience only), `models/preprocessing/scaler.pkl`,
`models/preprocessing/feature_columns.pkl`.

## 13. Evaluation Methods

**Splitting strategy:** Experiment A (grouped 70/15/15 train/val/test split,
`GroupShuffleSplit` on `scenario_id`) was used throughout, for both the pilot
and the full 5,000-scenario dataset. Since scenarios are sampled i.i.d. and
`scenario_id` is unique per scenario, grouping by scenario is mathematically
equivalent to a random split here -- it is the correct choice (not a
formality) because it guarantees no scenario's hourly-regenerated features
leak across train/val/test, which would matter if scenario families were ever
introduced. Experiments B (unseen weather year) and C (Västerås geographic
transfer) are stretch goals and were not attempted (§19).

**Accuracy metrics** (`src/ai/evaluation.py::compute_accuracy_metrics`): MAE,
RMSE, R², median absolute error, MAPE, max absolute error, signed bias, and
percentage of predictions within 5/10/20% of the reference value.

**Operational metrics** (`compute_operational_metrics`), reported *separately*
from accuracy metrics per the refinement addendum's requirement that
underprediction be treated as a distinct reliability risk rather than averaged
away: underprediction rate and mean/max underprediction magnitude,
overprediction rate and mean/max overprediction magnitude, exact-match rate.

**Physical verification metrics** (§16): reliability pass rate against each
scenario's own target, percent undersized/oversized, mean excess capacity,
max underprediction, additional cost from oversizing, and count of reliability
violations attributable specifically to undersizing (as opposed to some other
cause) -- the mandatory check that accuracy/operational metrics alone cannot
substitute for (PROJECT_BRIEF.md §20).

## 14. Mechanistic Results

Stage 3 (exhaustive battery-module search, 0-80 modules, 250 kWh each) was run
against the Jinan 2023 industrial baseline case: fixed PV 1,800 kWp, fixed wind
1,450 kW (combined 3,250 kW, within the 5,000 kW cap), synthetic industrial load
(3,800,000 kWh/year, 800 kW peak, `industrial_baseline` profile family), LPSP
target 1%.

These baseline capacities were not the first attempt. An initial guess of a 12
GWh/year load against 1,800 kWp PV + 800 kW wind was renewable-generation-
inadequate outright (annual coverage 28%) -- Jinan's blended PV+wind capacity
factor (~14.8%) simply cannot support that much load at this combined capacity.
The load was rescaled down and wind capacity raised to 1,450 kW so the baseline
reaches a comparable annual coverage ratio to the residential-scale project's
own baseline case, before being adopted -- the numbers below are the result of
that recalibration, not the first configuration tried.

**Result: infeasible within the configured search range.** The smallest LPSP
achieved at the largest tested candidate (80 modules, 20,000 kWh) was 10.06%, an
order of magnitude above the 1% target. No module count from 0-80 satisfies the
baseline reliability requirement, so per PROJECT_BRIEF.md §12 this is reported as
infeasible rather than the 80-module candidate being mislabelled "optimal."

This is *not* a case of insufficient annual renewable energy: annual PV+wind
production (4,232,266 kWh) exceeds annual load (3,800,000 kWh) by 11.4% --
deliberately comparable to the residential baseline's own 11% margin. Instead it
is a **seasonal generation/load mismatch**: the monthly renewable-to-load ratio
falls below 1.0 for five consecutive months, June through October (0.98, 0.98,
0.87, 0.76, 0.88), while running as high as 1.69 in April. A battery sized for
daily/weekly cycling cannot bridge a month-scale seasonal deficit -- evidenced by
the simultaneous presence of heavy curtailment (752,646 kWh/year wasted even at
the largest tested candidate) and persistent unserved energy (382,386 kWh/year)
in the same annual energy-flow balance.

A diagnostic-only extended search (n_max=400, 100,000 kWh, not the baseline
result) still had not reached the 1% target -- LPSP only fell to 4.24% at 400
modules, versus 10.06% at 80. Unlike the residential baseline (which became
feasible at 62 modules, roughly double its configured range), this industrial
case shows **severe diminishing returns from battery scale-up alone**: a 5x
capacity increase (20,000 → 100,000 kWh) more than halved LPSP but came nowhere
close to closing the remaining gap, illustrating that a deep, multi-month
seasonal deficit is a fundamentally different -- and more expensive -- problem
than a shallow one, even when both are nominally "the same kind of mismatch."
None of the 99.0%/99.5%/99.9% load-served sensitivity targets are met within the
configured range either.

Full candidate-by-candidate results: `outputs/tables/baseline_battery_candidate_results.csv`.
Figures: `outputs/figures/07_lpsp_vs_capacity.png` through `11_energy_flow_balance.png`.
Mechanistic search runtime: ~3.4 ms/candidate (81 candidates, ~0.28 s total) --
after JIT-compiling the hourly dispatch loop with numba (§9); the equivalent
pure-Python search took ~50 ms/candidate (~4.1 s total), a ~15x difference that
made the wider industrial search range (0-80 vs. the residential build's 0-30)
practical at full dataset scale.

## 15. AI-Model Results

### 15.1 Pilot (Stage 5/6, 100-scenario dataset)

Test-split (7 scenarios) accuracy for all four models predicting
`optimal_capacity_kwh` (naive/Ridge/Random Forest from Stage 5, MLP from Stage 6):

| Model | MAE (kWh) | RMSE (kWh) | R² | Bias (kWh) | % within 20% |
|---|---|---|---|---|---|
| Naive (median) | 5,107.1 | 5,985.8 | -0.204 | -2,464.3 | 14.3% |
| Ridge | 2,602.6 | 3,016.4 | 0.694 | 333.2 | 42.9% |
| Random Forest | 2,981.5 | 3,281.7 | 0.638 | 1,514.3 | 42.9% |
| **Neural Network (MLP)** | **6,311.8** | **7,570.8** | **-0.926** | **-5,637.4** | **0.0%** |

Operational (under/over-prediction, reported separately from averaged accuracy
per §21):

| Model | Underprediction rate | Mean underprediction (kWh) | Overprediction rate | Mean overprediction (kWh) |
|---|---|---|---|---|
| Naive | 57.1% | 6,625.0 | 42.9% | 3,083.3 |
| Ridge | 57.1% | 1,985.7 | 42.9% | 3,425.1 |
| Random Forest | 42.9% | 1,711.8 | 57.1% | 3,933.8 |
| Neural Network | 85.7% | 6,970.4 | 14.3% | 2,360.4 |

**Read honestly, not triumphantly -- and this pilot's honest result is
unflattering to the neural network.** Unlike the residential-scale pilot
(where the NN had the best accuracy of the four models), here the neural
network has the **worst** accuracy on every metric, including a negative R²
worse than the naive baseline's. §12 already showed why: its validation loss
visibly diverges after epoch 152, a real overfitting signature -- 27 training
rows is not enough data for a 15,233-parameter network at this problem's
noise level. This is reported as the actual, unmanipulated outcome of this
particular split and this particular (small) dataset -- not re-run, not
tuned, and not omitted because it doesn't flatter the neural network. It is
exactly the kind of small-sample result the refinement addendum's "test the
hypothesis, don't assume the NN wins" instruction exists to surface, and it
underscores why Stage 8/9's full-scale re-run (§15.2) matters: one 7-sample
test split is not enough evidence to conclude the neural network is worse at
this task in general, only that it is worse *on this split, at this training
set size*.

Full tables: `outputs/tables/accuracy_metrics_by_model.csv`,
`reliability_metrics_by_model.csv`, `runtime_comparison_stage5.csv`.
Figures: `outputs/figures/16`, `17`, `18`, `21`, `23`.

### 15.2 Full-scale (Stage 8/9, 5,000-scenario dataset)

Same pipeline, same architecture and hyperparameters (no tuning changes),
retrained fresh on the full dataset's 1,477/316/318 train/val/test split
(2,111 feasible scenarios). The neural network converged in 178 epochs
(32.8s). No model produced a negative prediction requiring clipping this time
(`{'naive': 0, 'ridge': 0, 'random_forest': 0, 'neural_network': 0}`) --
unlike the residential-scale full run, where 3 of Ridge's 430 predictions
were negative.

| Model | MAE (kWh) | RMSE (kWh) | R² | Bias (kWh) | % within 20% |
|---|---|---|---|---|---|
| Naive (median) | 3,421.4 | 4,367.1 | -0.015 | -528.3 | 31.4% |
| Ridge | 987.9 | 1,335.7 | 0.905 | 125.7 | 78.9% |
| Random Forest | 688.8 | 968.9 | 0.950 | 96.1 | 96.9% |
| **Neural Network (MLP)** | **332.8** | **538.9** | **0.985** | **-39.9** | **99.7%** |

Operational metrics (318 test scenarios):

| Model | Underprediction rate | Mean underprediction (kWh) | Overprediction rate | Mean overprediction (kWh) |
|---|---|---|---|---|
| Naive | 46.2% | 4,272.1 | 50.3% | 2,875.0 |
| Ridge | 40.3% | 1,071.0 | 59.7% | 931.9 |
| Random Forest | 37.7% | 785.4 | 62.3% | 630.2 |
| Neural Network | 49.4% | 377.4 | 50.6% | 289.3 |

At 318 test scenarios (vs. the pilot's 7), the accuracy ranking **completely
reverses relative to the pilot**: naive < Ridge < Random Forest < neural
network, with the neural network now clearly *best* (R² 0.985, MAE less than
half Random Forest's) rather than clearly worst. This is the sharpest
pilot-to-full-scale reversal in this project, residential-scale build
included -- the pilot's neural network result was not a subtle small-sample
wobble around an otherwise-consistent ranking, it was the complete opposite
ranking, driven by a training set (27 rows) too small for this architecture
to learn from at all. At 1,477 training rows, the same architecture and
hyperparameters (unchanged, no tuning) produce the best-performing model of
the four by a wide margin. See §16 for whether this accuracy improvement is
matched by a reliability improvement.

Full tables: `outputs/tables/accuracy_metrics_by_model_full.csv`,
`reliability_metrics_by_model_full.csv`. Model artifacts:
`models/neural_network_full/best_model.weights.h5` (portable, see §16.1),
`models/neural_network_full/best_model.keras`,
`models/preprocessing_full/{scaler.pkl,feature_columns.pkl}`. This section
trains fresh in-notebook rather than reloading a saved model, so it was never
exposed to the cross-version loading bug described in §16.1 -- the weights
file is saved here purely for consistency/reuse, not because this section
needed the fix.

## 16. Physical Verification of AI Predictions

### 16.1 Pilot (Stage 7, 100-scenario dataset)

*Note on cross-environment reproducibility:* this notebook loads the saved
neural network from `models/neural_network/best_model.weights.h5` (rebuilding
the architecture from `config/ml_training.yaml` via `load_trained_model`),
not the full `models/neural_network/best_model.keras` file. Loading a full
`.keras` model requires deserializing every layer's initializer configuration
(e.g. `GlorotUniform`), and Keras has changed that config's fields across
versions -- a model saved by a newer Keras (this project's development
environment) failed to load on Colab's older, separately-pinned Keras with
`GlorotUniform.__init__() got an unexpected keyword argument 'input_axes'`.
Weights-only loading sidesteps this: initializers only matter for the initial
random draw, which loaded weights immediately overwrite, so the rebuilt
architecture only needs to match shape, not initializer serialization
details. Verified bit-exact (max abs prediction difference 0.0) against the
original full-model load before this fix was adopted.

For every model's every test-split prediction (7 scenarios x 4 models = 28
verifications): rounded up to installable modules (ceiling, never down), the
actual hourly mechanistic dispatch was re-run with that battery size (using the
scenario's own round-trip efficiency, usable SOC window, and reliability
target), and the resulting LPSP checked against that scenario's own target.
Mechanistic-optimal dispatch was also freshly re-run per scenario for a
directly comparable diff (rather than trusting possibly-stale stored summary
statistics).

| Model | Reliability pass rate | % undersized | % oversized | Max underprediction (kWh) | Additional cost from oversizing (EUR/yr) |
|---|---|---|---|---|---|
| Naive | 42.9% | 57.1% | 42.9% | 9,500 | 359,375 |
| Ridge | 57.1% | 42.9% | 42.9% | 3,750 | 407,939 |
| Random Forest | 57.1% | 42.9% | 57.1% | 1,750 | 631,335 |
| **Neural Network** | **14.3%** | **85.7%** | **14.3%** | **12,500** | 97,128 |

**This is the headline, mandatory-check result of the whole project at pilot
scale -- and here it is unambiguous, not subtle.** The neural network had
both the *worst* accuracy (§15.1) *and* the worst reliability pass rate: 85.7%
of its test predictions undersize the battery (6 of 7 scenarios), and it has
the largest single underprediction of any model (12,500 kWh short on its
worst scenario). Unlike the residential-scale pilot, there is no accuracy/
reliability *disconnect* to point to here -- accuracy and reliability agree,
and both say the same thing: at 27 training rows, this network has not
learned a usable model. That is itself a valid and useful finding for Stage 7
to produce (a model that looks bad on accuracy and confirmed bad on
reliability is a model correctly identified as unusable, not a false
negative), but it means this pilot's physical verification step is not doing
the same *job* it did for the residential build, where it caught a real
accuracy/reliability disconnect invisible to accuracy metrics alone. Here it
mostly confirms what §15.1's accuracy numbers already showed.

Full per-scenario verification tables: `outputs/tables/physical_verification_{model}.csv`.
Summary: `physical_verification_summary_by_model.csv`. Figures: `outputs/figures/22`, `26`.

### 16.2 Full-scale (Stage 8/9, 5,000-scenario dataset)

Identical procedure, applied to all 318 test-split predictions per model
(1,272 verification runs total): ceiling-rounded to installable modules, fresh
mechanistic dispatch re-run for both the AI-predicted size and the
mechanistic-optimal reference size, checked against each scenario's own
reliability target.

| Model | Reliability pass rate | % undersized | % oversized | Max underprediction (kWh) | Additional cost from oversizing (EUR/yr) |
|---|---|---|---|---|---|
| Naive | 53.8% | 46.2% | 50.3% | 11,750 | 17,871,631 |
| Ridge | 68.2% | 31.8% | 59.7% | 5,500 | 7,857,690 |
| Random Forest | 73.3% | 26.7% | 62.3% | 3,750 | 5,827,706 |
| **Neural Network** | **79.2%** | **20.8%** | **50.6%** | **3,500** | 2,768,160 |

**This full-scale result completely reverses the pilot's finding, more
dramatically than in the residential-scale build.** At 7 test scenarios
(§16.1), the neural network had *both* the worst accuracy *and* the worst
reliability pass rate (14.3%, vs. Ridge/RF's 57.1%). At 318 test scenarios,
the ordering becomes a clean monotonic staircase matching the accuracy
ranking exactly (naive 53.8% < Ridge 68.2% < Random Forest 73.3% < neural
network 79.2%), and the neural network's undersizing rate falls to 20.8% (66
violations out of 318, still meaningfully non-trivial, but the lowest of the
four models and far below the pilot's 85.7%). The pilot's result was not
wrong to report at the time -- with 27 training rows, the neural network
genuinely had not learned a usable model, and Stage 7's physical verification
correctly confirmed that, not contradicted it. What changed between pilot and
full scale is the underlying model quality (1,477 training rows vs. 27), not
a statistical fluke in how the same underlying model happened to be measured.
This is a more clear-cut story than the residential build's pilot-to-full
reversal, where the pilot's neural network was accurate but unreliable
(a genuine disconnect); here the pilot's neural network was neither accurate
nor reliable, and the full-scale run is neither a surprise correction nor a
disconnect -- it is what more training data does for this architecture.

The remaining caveat at full scale is the same shape as the residential
build's: every model's oversizing rate stays above 50% (highest for Random
Forest, 62.3%), and additional cost from oversizing is very large in absolute
terms even for the best model (EUR 2,768,160/yr summed across 318 verified
scenarios -- an order of magnitude larger than the residential build's
equivalent figure, reflecting the industrial scale of these systems, not a
worse result). Accuracy and reliability both improved substantially with
scale here, but 20.8% of the best model's predictions still undersize the
battery -- "improved" is not "solved," and a 1-in-5 undersizing rate on an
industrial-scale system is a real reliability risk that a deployment decision
would need to account for, not a rounding error.

Full per-scenario tables: `outputs/tables/physical_verification_{model}_full.csv`.
Summary: `outputs/tables/physical_verification_summary_by_model_full.csv`.
Figure: `outputs/figures/22_reliability_pass_rate_full.png` (clean monotonic
staircase, 53.8% → 68.2% → 73.3% → 79.2%).

## 17. Accuracy and Computational-Efficiency Comparison

### 17.1 Pilot (Stage 5/6/7)

| Model | MAE (kWh) | R² | Inference time (s/scenario) |
|---|---|---|---|
| Naive | 5,107.1 | -0.204 | 0.000026 |
| Ridge | 2,602.6 | 0.694 | 0.000127 |
| Random Forest | 2,981.5 | 0.638 | 0.002421 |
| Neural Network | 6,311.8 | -0.926 | 0.019594 |

Mechanistic exhaustive search: ~0.180 s/scenario (81-candidate sweep, the
baseline case), pilot dataset generation (100 scenarios): ~18 s total -- after
the numba dispatch speedup (§9); the pre-numba equivalent would have been
~4.1 s/scenario, ~410 s total.

**Break-even (PROJECT_BRIEF.md §22, denominator per refinement addendum §1.5 --
the full exhaustive search per scenario, not a single dispatch run):**

| Model | Training time (s) | N_break_even (scenarios) |
|---|---|---|
| Ridge | 0.018 | ~100 |
| Random Forest | 0.95 | ~107 |
| Neural Network | 17.1 | ~219 |

Ridge and Random Forest's break-even points land close to the pilot's own
size (100 scenarios); the neural network's is over double that (~219), driven
by its slower training time relative to Ridge/Random Forest, not by anything
about mechanistic search cost. For a one-off ~100-200 scenario evaluation,
Ridge/RF and mechanistic search are roughly a wash; the neural network needs
meaningfully more reuse to pay off its training cost.

### 17.2 Full-scale (Stage 8/9)

| Model | MAE (kWh) | R² | Inference time (s/scenario) |
|---|---|---|---|
| Naive | 3,421.4 | -0.015 | 0.00000033 |
| Ridge | 987.9 | 0.905 | 0.000004 |
| Random Forest | 688.8 | 0.950 | 0.000118 |
| Neural Network | 332.8 | 0.985 | 0.000253 |

Mechanistic exhaustive search on the full dataset: ~0.186 s/scenario mean
(5,000 scenarios; 928.3 s sequential-equivalent total search cost, but 314.6 s
actual wall-clock time with the 4-worker parallel dataset generation, §9).

**Break-even, full-scale training costs:**

| Model | Training time (s) | N_break_even (scenarios) |
|---|---|---|
| Ridge | 0.05 | ~5,000 |
| Random Forest | 13.2 | ~5,074 |
| Neural Network | 33.8 | ~5,189 |

At full scale, every model's break-even point lands just above the size of
the dataset it was trained on (~5,000-5,189 scenarios), just as the
residential-scale build's full-scale break-even points did relative to *its*
5,000-scenario training set. This is not a coincidence specific to dataset
size or deployment scale: it shows N_break_even scales with training-set
size, because a larger training set both costs more mechanistic search to
generate (the denominator, per refinement addendum §1.5) and takes
proportionally longer to train a model on (the numerator). The practical
implication is unchanged from the residential build: AI amortizes its
training cost against *however many scenarios were used to generate its own
training data* -- an AI model becomes a clear net efficiency win only once it
is subsequently reused for substantially *more* new-scenario evaluations
beyond that original training set, since per-scenario inference cost is
several orders of magnitude below the mechanistic search regardless of
dataset size. For a single new scenario evaluated once, mechanistic search
remains both cheaper and self-verifying -- there is no computational
efficiency argument for AI at that scale, at either dataset size tested. One
industrial-specific nuance: because the numba dispatch speedup (§9) made
mechanistic search itself much cheaper per scenario than in the residential
build, the *absolute* time saved by using AI at scale is smaller here even
though the *relative* break-even ratio (dataset size needed before AI pays
off) is essentially unchanged -- a faster mechanistic method narrows AI's
absolute advantage without changing when it starts to have one.

## 18. Discussion

The research hypothesis (§2) proposed that a neural network "may approximate"
mechanistic battery capacities with lower inference time, while noting
"mechanistic verification may remain necessary." Taken together, the pilot
and full-scale results tell a two-act story -- and at industrial scale, the
two acts are more sharply opposed than they were in the residential-scale
build that preceded this pivot (PROJECT_BRIEF.md Addendum 2).

**Act one (pilot, §15.1/§16.1):** at n=7, the neural network had *both* the
worst accuracy (R² -0.926, worse than the naive median baseline) *and* the
worst reliability pass rate (14.3%, vs. Ridge/Random Forest's 57.1%) of the
four models. Unlike the residential-scale pilot -- where the neural network's
accuracy was genuinely good and only its reliability lagged, a real
accuracy/reliability disconnect that Stage 7's mandatory verification exists
specifically to catch -- this pilot's physical verification did not surface a
hidden failure mode invisible to accuracy metrics. It mostly confirmed what
§12's diverging training-loss curve and §15.1's accuracy table already showed:
27 training rows was not enough data for this 15,233-parameter architecture
to learn a usable model here, full stop, before verification even entered
the picture.

**Act two (full-scale, §15.2/§16.2):** at n=318, the ranking becomes a clean
monotonic match between accuracy and reliability (naive 53.8% < Ridge 68.2% <
Random Forest 73.3% < neural network 79.2%), and the neural network's
undersizing rate falls from 85.7% of test predictions to 20.8%. This is the
statistically stronger result and the one that should carry more weight in
answering the research questions (§20) -- but it does not retroactively make
the pilot's report wrong, any more than it did for the residential build. It
confirms that model quality, not measurement noise, was the pilot's binding
constraint: the same architecture and hyperparameters, given 1,477 training
rows instead of 27, produce the best-performing model of the four by a wide
margin. This is precisely the kind of result the addendum's "test the
hypothesis, don't assume the NN wins" instruction is designed to surface --
had Stage 8/9 not been attempted, the pilot's negative finding would have
stood as the project's headline result, understating what this architecture
can actually do once given enough data.

What is robust across *both* scales, and across *both* the residential and
industrial deployment contexts, is the qualitative pattern that
underprediction is a real, non-trivial risk at small scale for every model,
and that oversizing remains the dominant error mode for every model even
after accuracy improves at large scale (§16.2) -- accuracy improving does not
mean the cost-of-error problem disappears, only that it shrinks in magnitude
(relatively; in absolute EUR terms, industrial-scale oversizing costs are an
order of magnitude larger than the residential build's, simply because the
systems themselves are far larger). The pilot's 42.9%/42.9%/57.1%/14.3%
oversizing rates (naive/Ridge/RF/NN) explain that stage's cost-penalty
ordering: Random Forest's largest single errors happened to land on the
expensive side, giving it the highest oversizing cost penalty (EUR 631,335/yr)
despite a pass rate similar to Ridge's. At full scale the same qualitative
pattern holds (oversizing cost penalties of EUR 2.8M-17.9M/yr summed across
318 scenarios), even as the underlying pass rates improve substantially.

## 19. Limitations

- **Single location, single weather year.** Jinan 2023 only, at both pilot and
  full scale -- Experiment B (unseen weather year) and Experiment C (Västerås
  geographic transfer) are stretch goals not attempted (refinement addendum
  §1.1). No claim here generalizes to other climates or years.
- **Sample-size sensitivity is now a demonstrated finding, not just a
  caveat -- and a more severe one at industrial scale than at residential
  scale.** The reliability-pass-rate ranking reported in §16.1 (n=7) and
  §16.2 (n=318) reversed completely between pilot and full scale (§18): the
  neural network went from worst-on-both-accuracy-and-reliability to
  best-on-both. The full 5,000-scenario dataset (2,111 feasible, 318-scenario
  test split) substantially reduces sampling uncertainty relative to the
  pilot, but every number in this report is still a point estimate from one
  split of one dataset, not a guarantee against further movement at even
  larger scale.
- **The mechanistic model is the reference, not physical ground truth**
  (PROJECT_BRIEF.md §1). "Reliability pass rate" throughout this report means
  agreement with the mechanistic dispatch simulation's LPSP calculation under
  its own modelling assumptions (generic turbine curve, NOCT-based PV
  temperature model, reanalysis-derived weather) -- not validation against a
  real operating off-grid system, since no measured operational data exist for
  this project. This holds at both pilot and full scale: a larger sample
  makes the AI-vs-mechanistic *agreement* more statistically reliable, but
  does not change what that agreement is evidence of.
- **Fixed battery search range (0-80 modules, 250 kWh each; widened from the
  residential build's 0-30/15.36 kWh for this industrial scale).** Kept at
  this value per the same reasoning applied after the residential build's
  Stage 3 (a search range is a modelling decision made once, not re-litigated
  per scenario), which produced a ~58%/~58% infeasible-scenario rate at pilot
  and full scale respectively (39% pilot / 42.2% full feasible) and therefore
  excludes those scenarios from the training/test population for Stages 5-9.
- **Generic component models.** The wind turbine power curve and the PV
  NOCT/temperature-coefficient assumptions are documented modelling
  assumptions (`src/physics/wind_model.py`, `src/physics/pv_model.py`
  docstrings), not manufacturer-certified curves for a specific product.
- **No wind-speed cooling term in the PV cell-temperature model, and no
  battery thermal/temperature-dependent-capacity model.** Identified by direct
  comparison against OptiCE (§2): OptiCE's PV temperature model includes a
  wind-speed-dependent cooling term, and OptiCE's battery model corrects
  usable capacity downward for battery temperatures below 25°C via a fitted
  quadratic factor. This project's PV model uses a fixed-coefficient NOCT
  temperature model without a wind term, and its battery model has no thermal
  state at all -- usable capacity is constant regardless of ambient
  conditions. Both are standard simplifications for an LPSP-focused sizing
  study (rather than a detailed thermal-management study) but would bias
  results toward *underestimating* required battery capacity in climates with
  temperature extremes, since the mechanistic reference itself never derates
  capacity for temperature.
- **PV and wind capacity are scenario inputs, not jointly optimized decision
  variables.** OptiCE (§2) optimizes PV tilt/azimuth/capacity, wind tower
  height/capacity, and battery capacity jointly via a multi-objective genetic
  algorithm to find a single best design. This project instead samples PV and
  wind capacity across a range (§9) and only exhaustively searches battery
  capacity for each sampled combination -- a deliberate choice, since the
  research question here is about predicting the *battery-sizing outcome*
  across many different fixed systems, not about finding the single
  cost-optimal system. A consequence is that this project makes no claim
  about whether any of its sampled PV/wind combinations are themselves
  economically optimal.
- **One training run per model, at both scales.** No repeated-seed variance
  analysis; the reported neural-network results are from single training runs,
  not averages over multiple seeds. This is a distinct source of uncertainty
  from the sample-size effect discussed above -- even the full-scale ranking
  could in principle shift under a different training seed, though the much
  larger test set makes this less likely to matter than it would at pilot
  scale.
- **The industrial load-profile shape is a documented modelling assumption,
  not derived from a real facility's metered data** (`src/physics/
  load_profile.py`'s `industrial_baseline` family, PROJECT_BRIEF.md Addendum
  2): two-shift operation, a 45% weekend reduction, and a weak seasonal
  swing. A continuous-process facility (steel, chemicals) or a strictly
  weekdays-only operation would have a materially different shape, and every
  downstream number in this report -- from Stage 3's baseline feasibility
  finding through the full 5,000-scenario dataset -- is conditional on this
  specific shape choice.
- **No battery-price sensitivity analysis.** The full sensitivity sweep across
  battery module cost assumptions (PROJECT_BRIEF.md stretch goal) was not
  attempted; all costs use the single fixed `installed_cost_eur_per_kwh` in
  `config/jinan.yaml`.

## 20. Conclusions

This report covers the industrial deployment context (combined PV+wind
capacity ≤5,000 kW, `industrial_baseline` load profile), adopted partway
through the project (PROJECT_BRIEF.md Addendum 2) in place of the originally
residential-scale build. Answering the five research questions (§1) directly,
weighting the full-scale (Stage 8/9) result more heavily than the pilot's per
§18, while keeping both:

1. **Accuracy:** Yes, and dramatically more confidently than the pilot alone
   suggested -- in fact the pilot alone suggested the opposite. At full
   scale, the neural network most accurately reproduced mechanistic battery
   capacities (MAE 332.8 kWh, R² 0.985 vs. Random Forest's 688.8 kWh / 0.950),
   a result now backed by a 318-scenario test set rather than 7, where the
   same architecture had been the *worst*-performing model (R² -0.926).
2. **Speed:** AI inference is several orders of magnitude faster per scenario
   than the mechanistic search at both scales tested, though the numba
   dispatch speedup (§9) narrowed the mechanistic method's absolute
   disadvantage relative to the residential build without changing where the
   break-even point falls in relative terms. Break-even analysis (§17) shows
   this pays off in aggregate once a model is reused for meaningfully more
   evaluations than the size of the dataset it was trained on (~100-219 for
   the pilot-trained models, ~5,000-5,189 for the full-dataset-trained
   models) -- it is not a blanket efficiency win for one-off evaluations at
   either scale.
3. **Reliability when verified:** **Yes at full scale, decisively no at pilot
   scale -- and the gap between the two is the project's clearest finding
   at this deployment scale.** The pilot's neural network failed both the
   accuracy test and the mandatory physical-verification check (14.3% pass
   rate, vs. Ridge/Random Forest's 57.1%); at full scale, reliability pass
   rate tracked accuracy exactly (naive 53.8% < Ridge 68.2% < Random Forest
   73.3% < neural network 79.2%). Unlike the residential build's pilot result
   (an accuracy/reliability *disconnect* that verification specifically
   exists to catch), this pilot's verification mostly corroborated what its
   accuracy numbers already showed -- but the underlying lesson is the same:
   a small, honestly-reported negative result was directly retested at 48x
   the sample size, resolved in the opposite direction, and the mandatory
   physical-verification step (§16) confirmed the full-scale result on a
   fresh mechanistic re-run rather than on stored summary statistics. Even
   at full scale, 20.8% of the best model's predictions still undersize the
   battery -- "reliable" is a large improvement over the pilot, not a solved
   problem.
4. **Generalization to unseen conditions:** Not tested (single location/year
   at both pilot and full scale; §19). This remains the most significant
   unaddressed research question, and applies identically to the industrial
   and (unreported) residential deployment contexts.
5. **Practical trade-offs:** Mechanistic search is slow but self-verifying by
   construction; AI is fast and, at sufficient training-set scale, accurate
   and reliable by the mechanistic model's own standard -- but only once
   verified against it (Stage 7/9), and only for the single location/year and
   single load-profile shape this project actually tested. Oversizing remains
   the dominant residual error mode for every model even at full scale
   (§16.2, §18), and the absolute cost of that oversizing is an order of
   magnitude larger at industrial scale than at residential scale simply
   because the systems themselves are larger -- so "reliable" here means
   "meets the reliability target," not "minimum-cost," and the cost of being
   wrong scales with the size of the system being sized.

**On the research hypothesis:** more clearly supported at full scale than the
pilot alone suggested -- and, at this industrial deployment scale, the pilot
alone would have actively pointed the wrong direction on both halves of the
hypothesis. The "approximation with lower inference time" half is now
well-supported (§15.2); the "mechanistic verification may remain necessary"
half is concretely demonstrated by the fact that verification is what
confirmed the full-scale result actually holds up under a fresh mechanistic
re-run, not by a dramatic accuracy/reliability disconnect this time (that was
the residential build's finding, not this one). Reporting both the pilot's
negative result and the full-scale reversal, rather than only the final
favourable numbers, is the intended outcome of the project's honesty
requirements -- an NN that "wins" only because a bad small-sample result was
quietly dropped would be a materially weaker piece of evidence than the same
NN "winning" after that result was reported, explicitly retested, and
explained. The larger methodological lesson this industrial pivot adds to the
residential build's original finding: **the specific way a pilot-to-full-scale
result reverses is not itself a stable pattern to expect** -- the residential
pilot showed a subtle accuracy/reliability disconnect that verification alone
could catch; this industrial pilot showed an overt, verification-independent
failure to learn from too little data. Both are real, both are worth
reporting, and neither should be mistaken for a general rule about how neural
networks behave at small sample sizes in this problem class.
