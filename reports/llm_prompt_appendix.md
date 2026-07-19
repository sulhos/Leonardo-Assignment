# LLM Prompt Appendix

This file records the complete original build prompt given to Claude Code for this
project (PROJECT_BRIEF.md §29), plus a running log of subsequent prompts,
corrections, and responses as the project develops.

---

## 1. Original Build Prompt (verbatim)

The complete prompt is also kept at the project root as `PROJECT_BRIEF.md` (the
canonical, always-current copy consulted by each stage). It is reproduced below for
archival purposes as required by PROJECT_BRIEF.md §29.

# Project Brief for Claude Code: Battery Sizing — Mechanistic vs Neural Network Comparison

This file is the build instruction for Claude Code. It consists of two parts:

1. **Refinement addendum** (read this first — it overrides/clarifies the original spec below)
2. **Original detailed specification** (the full technical scope, kept as the canonical reference)

---

## PART 1: REFINEMENT ADDENDUM (read first)

This addendum was produced after reviewing the original spec for feasibility as a bachelor
thesis project with a multi-week timeline. Follow it; it takes precedence where it conflicts
with Part 2.

### 1.1 Scope tier: build an MVP core, treat the rest as stretch goals

The full spec in Part 2 (33 sections) is sized for a funded research project or a multi-person
capstone, not a single multi-week bachelor thesis build. Do not attempt all of it end to end.

**Core scope (build and fully validate this first):**
- Stage 1: Project scaffold
- Stage 2: Mechanistic components (weather, load, PV, wind, battery dispatch, energy-balance tests)
- Stage 3: Baseline battery optimization (Jinan 2023 only)
- Stage 4: Pilot scenario dataset (~100 scenarios, single location, single weather year is
  acceptable for the pilot)
- Stage 5: Baseline ML models (naive, linear/ridge, random forest or gradient boosting)
- Stage 6: Neural network (MLP)
- Stage 7: Physical AI verification (mandatory — do not skip this even under time pressure;
  it's the most important scientific check in the whole project)

**Stretch goals (attempt only after Core scope is complete, tested, and documented):**
- Stage 8: Full 5,000-scenario dataset (if time allows, scale up incrementally — e.g. 100 → 500
  → 1,000 → full — and re-validate at each step rather than jumping straight to 5,000)
- Experiment B (unseen weather year holdout) — requires multiple weather years downloaded
- Experiment C (Västerås geographic transfer) — explicitly optional in the original spec; keep
  it that way
- Full battery-price sensitivity (400/550/700 EUR/kWh) and reliability-target sensitivity sweeps
  beyond the baseline case
- Full polish pass on both written reports and the presentation outline

At every stage boundary, report honestly what was completed, what was cut, and what remains a
stretch goal. Do not silently expand scope to "finish everything" if it compromises testing or
correctness — a smaller, correctly validated core is more valuable than a larger, untested one.

### 1.2 Neural network framework: pin it explicitly

The original spec (Section 18) describes early stopping, best-model checkpointing, and negative-
prediction prevention without naming a framework. Default to **Keras/TensorFlow** for the MLP,
since those requirements map directly onto built-in Keras callbacks (`EarlyStopping`,
`ModelCheckpoint`) and a simple output-layer clamp (e.g. `ReLU` activation on the output, or
post-hoc `np.maximum(pred, 0)`) satisfies the non-negativity requirement cleanly. If PyTorch is
preferred instead, that's fine too, but say so explicitly in `src/ai/neural_network.py`'s
docstring and implement early stopping and checkpointing manually (PyTorch has no built-in
equivalent). Pick one and be consistent — don't mix frameworks across `src/ai/`.

### 1.3 Add a literature/related-work section to the technical report

Section 29's technical report outline jumps from "Introduction" straight to "Research objective"
with no related-work section. Add one: a short "Related Work" section (can be brief — a page or
so) situating this project against existing battery-sizing approaches (LPSP-based sizing methods,
tools like HOMER, and prior ML-for-sizing literature). This is normally expected in a bachelor
thesis rubric even in a mostly technical/software-engineering project. Do not fabricate citations
— use real, findable sources (web search is fine for this) or mark it clearly as "citations to be
completed" if sources aren't gathered yet.

### 1.4 Validate the NASA POWER API early, before building on top of it

Before writing the full weather pipeline in Stage 2, do a small standalone script/notebook cell
that hits the NASA POWER hourly API for a short date range around the Jinan coordinates and
confirms: what time standard the timestamps are actually in, what units the solar/wind/temperature
fields are actually returned in (don't trust assumed units — check them against the API docs and
sanity-check magnitudes), and whether hourly wind data is provided at 10m or 50m reference height
for the specific parameter used. Write these findings into
`data/README.md` before proceeding. If the hourly endpoint has coverage gaps or surprises for
either location, document them and adjust the plan rather than discovering it midway through
Stage 4.

### 1.5 Clarify the break-even runtime formula

In Section 22, `N_break_even`'s denominator (`mechanistic time per scenario - AI inference time
per scenario`) should use "mechanistic time per scenario" = the full exhaustive battery-module
search (all N_max candidates) for that scenario, not a single dispatch run — that's what the
mechanistic method actually costs per scenario in real use. State this explicitly wherever the
formula is computed or reported, so the comparison isn't accidentally overstated.

### 1.6 General reminders carried over from the original spec (worth re-emphasizing)

- Test the hypothesis; do not assume the neural network wins. Report negative or inconclusive
  results honestly, including in the final completion report.
- The mechanistic model is a reference method, not physical ground truth — never describe AI
  agreement with it as "validation against reality."
- Underprediction of battery capacity is a reliability risk, not just an error metric — always
  report it separately from averaged error metrics, and always run the physical verification step
  (Stage 7 / Section 20) before drawing conclusions about AI performance.
- Fit all scalers on training data only; avoid target leakage (no reference optimal capacity,
  module count, or cost as an ML input feature); use grouped/temporal splits, never a naive
  random split across near-identical scenarios.
- Keep notebooks thin — they call functions from `src/`, they don't contain the logic.
- Commands should work from the project root; use YAML config, type hints, logging (not print),
  fixed seeds, pathlib.

---

## PART 2: ORIGINAL DETAILED SPECIFICATION (canonical technical reference)

You are developing a complete bachelor-level research project titled:
"Comparison of Neural-Network and Mechanistic Models for Battery Sizing in an Off-Grid PV–Wind Energy System"

Your role is to build the complete Python project, including:
1. A physics-based/mechanistic battery-sizing model
2. A labelled scenario dataset generated using the mechanistic model
3. Conventional machine-learning baseline models
4. An artificial neural-network battery-sizing model
5. A fair comparison of accuracy, computational efficiency, reliability and generalizability
6. Jupyter notebooks, tests, figures, result tables and report structures

Do not create one large notebook containing all the logic. Build a modular, tested Python project. The notebooks should call reusable functions from the source modules.

Do not assume in advance that the neural network is more accurate or more efficient. Test the research hypothesis objectively and report negative or inconclusive findings honestly.

==================================================
1. RESEARCH OBJECTIVE
==================================================
The objective is to compare a mechanistic battery-sizing method with an AI-based neural-network method for an off-grid PV–wind system.
The project must investigate:
1. How accurately can a neural network reproduce battery capacities obtained from a mechanistic optimization?
2. How much faster is neural-network inference than a complete mechanistic battery-size search?
3. Does the AI-selected battery achieve the required off-grid reliability when verified using hourly mechanistic simulation?
4. How well does the AI model generalize to unseen weather years, load profiles and system configurations?
5. What are the practical advantages, limitations and risks of each approach?
The mechanistic model is the reference method, not necessarily physical truth.
If the neural network is trained using mechanistic outputs, describe its performance as agreement with the mechanistic reference. Do not claim validation against reality unless measured operational data are available.

==================================================
2. RESEARCH HYPOTHESIS
==================================================
Use the following balanced hypothesis:
"The neural-network model may approximate the battery capacities obtained from mechanistic optimization with substantially lower inference time when many system configurations must be evaluated. However, its accuracy and reliability depend on the representativeness of the training data, and mechanistic verification may remain necessary for unseen conditions."
The analysis must test this hypothesis rather than assume it is correct.

==================================================
3. SYSTEM BOUNDARY
==================================================
The system contains:
- Fixed photovoltaic generation capacity
- Fixed wind-generation capacity
- Battery storage
- Electrical load
- No grid connection
- No diesel or other backup generator
- No grid import
- No grid export
- Hourly simulation
- One complete local calendar year per scenario
PV and wind capacities are scenario inputs. They are not decision variables within an individual optimization.
The mechanistic optimization changes only the battery capacity.

==================================================
4. BASELINE CASE
==================================================
Location: Jinan, China
Coordinates: Latitude 36.65, Longitude 117.12
Local timezone: Asia/Shanghai
Baseline simulation year: 2023
Required number of local hourly time steps: 8,760
Load type: Synthetic residential or small residential-community load
Target annual electricity consumption: Approximately 30,000 kWh/year
Maximum hourly load: Not greater than 15 kW
Fixed baseline PV capacity: 20 kWp
Fixed baseline wind capacity: 10 kW
System type: Completely off-grid
Baseline reliability requirement: At least 99% of annual load energy served

Also include configuration support for:
Location: Västerås, Sweden
Approximate coordinates: Latitude 59.61, Longitude 16.54
Timezone: Europe/Stockholm
Do not use Västerås as the primary baseline unless required. Keep it available for geographic-transfer or sensitivity testing.

==================================================
5. BATTERY REFERENCE
==================================================
Use a configurable LFP battery based on the BYD Battery-Box Premium LVL.
Reference specifications:
- Chemistry: lithium iron phosphate, LFP
- Capacity per module: 15.36 kWh
- Rated DC power per module: 12.8 kW
- Approximate round-trip efficiency: 95%
- Charge efficiency: sqrt(0.95)
- Discharge efficiency: sqrt(0.95)
- Minimum operational SOC: 10%
- Maximum operational SOC: 95%
- Initial SOC: 50%
- Assumed economic lifetime: 10 years
- Project lifetime: 20 years
- Baseline installed cost: 550 EUR/kWh
- Cost sensitivity: 400, 550 and 700 EUR/kWh
- Candidate module counts: configurable integers, initially 0–30
Clearly distinguish: 1) Manufacturer specifications, 2) Modelling assumptions, 3) Economic assumptions.
Do not present 550 EUR/kWh as an official BYD retail price.
The model must support changing battery specifications through YAML configuration files.

==================================================
6. TEMPERATURE ASSUMPTION
==================================================
Assume the battery is installed in a temperature-managed enclosure.
Implement temperature-related battery derating as an optional configurable feature.
The model must allow:
- Charge-power derating below a configurable temperature
- Discharge-power derating if configured
- Optional thermal-management auxiliary load
- Normal operation when enclosure temperature remains within the permitted range
Do not create undocumented temperature rules.
Clearly state whether thermal-management energy is included or excluded in each experiment.
For the initial comparison, it is acceptable to use a controlled battery temperature and treat detailed thermal behaviour as a sensitivity analysis.

==================================================
7. WEATHER DATA
==================================================
Use the official NASA POWER hourly API to retrieve solar, temperature and wind data.
Requirements:
- Download enough dates around the selected calendar year to cover the complete local year after timezone conversion.
- Identify the time standard returned by the API.
- Convert timestamps correctly to the selected local timezone.
- Filter the selected local calendar year.
- Confirm exactly 8,760 hourly records for non-leap years.
- Handle daylight-saving-time complications correctly for Västerås.
- Detect missing timestamps.
- Detect duplicate timestamps.
- Check units explicitly.
- Save original downloaded data in data/raw.
- Save cleaned data in data/processed.
- Cache successful downloads.
- Make unit tests independent of internet access.
- Record the API URL, variables, units, coordinates, time standard, download date and processing steps.
- Never fabricate weather observations when an API request fails.
- Provide instructions for supplying a manual CSV if downloading fails.
Use multiple weather years if the API and project schedule permit. For example, use several years for scenario generation and hold out one complete year for testing.
Do not randomly divide hourly observations from the same year between training and testing.

==================================================
8. SYNTHETIC LOAD MODEL
==================================================
Build a reproducible hourly load-profile generator.
It must support:
- Morning demand peak
- Evening demand peak
- Lower nighttime demand
- Weekday/weekend differences
- Seasonal variation
- Random but reproducible variation
- Different peak-to-average ratios
- Different annual-consumption levels
- Different residential demand patterns
The baseline profile should:
- Consume approximately 30 MWh/year
- Remain at or below 15 kW peak demand
- Contain exactly 8,760 local hourly values
- Use a fixed random seed
- Be saved as CSV
Generate multiple load-profile families for machine-learning dataset construction. Do not create thousands of profiles differing only by a simple multiplication factor.
Include meaningful variation in timing, seasonality, morning/evening peaks, weekends and stochastic behaviour.
Produce:
- Representative winter-week plot
- Representative summer-week plot
- Annual load-duration curve
- Monthly energy-demand plot
- Summary statistics

==================================================
9. PV MODEL
==================================================
Implement an hourly mechanistic PV model for a configurable fixed PV capacity.
Use pvlib when technically appropriate.
Include:
- Solar position
- Plane-of-array irradiance
- Module or cell-temperature effect
- Inverter/conversion efficiency
- Configurable system losses
- Nighttime zero generation
- Output clipping at the configured system limit
- Clear treatment of tilt and azimuth
Check NASA POWER radiation units before using them.
Do not assume hourly radiation is in W/m² without verifying and converting it.
If required data are unavailable and a simplified PVWatts-style model is used:
- Document the equations
- Document all units
- Document assumptions
- Explain limitations
The output must always satisfy: 0 <= PV output <= configured maximum PV AC output

==================================================
10. WIND MODEL
==================================================
Implement an hourly mechanistic wind-generation model.
The baseline wind-turbine rated power is 10 kW.
Include:
- Wind speed from NASA POWER
- Documented reference height
- Configurable turbine hub height
- Power-law height adjustment
- Configurable wind-shear exponent
- Cut-in wind speed
- Rated wind speed
- Cut-out wind speed
- Transparent normalized turbine power curve
- Output constraint between zero and rated power
A suitable simplified power curve is:
P(v) = 0                                  for v < v_cut_in
P(v) = P_rated × (v³ - v_cut_in³) / (v_rated³ - v_cut_in³)   for v_cut_in <= v < v_rated
P(v) = P_rated                            for v_rated <= v <= v_cut_out
P(v) = 0                                  for v > v_cut_out
Keep all parameters configurable.
Explain the limitations of: reanalysis wind data, spatial resolution, height extrapolation, generic turbine power curves, missing turbulence and wake effects.

==================================================
11. MECHANISTIC BATTERY DISPATCH
==================================================
For every hour:
1. PV and wind generation directly supply the load.
2. Renewable surplus charges the battery.
3. During a renewable deficit, the battery discharges.
4. Remaining surplus becomes curtailed energy.
5. Remaining deficit becomes unserved energy.
Enforce: minimum and maximum SOC, charge efficiency, discharge efficiency, battery-capacity limit, charge-power limit, discharge-power limit, no simultaneous charging and discharging, no grid exchanges, no backup generator, no negative physical energy flows.
Use an internally consistent convention for energy entering the battery, energy stored, energy leaving the battery, battery losses. Document that convention.
The battery-state equation should follow the equivalent of:
E[t+1] = E[t] + charge_efficiency × charging_energy[t] - discharging_energy_delivered[t] / discharge_efficiency
The model must record: direct renewable supply, battery charging, battery discharging, curtailed energy, unserved energy, battery losses, SOC, renewable generation, load served.
Avoid initial-SOC bias. Implement one of these methods:
A. Repeated-year simulation until starting and ending SOC converge, or
B. A cyclic SOC solution
Explain and test the selected method.

==================================================
12. MECHANISTIC BATTERY OPTIMIZATION
==================================================
Use discrete exhaustive enumeration.
Battery module count: n = 0, 1, 2, ..., N_max
Battery capacity: E_battery = n × 15.36 kWh
For each candidate: 1) Run the complete hourly dispatch. 2) Calculate reliability. 3) Calculate curtailment. 4) Calculate battery utilization. 5) Calculate cost. 6) Record computational time.
Primary reliability metric: LPSP = total unserved energy / total load energy
Primary objective: Select the least-cost battery configuration satisfying LPSP <= 0.01
Because battery price increases with capacity, this will generally be the smallest feasible modular battery.
Also test: 99.0%, 99.5%, 99.9% load served.
If no candidate satisfies the reliability requirement, return: "No feasible battery size was found within the tested range for the selected fixed PV and wind capacities."
Do not call an infeasible candidate optimal.
If zero unserved energy cannot be obtained because of insufficient PV and wind production, clearly identify renewable-generation inadequacy rather than assuming that an infinitely large battery will solve it.

==================================================
13. MECHANISTIC PERFORMANCE METRICS
==================================================
For every candidate calculate: number of battery modules, nominal battery capacity, usable battery capacity, rated battery power, annual load, peak load, annual PV production, annual wind production, total renewable production, direct renewable consumption, load served, load-served fraction, LPSP, unserved energy, hours with unmet load, curtailed renewable energy, renewable utilization, battery charge energy, battery discharge energy, battery losses, minimum/maximum/mean SOC, battery throughput, equivalent full cycles, initial battery cost, replacement cost, present-value cost, equivalent annual cost, cost per kWh of load served, mechanistic optimization runtime.
Because the system has no grid or fossil generator, all served electricity is renewable. Do not misleadingly treat the renewable share of served energy as the main variable. Use load-served fraction, renewable utilization, curtailed energy, reliability.

==================================================
14. MACHINE-LEARNING DATASET GENERATION
==================================================
The neural network cannot be trained using only one year and one system configuration.
Use the mechanistic model to generate a labelled scenario dataset. Each dataset row represents one complete annual scenario.
Possible scenario inputs include: location, weather year, PV capacity, wind capacity, annual load, peak load, load factor, morning-peak magnitude, evening-peak magnitude, seasonal-load factor, weekend factor, renewable-to-load ratio, PV capacity factor, wind capacity factor, reliability target, battery efficiency, usable SOC fraction, temperature assumptions.
The target label is: Mechanistically optimized battery capacity in kWh
Also save: optimal number of modules, whether the scenario was feasible, reference LPSP, reference annualized cost, mechanistic optimization runtime.
Initial scenario ranges may include: PV capacity 10–40 kWp, wind capacity 5–25 kW, annual load 10–50 MWh/year, peak load 5–20 kW, reliability target 99.0–99.9%, round-trip battery efficiency 88–97%, usable SOC window 70–90%, multiple weather years, multiple load-pattern families.
Review these ranges for physical consistency. Do not generate unrealistic combinations blindly. For example: annual consumption must be compatible with peak demand; candidate renewable production must be checked against annual demand; clearly label infeasible scenarios; do not replace infeasible labels with arbitrary maximum battery sizes.
Start with a smaller pilot dataset, such as 100 scenarios, to verify correctness.
After validation, make the desired full dataset size configurable, for example: pilot 100, development 1,000, full experiment 5,000 or more, subject to runtime.
Implement parallel scenario generation if safe and beneficial, but preserve reproducibility.
Cache completed scenarios so interrupted generation can resume.
Record random seeds and configuration hashes.

==================================================
15. FEATURE ENGINEERING
==================================================
Create annual features that summarize the time-series conditions relevant to battery sizing.
LOAD FEATURES: annual load energy, peak load, mean load, load standard deviation, load factor, peak-to-average ratio, daytime-load share, nighttime-load share, seasonal variability, monthly maximum and minimum demand.
GENERATION FEATURES: PV capacity, wind capacity, annual PV generation, annual wind generation, total renewable generation, PV capacity factor, wind capacity factor, renewable-to-load ratio, monthly renewable variability, daily renewable variability.
NET-LOAD FEATURES: net load = load - PV generation - wind generation. Include maximum positive net load, mean positive net load, standard deviation of net load, total annual energy deficit before storage, total annual surplus before storage, maximum consecutive deficit duration, maximum cumulative deficit event, number of deficit hours, number of surplus hours, selected net-load quantiles, worst monthly renewable-to-load ratio.
SYSTEM FEATURES: reliability target, round-trip efficiency, usable SOC fraction, battery power-to-energy assumptions, location identifier if multiple locations are used.
Avoid data leakage. Do not include: mechanistically optimized battery capacity as an input, results that could only be known after choosing the optimal battery, reference optimal-module count as an input, reference optimal cost as an input.
Fit all feature scalers only on the training data.

==================================================
16. DATASET SPLITTING
==================================================
Avoid a naive random split that places nearly identical scenarios in both training and testing datasets.
Implement at least two evaluation strategies.
EXPERIMENT A: IN-DOMAIN TEST — Train, validate and test on different scenario configurations. Group related scenario variants so nearly identical cases remain in one split. Suggested proportions: 70% training, 15% validation, 15% testing.
EXPERIMENT B: UNSEEN WEATHER TEST — Hold out one complete weather year from training. Train using other weather years. Test only on the unseen year.
OPTIONAL EXPERIMENT C: GEOGRAPHIC TRANSFER — Train primarily on Jinan scenarios. Test on Västerås scenarios. Clearly identify this as an out-of-distribution transfer test. Do not expect strong transfer performance without geographic diversity in training.
Save split identifiers so all models use exactly the same training, validation and test observations.

==================================================
17. STATISTICAL AND MACHINE-LEARNING BASELINES
==================================================
Do not compare only the mechanistic model and neural network. Implement at least: 1) Naive/rule-based baseline, 2) Linear regression or ridge regression, 3) Random forest or gradient-boosting regression, 4) Neural-network regression.
The purpose is to determine whether the neural network adds value beyond simpler models.
Use the same features, dataset splits, test observations, evaluation metrics.
Perform reasonable hyperparameter selection using only training and validation data. Do not tune models using the test set.

==================================================
18. NEURAL-NETWORK MODEL
==================================================
Use a feedforward multilayer perceptron as the primary neural network.
A suitable initial architecture is: input layer matching the number of features, Dense 128 ReLU, Dropout 0.10, Dense 64 ReLU, Dense 32 ReLU, output layer one continuous value.
Target: Required battery capacity in kWh
Requirements: standardize numeric inputs, fit scalers only using training data, use an appropriate regression loss (MAE/MSE/Huber), monitor validation loss, use early stopping, save the best model, save preprocessing objects, record model architecture, record random seeds, plot training and validation loss, prevent negative capacity predictions, keep training reproducible as far as reasonably possible.
Do not implement an LSTM unless an additional experiment is explicitly justified. The primary task is tabular regression using engineered annual features.

==================================================
19. CONVERTING AI OUTPUT TO A REAL BATTERY
==================================================
The neural network predicts a continuous capacity: E_AI_predicted
Convert it to a commercially installable number of modules: N_AI = ceiling(E_AI_predicted / 15.36)
Installed AI-selected capacity: E_AI_installed = N_AI × 15.36
Never round down. Store both the raw continuous prediction and the rounded installed capacity.
Constrain predictions to the supported system range and flag extrapolations.

==================================================
20. PHYSICAL VERIFICATION OF AI PREDICTIONS
==================================================
Prediction-error metrics are not sufficient.
For every AI prediction in the test dataset: 1) Convert the prediction to an installed module count. 2) Insert that battery size into the hourly mechanistic dispatch model. 3) Run the complete annual simulation. 4) Calculate the achieved LPSP. 5) Determine whether the reliability constraint is satisfied. 6) Calculate cost and curtailment. 7) Compare against the mechanistic optimum.
Report: percentage of AI selections satisfying reliability, percentage undersized, percentage oversized, mean excess capacity, maximum underprediction, additional cost caused by overprediction, reliability violations caused by underprediction, difference in curtailed energy, difference in annualized cost.
This operational verification is mandatory.

==================================================
21. ACCURACY METRICS
==================================================
For continuous battery-capacity predictions calculate: MAE, RMSE, R², median absolute error, mean absolute percentage error (careful handling of zero targets), maximum absolute error, mean signed error/bias, percentage within ±5%/±10%/±20%.
Also calculate asymmetric operational metrics: underprediction rate, overprediction rate, mean underprediction, maximum underprediction, mean overprediction, reliability pass rate after mechanistic verification.
Battery underprediction is more dangerous than overprediction, so do not hide it inside average error metrics.

==================================================
22. COMPUTATIONAL-EFFICIENCY COMPARISON
==================================================
Measure runtime using a consistent timing method.
MECHANISTIC METHOD: runtime for one full battery-size search, runtime per candidate battery, runtime for multiple scenarios, approximate memory use if practical.
AI DEVELOPMENT: mechanistic dataset-generation time, feature-generation time, neural-network training time, model-selection time.
AI APPLICATION: inference time for one scenario, batch inference time, optional mechanistic verification time.
Clearly distinguish: one-time training cost, per-scenario inference cost, end-to-end development cost.
Calculate an approximate break-even number of scenario evaluations:
N_break_even = (dataset generation time + training time) / (mechanistic time per scenario - AI inference time per scenario)
Only calculate this if the denominator is positive and the assumptions are meaningful.
Do not claim that AI is computationally superior for a single case merely because inference is fast.

==================================================
23. ECONOMIC ANALYSIS
==================================================
Implement: initial battery investment, replacement in year 10, configurable real discount rate, present value, equivalent annual cost, cost per kWh of load served, battery-cost sensitivity, cost penalty caused by AI oversizing, reliability risk caused by AI undersizing.
Do not include grid-import, grid-export or fossil-fuel revenues. Keep economic parameters configurable.

==================================================
24. PROJECT ARCHITECTURE
==================================================
Create this structure:
battery-sizing-ai-comparison/
├── README.md
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── config/
│   ├── jinan.yaml
│   ├── vasteras.yaml
│   ├── scenario_generation.yaml
│   └── ml_training.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── scenarios/
│   ├── ml_dataset/
│   └── README.md
├── notebooks/
│   ├── 01_mechanistic_model.ipynb
│   ├── 02_scenario_dataset.ipynb
│   ├── 03_ml_baselines.ipynb
│   ├── 04_neural_network.ipynb
│   └── 05_model_comparison.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── weather.py
│   │   └── validation.py
│   ├── physics/
│   │   ├── __init__.py
│   │   ├── load_profile.py
│   │   ├── pv_model.py
│   │   ├── wind_model.py
│   │   ├── battery.py
│   │   ├── dispatch.py
│   │   ├── optimization.py
│   │   ├── economics.py
│   │   └── metrics.py
│   ├── scenarios/
│   │   ├── __init__.py
│   │   ├── sampling.py
│   │   ├── scenario_runner.py
│   │   └── dataset_builder.py
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── features.py
│   │   ├── splitting.py
│   │   ├── preprocessing.py
│   │   ├── baselines.py
│   │   ├── neural_network.py
│   │   ├── evaluation.py
│   │   └── physical_verification.py
│   └── visualization/
│       ├── __init__.py
│       └── plotting.py
├── models/
│   ├── neural_network/
│   ├── baselines/
│   └── preprocessing/
├── tests/
│   ├── test_weather.py
│   ├── test_load_profile.py
│   ├── test_pv_model.py
│   ├── test_wind_model.py
│   ├── test_battery.py
│   ├── test_dispatch.py
│   ├── test_energy_balance.py
│   ├── test_optimization.py
│   ├── test_features.py
│   ├── test_splitting.py
│   └── test_ai_verification.py
├── outputs/
│   ├── mechanistic/
│   ├── scenarios/
│   ├── ai/
│   ├── comparisons/
│   ├── figures/
│   ├── tables/
│   └── logs/
├── reports/
│   ├── technical_report.md
│   ├── decision_maker_report.md
│   └── llm_prompt_appendix.md
└── presentation/
    └── presentation_outline.md

==================================================
25. CODE-QUALITY REQUIREMENTS
==================================================
Use Python type hints. Use clear docstrings. Use pathlib for paths. Use YAML configuration. Use logging instead of scattered print statements. Fix random seeds. Avoid hard-coded scenario values inside modules. Keep equations transparent. Avoid unnecessary object-oriented complexity. Use vectorized calculations where appropriate. Write small, testable functions. Save intermediate outputs. Make long-running dataset generation resumable. Clearly report warnings and infeasible scenarios. Do not silently suppress exceptions. Do not invent missing data. Keep notebooks focused on explanation and results. Keep reusable calculations in src. Ensure commands work from the project root.

==================================================
26. VALIDATION AND AUTOMATED TESTS
==================================================
Implement automated tests confirming:
DATA: exactly 8,760 records for a non-leap local year, unique timestamps, no missing timestamps, correct timezone handling, no unexpected NaN values after processing, correct unit conversions.
LOAD: no negative demand, target annual consumption achieved within tolerance, peak demand respected, identical seed produces identical profile.
PV: no negative PV generation, PV zero at night within numerical tolerance, PV output does not exceed its configured limit.
WIND: no negative wind generation, wind output never exceeds rated capacity, correct behaviour below cut-in and above cut-out speeds.
BATTERY: SOC remains within bounds, charge and discharge power limits respected, no simultaneous charging and discharging, no energy is created, zero-capacity battery behaves correctly.
ENERGY BALANCE: for every hour, verify an internally consistent energy balance; annual residual within a strict numerical tolerance.
OPTIMIZATION: a feasible minimum candidate is selected correctly, an infeasible range is reported correctly, increasing battery capacity does not cause unexplained increases in unserved energy, modular capacity is calculated correctly.
AI: training-only scaler fitting, no target leakage, reproducible splits, no overlap between grouped train/test scenarios, nonnegative predictions, correct upward module rounding, physical verification uses predicted rather than reference battery capacity.
Tests must use small deterministic fixtures and must not require internet access.

==================================================
27. REQUIRED FIGURES
==================================================
MECHANISTIC MODEL: 1) Baseline annual load, PV and wind profiles, 2) Representative winter week, 3) Representative summer week, 4) Load-duration curve, 5) Monthly load and renewable production, 6) Wind-speed distribution and turbine power curve, 7) LPSP versus battery capacity, 8) Curtailment versus battery capacity, 9) Annualized cost versus battery capacity, 10) Representative SOC profiles, 11) Annual energy-flow balance.
DATASET: 12) Distributions of scenario inputs, 13) Distribution of mechanistic optimal battery capacities, 14) Feasible versus infeasible scenario counts, 15) Correlation matrix for selected features.
AI COMPARISON: 16) Training and validation loss, 17) Predicted versus reference battery capacity, 18) Residual distribution, 19) Error versus battery size, 20) Error versus renewable-to-load ratio, 21) Comparison of MAE/RMSE/R² across models, 22) Reliability pass rate across models, 23) Underprediction and overprediction rates, 24) Runtime comparison, 25) Accuracy–runtime comparison, 26) Cost penalty caused by predicted oversizing, 27) Reliability-target sensitivity, 28) Geographic-transfer results, if implemented.
Use readable labels and units. Save figures at publication-appropriate resolution.

==================================================
28. REQUIRED OUTPUT TABLES
==================================================
Save CSV files for: cleaned hourly weather, baseline hourly load, baseline PV and wind generation, mechanistic battery-candidate results, baseline optimal hourly dispatch, scenario inputs, scenario engineered features, mechanistic target labels, dataset split assignments, baseline-model predictions, neural-network predictions, continuous and rounded predictions, physically verified AI results, accuracy metrics by model, reliability metrics by model, runtime comparison, cost comparison, reliability sensitivity, battery-price sensitivity, annual energy balance.

==================================================
29. REPORTS
==================================================
Create structured initial drafts, but do not fabricate numerical results.
TECHNICAL REPORT (8–10 pages). Suggested structure: 1) Introduction, 2) Related work *(added per refinement addendum §1.3)*, 3) Research objective and questions, 4) System description and assumptions, 5) Weather and load data, 6) Mechanistic PV and wind models, 7) Battery-dispatch model, 8) Mechanistic battery optimization, 9) Scenario-dataset generation, 10) Feature engineering, 11) Machine-learning baseline models, 12) Neural-network architecture and training, 13) Evaluation methods, 14) Mechanistic results, 15) AI-model results, 16) Physical verification of AI predictions, 17) Accuracy and computational-efficiency comparison, 18) Discussion, 19) Limitations, 20) Conclusions.
DECISION-MAKER REPORT (4–5 pages). Focus on: why battery sizing matters, comparison of the two methods, cost consequences, reliability risks, when AI is useful, when mechanistic verification is necessary, interpretability, data requirements, implementation risks, recommendations.
LLM APPENDIX: Create reports/llm_prompt_appendix.md and include this complete prompt. Leave a structured section for later prompts, corrections and responses.

==================================================
30. PRESENTATION
==================================================
Create a 15-minute presentation outline followed by 5 minutes of questions.
Suggested slide sequence: 1) Problem and motivation, 2) Research objective and questions, 3) Off-grid system and baseline case, 4) Mechanistic battery-sizing method, 5) Scenario-dataset generation, 6) Neural-network method, 7) Validation and comparison design, 8) Mechanistic results, 9) AI prediction accuracy, 10) Physical reliability verification, 11) Runtime comparison, 12) Cost and risk implications, 13) Limitations, 14) Conclusions and recommendations.
Include approximate speaking time per slide. Provide a suggested division of slides among group members, but keep the number of members configurable.

==================================================
31. IMPLEMENTATION STAGES
==================================================
Do not attempt the complete 5,000-scenario experiment immediately. Proceed in controlled stages (see refinement addendum §1.1 for the MVP-vs-stretch breakdown of these stages).
STAGE 1: PROJECT SCAFFOLD — Create directory structure, configuration files, dependencies, README skeleton. Confirm project imports correctly.
STAGE 2: MECHANISTIC COMPONENTS — Implement load generation, weather acquisition, PV generation, wind generation, battery dispatch, energy-balance tests.
STAGE 3: BASELINE BATTERY OPTIMIZATION — Run the 2023 Jinan baseline, test candidate battery modules, determine feasibility, generate baseline figures and tables, report whether fixed generation is adequate.
STAGE 4: PILOT SCENARIO DATASET — Generate approximately 100 scenarios, verify feature calculations, inspect feasibility rates, estimate total runtime, correct unrealistic parameter combinations.
STAGE 5: BASELINE ML MODELS — Implement naive baseline, linear/ridge regression, tree-based model, evaluate using fixed dataset splits.
STAGE 6: NEURAL NETWORK — Implement preprocessing, train MLP, use early stopping, save the best model, generate initial prediction metrics.
STAGE 7: PHYSICAL AI VERIFICATION — Round predictions upward to modules, rerun hourly dispatch, measure achieved reliability, calculate over/undersizing.
STAGE 8: FULL SCENARIO DATASET — Only after pilot validation, generate the configured full dataset. Make generation resumable. Record runtime and failures.
STAGE 9: FINAL COMPARISON — Compare prediction accuracy, reliability, cost, computational efficiency. Test unseen weather. Optionally test geographic transfer.
STAGE 10: DOCUMENTATION — Complete notebooks, populate report drafts using actual results, complete presentation outline, run all automated tests.
After each stage: run the relevant tests, summarize what was completed, report errors and unresolved assumptions, commit or checkpoint changes if the workspace is a Git repository. Do not proceed past a scientifically invalid result without reporting it.

==================================================
32. FINAL COMPLETION REPORT
==================================================
At the end, provide a concise completion report containing: 1) Files created, 2) Commands required to install and run the project, 3) Data sources used, 4) Mechanistic assumptions, 5) AI-model configuration, 6) Number of generated scenarios, 7) Dataset-splitting approach, 8) Tests passed and failed, 9) Baseline optimal battery capacity, 10) Whether the baseline reliability target was feasible, 11) MAE/RMSE/R² for every predictive model, 12) AI underprediction rate, 13) AI reliability pass rate after physical verification, 14) Mechanistic runtime per scenario, 15) AI training time, 16) AI inference time, 17) Approximate computational break-even point, 18) Important limitations, 19) Items requiring human review.
Do not conceal failed tests, API failures, infeasible scenarios, poor neural-network performance or contradictory results.

==================================================
33. STARTING INSTRUCTION
==================================================
Start with Stage 1 only.
First inspect the existing workspace and report whether any relevant files already exist.
Then: 1) Create the project scaffold. 2) Create the configuration files. 3) Create dependency files. 4) Create an initial README. 5) Create placeholder modules with documented responsibilities. 6) Verify imports and paths. 7) Show the resulting project tree. 8) Report any assumptions requiring confirmation.
Do not yet generate thousands of scenarios or train the neural network.
After Stage 1 is complete, stop and provide a progress report so the architecture can be reviewed before Stage 2 begins.


---

## 2. Log of Subsequent Prompts, Corrections, and Responses

Entries are appended chronologically as the project progresses through the stages in
PROJECT_BRIEF.md §31. Each entry should record: the date, the prompt/correction given,
and a short summary of the response/outcome.

### Stage 1 — Project Scaffold

- **Prompt:** Initial build prompt above (refinement addendum + original spec).
  Starting instruction (§33) directed Claude Code to perform Stage 1 only: inspect
  the existing workspace, create the project scaffold, config files, dependency
  files, initial README, placeholder modules, verify imports, show the resulting
  tree, and report assumptions requiring confirmation, then stop for review.
- **Response/outcome:** Workspace was empty (no prior files) on branch
  `claude/repository-project-prompt-or0ida`. Full directory structure, config YAMLs,
  dependency files, README, all placeholder `src/` modules (documented, raising
  `NotImplementedError` until their implementing stage), notebook skeletons, test
  stubs, a real import/config smoke test, and report/presentation skeletons were
  created. See the Stage 1 progress report delivered in-conversation for the full
  list of assumptions flagged for confirmation before Stage 2 begins.

*(Further entries added as Stage 2 onward proceed.)*
