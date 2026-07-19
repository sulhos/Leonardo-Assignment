# Comparison of Neural-Network and Mechanistic Models for Battery Sizing in an Off-Grid PV–Wind Energy System

**Status:** structural draft created in Stage 1 (project scaffold). No results exist
yet — every numbered section below is a placeholder to be populated with real output
from later stages. Nothing in this document should be read as a finding.

---

## 1. Introduction

*TODO (Stage 10): motivation for off-grid battery sizing, why comparing a mechanistic
method against a neural network is worth investigating, and a one-paragraph roadmap of
the report.*

## 2. Related Work

*Added per refinement addendum §1.3 — this section is expected by the thesis rubric
and was missing from the original spec's outline.*

*TODO (Stage 10): brief (≈1 page) situating this project against:*
- *LPSP-based sizing methods for standalone PV/wind/battery systems*
- *Sizing tools such as HOMER (and similar techno-economic optimization tools)*
- *Prior machine-learning-for-sizing literature*

*Citations must be real and findable (web search is acceptable) or explicitly marked
"citations to be completed" — never fabricated (addendum §1.3).*

**Citations to be completed.**

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

*TODO: sampling ranges and consistency checks, pilot vs. full dataset size actually
achieved, caching/resumability (§14).*

## 10. Feature Engineering

*TODO: load/generation/net-load/system features, leakage-avoidance measures (§15).*

## 11. Machine-Learning Baseline Models

*TODO: naive, linear/ridge, tree-based baseline configuration and hyperparameter
selection procedure (§17).*

## 12. Neural-Network Architecture and Training

*TODO: architecture, framework (Keras/TensorFlow, per addendum §1.2), training
configuration, early stopping / checkpointing (§18).*

## 13. Evaluation Methods

*TODO: dataset-splitting strategy actually used (Experiment A always; B/C only if
attempted as stretch goals, §16), accuracy metrics, operational metrics (§21).*

## 14. Mechanistic Results

*TODO (populate from real output once Stage 3 runs): baseline optimal battery
capacity, feasibility at the baseline reliability target, key figures/tables.*

## 15. AI-Model Results

*TODO (populate from real output once Stage 5/6 run): accuracy metrics for every
model (naive, linear, tree, MLP).*

## 16. Physical Verification of AI Predictions

*TODO (populate from real output once Stage 7 runs — mandatory, not optional):
reliability pass rate, under/oversizing rates, cost/reliability consequences.*

## 17. Accuracy and Computational-Efficiency Comparison

*TODO: cross-model metric comparison; runtime comparison and break-even calculation.
Note per refinement addendum §1.5: the mechanistic-time term in the break-even
denominator is the full exhaustive battery-module search per scenario, not a single
dispatch run — state this explicitly wherever the formula is reported.*

## 18. Discussion

*TODO: interpret results against the research hypothesis (§2) without assuming the
neural network wins.*

## 19. Limitations

*TODO: single-location/single-year pilot scope if Stage 8 stretch goals were not
reached, mechanistic-model-as-reference caveat (never "validation against reality"
unless real operational data are available), generic turbine/PV assumptions, etc.*

## 20. Conclusions

*TODO: answer the five research-objective questions (§1) directly and honestly,
including negative or inconclusive results if that is what was found.*
