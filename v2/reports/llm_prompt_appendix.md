# V2 LLM Prompt Appendix

This is the V2 companion to the repository root's `reports/llm_prompt_appendix.md`
(which records V1's original build prompt, per `PROJECT_BRIEF.md` §29's
requirement to log "prompts, corrections, and responses" as the project
develops). V1's appendix stops after Stage 1; it was not kept up to date
through the rest of V1's build. This file exists to give V2 — the from-scratch
rebuild in this `v2/` subfolder — the complete record V1's own appendix
doesn't have.

**Fidelity note.** Three source documents are quoted verbatim here because the
exact text is available: the 11-task engineering-review brief (reproduced
below from `v2/reports/technical_report.md`'s own §2, which quoted it
verbatim), and the full diesel-hybrid-replacement prompt (§4, quoted from the
original uploaded file). Everything else — the V2 kickoff instruction, the
diesel-hybrid demonstration's originating prompt, and the per-task entries in
§2 — is reconstructed from commit messages, `v2/CROSS_CHECK_BRIEF.md`, and
`v2/DIESEL_HYBRID_DEMO.md`, since the literal original chat text for that
older work is not available in this session. Those entries are marked as
summaries, not transcripts, consistent with this project's own
no-fabrication standard applied to prompt history, not just numerical
results.

---

## 1. V2 Kickoff

**What was asked** (summarized from `v2/CROSS_CHECK_BRIEF.md` §1): an earlier
session had built a complete V1 pipeline — mechanistic PV/wind/battery model,
scenario dataset generator, four ML models including a neural network,
physical verification, and two written reports — for a diesel-backed hybrid
system. A separate engineering review of that V1 build produced an 11-task
improvement brief. The instruction was to build a V2.0 rebuild implementing
all 11 items. Three scope questions were resolved via `AskUserQuestion` before
starting: build V2 in a new `v2/` subfolder of the same branch (not a new
repository); proceed autonomously through all 11 tasks without stopping for
approval, except at the brief's own explicit checkpoint (Task 3, §2 below);
and drop an optional "keep one rule-of-thumb baseline" idea entirely — strict
mechanistic-vs-neural-network only.

**The 11-task brief** (quoted verbatim from `technical_report.md`'s original
§2, as it stood before the diesel-hybrid rewrite in §4):

1. Rewrite the mechanistic PV/wind/battery physics to strictly match the
   course lecture's exact equations, with hand-computed reference tests.
2. Remove every ML baseline except the neural network.
3. Resolve, with the thesis author, whether V2 studies a diesel-backed or an
   off-grid system (a checkpoint requiring explicit sign-off).
4. Fix a load-profile-shape coherence gap between the baseline showcase and
   the ML dataset.
5. Add cross-validation error bars to the NN's accuracy claim.
6. Add a learning curve across training-set sizes.
7. Add permutation feature importance.
8. Add a cost-sensitivity analysis.
9. Sharpen the computational-efficiency claim's text.
10. Trim the figure set to a lean, curated set.
11. Regenerate every affected artifact and rewrite the report end-to-end.

---

## 2. Log: V2 Tasks 1–11

| Task | Ask | Commit | Outcome |
|---|---|---|---|
| 1 | Lecture-exact PV/wind/battery physics | `eaadf2e` | Added the PV incidence-angle modifier, clearness-index/Erbs/Liu & Jordan as standalone tested functions (verified numerically identical to pvlib's isotropic model), switched wind turbine output to linear interpolation on a config-driven power curve, added battery self-discharge to the SOC equation. 11 new hand-computed-reference PV tests; full suite 173 passed. |
| 2 | Remove every ML baseline except the NN | `95c7700` | Deleted `src/ai/baselines.py` and its config section entirely — no rule-of-thumb reference either, per the kickoff's scope decision. 170/170 tests. |
| 4 | Fix load-shape coherence gap | `6272eb5` | The baseline showcase used `industrial_baseline` while every sampled ML scenario silently defaulted to `residential_baseline` — a real gap between what the flagship figure showed and what the model was trained on. Set the config to match what the code actually used everywhere. |
| **3** | **Checkpoint: diesel-backed vs. off-grid (requires sign-off)** | `891c8cd` | **See detailed reconstruction below — the one decision in this project most worth independent scrutiny.** |
| 5 | Cross-validation error bars | `e745045` | 10 independent 70/15/15 splits, full retraining each: R² = 0.9939 ± 0.0015, confirming the headline seed-42 result (R² = 0.9926) was representative, not a favorable draw. |
| 6 | Learning curve across training-set sizes | `2774fdc` | MAE 790 kWh (n=100) → 179 kWh (n≈3,168, the actual full training split) — clear plateau past ~2,000 scenarios. |
| 7 | Permutation feature importance | `a1b2648` | Top drivers: `monthly_renewable_variability`, `worst_monthly_renewable_to_load_ratio`, `reliability_target_load_served` — physically sensible for a hard-reliability off-grid objective. |
| 8 | Cost-sensitivity analysis | `625fd33` | Battery cost swept 220/300/400 EUR/kWh (closed-form, sanity-checked against the stored baseline LCOE): mean system LCOE 0.360/0.403/0.456 EUR/kWh — cost changes reported economics but never the selected battery size, since off-grid sizing is reliability-driven, not cost-driven. |
| 9 | Sharpen the efficiency claim | (folded into `b93b2af`) | Restated as regime-dependent: dramatic when batched, modest for single queries, rather than one flat multiplier. |
| 10 | Lean, curated figure set | `94246d1` | 10 figures, not an exhaustive regeneration of every V1 figure type: one baseline-physics pair, one dataset-composition pair, one figure per Task 5–8 analysis. |
| 11 | Regenerate artifacts, rewrite the report end-to-end | `b93b2af` | Full technical + decision-maker report rewrite for V2's off-grid scope. Both exported to PDF/Word. 173 tests green. |

### Task 3 in detail: the checkpoint decision

This is reconstructed from `v2/CROSS_CHECK_BRIEF.md` §2, written specifically
so an independent reviewer could scrutinize it — reproduced here for the same
reason.

The 11-task brief required stopping and confirming with the thesis author
whether V2 should study a diesel-backed system (continuing V1's later pivot)
or revert to the original off-grid specification — genuinely different
studies: diesel-backed makes reliability near-guaranteed by construction and
the search minimizes system LCOE; off-grid makes reliability a hard
pass/fail constraint and the search finds the smallest feasible battery.

The sequence of events:

1. An `AskUserQuestion` call was raised with three options (diesel, off-grid,
   both). It was interrupted by the harness twice (worker restarts), then
   accepted once, with the answer **"Run both configurations."**
2. The user's next message was a longer, detailed, reasoned argument
   recommending **off-grid only**, with a specific mandatory addition:
   re-tune the scenario sampling ranges so a large majority of sampled
   systems are renewable-adequate — verified on a pilot, before generating
   the full dataset — and keep genuinely infeasible scenarios honestly
   labelled rather than replaced with a maximum-battery guess.
3. The second, more detailed message was judged to supersede the first "run
   both" answer, and the build proceeded with **off-grid only** plus the
   sampling retune, without an explicit round-trip to reconcile the two
   answers (the session's resumption instruction at that point said not to
   ask further questions).

Naively flipping to off-grid under the diesel-era sampling ranges reproduced
a known trap: only 39/100 pilot scenarios were feasible (renewable/load
ratio too close to parity for any battery in the tested range to bridge a
seasonal shortfall). The ranges were re-tuned and validated empirically
before the full run: two 100-scenario pilots at 88% and 90% feasible, the
full 5,000-scenario run confirming 90.5%.

This decision was later fully superseded anyway — §4 below replaces the
off-grid study with diesel-hybrid entirely — so the question of whether
"off-grid only" was the right initial call is now moot for the current
report, but the reconstruction is kept here for the historical record and
because `v2/CROSS_CHECK_BRIEF.md` was written expecting independent review of
exactly this decision.

---

## 3. Diesel-Hybrid Demonstration (post-Task-11 addendum)

**What was asked** (summarized from `v2/DIESEL_HYBRID_DEMO.md`; the literal
original text is not available — it is described there as "an external task
prompt written for Codex but executed directly in this session instead"):
switch the diesel backup on, raise the baseline load so it visibly engages
(the completed off-grid study left diesel present but never dispatching,
curtailing ~60% of generated renewable energy), and produce the evidence —
an engagement-week figure, an energy-flow figure, and the renewable-share
vs. system-LCOE trade-off curve. Explicitly scoped as mechanistic-only: no
dataset regeneration, no retraining.

**Outcome** (`a03fdb1`, `90581ac`, `98e1305`): the originally-suggested
3.0–3.5 GWh/year target gave only 11–16% diesel share when actually tested
against the LCOE-optimal search — too faint. Six load levels were tested
empirically; 4.5 GWh/year at 1,200 kW peak was chosen, landing at 74.1%
renewable / 25.9% diesel share at the LCOE-optimal battery (24 modules,
6,000 kWh, system LCOE 0.288 EUR/kWh), comfortably inside a 65–85% target
band. Folded into both reports as an added section. This demonstration's
numbers are exactly what §4's Part A checkpoint re-validates before the full
replacement below.

---

## 4. Full Replacement: Diesel-Hybrid as the Sole Study (Parts A–G)

This is the prompt that produced the report as it currently stands. Quoted
in full, verbatim, from the uploaded file
(`CLAUDE_CODE_diesel_replace_offgrid.md`):

> # Claude Code task: make the diesel-hybrid the sole study (fully replace off-grid)
>
> You're working in the existing `battery-sizing-ai-comparison` repo. The current report studies a
> pure **off-grid** system for the neural-network comparison, with a diesel-hybrid section bolted on
> at the end as a mechanistic-only demonstration. **We are now replacing the off-grid study entirely.**
> The diesel-hybrid (PV + wind + battery + diesel) becomes the single system for the whole
> pipeline — dataset, neural network, verification, and report. The off-grid ML study, its dataset,
> its trained model, and all its figures should be removed from the report (they can remain in git
> history, but must not appear in the document or be presented as current results).
>
> This is a **full re-run plus an end-to-end report rewrite**, not an add-on section. Everything must
> come from real runs; do not fabricate numbers; keep the test suite green; keep implementation detail
> (file paths, "Task N", "V1/V2", "bug found") out of the report prose.
>
> ## Part A — Configure the diesel-hybrid baseline and verify it reproduces known numbers
>
> 1. Set the system to diesel-backed (`system.backup: diesel`), diesel rated at 1.25 × peak load.
>    Confirm the dispatch order: renewables serve load → surplus charges battery → deficit discharges
>    battery → residual deficit covered by diesel. With diesel present, the battery-sizing objective
>    is **minimum system LCOE** (not the off-grid "smallest feasible battery" rule).
> 2. Baseline load: **4.5 GWh/year at ~1,199 kW peak** (PV 1,800 kWp and wind 1,450 kW unchanged).
> 3. **Validation checkpoint — reproduce these baseline numbers before proceeding** (they were
>    validated previously; if you land far off, stop and investigate):
>    renewable share ≈ **74.1%**, diesel share ≈ **25.9%**, curtailment ≈ **18.4%**, diesel operating
>    hours ≈ **3,250/year**, fuel ≈ **683,550 L/year**, diesel CF ≈ **8.87%**, LPSP = **0**,
>    LCOE-optimal battery ≈ **24 modules / 6,000 kWh** at system LCOE ≈ **0.288 EUR/kWh**.
>
> ## Part B — Regenerate the 5,000-scenario dataset under the diesel regime
>
> 4. Update the scenario sampling load ranges to bracket the diesel baseline — suggest
>    `annual_load_kwh: [1,500,000, 7,000,000]` and `peak_load_kw: [400, 1,800]` (keep the PV/wind
>    capacity ranges and the 5,000 kW combined cap). The aim is a wide spread of renewable-to-load
>    ratios, from renewable-rich (low load → a large optimal battery is worth it) to load-heavy
>    (high load → diesel does most of the work and the optimal battery is small or zero).
> 5. **Every scenario is feasible under diesel** (diesel guarantees service), so there is no
>    feasibility filter anymore — the whole 5,000 are usable.
> 6. **Validate the target distribution on a 100-scenario pilot before the full run.** Under diesel
>    min-LCOE, renewable-poor scenarios often have an optimal battery of 0 kWh. Check that the target
>    is not degenerate — you want a healthy spread across the 0–80 module range, not (say) 70% of
>    scenarios at zero. If too many pile up at zero, narrow the upper load bound so more scenarios
>    have a non-trivial optimal battery, and report the distribution you settled on. Keep the
>    zero-battery scenarios that legitimately occur — they are real, informative outcomes, not errors.
>
> ## Part C — Retrain the neural network and redo the accuracy analyses
>
> 7. Recompute the 37 features (same leakage guard) and retrain the MLP (same architecture/config) on
>    the new diesel dataset. Report, all on the diesel data:
>    - single-split accuracy (MAE, RMSE, R², % within 10/20%),
>    - **10-split cross-validation** (mean ± std of MAE/RMSE/R²),
>    - a **learning curve** (MAE and R² vs. training-set size),
>    - **permutation feature importance** (top features + one-line physical interpretation — expect
>      the drivers to shift, since the target is now cost-optimal battery under diesel, not
>      reliability-constrained battery),
>    - the **predicted-vs-actual scatter** (1:1 line, R² annotated) as the headline accuracy figure.
>
> ## Part D — Add ONE simple sanity-check baseline (still outstanding)
>
> 8. Add a single trivial reference so the report can honestly answer "does the NN beat the obvious
>    approach?" — either a physics rule-of-thumb or a plain linear regression on the top ~3 features.
>    Report its MAE/R² next to the NN. This matters even more now: with many diesel scenarios having a
>    near-zero optimal battery and a target dominated by a few features, a trivial model may do well —
>    if it nearly matches the NN, say so honestly.
>
> ## Part E — Physical verification, reframed for diesel
>
> 9. Reliability is now ~100% by construction (diesel always covers residual load), so verification's
>    headline becomes **economic**: for each test prediction, round up to installable modules,
>    re-simulate, and report mean **extra system LCOE** vs. the true optimum, % over/under-sized, and
>    the cost of over/under-sizing. This is the mandatory check.
> 10. Also report a **reliability diagnostic** that keeps a reliability-flavoured signal: for each
>     NN-chosen battery, compute the LPSP that renewables + battery **alone** (without diesel) would
>     have achieved, and compare to the optimum's — i.e. how much more the AI's choice leans on diesel
>     than the true optimum would. This replaces the off-grid pass/fail check with a meaningful
>     diesel-era analogue.
>
> ## Part F — Cost sensitivity (now affects the target)
>
> 11. Sweep battery installed cost (e.g. 220 / 300 / 400 EUR/kWh) **and note an important difference
>     from the off-grid case**: under diesel min-LCOE the optimal battery *does* depend on battery
>     cost (cheaper battery → larger optimum), so the ML target itself shifts with the cost
>     assumption. Show how the optimal-battery distribution moves across the sweep, and flag in the
>     report that the target — and therefore the trained model — is conditioned on the assumed battery
>     price. Optionally add a diesel fuel-price sweep (±20%), since that also moves the optimum.
>
> ## Part G — Report rewrite (diesel-only, end to end)
>
> 12. Rewrite the report so the studied system is the diesel hybrid throughout. Remove the off-grid
>     dataset/model/figures and any "off-grid is the main study" framing. Update the research
>     questions: question 4 is no longer "does the battery meet reliability" (guaranteed by diesel) but
>     "how much does trusting the AI's battery cost economically, and how much more does it lean on
>     diesel than the optimum." Keep as current results:
>     - the diesel baseline (74.1% renewable / 25.9% diesel, 18.4% curtailment, 24 modules),
>     - the diesel-engagement week figure and the annual energy-flow figure,
>     - the renewable-share-vs-LCOE U-curve and the economic sweet spot (~74% renewable share),
>     - the retrained NN accuracy + CV + learning curve + feature importance + simple baseline,
>     - the reframed (economic + lean-on-diesel) verification,
>     - curtailment/diesel metrics throughout.
> 13. Keep a **short qualitative** note in the discussion that a diesel-free (off-grid) version of this
>     system would have to massively over-build renewables and curtail ~60% of generation to hit a
>     reliability target — as motivation for the diesel backup — but do **not** present off-grid as a
>     parallel study with its own results.
> 14. No development-log language. Report honestly anything that comes out differently from the current
>     (off-grid) report — in particular the NN accuracy will change on the new target, and that is
>     expected.
>
> ## Guardrails, order, deliverables
>
> - No fabricated numbers; fixed seeds; resumable generation; tests stay green.
> - Order: Part A (hit the checkpoint in step 3 before continuing) → Part B (validate the pilot target
>   distribution before the full run) → Part C → D → E → F → G.
> - Stop after Part A and after the Part B pilot, and report the numbers, so they can be checked before
>   the expensive full regeneration and retraining.
> - Deliver: the baseline validation numbers, the pilot target distribution, the retrained NN accuracy
>   (with CV error bars), learning curve, feature importance, the simple-baseline comparison, the
>   reframed verification results, the cost-sensitivity target shift, and the rewritten diesel-only
>   report.

### Log by part

| Part | What was delivered | Commit(s) |
|---|---|---|
| A (checkpoint) | Reran `diesel_hybrid_demo.py`, reproduced every checkpoint number exactly (renewable share 74.10%, diesel share 25.90%, curtailment 18.44%, 24 modules/6,000 kWh, LCOE 0.28811 EUR/kWh). Stopped and reported before continuing, per the prompt's explicit instruction. | `2a004ae` |
| B | Sampling ranges widened to `annual_load_kwh: [1.5M, 6.5M]`, `peak_load_kw: [400, 1800]` (trimmed from the suggested 7,000,000 upper bound — confirmed empirically unreachable within an 1,800 kW peak). 100-scenario pilot validated first: 100% feasible, only 1% at zero battery, 0% at the 80-module ceiling — no degenerate clustering, so the full run proceeded. Full 5,000-scenario dataset: mean 18.1 modules, std 8.5, spanning 0–46. | `ed3957f`, `7e728e0`, `565565a` |
| C | Recomputed 37 features, retrained the MLP: MAE 125.9 kWh, RMSE 168.2 kWh, R² 0.9934 (single split); 10-split CV R² 0.9939 ± 0.0004; learning curve MAE 588→111 kWh (100→3,500 training scenarios, still improving at full size, not plateaued like the off-grid case); permutation importance topped by `renewable_to_load_ratio` and `worst_monthly_renewable_to_load_ratio` — a different ranking than off-grid's reliability-driven top features, as expected for a cost-driven target. | `c859d59`, `f875eb8`, `a3b4145`, `4c058f8`, `99d1536` |
| D | Linear regression on the top 3 permutation-importance features: R² 0.453 vs. the full NN's 0.993 — the NN is not just reproducing what the obvious top features alone would already give you. | `64db1f4` |
| E | `physical_verification.py` extended to report mean extra system LCOE (0.00013 EUR/kWh, negligible) and the new lean-on-diesel diagnostic (mean difference −0.2 percentage points, only 6.9% of predictions lean more on diesel than the true optimum). | `ed3957f` (diagnostic), folded into `c859d59`'s run |
| F | Cost sensitivity re-run as a full mechanistic search (not the off-grid case's closed-form shortcut, since the selected candidate itself changes with cost) on a 1,000-scenario subsample: mean optimal battery 20.2/18.1/15.9 modules at 220/300/400 EUR/kWh — confirms the target genuinely shifts with the cost assumption. | `c7b2502`, `c7d142b` |
| G | Figures 20–27 generated; both reports rewritten end-to-end with diesel-hybrid as the sole studied system, research question 4 reframed, off-grid demoted to a one-paragraph qualitative note, no development-log language. Re-exported to PDF/Word. Full suite (178 tests) green. | `70c8a21`, `cb63bcf` |

**Approval to proceed without further checkpoints** (verbatim, this session):
> yea complete parts B through G

---

## 5. Post-Completion Follow-Ups (verbatim, this session)

Three follow-up requests after Parts A–G were delivered, each resolved in a
single exchange:

1. **"so is the report on word doc also complete, accomodating only the new results?"**
   Verified the exported `.docx`/`.pdf` files were regenerated in the same
   commit as the report rewrite (not stale), and confirmed by text-extraction
   that no leftover off-grid framing remained outside the one intentional
   qualitative sentence.

2. **"I think you still have kept some of the images from previous run as you
   can see in the image. can you please update the PV, Wind and Load profile
   for the entire year."** (with a screenshot of `01_annual_profile.png`,
   still showing the old off-grid baseline). Confirmed the figure was a
   genuine leftover — generated under the pre-diesel config and never
   regenerated even though the rewritten report no longer referenced it by
   that stale content. Regenerated it against the current diesel-hybrid
   baseline via `scripts/diesel_hybrid_demo.py` (which already computed the
   same load/PV/wind series for its other figures) and added it to the
   report's system-description section. Commit `b74d358`.

3. **"so is the report... the comparison of NN and mechanistic model is in
   computational cost? since the NN accuracy is almost the same as the
   mechanistic."** → **"so is there an image generated in the report
   depicting this with clear explanation?"** Confirmed no such figure
   existed (the efficiency section was text-only) and added one: a new
   `plot_computational_efficiency()` function and a committed measurement
   script (`scripts/measure_computational_efficiency_diesel.py`) that times
   the NN fresh rather than reusing an earlier ad-hoc figure, producing
   `28_computational_efficiency.png` (348 ms mechanistic vs. 58 ms
   single-call NN vs. 0.107 ms batched NN — 6× and 3,250× faster
   respectively), with the accompanying report text making explicit that the
   NN cannot be more accurate than its own training reference by
   construction, so compute cost is the axis that actually differentiates
   the two methods once accuracy is close. Commit `980c8f6`.

4. **This document** — "now I need one more thing. can you summarize all the
   prompts in a usefull way, document it as .md" — scoped to the full V2
   rebuild via a clarifying question, then written as this file.

---

## 6. Reproducing Any of the Above

```bash
cd v2
python3 -m pytest -q                                          # 178 passed
PYTHONPATH=. python3 scripts/diesel_hybrid_demo.py             # Part A baseline
PYTHONPATH=. python3 scripts/measure_computational_efficiency_diesel.py
```

Every numeric claim in this file traces to a specific committed script or
table under `v2/outputs/tables/`, `v2/data/scenarios/`, or `v2/models/` —
see `v2/reports/technical_report.md` for the fuller derivation of each, and
`v2/CROSS_CHECK_BRIEF.md` for a reproduction checklist written specifically
for independent review of the Task 3 checkpoint and the off-grid study that
§4 above has since superseded.
