# Presentation Outline: Mechanistic vs. Neural-Network Battery Sizing

**Status:** structural draft created in Stage 1. Content and speaking notes will be
populated with real results in Stage 10. Total time: 15 minutes presentation + 5
minutes questions (20 minutes total).

Number of group members is configurable — set `N_MEMBERS` below and adjust the slide
assignment table accordingly.

`N_MEMBERS = 1` *(placeholder — update if this is a group project)*

| # | Slide | Time | Content (TODO once results exist) | Assigned to |
|---|---|---|---|---|
| 1 | Problem and motivation | 1.0 min | Why off-grid battery sizing matters; cost/reliability tension | TBD |
| 2 | Research objective and questions | 1.0 min | The 5 research questions (PROJECT_BRIEF.md §1) | TBD |
| 3 | Off-grid system and baseline case | 1.5 min | System boundary, Jinan baseline parameters | TBD |
| 4 | Mechanistic battery-sizing method | 1.5 min | Dispatch model + exhaustive search | TBD |
| 5 | Scenario-dataset generation | 1.0 min | Sampling ranges, pilot size actually achieved | TBD |
| 6 | Neural-network method | 1.5 min | Architecture, framework (Keras/TF), training setup | TBD |
| 7 | Validation and comparison design | 1.0 min | Splitting strategy (Experiment A, +B/C if attempted) | TBD |
| 8 | Mechanistic results | 1.0 min | Baseline optimal battery, feasibility | TBD |
| 9 | AI prediction accuracy | 1.5 min | MAE/RMSE/R² across all models | TBD |
| 10 | Physical reliability verification | 1.5 min | Mandatory verification results — reliability pass rate | TBD |
| 11 | Runtime comparison | 1.0 min | Mechanistic vs. AI inference time, break-even point | TBD |
| 12 | Cost and risk implications | 1.0 min | Oversizing cost penalty, undersizing reliability risk | TBD |
| 13 | Limitations | 1.0 min | Scope actually achieved vs. stretch goals not reached | TBD |
| 14 | Conclusions and recommendations | 1.5 min | Answer the hypothesis honestly (§2) | TBD |

**Total:** 15.0 minutes + 5 minutes Q&A.

## Notes

- Do not populate slide content with placeholder or assumed numbers — only real
  output from the corresponding pipeline stage.
- Slide 10 (physical reliability verification) must not be cut for time even under
  presentation-length pressure; it is the project's most important scientific check
  (refinement addendum §1.1).
- If Stage 8 stretch goals (full 5,000-scenario dataset, unseen-weather holdout,
  geographic transfer) were not reached, say so explicitly on the limitations slide
  rather than presenting the pilot-scale result as the final one.
