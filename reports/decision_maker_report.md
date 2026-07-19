# Battery Sizing for Off-Grid PV–Wind Systems: Mechanistic vs. AI — Decision-Maker Summary

**Status:** structural draft created in Stage 1 (project scaffold). No results exist
yet. This document will be populated with real, tested findings in Stage 10.

*Target length: 4–5 pages once complete.*

---

## 1. Why Battery Sizing Matters

*TODO: plain-language framing — battery cost dominates off-grid system economics,
undersizing risks reliability, oversizing wastes capital.*

## 2. Comparison of the Two Methods

*TODO: one-paragraph plain-language description of the mechanistic (exhaustive
physics-based search) method vs. the AI (trained neural network) method.*

## 3. Cost Consequences

*TODO (populate from Stage 7/9 results): cost penalty of AI oversizing, using actual
computed figures — not estimates.*

## 4. Reliability Risks

*TODO (populate from Stage 7 results): reliability pass rate of AI-selected battery
sizes after mechanistic verification; consequences of underprediction, reported
plainly (this is a safety/reliability risk, not just a statistic).*

## 5. When AI Is Useful

*TODO: practical scenarios where fast inference matters (e.g. evaluating many
candidate system configurations), conditioned on the break-even analysis (§17 of the
technical report).*

## 6. When Mechanistic Verification Is Necessary

*TODO: state plainly that AI predictions should always be checked against mechanistic
simulation before being trusted for an actual installation, per the physical
verification step (mandatory, PROJECT_BRIEF.md §20).*

## 7. Interpretability

*TODO: contrast the transparency of the mechanistic model's equations against the
neural network's learned function.*

## 8. Data Requirements

*TODO: what it takes to build/retrain each method (weather data, scenario generation,
labelled training data) if applied to a new site.*

## 9. Implementation Risks

*TODO: risks of applying either method outside its validated range (e.g. new
geography, unseen weather, different battery chemistry).*

## 10. Recommendations

*TODO: concrete, evidence-based recommendation once results exist — do not
pre-conclude that either method is "better" before the comparison is actually run.*
