# Battery Sizing for an Off-Grid Solar + Wind System: What We Found (V2)

## The question

If you're building a solar + wind power system with no backup generator
and no grid connection, how big a battery do you need so the system
reliably meets demand? We compared two ways to answer that:

1. **The physics-based method**: simulate the system hour-by-hour for a
   full year, for every possible battery size, and pick the smallest one
   that keeps the lights on 99% of the time.
2. **The AI method**: train a neural network on thousands of these
   physics-based answers, then ask it to predict the right battery size
   directly for a new system, without re-running the year-long simulation.

This is a from-scratch redo ("V2") of an earlier version of this project,
rebuilt to match a specific course's exact engineering formulas, focused
tightly on this one comparison (no other AI models muddying the picture),
and switched to studying a fully **off-grid** system rather than one with
a diesel generator backing it up.

## Why off-grid, and what almost went wrong

Studying "off-grid" means reliability is a hard requirement: if the
battery can't cover a bad week, the system fails, full stop — there's no
generator to fall back on. That's the harder, more useful test of whether
the AI has really learned the right lesson.

But there's a trap: if you don't also make sure most of the systems you're
studying have *enough* solar and wind capacity relative to their demand,
almost none of them will be able to meet that reliability requirement no
matter how big a battery you install — a battery only smooths out short
gaps, it can't manufacture energy that was never generated in the first
place. We checked this before committing to the full study: an initial
test showed only 39 out of 100 sample systems could actually meet the
reliability target off-grid. We fixed the mix of systems being studied
(shrinking the typical electricity demand relative to the solar/wind
capacity already being tested) and re-checked twice more before running
the full study — landing at 88% and 90% of sample systems being solvable,
comfortably enough to learn from, while still keeping a genuine ~10%
minority of unsolvable cases so the results stay honest rather than
artificially perfect.

The full-scale study (5,000 example systems) came in at **90.5% solvable**,
confirming the fix worked.

## What we found

**The AI is very accurate.** Across ten separate tests (each time
retraining on a different random slice of the data), the AI's battery-size
predictions matched the physics-based answer to within about 163 kWh on
average, out of typical battery sizes in the thousands of kWh — a 99.4%
statistical fit. This wasn't a lucky one-off: repeating the test ten times
with different data splits gave nearly identical results every time.

**More training data helps — up to a point.** Accuracy improved sharply as
we gave the AI more example systems to learn from, but that improvement
essentially stopped after about 2,000 examples. Giving it the full
available ~3,200 barely helped at all. If we wanted a meaningfully better
model in the future, more of the same kind of data isn't the answer —
better information about each system would be.

**The AI learned the right lessons, not lucky guesses.** We tested which
pieces of information the AI relies on most by scrambling each one and
seeing how much its accuracy dropped. The top factors were exactly what an
engineer would expect to matter for an off-grid system: how much the solar
and wind output swings from month to month, how bad the single worst
month is, and how strict the reliability requirement is. That's
reassuring — it means the AI is reasoning about the right things.

**But trusting the AI blindly still carries real risk.** We took the AI's
predicted battery sizes, rounded them up to the nearest installable
battery unit (never down), and actually re-simulated each system for a
full year to see if it really held up. **94% did.** The other 6% fell
short of the reliability target even after rounding up — meaning if you
installed exactly what the AI suggested without double-checking, about 1
in 17 systems would come up short during a bad stretch of weather. This is
the main practical takeaway: **the AI is a fast, very good first estimate,
but a physics-based double-check before actually building the system is
still the responsible final step**, not an optional extra.

**Battery price matters, but doesn't change the recommended size.**
Testing battery prices from 220 to 400 EUR per kWh, the total system cost
per unit of electricity delivered ranged from about 0.36 to 0.46 EUR/kWh —
battery cost alone explains a third to nearly half of total system cost.
Notably, changing the assumed battery price does *not* change which
battery size gets recommended off-grid — that decision is driven entirely
by the reliability requirement, not by cost, since there's no cheaper
"good enough" option when reliability is a hard requirement.

**The AI really is much faster — but how much faster depends on how you
use it.** Running the full physics-based simulation takes about a third
of a second per system. If you ask the AI to evaluate many systems at once
(the realistic case for exploring lots of design options quickly), it's
about **2,500 times faster** per system. If you ask it to evaluate just
one system on its own, it's still faster — about **6.5 times** — but the
speed advantage shrinks a lot, because most of that single request's time
is just the overhead of asking the question at all, not the actual
computation. Either way, none of this includes the time it took to train
the AI in the first place (a few minutes, done once) — the speed payoff
only shows up once you're evaluating many new systems using that already-
trained model.

## Bottom line

For quickly narrowing down battery size across many candidate off-grid
system designs, the AI model is fast and reliably close to the physics-
based answer. For finalizing an actual system that will be built, the
physics-based check is still necessary — about 1 in 17 of the AI's
rounded-up recommendations would not have met the reliability target
without it. Used together — AI for fast exploration, physics-based
simulation for final confirmation — this gives the best of both: speed
during design, and confidence before construction.
