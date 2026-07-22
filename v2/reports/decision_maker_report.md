# Battery Sizing for a Diesel-Backed Solar + Wind System: What We Found

## The question

If you're building a solar + wind power system with a diesel generator as
backup, how big a battery is actually worth paying for? Bigger batteries
cost more but save on diesel fuel — the right size is an economic
trade-off, not a fixed requirement. We compared two ways to find that
size:

1. **The physics-based method**: simulate the system hour-by-hour for a
   full year, for every possible battery size, and pick the one that gives
   the lowest total cost of electricity.
2. **The AI method**: train a neural network on thousands of these
   physics-based answers, then ask it to predict the right battery size
   directly for a new system, without re-running the year-long simulation.

## Why a diesel backup, and what it changes

With a diesel generator that can always cover peak demand on its own, this
system essentially never fails to meet demand — the generator is the
safety net. That changes the central question: it's no longer "does the
battery keep the lights on," it's "how much battery is worth the capital
cost, given what it saves in diesel fuel."

For context: we also checked what would happen without the diesel
generator at all, using the same solar and wind capacity. To hit a strict
reliability target on renewables and battery alone, the system would need
to be built with a large surplus of solar and wind — so large that roughly
**60% of everything it generates would go to waste** (curtailed, with
nowhere to store or use it), compared to 18% when diesel is available to
soak up the rest of the demand. That's the practical reason a diesel
backup is part of this design: it lets solar, wind, and battery be sized
for cost-effectiveness rather than for surviving the worst possible week
on their own. We did not build this diesel-free version out as its own
separate study — it's just useful context for why the diesel-backed
design makes sense.

## The baseline system

At our reference case (a mid-sized industrial site, solar 1,800 kW and
wind 1,450 kW of capacity, load of 4.5 GWh/year), the cost-minimizing
choice is a **24-unit battery (6,000 kWh)**. With that battery, solar and
wind cover **74.1%** of the electricity used over the year, the diesel
generator covers the other **25.9%**, and only **18.4%** of the power
generated goes to waste. We checked, battery size by battery size, where
the sweet spot is: **adding battery is worth it up to about 74% renewable
share.** Below that point, every bit of battery you add saves more in
generator fuel than it costs. Beyond that point, it flips — the last bit
of battery capacity mostly sits idle waiting for rare, extreme weeks, and
costs more than the fuel it saves. Pushing renewable coverage from 74% up
toward 80% this way would cost about 30% more per unit of electricity than
stopping at the sweet spot, for only a small further gain in renewable
share.

## What we found about the AI

**The AI is very accurate.** Across ten separate tests (each time
retraining on a different random slice of 5,000 example systems, all of
which are solvable since diesel can always cover any shortfall), the AI's
battery-size predictions matched the physics-based answer to within about
124 kWh on average, out of typical battery sizes in the thousands of kWh —
a 99.4% statistical fit. This wasn't a lucky one-off: repeating the test
ten times with different data splits gave nearly identical results every
time.

**More training data keeps helping, more than in a simpler version of this
problem.** Accuracy improved sharply as we gave the AI more example
systems to learn from, and — unlike a version of this problem we tested
previously without a diesel backup, where extra data stopped helping past
a certain point — accuracy kept improving noticeably all the way to the
full ~3,500 training examples available. That makes sense: because the
right battery size here depends on the assumed battery price, not just a
fixed rule, there's a genuinely harder pattern for the AI to learn, so
more examples keep paying off.

**The AI learned the right lessons, and combining many clues beats relying
on just the top few.** We tested which pieces of information the AI relies
on most by scrambling each one and seeing how much its accuracy dropped.
The top factors were exactly what an engineer would expect to matter: how
much solar and wind output there is relative to demand overall and in the
worst month, and how often and how badly renewables fall short. We then
tried a much simpler method — a basic formula using only those top three
factors — and it was far less accurate (about ten times worse on average,
correctly explaining less than half the pattern versus over 99% for the
full AI). So while those top factors matter most, the AI's advantage comes
from weighing all of its available information together, not just the
obvious top few.

**Trusting the AI directly carries very little practical risk here —
unlike a version of this problem without a diesel backup.** We took the
AI's predicted battery sizes, rounded them up to the nearest installable
battery unit, and re-simulated each system for a full year. Because the
diesel generator is always there as a safety net, installing exactly what
the AI recommends essentially never causes a reliability shortfall. The
real question we checked instead was economic: **how much extra does it
cost to trust the AI's rounded prediction instead of the true
cost-optimal battery size?** On average, essentially nothing — about
0.00013 EUR per kWh, a small fraction of a percent of the system's total
cost, and every single prediction we checked landed within 5% of the
best-possible cost. We also checked whether trusting the AI makes the
system lean more heavily on diesel than it should: on average it doesn't
— if anything, the AI's choice leans very slightly *less* on diesel than
the ideal choice would, and only about 7% of cases lean more on diesel
than optimal, and even then only modestly. **The bottom line is different
here than for a system without backup power: because the diesel generator
absorbs the consequences of a slightly-wrong prediction, trusting the
AI's battery-size recommendation directly is a low-risk choice, not one
that requires a mandatory physics-based double-check before acting on it.**

**Battery price matters a lot here — including for which size gets
recommended, not just for the cost.** Testing battery prices from 220 to
400 EUR per kWh, the cost-minimizing battery size itself changed
noticeably: around 20 battery units on average at the cheaper price, down
to about 18 at the baseline price, down to about 16 at the higher price.
This is different from a version of this system without diesel backup,
where the battery price changes the total cost but not the recommended
size, because there the size is fixed by a reliability requirement rather
than economics. Here, a cheaper battery genuinely earns its keep by
displacing more diesel fuel, so a materially different battery price
would mean the AI needs to be retrained on new examples, not just reused
as-is.

**The AI really is much faster — but how much faster depends on how you
use it.** Running the full physics-based simulation takes about a third of
a second per system. If you ask the AI to evaluate many systems at once
(the realistic case for exploring lots of design options quickly), it's
about **3,300 times faster** per system. If you ask it to evaluate just
one system on its own, it's still faster — about **6 times** — but the
speed advantage shrinks a lot, because most of that single request's time
is just the overhead of asking the question at all, not the actual
computation. Either way, none of this includes the time it took to train
the AI in the first place (a few minutes, done once) — the speed payoff
only shows up once you're evaluating many new systems using that
already-trained model.

## Bottom line

For a system backed by a diesel generator, the AI model is fast, highly
accurate, and — because the generator absorbs the downside of a
slightly-wrong prediction — safe to trust directly for day-to-day battery
sizing decisions, without needing a mandatory physics-based re-check first.
The remaining caveats are practical rather than safety-critical: the
model's target is tied to the battery price assumption it was trained on,
so a materially different price means retraining, and more training
examples continue to help more than they would for a simpler,
reliability-driven version of this problem. Used together — AI for fast,
low-risk exploration and periodic physics-based checks to confirm the cost
assumptions still hold — this gives both speed and confidence in the
recommended battery size.
