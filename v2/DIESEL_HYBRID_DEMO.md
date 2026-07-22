# Diesel-hybrid demonstration: making the diesel backup visible

**Scope**: a focused, mechanistic-only change (config + physics + evidence),
following an external task prompt written for Codex but executed directly
in this session instead. Does **not** retrain the neural network or
regenerate the full 5,000-scenario dataset — those remain deferred follow-
up steps, exactly as the originating prompt specified.

## The situation this addresses

The completed V2 off-grid study (`reports/technical_report.md`, the Task 3
checkpoint) configured the system as pure off-grid: PV + wind + battery
only, diesel present in the code but switched off (`system.backup: none`).
To meet a 99% reliability target with only a small battery, PV and wind had
to be massively over-built relative to load — the baseline case generated
roughly **2.6× more energy per year than the load consumed** (~4.16 GWh
generated vs. ~1.6 GWh load), so **~60% of all generated renewable energy
was curtailed**, and the diesel generator (present in every config, sized
at 1.25× peak load) never actually dispatched.

This demonstration switches the diesel backup **on** and raises the load
so the diesel visibly engages during low-renewable periods, producing a
realistic hybrid PV + wind + battery + diesel system instead.

## What changed

`config/jinan.yaml`:
- `system.backup`: `none` → **`diesel`**
- `load.target_annual_kwh`: `1,600,000` → **`4,500,000`** (4.5 GWh/year)
- `load.max_hourly_kw`: `430` → **`1,200`** (achieved peak 1,199.4 kW)

`config/scenario_generation.yaml`:
- `ranges.annual_load_kwh`: `[800000, 2500000]` → **`[2500000, 6500000]`**
- `ranges.peak_load_kw`: `[150, 550]` → **`[700, 1700]`**
- (not yet used to regenerate a full dataset — see "What this does NOT do" below)

PV (1,800 kWp) and wind (1,450 kW) capacities are **unchanged** — only load
was raised, per the originating task's instruction to keep the renewable
side fixed and bring the load into balance with it.

`src/visualization/plotting.py`: added `plot_diesel_engagement_week`, a new
two-panel figure (renewable generation + load + diesel output on top,
battery SOC with min/max lines below) for a representative low-renewable
week — did not exist before this demo; everything else needed
(`plot_energy_flow_balance_with_diesel`, the diesel dispatch mechanics
itself) was already implemented and required no changes.

`tests/test_energy_balance.py`: added
`test_load_side_balance_holds_every_hour_with_diesel_backup`, extending the
existing load-side balance check to include diesel, parametrized across
battery sizes 0/1/3/8 modules.

## Why 4.5 GWh/year, not the originally-suggested 3.0–3.5 GWh/year

The originating prompt suggested targeting ~3.0–3.5 GWh/year at ~800–1,000
kW peak (renewable/load ratio ~1.2), aiming for 65–85% renewable share /
15–35% diesel share. Testing that range directly against an actual
LCOE-optimal battery search showed renewable share at **84–89%** —
diesel present but barely contributing, not a clear demonstration. Higher
loads were tested empirically (renewable share depends on hourly timing,
not just the annual ratio, so this had to be checked by actually running
the optimizer, not assumed):

| Annual load | Peak (achieved) | Ratio | Renewable share | Diesel share |
|---|---|---|---|---|
| 3.0 GWh | 800 kW | 1.39 | 89.2% | 10.8% |
| 3.2 GWh | 853 kW | 1.30 | 87.5% | 12.5% |
| 3.5 GWh | 933 kW | 1.19 | 84.0% | 16.0% |
| 4.0 GWh | 1,066 kW | 1.04 | 79.1% | 20.9% |
| **4.5 GWh** | **1,199 kW** | **0.92** | **74.1%** | **25.9%** |
| 5.0 GWh | 1,333 kW | 0.83 | 69.4% | 30.6% |

4.5 GWh/year was chosen as a clean, mid-window value — comfortably inside
the target 65–85% / 15–35% band, not at either edge.

## Results (from `scripts/diesel_hybrid_demo.py`, a real run — see that
script's output, also saved to `outputs/tables/diesel_hybrid_demo_summary.json`)

- **LCOE-optimal battery**: 24 modules, 6,000 kWh (system LCOE 0.28811 EUR/kWh)
- **Renewable share**: 74.10% | **Diesel share**: 25.90% | **LPSP**: 0.0 (diesel guarantees reliability, as designed)
- **Curtailment**: 18.44% of generated renewable energy (down sharply from ~60% under the off-grid baseline, since the larger load now absorbs most of the renewable production directly or via the battery)
- **Diesel operating hours**: 3,250 / 8,760 (37.1% of the year)
- **Diesel fuel consumption**: 683,550 L/year
- **Diesel rated power**: 1,499.3 kW (= 1.25 × 1,199.4 kW achieved peak, unchanged sizing convention)
- **Diesel capacity factor**: 8.87%
- **Energy balance**: verified exactly — max hourly residual 1.1×10⁻¹³ kWh (floating-point noise), annual residual 0.0 kWh. `load_kw` = `direct_supply_kwh` + `battery_discharge_kwh` + `diesel_output_kwh` + `still_unserved_kwh`, every hour, for the full 8,760-hour year.

**Figures** (`outputs/figures/`):
- `11_diesel_engagement_week.png` — the representative week (Dec 25–31,
  2023, the year's single highest-diesel-output week by inspection of
  weekly totals): renewables dip well below load for most of the week,
  the battery SOC sits near its floor almost the entire time (briefly
  recovering only twice, when a renewable surplus appears), and diesel
  output fills essentially the entire renewable/load gap for most hours.
- `12_energy_flow_balance_diesel_hybrid.png` — annual stacked-bar
  breakdown: renewable production splits into direct supply (2.29 GWh),
  battery charge (1.10 GWh), and curtailed (0.77 GWh); load is served by
  direct supply (2.29 GWh), battery discharge (1.04 GWh), and diesel
  (1.17 GWh), with zero still-unserved.
- `13_renewable_share_vs_lcoe.png` and `14_battery_capacity_vs_lcoe.png` —
  the full 0–80-module candidate sweep (reusing the same
  `plot_renewable_share_vs_lcoe`/`plot_metric_vs_battery_capacity`
  functions the pre-Task-11 build used), with the LCOE-minimizing
  candidate marked. See "Where does more battery stop paying for itself"
  below for the numbers behind these two figures.

### Where does more battery stop paying for itself?

The full 0–80-module sweep traces a clean U-shaped (backward-bending)
curve, exactly the OptiCE "renewable share vs. LCOE" shape the course
lecture material describes:

| Point | Modules | Battery capacity | System LCOE | Renewable share |
|---|---|---|---|---|
| Diesel-only (no battery) | 0 | 0 kWh | 0.346 EUR/kWh | 50.9% |
| **LCOE-minimizing (the search's own optimum)** | **24** | **6,000 kWh** | **0.288 EUR/kWh** | **74.1%** |
| Maximum tested battery | 80 | 20,000 kWh | 0.374 EUR/kWh | 80.4% |

**Reading this**: starting from diesel-only, each added battery module
saves more in diesel fuel than it costs in capital — LCOE falls by 0.058
EUR/kWh (17%) as renewable share climbs from 50.9% to 74.1%. Past 24
modules, the relationship flips: each additional module's capital cost now
exceeds what it saves in fuel, because the *marginal* battery capacity is
mostly covering rarer and rarer high-deficit hours rather than displacing
routine diesel runtime. Pushing all the way to the 80-module cap buys only
another 6.3 percentage points of renewable share (74.1% → 80.4%) but costs
0.086 EUR/kWh (30%) more than the optimum — a much worse trade than the
first 24 modules delivered. In short: **renewable share is worth paying
for up to ~74%; beyond that, each additional percentage point gets
markedly more expensive per kWh delivered.**

## What this does NOT do (explicitly deferred, per the originating task's own scope)

- **The full 5,000-scenario off-grid dataset, the trained neural network,
  and `reports/technical_report.md`/`decision_maker_report.md` are
  untouched.** They remain valid, already-generated results of the
  completed off-grid study (V2 Task 3 checkpoint) — generated under the
  *prior* values of `config/jinan.yaml` (`backup: none`,
  `target_annual_kwh: 1,600,000`) and `config/scenario_generation.yaml`
  (`annual_load_kwh: [800000, 2500000]`, `peak_load_kw: [150, 550]`), which
  this demo's config changes have now moved past. This is expected and
  intentional when a baseline config is deliberately changed, not a defect
  — to reproduce the off-grid study exactly, restore those old values (both
  noted above and in `config/jinan.yaml`/`config/scenario_generation.yaml`'s
  own comments).
- No new full-scale scenario dataset was generated under the new
  diesel-hybrid ranges, and the neural network was not retrained on one.
  Both remain explicitly deferred follow-up work.
- The off-grid code path (`system_backup="none"`) is untouched and fully
  functional — verified by the full 177-test suite passing, including the
  pre-existing off-grid-specific tests in `tests/test_optimization.py`.

## Reproducing this demonstration

```bash
cd v2
python3 -m pytest -q                          # 177 passed
PYTHONPATH=. python3 scripts/diesel_hybrid_demo.py
```
