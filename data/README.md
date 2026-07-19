# Data directory

This directory is git-ignored for large/generated content except for this README and
`.gitkeep` placeholders (see `.gitignore`). Nothing here should be treated as a
permanent source of truth — regenerate from `src/` + `config/`.

## Layout

- `raw/` — Unmodified downloads from the NASA POWER hourly API (and any manually
  supplied CSV fallback, per PROJECT_BRIEF.md §7). Never edited in place.
- `processed/` — Cleaned, timezone-converted, unit-verified hourly weather data ready
  for the PV/wind models, plus generated load profiles.
- `scenarios/` — Mechanistic scenario-generation outputs (inputs, labels, resumable
  cache) produced by `src/scenarios/`.
- `ml_dataset/` — Feature-engineered, split-labelled datasets ready for `src/ai/`.

## NASA POWER API validation (refinement addendum §1.4)

Before the full weather pipeline is built in Stage 2, a standalone validation check
must be run against the NASA POWER hourly API for a short date range around the Jinan
coordinates (36.65, 117.12) to confirm:

1. **Time standard** of the returned timestamps (NASA POWER hourly data is documented
   as UTC; this must be confirmed against actual response metadata, not assumed).
2. **Units** of the solar, wind, and temperature fields actually returned — checked
   against the current NASA POWER API documentation and sanity-checked by magnitude
   (e.g. GHI in W/m² should not silently be MJ/m²/hr or similar), not assumed from
   memory.
3. **Wind reference height** — whether the hourly wind-speed parameter used (e.g.
   `WS10M` vs a 50 m equivalent) is at 10 m or 50 m for both Jinan and Västerås, since
   this feeds directly into the power-law height adjustment in `src/physics/wind_model.py`.

**Status: not yet performed.** This is the first task of Stage 2, before any weather
pipeline code is written against assumed units/timestamps. Findings will be recorded
in this section (time standard, field units, wind reference height, any coverage gaps
or surprises for Jinan and/or Västerås) before Stage 2 proceeds further.

## Manual CSV fallback

If the NASA POWER API is unavailable, `src/data/weather.py` must support loading a
manually supplied CSV with the same schema as the processed output. Instructions for
this fallback will be documented here once `weather.py` is implemented in Stage 2.

## Reproducibility

- Weather downloads are cached in `raw/`; successful downloads are never re-fetched
  unnecessarily.
- Scenario generation in `scenarios/` is resumable and records the random seed and a
  configuration hash per PROJECT_BRIEF.md §14.
- No data in this directory is fabricated. If an API request fails, the pipeline must
  fail loudly (or fall back to the documented manual CSV path) rather than inventing
  observations.
