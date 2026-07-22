# Data directory

`raw/` and `processed/` (the NASA POWER weather cache) are committed directly, so the
repo is self-contained and reproducible without depending on the NASA POWER API being
reachable or unchanged at clone time. `scenarios/` and `ml_dataset/` are also
committed (see below) since they are explicit project deliverables. Everything here
is still regenerable from `src/` + `config/` if ever needed (`get_processed_weather`
re-fetches from the API when the cache is absent) -- committing it is a convenience
and reproducibility guarantee, not a claim that it's hand-authored or can't be
rebuilt.

## Layout

- `raw/` — Unmodified downloads from the NASA POWER hourly API (and any manually
  supplied CSV fallback, per PROJECT_BRIEF.md §7). Never edited in place.
- `processed/` — Cleaned, timezone-converted, unit-verified hourly weather data ready
  for the PV/wind models, plus generated load profiles.
- `scenarios/` — Mechanistic scenario-generation outputs (inputs, labels, resumable
  cache) produced by `src/scenarios/`.
- `ml_dataset/` — Feature-engineered, split-labelled datasets ready for `src/ai/`.

## NASA POWER API validation (refinement addendum §1.4)

**Status: performed.** Run via a standalone script (not part of `src/`) against
`https://power.larc.nasa.gov/api/temporal/hourly/point`, `community=RE`, for both
Jinan (36.65, 117.12) and Västerås (16.54, 59.61 — note lon/lat order) over a short
sample date range (2023-06-15 to 2023-06-17). Findings:

### 1. Time standard — do NOT assume UTC

By default the hourly endpoint returns **`time_standard: "LST"`** (Local Solar Time —
mean solar time derived from longitude, *not* the site's civil timezone, and with no
DST adjustment). This is confirmed directly in the JSON response's `header.time_standard`
field, not assumed. Using the default LST response and naively labelling it as the
site's civil local time would introduce an error of tens of minutes to over an hour
depending on longitude, and would not shift for DST.

**Decision:** always request the API with the query parameter **`time-standard=UTC`**
(confirmed working — `header.time_standard` then reports `"UTC"` for both Jinan and
Västerås). The pipeline converts those UTC timestamps to each site's local civil time
using its IANA timezone (`Asia/Shanghai` / `Europe/Stockholm`) via pandas
`tz_localize("UTC").tz_convert(timezone)`, which correctly handles the Västerås DST
transition. `src/data/weather.py` must always pass `time-standard=UTC` explicitly.

### 2. Field units — GHI is NOT plain W/m²

Per the response's own `parameters` metadata block (`data["parameters"][name]["units"]`):

| Parameter | Reported unit | Notes |
|---|---|---|
| `ALLSKY_SFC_SW_DWN` | **`Wh/m^2`** | Hourly energy density, not instantaneous W/m². Numerically equal to the average W/m² over that hour (since the accumulation window is exactly 1 h), so it can be used directly wherever an hourly-average irradiance in W/m² is expected — but it must be documented as Wh/m² accumulated-over-the-hour, not silently treated as an instantaneous reading. |
| `T2M` | `C` | Temperature at 2 m, degrees Celsius — as expected. |
| `WS10M` | `m/s` | Wind speed at 10 m — as expected. |
| `WS50M` | `m/s` | Wind speed at 50 m — as expected. |
| `WD10M` | `Degrees` | Wind direction at 10 m. |

Magnitude sanity check (Jinan, mid-June sample): GHI 0–912.5 Wh/m² (plausible for
clear-sky summer midday), WS10M 1.25–5.26 m/s, WS50M 1.39–8.21 m/s — all physically
reasonable, no unit-mismatch red flags (e.g. not MJ/m², not km/h).

### 3. Wind reference height

Both **`WS10M`** (10 m) and **`WS50M`** (50 m) are available on the hourly endpoint for
both Jinan and Västerås. **Decision:** use `WS10M` (10 m reference) as the model input
to `src/physics/wind_model.py`'s power-law height adjustment, since 10 m is closer to
the configured turbine hub height (15 m) than 50 m is — extrapolating a shorter
vertical distance is more defensible than extrapolating from 50 m down to 15 m.
`config/*.yaml` → `wind.weather_reference_height_m` is set to `10`.

### 4. Coverage / surprises

No coverage gaps observed in the sample range for either location. The API's `geometry`
block echoes back a modelled surface elevation for the queried point (Jinan ≈ 183 m,
Västerås ≈ 47 m) — informational only, not currently used by the model.

### 5. Elevation note (informational)

`config/jinan.yaml` and `config/vasteras.yaml` had placeholder `elevation_m` values
(51 m and 20 m respectively) that do not match the API's modelled elevation (183 m and
47 m). This has no effect on the current PV/wind equations (elevation is not yet
consumed anywhere), but is flagged here so it isn't silently forgotten if elevation
becomes model-relevant later (e.g. air-density-dependent wind power).

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
