"""NASA POWER hourly weather acquisition, cleaning, and caching.

Findings from the standalone API-validation check (refinement addendum §1.4,
full detail in `data/README.md`) that this module's behaviour depends on:

- The hourly endpoint defaults to Local Solar Time (`time_standard: "LST"`),
  which is neither UTC nor the site's civil timezone. This module always
  requests `time-standard=UTC` explicitly and converts to local civil time
  (with correct DST handling) using the site's IANA timezone.
- `ALLSKY_SFC_SW_DWN` (and DNI/DIFF) are reported in **Wh/m^2** (hourly
  energy density), not instantaneous W/m^2 -- numerically usable as an
  hourly-average irradiance, but never assumed without checking the
  response's own `parameters` metadata block.
- Wind speed is requested at 10 m (`WS10M`), matching the wind model's
  configured `weather_reference_height_m: 10` (closer to the 15 m default
  turbine hub height than the also-available 50 m parameter).

Never fabricates observations: a failed API request raises rather than
returning synthetic data, and callers may fall back to a manually supplied
CSV via `load_manual_csv_fallback`.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import PROJECT_ROOT, load_site_config

logger = logging.getLogger(__name__)

NASA_POWER_HOURLY_BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# Confirmed against the NASA POWER hourly API response metadata (addendum §1.4).
DEFAULT_PARAMETERS = [
    "ALLSKY_SFC_SW_DWN",  # GHI, Wh/m^2 (hourly)
    "ALLSKY_SFC_SW_DNI",  # DNI, Wh/m^2 (hourly)
    "ALLSKY_SFC_SW_DIFF",  # DHI, Wh/m^2 (hourly)
    "T2M",  # air temperature at 2 m, C
    "WS10M",  # wind speed at 10 m, m/s
]

EXPECTED_UNITS = {
    "ALLSKY_SFC_SW_DWN": "Wh/m^2",
    "ALLSKY_SFC_SW_DNI": "Wh/m^2",
    "ALLSKY_SFC_SW_DIFF": "Wh/m^2",
    "T2M": "C",
    "WS10M": "m/s",
}

# Loose physical plausibility bounds for the magnitude sanity check.
SANITY_BOUNDS = {
    "ALLSKY_SFC_SW_DWN": (0.0, 1500.0),
    "ALLSKY_SFC_SW_DNI": (0.0, 1500.0),
    "ALLSKY_SFC_SW_DIFF": (0.0, 1500.0),
    "T2M": (-60.0, 60.0),
    "WS10M": (0.0, 60.0),
}

FILL_VALUE = -999.0
REQUEST_TIMEOUT_SECONDS = 60


class WeatherAPIError(RuntimeError):
    """Raised when the NASA POWER API request fails or returns unusable data."""


def fetch_hourly_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    parameters: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch raw hourly weather data from the NASA POWER API in UTC.

    Args:
        latitude: Site latitude in decimal degrees.
        longitude: Site longitude in decimal degrees.
        start_date: Inclusive start date, "YYYYMMDD".
        end_date: Inclusive end date, "YYYYMMDD".
        parameters: NASA POWER hourly parameter codes to request. Defaults to
            `DEFAULT_PARAMETERS`.

    Returns:
        DataFrame indexed by a UTC `DatetimeIndex`, one column per parameter,
        with the fill value (-999.0) replaced by NaN. `df.attrs["units"]`
        holds the units reported by the API for each parameter, and
        `df.attrs["raw_response"]` holds the full parsed JSON response.

    Raises:
        WeatherAPIError: If the request fails or the response cannot be parsed.
    """
    params = parameters or DEFAULT_PARAMETERS
    query = {
        "parameters": ",".join(params),
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": start_date,
        "end": end_date,
        "format": "JSON",
        "time-standard": "UTC",
    }

    logger.info(
        "Fetching NASA POWER hourly data: lat=%s lon=%s %s..%s params=%s",
        latitude, longitude, start_date, end_date, params,
    )
    try:
        response = requests.get(NASA_POWER_HOURLY_BASE_URL, params=query, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise WeatherAPIError(f"NASA POWER API request failed: {exc}") from exc

    header = data.get("header", {})
    if header.get("time_standard") != "UTC":
        raise WeatherAPIError(
            f"Expected time_standard=UTC in API response, got {header.get('time_standard')!r}. "
            "The API contract may have changed -- do not proceed without re-validating."
        )

    properties = data.get("properties", {}).get("parameter", {})
    if not properties:
        raise WeatherAPIError(f"NASA POWER API response contained no parameter data: {data}")

    series = {}
    for name in params:
        if name not in properties:
            raise WeatherAPIError(f"Requested parameter {name!r} missing from API response.")
        values = properties[name]
        index = pd.to_datetime(list(values.keys()), format="%Y%m%d%H", utc=True)
        series[name] = pd.Series(values.values(), index=index, dtype="float64").sort_index()

    df = pd.DataFrame(series)
    df = df.replace(FILL_VALUE, pd.NA).astype("float64")
    df.attrs["units"] = data.get("parameters", {})
    df.attrs["raw_response"] = data
    return df


def validate_units(df: pd.DataFrame, parameters: list[str] | None = None) -> None:
    """Confirm the API's reported units match `EXPECTED_UNITS` and that
    values fall within loose physical plausibility bounds.

    Raises:
        WeatherAPIError: If a unit mismatch or an out-of-range value is found.
    """
    params = parameters or list(df.columns)
    reported_units = df.attrs.get("units", {})

    for name in params:
        expected = EXPECTED_UNITS.get(name)
        reported = reported_units.get(name, {}).get("units")
        if expected is not None and reported is not None and reported != expected:
            raise WeatherAPIError(
                f"Unit mismatch for {name}: expected {expected!r}, API reported {reported!r}. "
                "Re-run the addendum §1.4 validation before trusting this data."
            )

        if name in SANITY_BOUNDS:
            lower, upper = SANITY_BOUNDS[name]
            column = df[name].dropna()
            if column.empty:
                continue
            if column.min() < lower or column.max() > upper:
                raise WeatherAPIError(
                    f"{name} values out of plausible range [{lower}, {upper}]: "
                    f"min={column.min()}, max={column.max()}"
                )


def convert_to_local_year(raw: pd.DataFrame, timezone: str, year: int) -> pd.DataFrame:
    """Convert UTC weather data to a local-timezone, single-calendar-year
    hourly series, verifying record count and detecting missing/duplicate
    timestamps.

    Args:
        raw: UTC-indexed DataFrame from `fetch_hourly_weather`, covering at
            least the requested local year plus timezone-offset buffer.
        timezone: IANA timezone name (e.g. "Asia/Shanghai").
        year: Local calendar year to extract.

    Returns:
        DataFrame indexed by a local-timezone `DatetimeIndex`, containing
        exactly one row per hour of the requested local year (8,760 rows for
        a non-leap year, 8,784 for a leap year).

    Raises:
        WeatherAPIError: If timestamps are missing, duplicated, or the
            record count does not match the expected length.
    """
    if raw.index.tz is None:
        raise WeatherAPIError("Expected a UTC tz-aware index; got a naive index.")

    local = raw.tz_convert(timezone)

    if local.index.has_duplicates:
        dupes = local.index[local.index.duplicated()].tolist()
        raise WeatherAPIError(f"Duplicate local timestamps detected: {dupes[:5]}...")

    start_local = pd.Timestamp(year=year, month=1, day=1, hour=0, tz=timezone)
    end_local = pd.Timestamp(year=year, month=12, day=31, hour=23, tz=timezone)
    local_year = local.loc[(local.index >= start_local) & (local.index <= end_local)]

    expected_index = pd.date_range(start=start_local, end=end_local, freq="h", tz=timezone)
    missing = expected_index.difference(local_year.index)
    if len(missing) > 0:
        raise WeatherAPIError(
            f"Missing {len(missing)} local timestamps for {year} in {timezone}: "
            f"{list(missing[:5])}..."
        )

    local_year = local_year.reindex(expected_index)
    expected_length = len(expected_index)
    if len(local_year) != expected_length:
        raise WeatherAPIError(
            f"Expected {expected_length} local hourly records for {year}, got {len(local_year)}."
        )

    local_year.attrs = raw.attrs
    return local_year


def load_manual_csv_fallback(path: Path) -> pd.DataFrame:
    """Load a manually supplied CSV in place of a failed API download.

    Expected schema: a "timestamp" column parseable as ISO-8601 local
    civil time (tz-aware or naive-but-documented-local), plus one column
    per entry in `DEFAULT_PARAMETERS` using the same units as
    `EXPECTED_UNITS`. This fallback is documented here (rather than only in
    code) so it can be followed without reading source: if the NASA POWER
    API is unavailable, prepare a CSV matching this schema and pass its path
    to `get_processed_weather(..., manual_csv_path=path)`.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Manual weather CSV not found: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    logger.warning("Loaded manual CSV weather fallback from %s -- API data was not used.", path)
    return df


def _raw_cache_path(site: str, year: int) -> Path:
    return PROJECT_ROOT / "data" / "raw" / f"{site}_{year}_raw.json"


def _processed_cache_path(site: str, year: int) -> Path:
    return PROJECT_ROOT / "data" / "processed" / f"{site}_{year}_weather.csv"


def get_processed_weather(
    site: str,
    year: int,
    use_cache: bool = True,
    manual_csv_path: Path | None = None,
) -> pd.DataFrame:
    """High-level entry point: return cleaned, local-year hourly weather data
    for a configured site, using the raw/processed cache where possible.

    Args:
        site: Site config name (e.g. "jinan", "vasteras").
        year: Local calendar year to retrieve.
        use_cache: If True, reuse a cached processed CSV (or raw JSON) rather
            than re-fetching from the API.
        manual_csv_path: If given, load from this CSV instead of the API
            (addendum's manual-CSV fallback), skipping caching.

    Returns:
        Local-year hourly weather DataFrame (see `convert_to_local_year`).
    """
    if manual_csv_path is not None:
        return load_manual_csv_fallback(manual_csv_path)

    config = load_site_config(site)
    latitude = config["site"]["latitude"]
    longitude = config["site"]["longitude"]
    timezone = config["site"]["timezone"]

    processed_path = _processed_cache_path(site, year)
    if use_cache and processed_path.is_file():
        logger.info("Loading cached processed weather from %s", processed_path)
        df = pd.read_csv(processed_path, parse_dates=["timestamp"], index_col="timestamp")
        # CSV round-trips lose the IANA zone (pandas restores a fixed UTC
        # offset instead), which silently breaks DST-awareness for later
        # comparisons/merges -- always re-attach the real IANA timezone.
        df.index = df.index.tz_convert(timezone)
        return df

    raw_path = _raw_cache_path(site, year)
    if use_cache and raw_path.is_file():
        logger.info("Loading cached raw weather from %s", raw_path)
        with raw_path.open("r", encoding="utf-8") as f:
            cached = json.load(f)
        raw = _raw_json_to_dataframe(cached)
    else:
        # Buffer 2 days on each side: covers any UTC offset (max +-14h) and
        # DST transitions, guaranteeing full local-year coverage after conversion.
        start = dt.date(year, 1, 1) - dt.timedelta(days=2)
        end = dt.date(year, 12, 31) + dt.timedelta(days=2)
        raw = fetch_hourly_weather(
            latitude, longitude, start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
        )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("w", encoding="utf-8") as f:
            json.dump(raw.attrs["raw_response"], f)
        logger.info("Cached raw weather to %s", raw_path)

    validate_units(raw)
    local_year = convert_to_local_year(raw, timezone, year)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    local_year.to_csv(processed_path, index_label="timestamp")
    logger.info("Cached processed weather to %s", processed_path)
    return local_year


def _raw_json_to_dataframe(data: dict[str, Any]) -> pd.DataFrame:
    """Reconstruct the `fetch_hourly_weather` DataFrame from a cached raw
    JSON response (avoids re-hitting the API for cached downloads)."""
    properties = data["properties"]["parameter"]
    series = {}
    for name, values in properties.items():
        index = pd.to_datetime(list(values.keys()), format="%Y%m%d%H", utc=True)
        series[name] = pd.Series(values.values(), index=index, dtype="float64").sort_index()
    df = pd.DataFrame(series).replace(FILL_VALUE, pd.NA).astype("float64")
    df.attrs["units"] = data.get("parameters", {})
    df.attrs["raw_response"] = data
    return df
