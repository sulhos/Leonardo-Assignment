"""NASA POWER hourly weather acquisition, cleaning, and caching.

Responsibilities (implemented in Stage 2, after the API validation step described
in PROJECT_BRIEF.md refinement addendum §1.4 and recorded in data/README.md):

- Query the NASA POWER hourly API for a date range covering the requested local
  calendar year (with enough surrounding days to survive timezone conversion).
- Parse the API's documented time standard and convert timestamps to the site's
  local timezone (handling DST transitions for Vasteras).
- Filter to exactly the requested local calendar year and verify record counts
  (8,760 for a non-leap year).
- Detect missing and duplicate timestamps.
- Verify and, if necessary, convert field units against the NASA POWER API
  documentation (never assumed).
- Cache successful raw downloads under data/raw/, and write cleaned output to
  data/processed/.
- Support a manual CSV fallback if the API is unavailable.
- Never fabricate observations for a failed request.

This module intentionally contains no implementation yet -- it is a Stage 1
placeholder. Functions raise NotImplementedError until Stage 2.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

NASA_POWER_HOURLY_BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"


def fetch_hourly_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    parameters: list[str],
) -> pd.DataFrame:
    """Fetch raw hourly weather data from the NASA POWER API.

    Args:
        latitude: Site latitude in decimal degrees.
        longitude: Site longitude in decimal degrees.
        start_date: Inclusive start date, "YYYYMMDD".
        end_date: Inclusive end date, "YYYYMMDD".
        parameters: NASA POWER hourly parameter codes to request.

    Returns:
        Raw API response as a DataFrame, timestamps in the API's native time
        standard (not yet converted to local time).
    """
    raise NotImplementedError("Implemented in Stage 2, after API validation (addendum §1.4).")


def convert_to_local_year(
    raw: pd.DataFrame, timezone: str, year: int
) -> pd.DataFrame:
    """Convert raw (native time standard) weather data to a local-timezone,
    single-calendar-year hourly series, verifying record count and detecting
    missing/duplicate timestamps."""
    raise NotImplementedError("Implemented in Stage 2.")


def validate_units(raw: pd.DataFrame, parameters: list[str]) -> None:
    """Sanity-check that returned field units match NASA POWER API documentation
    (e.g. solar irradiance magnitude, wind speed reference height)."""
    raise NotImplementedError("Implemented in Stage 2.")


def load_manual_csv_fallback(path: Path) -> pd.DataFrame:
    """Load a manually supplied CSV in place of a failed API download.

    The expected schema will be documented in data/README.md once defined.
    """
    raise NotImplementedError("Implemented in Stage 2.")


def get_processed_weather(site: str, year: int, use_cache: bool = True) -> pd.DataFrame:
    """High-level entry point: return cleaned, local-year hourly weather data
    for a configured site, using the raw/processed cache where possible."""
    raise NotImplementedError("Implemented in Stage 2.")
