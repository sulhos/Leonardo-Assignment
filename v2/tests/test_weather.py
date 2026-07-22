"""Tests for `src.data.weather` (PROJECT_BRIEF.md §26).

All tests here use small, in-memory, deterministic fixtures -- no NASA POWER
API calls. The real API-validation findings (time standard, units, wind
reference height) are documented in `data/README.md` (addendum §1.4); these
tests check that `src/data/weather.py`'s local logic (timezone conversion,
missing/duplicate detection, unit checking) behaves correctly given
already-fetched data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.weather import (
    DEFAULT_PARAMETERS,
    EXPECTED_UNITS,
    WeatherAPIError,
    convert_to_local_year,
    validate_units,
)


def _make_utc_raw(start_utc: str, end_utc: str, columns: list[str] | None = None) -> pd.DataFrame:
    cols = columns or DEFAULT_PARAMETERS
    index = pd.date_range(start=start_utc, end=end_utc, freq="h", tz="UTC")
    data = {c: [50.0] * len(index) for c in cols}
    df = pd.DataFrame(data, index=index)
    df.attrs["units"] = {c: {"units": EXPECTED_UNITS.get(c, "unknown")} for c in cols}
    return df


def test_convert_to_local_year_non_leap_jinan() -> None:
    raw = _make_utc_raw("2022-12-29T00:00:00Z", "2024-01-03T23:00:00Z")
    local = convert_to_local_year(raw, "Asia/Shanghai", 2023)
    assert len(local) == 8760
    assert local.index[0] == pd.Timestamp("2023-01-01 00:00:00", tz="Asia/Shanghai")
    assert local.index[-1] == pd.Timestamp("2023-12-31 23:00:00", tz="Asia/Shanghai")


def test_convert_to_local_year_handles_dst_vasteras() -> None:
    raw = _make_utc_raw("2022-12-29T00:00:00Z", "2024-01-03T23:00:00Z")
    local = convert_to_local_year(raw, "Europe/Stockholm", 2023)
    assert len(local) == 8760
    assert str(local.index.tz) == "Europe/Stockholm"


def test_convert_to_local_year_no_duplicate_timestamps() -> None:
    raw = _make_utc_raw("2022-12-29T00:00:00Z", "2024-01-03T23:00:00Z")
    local = convert_to_local_year(raw, "Asia/Shanghai", 2023)
    assert not local.index.has_duplicates


def test_convert_to_local_year_detects_missing_timestamps() -> None:
    raw = _make_utc_raw("2022-12-29T00:00:00Z", "2024-01-03T23:00:00Z")
    raw = raw.drop(raw.index[100])  # remove one hour from the middle of the buffer/year
    with pytest.raises(WeatherAPIError, match="Missing"):
        convert_to_local_year(raw, "Asia/Shanghai", 2023)


def test_convert_to_local_year_no_nans_when_input_complete() -> None:
    raw = _make_utc_raw("2022-12-29T00:00:00Z", "2024-01-03T23:00:00Z")
    local = convert_to_local_year(raw, "Asia/Shanghai", 2023)
    assert local.isna().sum().sum() == 0


def test_convert_to_local_year_rejects_naive_index() -> None:
    raw = _make_utc_raw("2022-12-29T00:00:00Z", "2024-01-03T23:00:00Z")
    raw.index = raw.index.tz_localize(None)
    with pytest.raises(WeatherAPIError, match="tz-aware"):
        convert_to_local_year(raw, "Asia/Shanghai", 2023)


def test_validate_units_passes_for_expected_units() -> None:
    raw = _make_utc_raw("2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z")
    validate_units(raw)  # should not raise


def test_validate_units_rejects_mismatched_units() -> None:
    raw = _make_utc_raw("2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z")
    raw.attrs["units"]["T2M"] = {"units": "F"}  # wrong unit, should have been "C"
    with pytest.raises(WeatherAPIError, match="Unit mismatch"):
        validate_units(raw)


def test_validate_units_rejects_implausible_magnitude() -> None:
    raw = _make_utc_raw("2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z")
    raw["T2M"] = 500.0  # not a plausible air temperature in Celsius
    with pytest.raises(WeatherAPIError, match="out of plausible range"):
        validate_units(raw)
