"""Tests for `src.scenarios.sampling`, `scenario_runner`, and `dataset_builder`
(Stage 4). Not part of the original Stage-1 test-file list (PROJECT_BRIEF.md
§24 didn't anticipate a dedicated scenarios test file), added because these
modules have real behaviour worth covering. Uses small deterministic,
synthetic (non-downloaded) fixtures -- no internet access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.scenarios.dataset_builder import build_dataset
from src.scenarios.sampling import check_scenario_consistency, sample_scenarios
from src.scenarios.scenario_runner import run_scenario

RANGES = {
    "pv_capacity_kwp": [10, 40],
    "wind_capacity_kw": [5, 25],
    "annual_load_kwh": [10000, 50000],
    "peak_load_kw": [5, 20],
    "reliability_target_load_served": [0.990, 0.999],
    "round_trip_efficiency": [0.88, 0.97],
    "usable_soc_window_fraction": [0.70, 0.90],
}


def _valid_scenario() -> dict:
    return {
        "scenario_id": 0,
        "location": "jinan",
        "timezone": "Asia/Shanghai",
        "weather_year": 2023,
        "random_seed": 42,
        "pv_capacity_kwp": 20.0,
        "wind_capacity_kw": 10.0,
        "annual_load_kwh": 30000.0,
        "peak_load_kw": 15.0,
        "reliability_target_load_served": 0.99,
        "round_trip_efficiency": 0.95,
        "usable_soc_window_fraction": 0.85,
    }


def test_check_scenario_consistency_accepts_valid_scenario() -> None:
    is_consistent, reason = check_scenario_consistency(_valid_scenario())
    assert is_consistent
    assert reason is None


def test_check_scenario_consistency_rejects_incompatible_peak_annual_combo() -> None:
    scenario = _valid_scenario()
    scenario["annual_load_kwh"] = 50000.0
    scenario["peak_load_kw"] = 5.0  # far too low a peak for this much annual energy
    is_consistent, reason = check_scenario_consistency(scenario)
    assert not is_consistent
    assert "Incompatible peak_load_kw" in reason


@pytest.mark.parametrize(
    "field,value",
    [
        ("pv_capacity_kwp", 0.0),
        ("wind_capacity_kw", -1.0),
        ("annual_load_kwh", 0.0),
        ("peak_load_kw", 0.0),
        ("reliability_target_load_served", 1.5),
        ("round_trip_efficiency", 0.0),
        ("usable_soc_window_fraction", 1.0),
    ],
)
def test_check_scenario_consistency_rejects_invalid_values(field: str, value: float) -> None:
    scenario = _valid_scenario()
    scenario[field] = value
    is_consistent, _ = check_scenario_consistency(scenario)
    assert not is_consistent


def test_check_scenario_consistency_no_cap_by_default() -> None:
    # Without max_combined_pv_wind_capacity_kw, any positive PV+wind combo
    # is accepted (residential-style config, no cap configured).
    scenario = _valid_scenario()
    scenario["pv_capacity_kwp"] = 3000.0
    scenario["wind_capacity_kw"] = 3000.0
    is_consistent, _ = check_scenario_consistency(scenario)
    assert is_consistent


def test_check_scenario_consistency_rejects_over_combined_cap() -> None:
    scenario = _valid_scenario()
    scenario["pv_capacity_kwp"] = 3000.0
    scenario["wind_capacity_kw"] = 2500.0  # combined 5500 > 5000 cap
    is_consistent, reason = check_scenario_consistency(scenario, max_combined_pv_wind_capacity_kw=5000.0)
    assert not is_consistent
    assert "Combined PV+wind capacity" in reason


def test_check_scenario_consistency_accepts_under_combined_cap() -> None:
    scenario = _valid_scenario()
    scenario["pv_capacity_kwp"] = 1800.0
    scenario["wind_capacity_kw"] = 800.0  # combined 2600 < 5000 cap
    is_consistent, reason = check_scenario_consistency(scenario, max_combined_pv_wind_capacity_kw=5000.0)
    assert is_consistent
    assert reason is None


def test_sample_scenarios_respects_combined_pv_wind_cap() -> None:
    industrial_ranges = {
        "pv_capacity_kwp": [500, 3500],
        "wind_capacity_kw": [200, 2000],
        "annual_load_kwh": [3_000_000, 21_000_000],
        "peak_load_kw": [500, 4200],
        "reliability_target_load_served": [0.990, 0.999],
        "round_trip_efficiency": [0.88, 0.97],
        "usable_soc_window_fraction": [0.70, 0.90],
        "max_combined_pv_wind_capacity_kw": 5000,
    }
    scenarios = sample_scenarios(15, industrial_ranges, 42, "jinan", "Asia/Shanghai", 2023)
    for s in scenarios:
        assert s["pv_capacity_kwp"] + s["wind_capacity_kw"] <= 5000


def test_sample_scenarios_reproducible_with_same_seed() -> None:
    a = sample_scenarios(10, RANGES, 42, "jinan", "Asia/Shanghai", 2023)
    b = sample_scenarios(10, RANGES, 42, "jinan", "Asia/Shanghai", 2023)
    assert a == b


def test_sample_scenarios_different_seed_differs() -> None:
    a = sample_scenarios(10, RANGES, 42, "jinan", "Asia/Shanghai", 2023)
    b = sample_scenarios(10, RANGES, 7, "jinan", "Asia/Shanghai", 2023)
    assert a != b


def test_sample_scenarios_returns_requested_count_with_unique_ids() -> None:
    scenarios = sample_scenarios(15, RANGES, 42, "jinan", "Asia/Shanghai", 2023)
    assert len(scenarios) == 15
    assert len({s["scenario_id"] for s in scenarios}) == 15


def test_sample_scenarios_all_within_configured_ranges() -> None:
    scenarios = sample_scenarios(20, RANGES, 42, "jinan", "Asia/Shanghai", 2023)
    for s in scenarios:
        for field, (low, high) in RANGES.items():
            assert low <= s[field] <= high, f"{field}={s[field]} outside [{low}, {high}]"


def _synthetic_full_year_weather() -> pd.DataFrame:
    """A deterministic, non-downloaded stand-in for a full local year of
    weather -- enough structure (day/night, seasonal wind variation) to
    exercise the full mechanistic pipeline without any I/O."""
    index = pd.date_range("2023-01-01", periods=8760, freq="h", tz="Asia/Shanghai")
    hour = index.hour.to_numpy()
    day_of_year = index.dayofyear.to_numpy()

    daylight = (hour >= 6) & (hour <= 18)
    seasonal = 1.0 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    ghi = np.where(daylight, 700 * np.sin(np.pi * (hour - 6) / 12) * seasonal, 0.0).clip(min=0)
    dni = ghi * 0.7
    dhi = ghi * 0.3
    temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.where(daylight, 5, 0)
    wind_speed = 3 + 2 * np.abs(np.sin(2 * np.pi * day_of_year / 30))

    return pd.DataFrame(
        {
            "ALLSKY_SFC_SW_DWN": ghi,
            "ALLSKY_SFC_SW_DNI": dni,
            "ALLSKY_SFC_SW_DIFF": dhi,
            "T2M": temp,
            "WS10M": wind_speed,
        },
        index=index,
    )


SITE_CONFIG = {
    "site": {"latitude": 36.65, "longitude": 117.12, "timezone": "Asia/Shanghai"},
    "system": {"backup": "diesel"},
    "pv": {"tilt_deg": 36.65, "azimuth_deg": 180.0, "system_losses_fraction": 0.14, "inverter_efficiency": 0.96,
           "installed_cost_eur_per_kwp": 700, "economic_lifetime_years": 25, "om_cost_fraction_per_year": 0.015},
    "wind": {"weather_reference_height_m": 10.0, "hub_height_m": 15.0, "wind_shear_exponent": 0.14,
             "power_curve_speed_power_fraction": [
                 [0.0, 0.0], [2.5, 0.0], [7.0, 0.2489], [11.0, 1.0], [25.0, 1.0], [25.01, 0.0], [50.0, 0.0],
             ],
             "installed_cost_eur_per_kw": 1200, "economic_lifetime_years": 20, "om_cost_fraction_per_year": 0.025},
    "battery": {"module_capacity_kwh": 15.36, "module_rated_power_kw": 12.8, "initial_soc_fraction": 0.5,
                "self_discharge_rate_per_hour": 0.0,
                "installed_cost_eur_per_kwh": 550, "economic_lifetime_years": 10, "project_lifetime_years": 20},
    "diesel": {"sizing_factor": 1.25, "fuel_curve_intercept_l_per_kwh_rated": 0.08145,
               "fuel_curve_slope_l_per_kwh_output": 0.246, "fuel_price_eur_per_l": 0.9,
               "installed_cost_eur_per_kw": 650, "om_cost_fraction_per_year": 0.03,
               "economic_lifetime_years": 15, "project_lifetime_years": 20},
    "economics": {"real_discount_rate": 0.05},
}


def test_run_scenario_label_fields_match_leakage_columns() -> None:
    from src.ai.features import LEAKAGE_COLUMNS

    weather = _synthetic_full_year_weather()
    scenario = _valid_scenario()
    row = run_scenario(scenario, SITE_CONFIG, weather, n_max=5)

    for column in LEAKAGE_COLUMNS:
        assert column in row, f"{column} (used by the leakage guard) missing from run_scenario output"


def test_run_scenario_returns_original_scenario_fields() -> None:
    weather = _synthetic_full_year_weather()
    scenario = _valid_scenario()
    row = run_scenario(scenario, SITE_CONFIG, weather, n_max=5)
    for key, value in scenario.items():
        assert row[key] == value


def test_build_dataset_resumable(tmp_path) -> None:
    site_config = {**SITE_CONFIG, "site": {**SITE_CONFIG["site"]}}

    # Monkeypatch get_processed_weather via a small n_max/scenario count and
    # a real (cached) weather fetch would require internet; instead call
    # build_dataset against a temp dir using the real Jinan config, but keep
    # n_scenarios tiny so this test is fast even if it needs the network-free
    # cached weather from other tests' runs. If no cache is present, skip.
    from src.config import PROJECT_ROOT, load_site_config

    cached_weather_path = PROJECT_ROOT / "data" / "processed" / "jinan_2023_weather.csv"
    if not cached_weather_path.is_file():
        pytest.skip("No cached Jinan 2023 weather available; skipping dataset_builder integration test.")

    full_site_config = load_site_config("jinan")
    out_dir = tmp_path / "scenarios"

    df1 = build_dataset(3, out_dir, random_seed=1, site_config=full_site_config, ranges=RANGES, n_max=3, dataset_name="unit_test")
    assert len(df1) == 3

    df2 = build_dataset(5, out_dir, random_seed=1, site_config=full_site_config, ranges=RANGES, n_max=3, dataset_name="unit_test", resume=True)
    assert len(df2) == 5
    assert set(df1["scenario_id"]).issubset(set(df2["scenario_id"]))


def test_build_dataset_refuses_to_resume_under_a_different_config(tmp_path) -> None:
    # Regression test for a real bug hit during the industrial-scale pivot:
    # resume=True (the default) silently returned an entirely stale dataset
    # generated under the OLD (residential) config, because every
    # scenario_id in the new (industrial) request already existed in the
    # old CSV -- no error, no new rows, just quietly wrong data.
    from src.config import PROJECT_ROOT, load_site_config

    cached_weather_path = PROJECT_ROOT / "data" / "processed" / "jinan_2023_weather.csv"
    if not cached_weather_path.is_file():
        pytest.skip("No cached Jinan 2023 weather available; skipping dataset_builder integration test.")

    full_site_config = load_site_config("jinan")
    out_dir = tmp_path / "scenarios"

    build_dataset(3, out_dir, random_seed=1, site_config=full_site_config, ranges=RANGES, n_max=3, dataset_name="unit_test")

    different_ranges = {**RANGES, "pv_capacity_kwp": [100, 200]}
    with pytest.raises(RuntimeError, match="Refusing to resume"):
        build_dataset(
            3, out_dir, random_seed=1, site_config=full_site_config, ranges=different_ranges,
            n_max=3, dataset_name="unit_test", resume=True,
        )
