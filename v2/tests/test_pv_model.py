"""Tests for `src.physics.pv_model` (PROJECT_BRIEF.md §26).

Uses a small synthetic (not downloaded) weather fixture: a single local day
at the Jinan coordinates with a symmetric daytime GHI/DNI/DHI bell curve and
zero irradiance at night, so tests do not require internet access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib
import pytest

from src.physics.pv_model import (
    clearness_index,
    compute_pv_generation,
    erbs_decomposition,
    extraterrestrial_ghi,
    incidence_angle_modifier,
    liu_jordan_transposition,
)

LATITUDE = 36.65
LONGITUDE = 117.12
TILT_DEG = 36.65
AZIMUTH_DEG = 180.0


def _synthetic_day_weather() -> pd.DataFrame:
    index = pd.date_range("2023-06-15 00:00", periods=24, freq="h", tz="Asia/Shanghai")
    hour = index.hour.to_numpy()
    # Daylight roughly 05:00-19:00 at this latitude in June; bell-shaped GHI peaking at noon.
    daylight = (hour >= 5) & (hour <= 19)
    ghi = np.where(daylight, 800 * np.sin(np.pi * (hour - 5) / 14), 0.0)
    ghi = np.clip(ghi, 0, None)
    dni = ghi * 0.75  # rough direct fraction, illustrative only
    dhi = ghi * 0.30
    temp = 25.0 + 5 * np.sin(np.pi * (hour - 5) / 14)
    temp = np.where(daylight, temp, 20.0)
    return pd.DataFrame(
        {
            "ALLSKY_SFC_SW_DWN": ghi,
            "ALLSKY_SFC_SW_DNI": dni,
            "ALLSKY_SFC_SW_DIFF": dhi,
            "T2M": temp,
            "WS10M": 3.0,
        },
        index=index,
    )


def _compute(weather: pd.DataFrame, capacity_kwp: float = 20.0) -> pd.Series:
    return compute_pv_generation(
        weather=weather,
        capacity_kwp=capacity_kwp,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        tilt_deg=TILT_DEG,
        azimuth_deg=AZIMUTH_DEG,
        system_losses_fraction=0.14,
        inverter_efficiency=0.96,
    )


def test_no_negative_pv_generation() -> None:
    pv = _compute(_synthetic_day_weather())
    assert (pv >= 0).all()


def test_pv_zero_at_night() -> None:
    weather = _synthetic_day_weather()
    pv = _compute(weather)
    night_mask = weather["ALLSKY_SFC_SW_DWN"] == 0
    assert night_mask.sum() > 0
    assert (pv[night_mask] == 0).all()


def test_pv_never_exceeds_capacity() -> None:
    capacity_kwp = 20.0
    pv = _compute(_synthetic_day_weather(), capacity_kwp=capacity_kwp)
    assert (pv <= capacity_kwp).all()


def test_pv_positive_during_peak_daylight() -> None:
    pv = _compute(_synthetic_day_weather())
    assert pv.loc[pv.index.hour == 12].iloc[0] > 0


def test_missing_required_column_raises() -> None:
    weather = _synthetic_day_weather().drop(columns=["T2M"])
    with pytest.raises(ValueError, match="missing required columns"):
        _compute(weather)


# --- Lecture-equation hand-computed reference tests (Task 1a) ---
# Each expected value below is independently hand-computed from the lecture's
# own formula (module docstring), not derived by calling the function under
# test with different inputs.


def test_extraterrestrial_ghi_matches_hand_computed_reference() -> None:
    # EGHI = 1367 * (1 + 0.033*cos(2*pi*172/365.25)) * sin(radians(60))
    #      = 1145.4402... W/m^2 (day 172 ~ June 21, solar elevation 60 deg)
    eghi = extraterrestrial_ghi(day_of_year=172, solar_elevation_deg=60.0)
    assert eghi == pytest.approx(1145.4402, abs=0.01)


def test_extraterrestrial_ghi_zero_below_horizon() -> None:
    assert extraterrestrial_ghi(day_of_year=100, solar_elevation_deg=-5.0) == 0.0


def test_clearness_index_matches_hand_computed_reference() -> None:
    # k_t = 700 / 1145.4402 = 0.6111...
    k_t = clearness_index(ghi=700.0, eghi=1145.4402)
    assert k_t == pytest.approx(0.611119, abs=1e-5)


def test_clearness_index_zero_at_night() -> None:
    assert clearness_index(ghi=0.0, eghi=0.0) == 0.0


def test_erbs_decomposition_matches_hand_computed_reference_all_three_regimes() -> None:
    # Low regime (k_t=0.15 <= 0.22): DHI = (1 - 0.09*0.15)*500 = 493.25
    dhi_low = erbs_decomposition(ghi=np.array([500.0]), k_t=np.array([0.15]))
    assert dhi_low[0] == pytest.approx(493.25, abs=0.01)

    # Mid regime (k_t=0.5, 0.22 < k_t <= 0.8): cubic polynomial * 500 = 329.575
    dhi_mid = erbs_decomposition(ghi=np.array([500.0]), k_t=np.array([0.5]))
    assert dhi_mid[0] == pytest.approx(329.575, abs=0.01)

    # High regime (k_t=0.9 > 0.8): DHI = 0.165*500 = 82.5
    dhi_high = erbs_decomposition(ghi=np.array([500.0]), k_t=np.array([0.9]))
    assert dhi_high[0] == pytest.approx(82.5, abs=0.01)


def test_liu_jordan_transposition_matches_hand_computed_reference() -> None:
    # GHI=600, DHI=150, AOI=20deg, zenith=40deg, tilt=36.65deg, albedo=0.25:
    # R_b = cos(20)/cos(40) = 1.226682..., beam=R_b*(600-150)=552.007
    # sky=150*(1+cos(36.65))/2=135.172, ground=0.25*600*(1-cos(36.65))/2=14.828
    # GTI = 552.007 + 135.172 + 14.828 = 702.007 W/m^2
    gti = liu_jordan_transposition(
        ghi=np.array([600.0]), dhi=np.array([150.0]), aoi_deg=np.array([20.0]),
        zenith_deg=np.array([40.0]), tilt_deg=36.65, ground_albedo=0.25,
    )
    assert gti[0] == pytest.approx(702.007, abs=0.01)


def test_liu_jordan_transposition_no_beam_when_sun_behind_panel() -> None:
    # AOI=100deg (sun behind the panel plane): R_b clips to 0, only diffuse/ground remain.
    gti = liu_jordan_transposition(
        ghi=np.array([600.0]), dhi=np.array([150.0]), aoi_deg=np.array([100.0]),
        zenith_deg=np.array([40.0]), tilt_deg=36.65, ground_albedo=0.25,
    )
    expected_diffuse_and_ground = 150.0 * (1 + np.cos(np.radians(36.65))) / 2 + 0.25 * 600.0 * (
        1 - np.cos(np.radians(36.65))
    ) / 2
    assert gti[0] == pytest.approx(expected_diffuse_and_ground, abs=0.01)


def test_incidence_angle_modifier_matches_hand_computed_reference() -> None:
    # IAM(aoi=30deg) = 1 - 0.05*(1/cos(30deg) - 1) = 0.992265...
    iam = incidence_angle_modifier(np.array([30.0]))
    assert iam[0] == pytest.approx(0.992265, abs=1e-5)


def test_incidence_angle_modifier_is_one_at_normal_incidence() -> None:
    iam = incidence_angle_modifier(np.array([0.0]))
    assert iam[0] == pytest.approx(1.0, abs=1e-9)


def test_incidence_angle_modifier_clipped_to_unit_interval() -> None:
    iam = incidence_angle_modifier(np.array([89.9, 0.0]))
    assert (iam >= 0.0).all() and (iam <= 1.0).all()


def test_pvlib_isotropic_matches_lecture_liu_jordan_formula_componentwise() -> None:
    """Numerically verifies the module docstring's equivalence claim: pvlib's
    'isotropic' transposition model computes each Liu & Jordan term
    (sky-diffuse, ground-reflected, and beam-via-DNI*cos(AOI)) identically
    to the lecture's own formula, using real (not synthetic) weather data
    where GHI/DNI/DHI are not perfectly self-consistent."""
    weather = pd.read_csv("data/processed/jinan_2023_weather.csv", index_col=0, parse_dates=True)
    sample = weather[weather["ALLSKY_SFC_SW_DWN"] > 50].iloc[::500].head(8)

    tilt_deg, azimuth_deg, albedo = 36.65, 180.0, 0.25
    tilt_rad = np.radians(tilt_deg)
    solpos = pvlib.solarposition.get_solarposition(sample.index, 36.65, 117.12)
    aoi_deg = pvlib.irradiance.aoi(tilt_deg, azimuth_deg, solpos["apparent_zenith"], solpos["azimuth"])

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt_deg, surface_azimuth=azimuth_deg,
        solar_zenith=solpos["apparent_zenith"], solar_azimuth=solpos["azimuth"],
        dni=sample["ALLSKY_SFC_SW_DNI"], ghi=sample["ALLSKY_SFC_SW_DWN"], dhi=sample["ALLSKY_SFC_SW_DIFF"],
        albedo=albedo,
    )

    ghi = sample["ALLSKY_SFC_SW_DWN"].to_numpy()
    dhi = sample["ALLSKY_SFC_SW_DIFF"].to_numpy()
    dni = sample["ALLSKY_SFC_SW_DNI"].to_numpy()
    cos_aoi_clipped = np.clip(np.cos(np.radians(aoi_deg.to_numpy())), 0.0, None)

    lecture_sky_diffuse = dhi * (1 + np.cos(tilt_rad)) / 2
    lecture_ground_diffuse = albedo * ghi * (1 - np.cos(tilt_rad)) / 2
    lecture_beam_from_dni = dni * cos_aoi_clipped

    np.testing.assert_allclose(poa["poa_sky_diffuse"].to_numpy(), lecture_sky_diffuse, atol=1e-6)
    np.testing.assert_allclose(poa["poa_ground_diffuse"].to_numpy(), lecture_ground_diffuse, atol=1e-6)
    np.testing.assert_allclose(poa["poa_direct"].to_numpy(), lecture_beam_from_dni, atol=1e-6)
