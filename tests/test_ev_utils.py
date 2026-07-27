import pytest

from utils.EVs.ev_demand import EVDemandConfig
from utils.EVs.ev_utils import (
    assign_income_midpoints,
    assign_nhts_income_bucket,
    assign_urban_from_metro,
    get_census_division_for_state,
    parse_release_for_weather_map,
    resstock_temp_power_mult,
    yuksel_michalek_miles_to_kwh,
)


@pytest.fixture
def test_config():
    return EVDemandConfig(state="NY", release="res_2024_tmy3_2")


def test_get_census_division_for_state():
    assert get_census_division_for_state("NY") == 2
    assert get_census_division_for_state("CA") == 9
    assert get_census_division_for_state("TX") == 7
    assert get_census_division_for_state("FL") == 5
    assert get_census_division_for_state("IL") == 3
    assert get_census_division_for_state("OH") == 3


def test_assign_urban_from_metro():
    assert assign_urban_from_metro("In metro area, principal city") == 1
    assert assign_urban_from_metro("In metro area, not/partially in principal city") == 1
    assert assign_urban_from_metro("Not/partially in metro area") == 2
    with pytest.raises(ValueError, match="Unknown ResStock metro status"):
        assign_urban_from_metro("urban")


def test_assign_nhts_income_bucket():
    """Test the assign_nhts_income_bucket function with key income values."""

    # Test 4 key cases covering different buckets
    assert assign_nhts_income_bucket(5000) == 1  # Low income
    assert assign_nhts_income_bucket(50000) == 6  # Middle income
    assert assign_nhts_income_bucket(150000) == 10  # High income
    assert assign_nhts_income_bucket(250000) == 11  # Very high income


def test_assign_income_midpoints():
    """Test the assign_income_midpoints function with various income range strings."""

    # Test 4 key cases covering different scenarios
    assert assign_income_midpoints("60000-69999") == 64999  # Range midpoint
    assert assign_income_midpoints("0-10000") == 5000  # Low range
    assert assign_income_midpoints("200000") == 200000  # Single value
    assert assign_income_midpoints(None) is None  # None input


def test_parse_release_for_weather_map():
    assert parse_release_for_weather_map("res_2024_tmy3_2") == ("resstock", "2024", "tmy3", "2")
    assert parse_release_for_weather_map("resstock_2024_amy2018_2") == (
        "resstock",
        "2024",
        "amy2018",
        "2",
    )
    with pytest.raises(ValueError, match="Unrecognized release key"):
        parse_release_for_weather_map("not_a_release")


def test_resstock_temp_power_mult_anchors():
    # OpenStudio-HPXML / Speake curve anchors (°F → multiplier).
    assert resstock_temp_power_mult(70.0) == pytest.approx(0.991, abs=0.01)
    assert resstock_temp_power_mult(0.0) == pytest.approx(2.256, abs=0.01)
    assert resstock_temp_power_mult(32.0) == pytest.approx(1.413, abs=0.01)
    assert resstock_temp_power_mult(100.0) == pytest.approx(1.168, abs=0.01)
    # Clip below 0°F / above 100°F to bounds.
    assert resstock_temp_power_mult(-20.0) == pytest.approx(resstock_temp_power_mult(0.0))
    assert resstock_temp_power_mult(120.0) == pytest.approx(resstock_temp_power_mult(100.0))


def test_yuksel_michalek_miles_to_kwh_still_available():
    # Mild Leaf-like intensity ~0.27 kWh/mi at ~70°F.
    assert yuksel_michalek_miles_to_kwh(100.0, 70.0) == pytest.approx(26.88, rel=0.02)


def test_load_resstock_weather_station_temps_hour_ending(tmp_path):
    from datetime import datetime

    import polars as pl

    from utils.EVs.charging import build_hours_base
    from utils.EVs.ev_utils import (
        build_bldg_hourly_temp_f,
        load_resstock_weather_station_temps,
    )

    # Minimal hour-ending TMY snippet: 01:00 is first hour of Jan 1 (clock hour 0).
    weather_csv = tmp_path / "G2400310.csv"
    weather_csv.write_text(
        "date_time,Dry Bulb Temperature [°C]\n"
        "1997-01-01 01:00:00,0.0\n"  # → hour 0, 32°F
        "1997-01-01 02:00:00,10.0\n"  # → hour 1, 50°F
        "1997-01-02 00:00:00,20.0\n"  # → Jan 1 hour 23
    )
    temps = load_resstock_weather_station_temps(weather_csv)
    assert temps.filter((pl.col("month") == 1) & (pl.col("day") == 1) & (pl.col("hour") == 0))[
        "temp_f"
    ][0] == pytest.approx(32.0)
    assert temps.filter((pl.col("month") == 1) & (pl.col("day") == 1) & (pl.col("hour") == 1))[
        "temp_f"
    ][0] == pytest.approx(50.0)
    assert temps.filter((pl.col("month") == 1) & (pl.col("day") == 1) & (pl.col("hour") == 23))[
        "temp_f"
    ][0] == pytest.approx(68.0)

    # Fill remaining hours so a 1-day hours_base can join.
    full = pl.DataFrame({
        "month": [1] * 24,
        "day": [1] * 24,
        "hour": list(range(24)),
        "temp_f": [32.0] * 24,
    })
    hours_base = build_hours_base(datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 23))
    hourly = build_bldg_hourly_temp_f(
        hours_base=hours_base,
        bldg_stations=pl.DataFrame({"bldg_id": ["b1"], "weather_station_name": ["G2400310"]}),
        station_temps={"G2400310": full},
    )
    assert hourly.height == 24
    assert hourly["temp_f"].null_count() == 0
