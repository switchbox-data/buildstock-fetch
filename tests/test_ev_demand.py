from datetime import datetime
from itertools import pairwise
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from utils.ev_demand import (
    HOURS_PER_YEAR,
    EVDemandCalculator,
    VehicleProfile,
    nhts_departure_hour,
    nhts_arrival_hour,
    summarize_nhts_match_catalog,
)


# Test data fixtures
@pytest.fixture
def mock_nhts_data():
    data = {
        "hh_vehicle_id": ["v1", "v2", "v3", "v4", "v4", "v1", "v3"],  # Added weekend trips for v1 and v3
        "income_bucket": [1, 2, 2, 3, 3, 1, 2],  # v2 and v3 both match b2's income=2
        "occupants": [2, 3, 3, 4, 4, 2, 3],  # v2 and v3 both match b2's occupants=3
        "vehicles": [1, 2, 2, 1, 1, 1, 2],  # v2 and v3 both from 2-vehicle households
        "weekday": [2, 2, 2, 2, 2, 1, 1],  # Added weekend trips (1) for v1 and v3
        "start_time": [800, 900, 1000, 800, 1300, 1100, 1400],  # Added weekend start times
        "end_time": [1700, 1800, 1900, 1200, 1700, 1500, 1800],  # Added weekend end times
        "miles_driven": [20.0, 30.0, 40.0, 10.0, 15.0, 25.0, 35.0],  # Added weekend miles
        "trip_weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Added weekend weights
    }
    return pl.DataFrame(data)  # Return eager DataFrame to match production


@pytest.fixture
def mock_metadata():
    data = {
        "bldg_id": ["b1", "b2", "b3"],
        "income_bucket": [1, 2, 3],
        "occupants": [2, 3, 4],
        "vehicles": [1, 2, 1],  # b1 has 1 vehicle, b2 has 2, b3 has 1
        "metro": ["urban", "suburban", "rural"],
    }
    return pl.DataFrame(data)


@pytest.fixture
def mock_metadata_with_zero():
    data = {
        "bldg_id": ["b1", "b2", "b3", "b4"],  # Added b4 with 0 vehicles
        "income_bucket": [1, 2, 3, 2],
        "occupants": [2, 3, 4, 1],
        "vehicles": [1, 2, 1, 0],  # b4 has 0 vehicles
        "metro": ["urban", "suburban", "rural", "urban"],
    }
    return pl.DataFrame(data)


@pytest.fixture
def calculator(mock_nhts_data, mock_metadata):
    return EVDemandCalculator(
        metadata_df=mock_metadata,
        nhts_df=mock_nhts_data,
        pums_df=mock_metadata,  # Using same data for simplicity
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 1, 7),
        random_state=42,
    )


def test_nhts_hour_conversion():
    assert nhts_departure_hour(830) == 8
    assert nhts_departure_hour(1700) == 17
    assert nhts_arrival_hour(1700) == 17
    assert nhts_arrival_hour(1715) == 18
    assert nhts_arrival_hour(1450) == 15


def test_find_best_matches(calculator):
    # Test exact match for single vehicle
    match_type, vehicle_ids = calculator.find_best_matches(
        target_income=1, target_occupants=2, target_vehicles=1, num_samples=1, weekday=True
    )
    assert match_type == "exact"
    assert vehicle_ids == ["v1"]

    # Test exact match for multiple vehicles
    match_type, vehicle_ids = calculator.find_best_matches(
        target_income=2, target_occupants=3, target_vehicles=2, num_samples=2, weekday=True
    )
    assert match_type == "exact"
    assert len(vehicle_ids) == 2
    assert set(vehicle_ids) == {"v2", "v3"}  # Should get both vehicles with matching characteristics

    # Test partial match (income and occupants only)
    match_type, vehicle_ids = calculator.find_best_matches(
        target_income=2,
        target_occupants=3,
        target_vehicles=1,  # Different from data
        num_samples=1,
        weekday=True,
    )
    assert match_type == "income_occupants"
    assert vehicle_ids[0] in ["v2", "v3"]

    # Test income-only match
    match_type, vehicle_ids = calculator.find_best_matches(
        target_income=3,
        target_occupants=2,  # Different from data
        target_vehicles=1,  # Different from data
        num_samples=1,
        weekday=True,
    )
    assert match_type == "income_only"
    assert vehicle_ids[0] == "v4"  # v4 has income=3

    # Test closest income match
    match_type, vehicle_ids = calculator.find_best_matches(
        target_income=4,  # Not in data
        target_occupants=2,
        target_vehicles=1,
        num_samples=1,
        weekday=True,
    )
    assert match_type == "closest_income"
    assert vehicle_ids[0] in ["v1", "v2", "v3", "v4"]


def test_sample_vehicle_profiles(calculator):
    profiles = calculator.sample_vehicle_profiles(calculator.metadata_df, calculator.nhts_df)

    # Check that we got profiles for each vehicle
    expected_vehicle_count = calculator.metadata_df["vehicles"].sum()
    assert len(profiles) == expected_vehicle_count  # Should be 4 (b1:1, b2:2, b3:1)

    # Expected profiles with calculator's random_state=42
    expected_profiles = {
        ("b1", 1): {  # Building 1 has 1 vehicle (matches v1)
            "weekday_departure_hour": [8],
            "weekday_arrival_hour": [17],
            "weekday_miles": [20.0],
            "weekday_trip_weights": [1.0],
            "weekend_departure_hour": [11],  # Now has weekend trips
            "weekend_arrival_hour": [15],
            "weekend_miles": [25.0],
            "weekend_trip_weights": [1.0],
            "weekday_trip_ids": [1],
            "weekend_trip_ids": [1],
        },
        ("b2", 1): {  # Building 2 first vehicle (matches v3)
            "weekday_departure_hour": [10],
            "weekday_arrival_hour": [19],
            "weekday_miles": [40.0],
            "weekday_trip_weights": [1.0],
            "weekend_departure_hour": [14],  # v3 has weekend trips
            "weekend_arrival_hour": [18],
            "weekend_miles": [35.0],
            "weekend_trip_weights": [1.0],
            "weekday_trip_ids": [1],
            "weekend_trip_ids": [1],
        },
        ("b2", 2): {  # Building 2 second vehicle (matches v2)
            "weekday_departure_hour": [9],
            "weekday_arrival_hour": [18],
            "weekday_miles": [30.0],
            "weekday_trip_weights": [1.0],
            "weekend_departure_hour": [],
            "weekend_arrival_hour": [],
            "weekend_miles": [],
            "weekend_trip_weights": [],
            "weekday_trip_ids": [1],
            "weekend_trip_ids": [],
        },
        ("b3", 1): {  # Building 3 has 1 vehicle (matches v4)
            "weekday_departure_hour": [8, 13],  # Two trips on weekdays
            "weekday_arrival_hour": [12, 17],
            "weekday_miles": [10.0, 15.0],
            "weekday_trip_weights": [1.0, 1.0],
            "weekend_departure_hour": [],  # v4 has no weekend trips
            "weekend_arrival_hour": [],
            "weekend_miles": [],
            "weekend_trip_weights": [],
            "weekday_trip_ids": [1, 2],
            "weekend_trip_ids": [],
        },
    }
    print(profiles)
    # Check that we got all expected profiles
    assert set(profiles.keys()) == set(expected_profiles.keys())

    # Check each profile matches expected values
    for (bldg_id, vehicle_id), profile in profiles.items():
        print(profile)
        expected = expected_profiles[(bldg_id, vehicle_id)]
        # Check that profile matches its key values
        assert profile.bldg_id == bldg_id
        assert profile.vehicle_id == vehicle_id

        # Check exact values for weekday trips
        assert len(profile.weekday_departure_hour) == len(expected["weekday_departure_hour"])
        assert profile.weekday_departure_hour == expected["weekday_departure_hour"]
        assert profile.weekday_arrival_hour == expected["weekday_arrival_hour"]
        assert profile.weekday_miles == expected["weekday_miles"]
        assert profile.weekday_trip_weights == expected["weekday_trip_weights"]

        # Check exact values for weekend trips
        assert len(profile.weekend_departure_hour) == len(expected["weekend_departure_hour"])
        assert profile.weekend_departure_hour == expected["weekend_departure_hour"]
        assert profile.weekend_arrival_hour == expected["weekend_arrival_hour"]
        assert profile.weekend_miles == expected["weekend_miles"]
        assert profile.weekend_trip_weights == expected["weekend_trip_weights"]

        # Check trip IDs
        assert profile.weekday_trip_ids == expected["weekday_trip_ids"]
        assert profile.weekend_trip_ids == expected["weekend_trip_ids"]


def test_sample_vehicle_profiles_match_catalog(calculator):
    profiles, catalog = calculator.sample_vehicle_profiles(
        calculator.metadata_df,
        calculator.nhts_df,
        return_catalog=True,
    )

    assert len(profiles) == catalog.filter(pl.col("nhts_vehicle_matched")).height
    assert catalog.height == calculator.metadata_df["vehicles"].sum()

    summary = summarize_nhts_match_catalog(catalog)
    assert summary.filter(pl.col("metric") == "vehicle_slots")["count"][0] == 4
    assert summary.filter(pl.col("metric") == "missing_weekend_trip_profile")["count"][0] == 2

    missing_weekend = catalog.filter(pl.col("nhts_vehicle_matched") & ~pl.col("has_weekend_trips"))
    assert missing_weekend.height == 2
    assert set(missing_weekend.select("bldg_id", "vehicle_slot").rows()) == {("b2", 2), ("b3", 1)}

    vehicle_slots_with_any_gap = catalog.filter(
        ~pl.col("nhts_vehicle_matched") | ~pl.col("has_weekday_trips") | ~pl.col("has_weekend_trips")
    ).height
    gap_summary = summary.filter(pl.col("metric") == "vehicle_slots_with_any_gap")
    assert gap_summary["count"][0] == vehicle_slots_with_any_gap
    assert gap_summary["share_of_vehicle_slots"][0] == pytest.approx(vehicle_slots_with_any_gap / 4)


def test_sample_vehicle_profiles_zero_vehicles(calculator, mock_nhts_data, mock_metadata_with_zero):
    # Create new calculator with metadata that includes a zero-vehicle building
    calculator = EVDemandCalculator(
        metadata_df=mock_metadata_with_zero,
        nhts_df=mock_nhts_data,
        pums_df=mock_metadata_with_zero,
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 1, 7),
        random_state=42,
    )

    profiles = calculator.sample_vehicle_profiles(calculator.metadata_df, calculator.nhts_df)

    # Check that we got profiles for each vehicle (excluding the 0-vehicle building)
    expected_vehicle_count = mock_metadata_with_zero["vehicles"].sum()  # Should be 4 (b1:1, b2:2, b3:1, b4:0)
    assert len(profiles) == expected_vehicle_count

    # Verify b4 has no profiles
    assert not any(bldg_id == "b4" for (bldg_id, _) in profiles)

    # Verify other buildings still have their profiles
    assert ("b1", 1) in profiles
    assert ("b2", 1) in profiles
    assert ("b2", 2) in profiles
    assert ("b3", 1) in profiles


def test_generate_daily_schedules(calculator):
    # Create a sample profile with known values
    profile = VehicleProfile(
        bldg_id="b1",
        vehicle_id=1,
        weekday_departure_hour=[8],
        weekday_arrival_hour=[17],
        weekday_miles=[20.0],
        weekday_trip_weights=[1.0],
        weekend_departure_hour=[10],
        weekend_arrival_hour=[19],
        weekend_miles=[25.0],
        weekend_trip_weights=[1.0],
        weekday_trip_ids=[1],
        weekend_trip_ids=[1],
    )

    schedules = calculator._generate_vehicle_daily_trip_schedules(profile)

    expected_schedules = [
        # Weekend days (Sat-Sun)
        {"date": datetime(2022, 1, 1), "departure_hour": 10, "arrival_hour": 18, "miles_driven": 22.22029970},
        {"date": datetime(2022, 1, 2), "departure_hour": 10, "arrival_hour": 19, "miles_driven": 25.79725546},
        # Weekdays (Mon-Fri)
        {"date": datetime(2022, 1, 3), "departure_hour": 8, "arrival_hour": 17, "miles_driven": 18.83824373},
        {"date": datetime(2022, 1, 4), "departure_hour": 8, "arrival_hour": 17, "miles_driven": 18.94966038},
        {"date": datetime(2022, 1, 5), "departure_hour": 8, "arrival_hour": 17, "miles_driven": 19.14390787},
        {"date": datetime(2022, 1, 6), "departure_hour": 8, "arrival_hour": 17, "miles_driven": 18.51518632},
        {"date": datetime(2022, 1, 7), "departure_hour": 8, "arrival_hour": 16, "miles_driven": 20.24443833},
    ]
    print(schedules)
    assert len(schedules) == len(expected_schedules)

    for actual, expected in zip(schedules.iter_rows(named=True), expected_schedules, strict=True):
        assert actual["bldg_id"] == "b1"
        assert actual["vehicle_id"] == 1
        assert actual["date"] == expected["date"]
        assert actual["departure_hour"] == expected["departure_hour"]
        assert actual["arrival_hour"] == expected["arrival_hour"]
        assert pytest.approx(actual["miles_driven"], rel=1e-8) == expected["miles_driven"]


def test_normalize_day_trip_times_enforces_order_and_non_overlap(calculator):
    departures = np.array([12, 8])
    home_hours = np.array([11, 17])  # first trip inverted; second overlaps first chronologically

    dep, home, keep = calculator._normalize_day_trip_times(departures, home_hours)

    assert keep.tolist() == [True, True]
    assert dep.tolist() == [17, 8]
    assert home.tolist() == [18, 17]
    assert (home > dep).all()

    chronological = sorted(zip(dep, home, strict=True))
    for (_, prev_home), (next_dep, _) in pairwise(chronological):
        assert next_dep >= prev_home


def test_generate_daily_schedules_no_invalid_or_overlapping_trips(calculator):
    profile = VehicleProfile(
        bldg_id="b1",
        vehicle_id=1,
        weekday_departure_hour=[8, 13],
        weekday_arrival_hour=[12, 17],
        weekday_miles=[20.0, 10.0],
        weekday_trip_weights=[1.0, 1.0],
        weekend_departure_hour=[10, 15],
        weekend_arrival_hour=[14, 18],
        weekend_miles=[25.0, 5.0],
        weekend_trip_weights=[1.0, 1.0],
        weekday_trip_ids=[1, 2],
        weekend_trip_ids=[1, 2],
    )

    schedules = calculator._generate_vehicle_daily_trip_schedules(profile, rng=np.random.RandomState(0))

    for day_trips in schedules.partition_by("date", as_dict=False):
        day_trips = day_trips.sort("departure_hour")
        for row in day_trips.iter_rows(named=True):
            assert row["arrival_hour"] > row["departure_hour"]

        for prev, nxt in pairwise(day_trips.iter_rows(named=True)):
            assert nxt["departure_hour"] >= prev["arrival_hour"]


def test_build_hours_base_matches_instance_date_range(calculator):
    hours_base = calculator._build_hours_base()
    assert hours_base.height == calculator.num_hours
    assert hours_base["timestamp"][0] == datetime(2022, 1, 1, 0, 0, 0)
    assert hours_base["timestamp"][-1] == datetime(2022, 1, 7, 23, 0, 0)


def test_generate_vehicle_presence_schedules_marks_trip_hours_away(calculator):
    profile = VehicleProfile(
        bldg_id="b1",
        vehicle_id=1,
        weekday_departure_hour=[8],
        weekday_arrival_hour=[17],
        weekday_miles=[20.0],
        weekday_trip_weights=[1.0],
        weekend_departure_hour=[10],
        weekend_arrival_hour=[19],
        weekend_miles=[25.0],
        weekend_trip_weights=[1.0],
        weekday_trip_ids=[1],
        weekend_trip_ids=[1],
    )
    trip_schedules = calculator._generate_vehicle_daily_trip_schedules(profile, rng=np.random.RandomState(0))
    presence = calculator.generate_vehicle_presence_schedules(
        trip_schedules,
        vehicle_keys=[("b1", 1)],
    )[("b1", 1)]

    assert presence.height == calculator.num_hours
    assert presence.filter(pl.col("can_charge") != pl.col("at_home")).is_empty()
    assert presence.filter(pl.col("at_home") & pl.col("away_from_home")).is_empty()
    assert presence.filter(pl.col("at_home") & pl.col("can_charge").is_null()).is_empty()
    assert presence.filter(pl.col("at_home") & pl.col("away_from_home").is_null()).is_empty()

    weekday_away = presence.filter(pl.col("away_from_home"))
    assert weekday_away.height > 0
    assert weekday_away["at_home"].not_().all()
    assert not weekday_away["can_charge"].any()


def test_generate_vehicle_presence_schedules_all_home_without_trips(calculator):
    presence = calculator.generate_vehicle_presence_schedules(
        pl.DataFrame({
            "bldg_id": [],
            "vehicle_id": [],
            "date": [],
            "departure_hour": [],
            "arrival_hour": [],
            "miles_driven": [],
        }),
        vehicle_keys=[("b1", 1)],
    )[("b1", 1)]

    assert presence.height == calculator.num_hours
    assert presence["at_home"].all()
    assert presence["can_charge"].all()
    assert not presence["away_from_home"].any()


def test_generate_vehicle_soc_schedules_energy_balance(calculator):
    profile = VehicleProfile(
        bldg_id="b1",
        vehicle_id=1,
        weekday_departure_hour=[8],
        weekday_arrival_hour=[17],
        weekday_miles=[20.0],
        weekday_trip_weights=[1.0],
        weekend_departure_hour=[10],
        weekend_arrival_hour=[19],
        weekend_miles=[25.0],
        weekend_trip_weights=[1.0],
        weekday_trip_ids=[1],
        weekend_trip_ids=[1],
    )
    battery_capacity_kwh = 90.0
    trip_schedules = calculator._generate_vehicle_daily_trip_schedules(profile, rng=np.random.RandomState(0))
    soc_schedule = calculator.generate_vehicle_soc_schedules(
        trip_schedules,
        vehicle_keys=[("b1", 1)],
        battery_capacity_kwh=battery_capacity_kwh,
        kwh_per_mile=0.30,
        charger_power_kw=7.2,
    )[("b1", 1)]

    assert soc_schedule.height == calculator.num_hours
    assert soc_schedule["soc_kwh"].min() >= 0.0
    assert soc_schedule["soc_kwh"].max() <= battery_capacity_kwh + 1e-9

    expected_discharge = trip_schedules["miles_driven"].sum() * 0.30
    assert soc_schedule["discharge_kwh"].sum() == pytest.approx(expected_discharge, rel=1e-6)
    assert soc_schedule["charge_kwh"].sum() == pytest.approx(expected_discharge, rel=1e-6)
    assert not soc_schedule["soc_underflow"].any()


def test_generate_vehicle_soc_schedules_flags_underflow(calculator):
    profile = VehicleProfile(
        bldg_id="b1",
        vehicle_id=1,
        weekday_departure_hour=[8],
        weekday_arrival_hour=[10],
        weekday_miles=[100.0],
        weekday_trip_weights=[1.0],
        weekend_departure_hour=[10],
        weekend_arrival_hour=[12],
        weekend_miles=[100.0],
        weekend_trip_weights=[1.0],
        weekday_trip_ids=[1],
        weekend_trip_ids=[1],
    )
    trip_schedules = calculator._generate_vehicle_daily_trip_schedules(profile, rng=np.random.RandomState(0))
    soc_schedule = calculator.generate_vehicle_soc_schedules(
        trip_schedules,
        vehicle_keys=[("b1", 1)],
        battery_capacity_kwh=5.0,
        kwh_per_mile=0.30,
        charger_power_kw=7.2,
    )[("b1", 1)]

    assert soc_schedule["soc_underflow"].any()
    assert soc_schedule["soc_kwh"].min() == 0.0


def test_generate_vehicle_soc_schedules_uses_prebuilt_presence(calculator):
    profile = VehicleProfile(
        bldg_id="b1",
        vehicle_id=1,
        weekday_departure_hour=[8],
        weekday_arrival_hour=[17],
        weekday_miles=[20.0],
        weekday_trip_weights=[1.0],
        weekend_departure_hour=[10],
        weekend_arrival_hour=[19],
        weekend_miles=[25.0],
        weekend_trip_weights=[1.0],
        weekday_trip_ids=[1],
        weekend_trip_ids=[1],
    )
    trip_schedules = calculator._generate_vehicle_daily_trip_schedules(profile, rng=np.random.RandomState(0))
    presence_by_vehicle = calculator.generate_vehicle_presence_schedules(
        trip_schedules,
        vehicle_keys=[("b1", 1)],
    )
    soc_kwargs = {
        "battery_capacity_kwh": 90.0,
        "kwh_per_mile": 0.30,
        "charger_power_kw": 7.2,
    }
    soc_direct = calculator.generate_vehicle_soc_schedules(
        trip_schedules,
        vehicle_keys=[("b1", 1)],
        **soc_kwargs,
    )[("b1", 1)]
    soc_from_presence = calculator.generate_vehicle_soc_schedules(
        trip_schedules,
        presence_by_vehicle=presence_by_vehicle,
        **soc_kwargs,
    )[("b1", 1)]

    assert soc_direct.columns == soc_from_presence.columns
    for column in soc_direct.columns:
        if soc_direct[column].dtype == pl.Boolean:
            assert soc_direct[column].to_list() == soc_from_presence[column].to_list()
        else:
            assert soc_direct[column].to_list() == pytest.approx(soc_from_presence[column].to_list(), rel=1e-9, abs=1e-9)


def test_compute_hourly_soc_is_beginning_of_hour(calculator):
    at_home = np.array([False, True])
    discharge = np.array([10.0, 0.0])
    charge_kwh = calculator._schedule_immediate_charging(
        at_home,
        discharge,
        battery_capacity_kwh=90.0,
        charger_power_kw=7.2,
        initial_soc_kwh=90.0,
    )
    soc_kwh, _ = calculator._compute_hourly_soc(
        discharge,
        charge_kwh,
        initial_soc_kwh=90.0,
    )

    assert soc_kwh[0] == pytest.approx(90.0)
    assert soc_kwh[1] == pytest.approx(80.0)


def test_cost_minimizing_charging_shifts_to_cheaper_hours(calculator):
    prices = np.array([0.20, 0.20, 0.10, 0.10])
    at_home = np.array([True, True, True, False])
    discharge = np.array([0.0, 0.0, 0.0, 10.0])

    immediate_charge = calculator._schedule_immediate_charging(
        at_home,
        discharge,
        battery_capacity_kwh=90.0,
        charger_power_kw=7.2,
        initial_soc_kwh=5.0,
    )
    optimized_charge, optimized_shed = calculator._schedule_cost_minimizing_charging(
        at_home,
        discharge,
        battery_capacity_kwh=90.0,
        charger_power_kw=7.2,
        initial_soc_kwh=5.0,
        hourly_price_usd_per_kwh=prices,
    )
    _, optimized_underflow = calculator._compute_hourly_soc(
        discharge,
        optimized_charge,
        initial_soc_kwh=5.0,
    )

    immediate_cost = float((immediate_charge * prices).sum())
    optimized_cost = float((optimized_charge * prices).sum())

    assert not optimized_underflow.any()
    assert optimized_shed.sum() == pytest.approx(0.0, abs=1e-6)
    assert optimized_charge.sum() == pytest.approx(5.0, rel=1e-6)
    assert optimized_cost <= immediate_cost + 1e-9
    assert optimized_charge[2] == pytest.approx(5.0, rel=1e-6)
    assert optimized_cost == pytest.approx(5.0 * 0.10, rel=1e-6)


def test_cost_minimizing_charging_is_feasible_when_trip_exceeds_initial_soc(calculator):
    """Shed load keeps the LP feasible when away-hour trip draw exceeds start-of-hour SOC."""
    at_home = np.array([False, True, True, True])
    discharge = np.array([10.0, 0.0, 0.0, 0.0])
    prices = np.array([0.10, 0.10, 0.10, 0.10])

    charge, shed = calculator._schedule_cost_minimizing_charging(
        at_home,
        discharge,
        battery_capacity_kwh=90.0,
        charger_power_kw=7.2,
        initial_soc_kwh=5.0,
        hourly_price_usd_per_kwh=prices,
    )

    assert shed[0] == pytest.approx(5.0, rel=1e-6)
    assert charge.sum() == pytest.approx(0.0, abs=0.1)


def test_cost_minimizing_charging_sheds_trip_when_cheaper_than_charging(calculator):
    prices = np.array([1.00, 1.00, 0.10, 0.10])
    at_home = np.array([False, True, True, True])
    discharge = np.array([10.0, 0.0, 0.0, 0.0])

    charge, shed = calculator._schedule_cost_minimizing_charging(
        at_home,
        discharge,
        battery_capacity_kwh=90.0,
        charger_power_kw=7.2,
        initial_soc_kwh=5.0,
        hourly_price_usd_per_kwh=prices,
        shed_load_penalty_usd_per_kwh=0.50,
    )
    _, underflow = calculator._compute_hourly_soc(
        discharge,
        charge,
        initial_soc_kwh=5.0,
    )

    assert underflow[0]
    assert shed[0] == pytest.approx(5.0, rel=1e-6)
    assert charge.sum() == pytest.approx(0.0, abs=1e-6)


def test_generate_vehicle_soc_schedules_cost_minimizing_not_more_expensive(calculator):
    profile = VehicleProfile(
        bldg_id="b1",
        vehicle_id=1,
        weekday_departure_hour=[8],
        weekday_arrival_hour=[17],
        weekday_miles=[20.0],
        weekday_trip_weights=[1.0],
        weekend_departure_hour=[10],
        weekend_arrival_hour=[19],
        weekend_miles=[25.0],
        weekend_trip_weights=[1.0],
        weekday_trip_ids=[1],
        weekend_trip_ids=[1],
    )
    trip_schedules = calculator._generate_vehicle_daily_trip_schedules(profile, rng=np.random.RandomState(0))
    hours_base = calculator._build_hours_base()
    prices = np.where(hours_base["hour"].to_numpy() < 12, 0.10, 0.20)

    immediate = calculator.generate_vehicle_soc_schedules(
        trip_schedules,
        vehicle_keys=[("b1", 1)],
        charging_strategy="immediate",
    )[("b1", 1)]
    optimized = calculator.generate_vehicle_soc_schedules(
        trip_schedules,
        vehicle_keys=[("b1", 1)],
        charging_strategy="cost_minimizing",
        hourly_price_usd_per_kwh=prices,
    )[("b1", 1)]

    immediate_cost = float((immediate["charge_kwh"].to_numpy() * prices).sum())
    optimized_cost = float((optimized["charge_kwh"].to_numpy() * prices).sum())

    assert optimized["discharge_kwh"].sum() == pytest.approx(immediate["discharge_kwh"].sum(), rel=1e-6)
    assert optimized["charge_kwh"].sum() == pytest.approx(immediate["charge_kwh"].sum(), rel=1e-6)
    assert optimized_cost <= immediate_cost + 1e-6
    assert not optimized["soc_underflow"].any()


def test_immediate_charging_stays_full_without_trips(calculator):
    at_home = np.ones(HOURS_PER_YEAR, dtype=bool)
    discharge = np.zeros(HOURS_PER_YEAR, dtype=np.float64)
    charge_kwh = calculator._schedule_immediate_charging(
        at_home,
        discharge,
        battery_capacity_kwh=90.0,
        charger_power_kw=7.2,
        initial_soc_kwh=90.0,
    )
    soc_kwh, soc_underflow = calculator._compute_hourly_soc(
        discharge,
        charge_kwh,
        initial_soc_kwh=90.0,
    )

    assert soc_kwh[0] == pytest.approx(90.0)
    assert soc_kwh[-1] == pytest.approx(90.0)
    assert charge_kwh.sum() == pytest.approx(0.0)
    assert not soc_underflow.any()


def test_off_peak_charging_never_charges_during_peak(calculator):
    from datetime import date

    hours_base = pl.DataFrame({
        "hour_index": list(range(24)),
        "date": [date(2022, 1, 1)] * 24,
        "hour": list(range(24)),
        "timestamp": [datetime(2022, 1, 1, hour) for hour in range(24)],
    })
    is_off_peak = EVDemandCalculator._build_is_off_peak(hours_base)
    at_home = np.array([hour < 8 or hour >= 17 for hour in range(24)])
    discharge = np.zeros(24, dtype=np.float64)
    discharge[8:17] = 6.0 / 9.0

    vehicle_trips = pl.DataFrame({
        "bldg_id": ["b1"],
        "vehicle_id": [1],
        "date": [datetime(2022, 1, 1)],
        "departure_hour": [8],
        "arrival_hour": [17],
        "miles_driven": [20.0],
    })
    charge_allowed, soc_target_kwh = EVDemandCalculator._build_off_peak_charging_params(
        at_home,
        discharge,
        hours_base,
        vehicle_trips,
        battery_capacity_kwh=90.0,
        is_off_peak=is_off_peak,
    )
    charge_kwh = EVDemandCalculator._schedule_off_peak_charging(
        at_home,
        discharge,
        charge_allowed=charge_allowed,
        soc_target_kwh=soc_target_kwh,
        battery_capacity_kwh=90.0,
        charger_power_kw=7.2,
        initial_soc_kwh=18.0,
    )

    for peak_hour in range(17, 22):
        assert charge_kwh[peak_hour] == pytest.approx(0.0)
    assert charge_kwh.sum() > 0.0


def test_off_peak_charging_blocks_between_trip_home_hours(calculator):
    from datetime import date

    hours_base = pl.DataFrame({
        "hour_index": list(range(24)),
        "date": [date(2022, 1, 1)] * 24,
        "hour": list(range(24)),
        "timestamp": [datetime(2022, 1, 1, hour) for hour in range(24)],
    })
    is_off_peak = EVDemandCalculator._build_is_off_peak(hours_base)
    # Home 0-7, away 8-11, home 12-13 (off-peak lunch), away 14-16, home 17-23.
    at_home = np.array([hour < 8 or hour in (12, 13) or hour >= 17 for hour in range(24)])
    discharge = np.zeros(24, dtype=np.float64)
    discharge[8:12] = 4.0 / 4.0
    discharge[14:17] = 2.0 / 3.0

    vehicle_trips = pl.DataFrame({
        "bldg_id": ["b1", "b1"],
        "vehicle_id": [1, 1],
        "date": [datetime(2022, 1, 1), datetime(2022, 1, 1)],
        "departure_hour": [8, 14],
        "arrival_hour": [12, 17],
        "miles_driven": [13.33, 6.67],
    })
    charge_allowed, _ = EVDemandCalculator._build_off_peak_charging_params(
        at_home,
        discharge,
        hours_base,
        vehicle_trips,
        battery_capacity_kwh=90.0,
        is_off_peak=is_off_peak,
    )

    assert not charge_allowed[12]
    assert not charge_allowed[13]


def test_off_peak_charging_no_emergency_override_after_low_soc_return(calculator):
    from datetime import date

    hours_base = pl.DataFrame({
        "hour_index": list(range(24)),
        "date": [date(2022, 1, 1)] * 24,
        "hour": list(range(24)),
        "timestamp": [datetime(2022, 1, 1, hour) for hour in range(24)],
    })
    is_off_peak = EVDemandCalculator._build_is_off_peak(hours_base)
    at_home = np.array([hour < 8 or hour >= 17 for hour in range(24)])
    discharge = np.zeros(24, dtype=np.float64)
    discharge[8:17] = 30.0 / 9.0

    vehicle_trips = pl.DataFrame({
        "bldg_id": ["b1"],
        "vehicle_id": [1],
        "date": [datetime(2022, 1, 1)],
        "departure_hour": [8],
        "arrival_hour": [17],
        "miles_driven": [100.0],
    })
    charge_allowed, soc_target_kwh = EVDemandCalculator._build_off_peak_charging_params(
        at_home,
        discharge,
        hours_base,
        vehicle_trips,
        battery_capacity_kwh=90.0,
        is_off_peak=is_off_peak,
    )
    charge_kwh = EVDemandCalculator._schedule_off_peak_charging(
        at_home,
        discharge,
        charge_allowed=charge_allowed,
        soc_target_kwh=soc_target_kwh,
        battery_capacity_kwh=90.0,
        charger_power_kw=7.2,
        initial_soc_kwh=10.0,
    )
    immediate_charge = EVDemandCalculator._schedule_immediate_charging(
        at_home,
        discharge,
        battery_capacity_kwh=90.0,
        charger_power_kw=7.2,
        initial_soc_kwh=10.0,
    )
    soc_kwh, _ = EVDemandCalculator._compute_hourly_soc(
        discharge,
        charge_kwh,
        initial_soc_kwh=10.0,
    )

    assert charge_kwh[17:22].sum() == pytest.approx(0.0)
    assert immediate_charge[17:22].sum() > 0.0
    assert soc_kwh[17] < soc_target_kwh[17]


def test_generate_vehicle_soc_schedules_off_peak(calculator):
    profile = VehicleProfile(
        bldg_id="b1",
        vehicle_id=1,
        weekday_departure_hour=[8],
        weekday_arrival_hour=[17],
        weekday_miles=[20.0],
        weekday_trip_weights=[1.0],
        weekend_departure_hour=[10],
        weekend_arrival_hour=[19],
        weekend_miles=[25.0],
        weekend_trip_weights=[1.0],
        weekday_trip_ids=[1],
        weekend_trip_ids=[1],
    )
    trip_schedules = calculator._generate_vehicle_daily_trip_schedules(profile, rng=np.random.RandomState(0))
    hours_base = calculator._build_hours_base()
    peak_hours = list(range(17, 22))

    immediate = calculator.generate_vehicle_soc_schedules(
        trip_schedules,
        vehicle_keys=[("b1", 1)],
        charging_strategy="immediate",
    )[("b1", 1)]
    off_peak = calculator.generate_vehicle_soc_schedules(
        trip_schedules,
        vehicle_keys=[("b1", 1)],
        charging_strategy="off_peak",
    )[("b1", 1)]

    peak_charge_kwh = off_peak.filter(pl.col("timestamp").dt.hour().is_in(peak_hours))["charge_kwh"].sum()
    assert peak_charge_kwh == pytest.approx(0.0)
    assert off_peak["charge_kwh"].sum() <= immediate["charge_kwh"].sum() + 1e-6


def test_vehicle_hourly_schedules_to_dataframe(calculator):
    profile = VehicleProfile(
        bldg_id="b1",
        vehicle_id=1,
        weekday_departure_hour=[8],
        weekday_arrival_hour=[17],
        weekday_miles=[20.0],
        weekday_trip_weights=[1.0],
        weekend_departure_hour=[10],
        weekend_arrival_hour=[19],
        weekend_miles=[25.0],
        weekend_trip_weights=[1.0],
        weekday_trip_ids=[1],
        weekend_trip_ids=[1],
    )
    trip_schedules = calculator._generate_vehicle_daily_trip_schedules(profile, rng=np.random.RandomState(0))
    soc_by_vehicle = calculator.generate_vehicle_soc_schedules(
        trip_schedules,
        vehicle_keys=[("b1", 1)],
    )
    soc_df = EVDemandCalculator.vehicle_hourly_schedules_to_dataframe(soc_by_vehicle)

    assert soc_df.height == calculator.num_hours
    assert soc_df["bldg_id"].to_list() == ["b1"] * calculator.num_hours
    assert soc_df["vehicle_id"].to_list() == [1] * calculator.num_hours


@patch("utils.ev_demand.EVDemandCalculator.generate_daily_trip_schedules")
@patch("utils.ev_demand.EVDemandCalculator.sample_vehicle_profiles")
@patch("utils.ev_demand.EVDemandCalculator.predict_num_vehicles")
def test_match_and_generate_trip_schedules(predict_vehicles, sample_profiles, generate_schedule, calculator):
    # Setup expected data
    metadata = calculator.metadata_df
    predict_vehicles.return_value = metadata

    profile = VehicleProfile(
        bldg_id="b1",
        vehicle_id=1,
        weekday_departure_hour=[8],
        weekday_arrival_hour=[17],
        weekday_miles=[20.0],
        weekday_trip_weights=[1.0],
        weekend_departure_hour=[10],
        weekend_arrival_hour=[19],
        weekend_miles=[25.0],
        weekend_trip_weights=[1.0],
        weekday_trip_ids=[1],
        weekend_trip_ids=[1],
    )
    sample_profiles.return_value = {("b1", 1): profile}

    # Expected schedule data
    schedule_data = {
        "bldg_id": ["b1", "b1"],
        "vehicle_id": [1, 1],
        "date": [datetime(2022, 1, 1), datetime(2022, 1, 2)],
        "departure_hour": [10, 10],
        "arrival_hour": [19, 19],
        "miles_driven": [25.0, 25.0],
    }
    generate_schedule.return_value = pl.DataFrame(schedule_data)

    # Run the function
    result = calculator.match_and_generate_trip_schedules()

    # Verify exact expected output
    assert isinstance(result, pl.DataFrame)
    assert result.shape == (2, 6)  # 2 rows, 6 columns

    # Check exact values
    assert result["bldg_id"].to_list() == schedule_data["bldg_id"]
    assert result["vehicle_id"].to_list() == schedule_data["vehicle_id"]
    assert result["departure_hour"].to_list() == schedule_data["departure_hour"]
    assert result["arrival_hour"].to_list() == schedule_data["arrival_hour"]
    assert result["miles_driven"].to_list() == schedule_data["miles_driven"]
    assert [d.strftime("%Y-%m-%d") for d in result["date"]] == [d.strftime("%Y-%m-%d") for d in schedule_data["date"]]

    # Verify mock calls
    predict_vehicles.assert_called_once()
    sample_profiles.assert_called_once()
    generate_schedule.assert_called_once()
