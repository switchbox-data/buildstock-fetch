from datetime import datetime
from unittest.mock import patch

import polars as pl
import pytest

from utils.ev_demand import EVDemandCalculator, VehicleProfile


# Test data fixtures
@pytest.fixture
def mock_nhts_data():
    data = {
        "hh_vehicle_id": ["v1", "v2", "v3", "v4", "v4"],  # v4 has multiple trips
        "income_bucket": [1, 2, 2, 3, 3],  # v2 and v3 both match b2's income=2
        "occupants": [2, 3, 3, 4, 4],  # v2 and v3 both match b2's occupants=3
        "vehicles": [1, 2, 2, 1, 1],  # v2 and v3 both from 2-vehicle households
        "weekday": [2, 2, 2, 2, 2],  # All weekday trips
        "start_time": [800, 900, 1000, 800, 1300],
        "end_time": [1700, 1800, 1900, 1200, 1700],
        "miles_driven": [20.0, 30.0, 40.0, 10.0, 15.0],
        "trip_weight": [1.0, 1.0, 1.0, 1.0, 1.0],
    }
    return pl.DataFrame(data)


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
            "weekend_departure_hour": [],  # No weekend trips
            "weekend_arrival_hour": [],
            "weekend_miles": [],
            "weekend_trip_weights": [],
            "weekday_trip_ids": [1],
            "weekend_trip_ids": [],
        },
        ("b2", 1): {  # Building 2 first vehicle (matches v2)
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
        ("b2", 2): {  # Building 2 second vehicle (matches v3)
            "weekday_departure_hour": [10],
            "weekday_arrival_hour": [19],
            "weekday_miles": [40.0],
            "weekday_trip_weights": [1.0],
            "weekend_departure_hour": [],
            "weekend_arrival_hour": [],
            "weekend_miles": [],
            "weekend_trip_weights": [],
            "weekday_trip_ids": [1],
            "weekend_trip_ids": [],
        },
        ("b3", 1): {  # Building 3 has 1 vehicle (matches v3)
            "weekday_departure_hour": [8, 13],  # Two trips on weekdays
            "weekday_arrival_hour": [12, 17],
            "weekday_miles": [10.0, 15.0],
            "weekday_trip_weights": [1.0, 1.0],
            "weekend_departure_hour": [],  # No weekend trips
            "weekend_arrival_hour": [],
            "weekend_miles": [],
            "weekend_trip_weights": [],
            "weekday_trip_ids": [1, 2],
            "weekend_trip_ids": [],
        },
    }

    # Check that we got all expected profiles
    assert set(profiles.keys()) == set(expected_profiles.keys())

    # Check each profile matches expected values
    for (bldg_id, vehicle_id), profile in profiles.items():
        print(profile)
        expected = expected_profiles[(bldg_id, vehicle_id)]
        # Check that profile matches its key values
        assert profile.building_id == bldg_id
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
        building_id="b1",
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

    # Generate schedules - should be reproducible with seed 42
    schedules = calculator.generate_daily_schedules(profile)

    # Expected values with seed 42 (actual values from running the function)
    expected_schedules = [
        # Weekend days (Sat-Sun)
        {"date": datetime(2022, 1, 1), "departure_hour": 10, "arrival_hour": 19, "miles_driven": 22.22029970},
        {"date": datetime(2022, 1, 2), "departure_hour": 10, "arrival_hour": 19, "miles_driven": 25.79725546},
        # Weekdays (Mon-Fri)
        {"date": datetime(2022, 1, 3), "departure_hour": 8, "arrival_hour": 17, "miles_driven": 20.55808258},
        {"date": datetime(2022, 1, 4), "departure_hour": 8, "arrival_hour": 17, "miles_driven": 22.02103057},
        {"date": datetime(2022, 1, 5), "departure_hour": 8, "arrival_hour": 17, "miles_driven": 18.83824373},
        {"date": datetime(2022, 1, 6), "departure_hour": 8, "arrival_hour": 17, "miles_driven": 18.94966039},
        {"date": datetime(2022, 1, 7), "departure_hour": 8, "arrival_hour": 17, "miles_driven": 14.77490197},
    ]

    assert len(schedules) == len(expected_schedules)

    for actual, expected in zip(schedules, expected_schedules):
        assert actual.building_id == "b1"
        assert actual.vehicle_id == 1
        assert actual.date == expected["date"]
        assert actual.departure_hour == expected["departure_hour"]
        assert actual.arrival_hour == expected["arrival_hour"]
        assert pytest.approx(actual.miles_driven, rel=1e-8) == expected["miles_driven"]


@patch("utils.ev_demand.EVDemandCalculator._generate_annual_trip_schedule")
@patch("utils.ev_demand.EVDemandCalculator.sample_vehicle_profiles")
@patch("utils.ev_demand.EVDemandCalculator.predict_num_vehicles")
def test_generate_trip_schedules(predict_vehicles, sample_profiles, generate_schedule, calculator):
    # Setup expected data
    metadata = calculator.metadata_df
    predict_vehicles.return_value = metadata

    profile = VehicleProfile(
        building_id="b1",
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
        "building_id": ["b1", "b1"],
        "vehicle_id": [1, 1],
        "date": [datetime(2022, 1, 1), datetime(2022, 1, 2)],
        "departure_hour": [10, 10],
        "arrival_hour": [19, 19],
        "miles_driven": [25.0, 25.0],
    }
    generate_schedule.return_value = pl.DataFrame(schedule_data)

    # Run the function
    result = calculator.generate_trip_schedules()

    # Verify exact expected output
    assert isinstance(result, pl.DataFrame)
    assert result.shape == (2, 6)  # 2 rows, 6 columns

    # Check exact values
    assert result["building_id"].to_list() == schedule_data["building_id"]
    assert result["vehicle_id"].to_list() == schedule_data["vehicle_id"]
    assert result["departure_hour"].to_list() == schedule_data["departure_hour"]
    assert result["arrival_hour"].to_list() == schedule_data["arrival_hour"]
    assert result["miles_driven"].to_list() == schedule_data["miles_driven"]
    assert [d.strftime("%Y-%m-%d") for d in result["date"]] == [d.strftime("%Y-%m-%d") for d in schedule_data["date"]]

    # Verify mock calls
    predict_vehicles.assert_called_once()
    sample_profiles.assert_called_once()
    generate_schedule.assert_called_once()
