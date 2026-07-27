from datetime import datetime
from itertools import pairwise
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from utils.EVs.NHTSProfileSampler import (
    TripProfile,
    VehicleProfile,
    nhts_arrival_hour,
    nhts_departure_hour,
    summarize_nhts_match_catalog,
    NHTSProfileSampler,
)
from utils.EVs.nhts_tours import trips_as_singleton_tours
from utils.EVs.TripScheduleGenerator import TripScheduleGenerator
from utils.EVs.charging import (
    build_hours_base,
    build_is_off_peak,
    build_off_peak_charging_params,
    compute_hourly_soc,
    schedule_cost_minimizing_charging,
    schedule_immediate_charging,
    schedule_off_peak_charging,
    schedule_off_peak_immediate_charging,
)
from utils.EVs.ev_demand import EVDemandCalculator

HOURS_PER_YEAR = 8760


def make_trip_profile(
    trip_departure_hours: list[int],
    trip_arrival_hours: list[int],
    trip_miles_driven: list[float],
    trip_weights: list[float] | None = None,
    trip_ids: list[int] | None = None,
) -> TripProfile:
    """Test helper: one tour per leg (explicit; no TripProfile auto-fill)."""
    n = len(trip_departure_hours)
    return trips_as_singleton_tours(
        trip_departure_hours=trip_departure_hours,
        trip_arrival_hours=trip_arrival_hours,
        trip_miles_driven=trip_miles_driven,
        trip_weights=trip_weights or [1.0] * n,
        trip_ids=trip_ids,
    )


def num_hours_for_range(start_date: datetime, end_date: datetime) -> int:
    start_hour = start_date.replace(minute=0, second=0, microsecond=0)
    end_hour = end_date.replace(minute=0, second=0, microsecond=0)
    return int((end_hour - start_hour).total_seconds() // 3600) + 1


def make_ev_attributes(
    keys: list[tuple[str, int]],
    *,
    battery_capacity_kwh: float = 90.0,
    kwh_per_mile: float = 0.30,
) -> pl.DataFrame:
    """Minimal per-vehicle battery attrs for SOC tests."""
    return pl.DataFrame({
        "bldg_id": [k[0] for k in keys],
        "vehicle_id": [k[1] for k in keys],
        "battery_capacity_kwh": [battery_capacity_kwh] * len(keys),
        "kwh_per_mile": [kwh_per_mile] * len(keys),
    })


def make_vehicle_profile(
    bldg_id: str = "b1",
    vehicle_id: int = 1,
    weekday: TripProfile | None = None,
    weekend: TripProfile | None = None,
) -> VehicleProfile:
    return VehicleProfile(
        bldg_id=bldg_id,
        vehicle_id=vehicle_id,
        weekday=weekday or TripProfile(),
        weekend=weekend or TripProfile(),
    )


# Test data fixtures
@pytest.fixture
def mock_nhts_data():
    # urban: 1=urban, 2=rural (NHTS URBRUR). v1–v3 urban; v4 rural.
    data = {
        "hh_vehicle_id": ["v1", "v2", "v3", "v4", "v4", "v1", "v2", "v3"],
        "income_bucket": [1, 2, 2, 3, 3, 1, 2, 2],
        "occupants": [2, 3, 3, 4, 4, 2, 3, 3],
        "vehicles": [1, 2, 2, 1, 1, 1, 2, 2],
        "urban": [1, 1, 1, 2, 2, 1, 1, 1],
        "weekday": [2, 2, 2, 2, 2, 1, 1, 1],
        "start_time": [800, 900, 1000, 800, 1300, 1100, 1200, 1400],
        "end_time": [1700, 1800, 1900, 1200, 1700, 1500, 1600, 1800],
        "miles_driven": [20.0, 30.0, 40.0, 10.0, 15.0, 25.0, 28.0, 35.0],
        "trip_weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    }
    return pl.DataFrame(data)  # Return eager DataFrame to match production


@pytest.fixture
def mock_metadata():
    data = {
        "bldg_id": ["b1", "b2", "b3"],
        "income_bucket": [1, 2, 3],
        "occupants": [2, 3, 4],
        "vehicles": [1, 2, 1],  # b1 has 1 vehicle, b2 has 2, b3 has 1
        "metro": [
            "In metro area, principal city",
            "In metro area, not/partially in principal city",
            "Not/partially in metro area",
        ],
        "urban": [1, 1, 2],
    }
    return pl.DataFrame(data)


@pytest.fixture
def mock_metadata_with_zero():
    data = {
        "bldg_id": ["b1", "b2", "b3", "b4"],  # Added b4 with 0 vehicles
        "income_bucket": [1, 2, 3, 2],
        "occupants": [2, 3, 4, 1],
        "vehicles": [1, 2, 1, 0],  # b4 has 0 vehicles
        "metro": [
            "In metro area, principal city",
            "In metro area, not/partially in principal city",
            "Not/partially in metro area",
            "In metro area, principal city",
        ],
        "urban": [1, 1, 2, 1],
    }
    return pl.DataFrame(data)


@pytest.fixture
def calculator(mock_nhts_data, mock_metadata, ev_ownership_df, ev_battery_df, ev_autonomie_df):
    return EVDemandCalculator(
        metadata_df=mock_metadata,
        nhts_df=mock_nhts_data,
        ev_ownership_df=ev_ownership_df,
        ev_battery_df=ev_battery_df,
        ev_autonomie_df=ev_autonomie_df,
        start_date=datetime(2022, 1, 1, 4),
        end_date=datetime(2022, 1, 8, 3),
        pums_df=mock_metadata,  # Using same data for simplicity
        random_state=42,
    )


def test_nhts_hour_conversion():
    assert nhts_departure_hour(830) == 8
    assert nhts_departure_hour(1700) == 17
    assert nhts_arrival_hour(1700) == 17
    assert nhts_arrival_hour(1715) == 18
    assert nhts_arrival_hour(1450) == 15


def test_match_skips_vehicle_tier_by_default(calculator):
    match_type, vehicle_ids = calculator.nhts_sampler.match(
        target_income=1,
        target_urban=1,
        target_occupants=2,
        target_vehicles=1,
        num_samples=1,
        weekday=True,
    )
    assert match_type == "urban_income_occupants"
    assert vehicle_ids == ["v1"]


def test_match_with_vehicle_tier(calculator):
    # Test urban + exact vehicle match for single vehicle
    match_type, vehicle_ids = calculator.nhts_sampler.match(
        target_income=1,
        target_urban=1,
        target_occupants=2,
        target_vehicles=1,
        num_samples=1,
        weekday=True,
        match_on_vehicles=True,
    )
    assert match_type == "exact"
    assert vehicle_ids == ["v1"]

    # Test urban + exact match for multiple vehicles
    match_type, vehicle_ids = calculator.nhts_sampler.match(
        target_income=2,
        target_urban=1,
        target_occupants=3,
        target_vehicles=2,
        num_samples=2,
        weekday=True,
        match_on_vehicles=True,
    )
    assert match_type == "exact"
    assert len(vehicle_ids) == 2
    assert set(vehicle_ids) == {"v2", "v3"}  # Should get both vehicles with matching characteristics

    # Test partial match (urban + income and occupants only)
    match_type, vehicle_ids = calculator.nhts_sampler.match(
        target_income=2,
        target_urban=1,
        target_occupants=3,
        target_vehicles=1,  # Different from data
        num_samples=1,
        weekday=True,
        match_on_vehicles=True,
    )
    assert match_type == "urban_income_occupants"
    assert vehicle_ids[0] in ["v2", "v3"]

    # Test income-only match after urban tiers miss (income=3 is rural-only in fixture)
    match_type, vehicle_ids = calculator.nhts_sampler.match(
        target_income=3,
        target_urban=1,  # urban request; v4 is rural → drop urban
        target_occupants=2,  # Different from data
        target_vehicles=1,  # Different from data
        num_samples=1,
        weekday=True,
        match_on_vehicles=True,
    )
    assert match_type == "income"
    assert vehicle_ids[0] == "v4"  # v4 has income=3

    # Test closest income match
    match_type, vehicle_ids = calculator.nhts_sampler.match(
        target_income=4,  # Not in data
        target_urban=1,
        target_occupants=2,
        target_vehicles=1,
        num_samples=1,
        weekday=True,
        match_on_vehicles=True,
    )
    assert match_type == "closest_income"
    assert vehicle_ids[0] in ["v1", "v2", "v3", "v4"]


def test_match_prefers_urban_over_income_only_cross_urban(calculator):
    """Rural building with income=3 matches rural v4 via exact (urban held)."""
    match_type, vehicle_ids = calculator.nhts_sampler.match(
        target_income=3,
        target_urban=2,
        target_occupants=4,
        target_vehicles=1,
        num_samples=1,
        weekday=True,
        match_on_vehicles=True,
    )
    assert match_type == "exact"
    assert vehicle_ids == ["v4"]


def test_match_prefers_urban_income_over_income_occupants():
    """Keep urban/rural when dropping occupants; place-type beats HH size for VMT.

    Pool has rural income=2 with occupants=4 and urban income=2 with occupants=3.
    A rural target at income=2, occupants=3 should hit urban_income (rural peer)
    before income_occupants (cross-urban same HH size).
    """
    nhts = pl.DataFrame({
        "hh_vehicle_id": ["urban_occ3", "rural_occ4", "urban_occ3", "rural_occ4"],
        "income_bucket": [2, 2, 2, 2],
        "occupants": [3, 4, 3, 4],
        "vehicles": [1, 1, 1, 1],
        "urban": [1, 2, 1, 2],
        "weekday": [2, 2, 1, 1],
        "start_time": [900, 800, 1100, 1000],
        "end_time": [1700, 1200, 1500, 1400],
        "miles_driven": [30.0, 25.0, 28.0, 22.0],
        "trip_weight": [1.0, 1.0, 1.0, 1.0],
    })
    sampler = NHTSProfileSampler(nhts_df=nhts, random_state=42)
    match_type, vehicle_ids = sampler.match(
        target_income=2,
        target_urban=2,
        target_occupants=3,
        target_vehicles=1,
        num_samples=1,
        weekday=True,
    )
    assert match_type == "urban_income"
    assert vehicle_ids == ["rural_occ4"]


def test_match_falls_back_to_income_occupants_when_urban_income_misses(calculator):
    """When no same-urban pool exists at the income, cross urban to keep occupants."""
    match_type, vehicle_ids = calculator.nhts_sampler.match(
        target_income=2,
        target_urban=2,  # rural; fixture has income=2 only as urban v2/v3
        target_occupants=3,
        target_vehicles=1,
        num_samples=1,
        weekday=True,
    )
    assert match_type == "income_occupants"
    assert vehicle_ids[0] in ["v2", "v3"]

def test_sample_uses_sampler_nhts_by_default(
    mock_nhts_data, mock_metadata, ev_ownership_df, ev_battery_df, ev_autonomie_df
):
    calculator_kwargs = {
        "metadata_df": mock_metadata,
        "nhts_df": mock_nhts_data,
        "ev_ownership_df": ev_ownership_df,
        "ev_battery_df": ev_battery_df,
        "ev_autonomie_df": ev_autonomie_df,
        "start_date": datetime(2022, 1, 1, 4),
        "end_date": datetime(2022, 1, 8, 3),
        "random_state": 42,
    }
    profiles_explicit = EVDemandCalculator(**calculator_kwargs).nhts_sampler.sample(
        mock_metadata,
        mock_nhts_data,
    )
    profiles_default = EVDemandCalculator(**calculator_kwargs).nhts_sampler.sample(
        mock_metadata
    )
    assert profiles_explicit == profiles_default


def test_sample(calculator):
    profiles = calculator.nhts_sampler.sample(
        calculator.metadata_df, calculator.nhts_df
    )

    # Check that we got profiles for each vehicle
    expected_vehicle_count = calculator.metadata_df["vehicles"].sum()
    assert len(profiles) == expected_vehicle_count  # Should be 4 (b1:1, b2:2, b3:1)

    # Expected profiles with calculator's random_state=42
    expected_profiles = {
        ("b1", 1): {
            "weekday": {
                "trip_departure_hours": [8],
                "trip_arrival_hours": [17],
                "trip_miles_driven": [20.0],
                "trip_weights": [1.0],
                "trip_ids": [1],
            },
            "weekend": {
                "trip_departure_hours": [11],
                "trip_arrival_hours": [15],
                "trip_miles_driven": [25.0],
                "trip_weights": [1.0],
                "trip_ids": [1],
            },
        },
        ("b2", 1): {
            "weekday": {
                "trip_departure_hours": [10],
                "trip_arrival_hours": [19],
                "trip_miles_driven": [40.0],
                "trip_weights": [1.0],
                "trip_ids": [1],
            },
            "weekend": {
                "trip_departure_hours": [12],
                "trip_arrival_hours": [16],
                "trip_miles_driven": [28.0],
                "trip_weights": [1.0],
                "trip_ids": [1],
            },
        },
        ("b2", 2): {
            "weekday": {
                "trip_departure_hours": [9],
                "trip_arrival_hours": [18],
                "trip_miles_driven": [30.0],
                "trip_weights": [1.0],
                "trip_ids": [1],
            },
            "weekend": {
                "trip_departure_hours": [14],
                "trip_arrival_hours": [18],
                "trip_miles_driven": [35.0],
                "trip_weights": [1.0],
                "trip_ids": [1],
            },
        },
        ("b3", 1): {
            "weekday": {
                "trip_departure_hours": [8, 13],
                "trip_arrival_hours": [12, 17],
                "trip_miles_driven": [10.0, 15.0],
                "trip_weights": [1.0, 1.0],
                "trip_ids": [1, 2],
            },
            "weekend": {
                "trip_departure_hours": [14],
                "trip_arrival_hours": [18],
                "trip_miles_driven": [35.0],
                "trip_weights": [1.0],
                "trip_ids": [1],
            },
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

        expected_weekday = expected["weekday"]
        assert profile.weekday.trip_departure_hours == expected_weekday["trip_departure_hours"]
        assert profile.weekday.trip_arrival_hours == expected_weekday["trip_arrival_hours"]
        assert profile.weekday.trip_miles_driven == expected_weekday["trip_miles_driven"]
        assert profile.weekday.trip_weights == expected_weekday["trip_weights"]
        assert profile.weekday.trip_ids == expected_weekday["trip_ids"]

        expected_weekend = expected["weekend"]
        assert profile.weekend.trip_departure_hours == expected_weekend["trip_departure_hours"]
        assert profile.weekend.trip_arrival_hours == expected_weekend["trip_arrival_hours"]
        assert profile.weekend.trip_miles_driven == expected_weekend["trip_miles_driven"]
        assert profile.weekend.trip_weights == expected_weekend["trip_weights"]
        assert profile.weekend.trip_ids == expected_weekend["trip_ids"]


def test_sample_match_catalog(calculator):
    profiles, catalog = calculator.nhts_sampler.sample(
        calculator.metadata_df,
        calculator.nhts_df,
        return_catalog=True,
    )

    assert len(profiles) == catalog.filter(pl.col("nhts_vehicle_matched")).height
    assert catalog.height == calculator.metadata_df["vehicles"].sum()

    summary = summarize_nhts_match_catalog(catalog)
    assert summary.filter(pl.col("metric") == "vehicle_slots")["count"][0] == 4
    assert summary.filter(pl.col("metric") == "missing_weekend_trip_profile")["count"][0] == 0

    missing_weekend = catalog.filter(pl.col("nhts_vehicle_matched") & ~pl.col("has_weekend_trips"))
    assert missing_weekend.height == 0

    vehicle_slots_with_any_gap = catalog.filter(
        ~pl.col("nhts_vehicle_matched") | ~pl.col("has_weekday_trips") | ~pl.col("has_weekend_trips")
    ).height
    assert vehicle_slots_with_any_gap == 0


def test_sample_zero_vehicles(
    calculator, mock_nhts_data, mock_metadata_with_zero, ev_ownership_df, ev_battery_df, ev_autonomie_df
):
    # Create new calculator with metadata that includes a zero-vehicle building
    calculator = EVDemandCalculator(
        metadata_df=mock_metadata_with_zero,
        nhts_df=mock_nhts_data,
        ev_ownership_df=ev_ownership_df,
        ev_battery_df=ev_battery_df,
        ev_autonomie_df=ev_autonomie_df,
        start_date=datetime(2022, 1, 1, 4),
        end_date=datetime(2022, 1, 8, 3),
        pums_df=mock_metadata_with_zero,
        random_state=42,
    )

    profiles = calculator.nhts_sampler.sample(
        calculator.metadata_df, calculator.nhts_df
    )

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
    profile = make_vehicle_profile(
        weekday=make_trip_profile([8], [17], [20.0]),
        weekend=make_trip_profile([10], [19], [25.0]),
    )

    schedules = calculator.trip_schedule_generator.generate_daily_trip_schedule(profile)

    expected_schedules = [
        # Weekend travel days (Sat-Sun) — independent dep/arr offsets can stretch duration
        {"travel_date": datetime(2022, 1, 1), "trip_departure_hour": 10, "trip_arrival_hour": 21, "trip_miles_driven": 26.61922135},
        {"travel_date": datetime(2022, 1, 2), "trip_departure_hour": 10, "trip_arrival_hour": 19, "trip_miles_driven": 28.80757464},
        # Weekday travel days (Mon-Fri)
        {"travel_date": datetime(2022, 1, 3), "trip_departure_hour": 7, "trip_arrival_hour": 18, "trip_miles_driven": 23.15842563},
        {"travel_date": datetime(2022, 1, 4), "trip_departure_hour": 6, "trip_arrival_hour": 19, "trip_miles_driven": 21.53486946},
        {"travel_date": datetime(2022, 1, 5), "trip_departure_hour": 8, "trip_arrival_hour": 17, "trip_miles_driven": 19.07316461},
        {"travel_date": datetime(2022, 1, 6), "trip_departure_hour": 8, "trip_arrival_hour": 17, "trip_miles_driven": 19.06854049},
        {"travel_date": datetime(2022, 1, 7), "trip_departure_hour": 8, "trip_arrival_hour": 17, "trip_miles_driven": 17.97433776},
    ]
    assert len(schedules) == len(expected_schedules)

    for actual, expected in zip(schedules.iter_rows(named=True), expected_schedules, strict=True):
        assert actual["bldg_id"] == "b1"
        assert actual["vehicle_id"] == 1
        assert actual["travel_date"] == expected["travel_date"]
        assert actual["trip_departure_date"] == expected["travel_date"]
        assert actual["trip_arrival_date"] == expected["travel_date"]
        assert actual["trip_departure_hour"] == expected["trip_departure_hour"]
        assert actual["trip_arrival_hour"] == expected["trip_arrival_hour"]
        assert pytest.approx(actual["trip_miles_driven"], rel=1e-8) == expected["trip_miles_driven"]
        # Singleton-tour profiles: tour window matches the drive interval.
        assert actual["tour_id"] == 1
        assert actual["tour_departure_hour"] == actual["trip_departure_hour"]
        assert actual["tour_arrival_hour"] == actual["trip_arrival_hour"]


def test_normalize_day_trip_times_enforces_order_and_non_overlap():
    # Travel-day extended hours (same as clock hours when both are >= 4am)
    departures = np.array([12, 8])
    arrival_hours = np.array([11, 17])  # first trip inverted; second overlaps first chronologically

    gen = TripScheduleGenerator(
        start_date=datetime(2022, 1, 1, 4),
        end_date=datetime(2022, 1, 2, 3),
    )
    dep, arrival, keep = gen._normalize_day_trip_times(departures, arrival_hours)

    assert keep.tolist() == [True, True]
    assert dep.tolist() == [17, 8]
    assert arrival.tolist() == [18, 17]
    assert (arrival > dep).all()

    chronological = sorted(zip(dep, arrival, strict=True))
    for (_, prev_arrival), (next_dep, _) in pairwise(chronological):
        assert next_dep >= prev_arrival


def test_generate_daily_schedules_no_invalid_trips_or_overlapping_tours(calculator):
    """Drive intervals stay valid; tour away windows on a day do not overlap."""
    profile = make_vehicle_profile(
        weekday=make_trip_profile([8, 13], [12, 17], [20.0, 10.0]),
        weekend=make_trip_profile([10, 15], [14, 18], [25.0, 5.0], trip_ids=[1, 2]),
    )

    schedules = calculator.trip_schedule_generator.generate_daily_trip_schedule(
        profile, rng=np.random.RandomState(0)
    )

    for day_trips in schedules.partition_by("travel_date", as_dict=False):
        for row in day_trips.iter_rows(named=True):
            dep_key = (row["trip_departure_date"], row["trip_departure_hour"])
            arr_key = (row["trip_arrival_date"], row["trip_arrival_hour"])
            assert arr_key > dep_key

        tours = list(
            day_trips.unique(
                subset=[
                    "tour_id",
                    "tour_departure_date",
                    "tour_departure_hour",
                    "tour_arrival_date",
                    "tour_arrival_hour",
                ]
            )
            .sort(
                [
                    "tour_departure_date",
                    "tour_departure_hour",
                    "tour_arrival_date",
                    "tour_arrival_hour",
                ]
            )
            .iter_rows(named=True)
        )
        for prev, nxt in pairwise(tours):
            prev_arr = (prev["tour_arrival_date"], prev["tour_arrival_hour"])
            next_dep = (nxt["tour_departure_date"], nxt["tour_departure_hour"])
            assert next_dep >= prev_arr


def test_per_trip_offsets_can_stretch_duration_and_differ_across_legs():
    """Independent dep/arr offsets (Turk-style) can change trip length; legs are not rigid-shifted together."""
    from utils.EVs.NHTSProfileSampler import TripProfile

    weekday = TripProfile(
        trip_departure_hours=[8, 16],
        trip_arrival_hours=[9, 17],
        trip_miles_driven=[10.0, 10.0],
        trip_weights=[1.0, 1.0],
        trip_ids=[1, 2],
        tour_ids=[1, 1],
        tour_departure_hours=[8],
        tour_arrival_hours=[17],
        tour_ends_away=[False],
    )
    profile = make_vehicle_profile(weekday=weekday, weekend=TripProfile())
    gen = TripScheduleGenerator(
        start_date=datetime(2022, 1, 3, 4),
        end_date=datetime(2022, 1, 4, 3),
        random_state=0,
        time_offsets=(-1, 1),
        time_offset_probabilities=(0.5, 0.5),
        miles_noise_std_fraction=0.0,
    )

    class _ForcedOffsets(np.random.RandomState):
        """First choice → all dep -1; second → all arr +1; later draws use parent RNG."""

        def __init__(self):
            super().__init__(0)
            self._choice_calls = 0

        def choice(self, a, size=None, replace=True, p=None):  # noqa: A003
            self._choice_calls += 1
            if self._choice_calls == 1:
                return np.full(size, -1, dtype=int)
            if self._choice_calls == 2:
                return np.full(size, 1, dtype=int)
            return super().choice(a, size=size, replace=replace, p=p)

    trips = gen.generate_daily_trip_schedule(profile, rng=_ForcedOffsets())
    assert trips.height == 2
    # Base legs were 1 hour; dep -1 and arr +1 stretch each to 3 hours.
    durations = (
        trips["trip_arrival_hour"] - trips["trip_departure_hour"]
    ).to_list()
    assert durations == [3, 3]
    assert trips["tour_departure_hour"].unique().to_list() == [
        int(trips["trip_departure_hour"].min())
    ]
    assert trips["tour_arrival_hour"].unique().to_list() == [
        int(trips["trip_arrival_hour"].max())
    ]

def test_overnight_trip_spans_midnight(calculator):
    """NHTS early-morning / overnight legs land on the next calendar morning."""
    profile = make_vehicle_profile(
        weekday=make_trip_profile([22], [2], [30.0]),  # 10pm → 2am on travel day
        weekend=make_trip_profile([22], [2], [30.0]),
    )
    # Single weekday travel day: Mon 2022-01-03
    gen = TripScheduleGenerator(
        start_date=datetime(2022, 1, 3, 4),
        end_date=datetime(2022, 1, 4, 3),
        random_state=0,
        time_offsets=(0,),
        time_offset_probabilities=(1.0,),
        miles_noise_std_fraction=0.0,
    )
    schedules = gen.generate_daily_trip_schedule(profile, rng=np.random.RandomState(0))
    assert schedules.height == 1
    row = schedules.row(0, named=True)
    assert row["travel_date"] == datetime(2022, 1, 3)
    assert row["trip_departure_date"] == datetime(2022, 1, 3)
    assert row["trip_departure_hour"] == 22
    assert row["trip_arrival_date"] == datetime(2022, 1, 4)
    assert row["trip_arrival_hour"] == 2

    presence = calculator.charging_simulator.generate_presence(
        schedules,
        hours_base=build_hours_base(datetime(2022, 1, 3, 4), datetime(2022, 1, 4, 3)),
    )[("b1", 1)]
    away = presence.filter(pl.col("away_from_home"))
    away_hours = set(zip(away["timestamp"].dt.date().to_list(), away["timestamp"].dt.hour().to_list()))
    assert (datetime(2022, 1, 3).date(), 22) in away_hours
    assert (datetime(2022, 1, 3).date(), 23) in away_hours
    assert (datetime(2022, 1, 4).date(), 0) in away_hours
    assert (datetime(2022, 1, 4).date(), 1) in away_hours
    assert (datetime(2022, 1, 4).date(), 2) not in away_hours


def test_build_tours_from_nhts_legs_school_dropoff_commute():
    """Home→dropoff→work→dropoff→home becomes one tour with four drive legs."""
    from utils.EVs.nhts_tours import build_tours_from_legs, nhts_arrival_hour

    # Real pattern from NHTS household 9000013847 (times HHMM, purposes WHY*).
    start_times = [840, 850, 1715, 1805]
    end_times = [845, 925, 1800, 1815]
    why_from = [1, 10, 3, 10]  # home, dropoff, work, dropoff
    why_to = [10, 3, 10, 1]  # dropoff, work, dropoff, home
    miles = [2.8, 33.5, 31.7, 3.1]
    weights = [1.0, 1.0, 1.0, 1.0]

    day = build_tours_from_legs(
        start_times=start_times,
        end_times=end_times,
        trip_miles_driven=miles,
        trip_weights=weights,
        why_from=why_from,
        why_to=why_to,
    )

    assert day.tour_ids == [1, 1, 1, 1]
    assert len(day.tour_departure_hours) == 1
    assert day.tour_departure_hours[0] == 8  # STRTTIME 840 → hour after tour chain
    assert day.tour_arrival_hours[0] == nhts_arrival_hour(1815)  # 18:15 → exclusive hour 19
    assert day.trip_departure_hours == [8, 8, 17, 18]
    assert day.trip_arrival_hours == [9, 10, 18, 19]
    assert day.trip_miles_driven == miles
    assert day.tour_ends_away == [False]


def test_long_work_dwell_stays_one_tour():
    """Home→work (park all day)→home is one tour; dwell length does not split."""
    from utils.EVs.nhts_tours import build_tours_from_legs

    day = build_tours_from_legs(
        start_times=[800, 1700],
        end_times=[830, 1730],
        trip_miles_driven=[15.0, 15.0],
        trip_weights=[1.0, 1.0],
        why_from=[1, 3],  # home → work, work → home
        why_to=[3, 1],
    )
    assert day.tour_ids == [1, 1]
    assert day.tour_departure_hours == [8]
    assert day.tour_arrival_hours == [18]  # 1730 → exclusive hour 18
    assert len(day.tour_departure_hours) == 1


def test_presence_uses_tour_discharge_uses_drive_legs(calculator):
    """Mid-tour parking is away (no home charge) but has zero discharge / no temp draw."""
    from utils.EVs.NHTSProfileSampler import TripProfile

    # One tour 8→17 away; drives only 8→9 and 16→17 (parked at work 9–16).
    weekday = TripProfile(
        trip_departure_hours=[8, 16],
        trip_arrival_hours=[9, 17],
        trip_miles_driven=[10.0, 10.0],
        trip_weights=[1.0, 1.0],
        trip_ids=[1, 2],
        tour_ids=[1, 1],
        tour_departure_hours=[8],
        tour_arrival_hours=[17],
        tour_ends_away=[False],
    )
    profile = make_vehicle_profile(weekday=weekday, weekend=TripProfile())
    gen = TripScheduleGenerator(
        start_date=datetime(2022, 1, 3, 4),  # Monday only
        end_date=datetime(2022, 1, 4, 3),
        random_state=0,
        time_offsets=(0,),
        time_offset_probabilities=(1.0,),
        miles_noise_std_fraction=0.0,
    )
    trips = gen.generate_daily_trip_schedule(profile, rng=np.random.RandomState(0))
    assert trips.height == 2
    assert set(trips["tour_id"].to_list()) == {1}
    assert trips["tour_departure_hour"].unique().to_list() == [8]
    assert trips["tour_arrival_hour"].unique().to_list() == [17]

    hours_base = build_hours_base(datetime(2022, 1, 3, 4), datetime(2022, 1, 4, 3))
    presence = calculator.charging_simulator.generate_presence(trips, hours_base=hours_base)[("b1", 1)]
    day = presence.filter(pl.col("timestamp").dt.date() == datetime(2022, 1, 3).date())
    # Away for tour window hours 8..16 inclusive (arrival exclusive at 17).
    away_hours = set(day.filter(~pl.col("at_home"))["timestamp"].dt.hour().to_list())
    assert away_hours == set(range(8, 17))

    attrs = make_ev_attributes([("b1", 1)], kwh_per_mile=0.3)
    soc = calculator.charging_simulator.generate_soc(
        trips,
        ev_attributes=attrs,
        hours_base=hours_base,
    )
    soc_day = soc.filter(pl.col("timestamp").dt.date() == datetime(2022, 1, 3).date())
    discharge_by_hour = {
        int(r["timestamp"].hour): float(r["discharge_kwh"])
        for r in soc_day.iter_rows(named=True)
    }
    # Discharge only on drive hours 8 and 16; parked 9–15 is away but 0 kWh.
    assert discharge_by_hour[8] == pytest.approx(10.0 * 0.3)
    assert discharge_by_hour[16] == pytest.approx(10.0 * 0.3)
    for hour in range(9, 16):
        assert discharge_by_hour[hour] == 0.0
    assert discharge_by_hour[17] == 0.0
    # Still away (not charging) during parked work hours.
    at_home_by_hour = {
        int(r["timestamp"].hour): bool(r["at_home"])
        for r in soc_day.iter_rows(named=True)
    }
    assert at_home_by_hour[12] is False
    assert at_home_by_hour[17] is True


def test_build_hours_base_matches_instance_date_range(calculator):
    hours_base = build_hours_base(calculator.start_date, calculator.end_date)
    assert hours_base.height == num_hours_for_range(calculator.start_date, calculator.end_date)
    assert hours_base["timestamp"][0] == datetime(2022, 1, 1, 4, 0, 0)
    assert hours_base["timestamp"][-1] == datetime(2022, 1, 8, 3, 0, 0)


def _vehicle_hourly_schedule(df: pl.DataFrame, bldg_id: str, vehicle_id: int) -> pl.DataFrame:
    return df.filter(
        (pl.col("bldg_id") == bldg_id) & (pl.col("vehicle_id") == vehicle_id)
    ).drop("bldg_id", "vehicle_id")


def test_generate_presence_schedules_marks_trip_hours_away(calculator):
    profile = make_vehicle_profile(
        weekday=make_trip_profile([8], [17], [20.0]),
        weekend=make_trip_profile([10], [19], [25.0]),
    )
    trip_schedules = calculator.trip_schedule_generator.generate_daily_trip_schedule(
        profile, rng=np.random.RandomState(0)
    )
    presence = calculator.charging_simulator.generate_presence(
        trip_schedules,
        vehicle_keys=[("b1", 1)],
    )[("b1", 1)]

    assert presence.height == num_hours_for_range(calculator.start_date, calculator.end_date)
    assert presence.filter(pl.col("at_home") & pl.col("away_from_home")).is_empty()
    assert presence.filter(pl.col("at_home") & pl.col("away_from_home").is_null()).is_empty()

    weekday_away = presence.filter(pl.col("away_from_home"))
    assert weekday_away.height > 0
    assert weekday_away["at_home"].not_().all()


def test_generate_presence_schedules_all_home_without_trips(calculator):
    presence = calculator.charging_simulator.generate_presence(
        pl.DataFrame({
            "bldg_id": [],
            "vehicle_id": [],
            "travel_date": [],
            "trip_departure_date": [],
            "trip_departure_hour": [],
            "trip_arrival_date": [],
            "trip_arrival_hour": [],
            "trip_miles_driven": [],
        }),
        vehicle_keys=[("b1", 1)],
    )[("b1", 1)]

    assert presence.height == num_hours_for_range(calculator.start_date, calculator.end_date)
    assert presence["at_home"].all()
    assert not presence["away_from_home"].any()
    assert "can_charge" not in presence.columns


def test_generate_soc_schedules_energy_balance(calculator):
    profile = make_vehicle_profile(
        weekday=make_trip_profile([8], [17], [20.0]),
        weekend=make_trip_profile([10], [19], [25.0]),
    )
    battery_capacity_kwh = 90.0
    kwh_per_mile = 0.30
    attrs = make_ev_attributes([("b1", 1)], battery_capacity_kwh=battery_capacity_kwh, kwh_per_mile=kwh_per_mile)
    trip_schedules = calculator.trip_schedule_generator.generate_daily_trip_schedule(
        profile, rng=np.random.RandomState(0)
    )
    soc_schedule = _vehicle_hourly_schedule(
        calculator.generate_soc_schedules(
            trip_schedules,
            vehicle_keys=[("b1", 1)],
            ev_attributes=attrs,
            charger_power_kw=7.2,
        ),
        "b1",
        1,
    )

    assert soc_schedule.height == num_hours_for_range(calculator.start_date, calculator.end_date)
    assert soc_schedule["soc_kwh"].min() >= 0.0
    assert soc_schedule["soc_kwh"].max() <= battery_capacity_kwh + 1e-9

    expected_discharge = trip_schedules["trip_miles_driven"].sum() * kwh_per_mile
    assert soc_schedule["discharge_kwh"].sum() == pytest.approx(expected_discharge, rel=1e-6)
    assert soc_schedule["charge_kwh"].sum() == pytest.approx(expected_discharge, rel=1e-6)
    assert not soc_schedule["soc_underflow"].any()


def test_generate_soc_schedules_flags_underflow(calculator):
    profile = make_vehicle_profile(
        weekday=make_trip_profile([8], [10], [100.0]),
        weekend=make_trip_profile([10], [12], [100.0]),
    )
    trip_schedules = calculator.trip_schedule_generator.generate_daily_trip_schedule(
        profile, rng=np.random.RandomState(0)
    )
    attrs = make_ev_attributes([("b1", 1)], battery_capacity_kwh=5.0, kwh_per_mile=0.30)
    soc_schedule = _vehicle_hourly_schedule(
        calculator.generate_soc_schedules(
            trip_schedules,
            vehicle_keys=[("b1", 1)],
            ev_attributes=attrs,
            charger_power_kw=7.2,
        ),
        "b1",
        1,
    )

    assert soc_schedule["soc_underflow"].any()
    assert soc_schedule["soc_kwh"].min() == 0.0


def test_generate_soc_schedules_uses_prebuilt_presence(calculator):
    profile = make_vehicle_profile(
        weekday=make_trip_profile([8], [17], [20.0]),
        weekend=make_trip_profile([10], [19], [25.0]),
    )
    trip_schedules = calculator.trip_schedule_generator.generate_daily_trip_schedule(
        profile, rng=np.random.RandomState(0)
    )
    presence_by_vehicle = calculator.charging_simulator.generate_presence(
        trip_schedules,
        vehicle_keys=[("b1", 1)],
    )
    soc_kwargs = {
        "ev_attributes": make_ev_attributes([("b1", 1)]),
        "charger_power_kw": 7.2,
    }
    soc_direct = _vehicle_hourly_schedule(
        calculator.generate_soc_schedules(
            trip_schedules,
            vehicle_keys=[("b1", 1)],
            **soc_kwargs,
        ),
        "b1",
        1,
    )
    soc_from_presence = _vehicle_hourly_schedule(
        calculator.generate_soc_schedules(
            trip_schedules,
            presence_by_vehicle=presence_by_vehicle,
            **soc_kwargs,
        ),
        "b1",
        1,
    )

    assert soc_direct.columns == soc_from_presence.columns
    for column in soc_direct.columns:
        if soc_direct[column].dtype == pl.Boolean:
            assert soc_direct[column].to_list() == soc_from_presence[column].to_list()
        else:
            assert soc_direct[column].to_list() == pytest.approx(soc_from_presence[column].to_list(), rel=1e-9, abs=1e-9)


def test_compute_hourly_soc_is_beginning_of_hour():
    at_home = np.array([False, True])
    discharge = np.array([10.0, 0.0])
    charge_kwh = schedule_immediate_charging(
        at_home,
        discharge,
        battery_capacity_kwh=90.0,
        charger_power_kw=7.2,
        initial_soc_kwh=90.0,
    )
    soc_kwh, _ = compute_hourly_soc(
        discharge,
        charge_kwh,
        initial_soc_kwh=90.0,
    )

    assert soc_kwh[0] == pytest.approx(90.0)
    assert soc_kwh[1] == pytest.approx(80.0)


def test_cost_minimizing_charging_shifts_to_cheaper_hours():
    prices = np.array([0.20, 0.20, 0.10, 0.10])
    at_home = np.array([True, True, True, False])
    discharge = np.array([0.0, 0.0, 0.0, 10.0])

    immediate_charge = schedule_immediate_charging(
        at_home,
        discharge,
        battery_capacity_kwh=90.0,
        charger_power_kw=7.2,
        initial_soc_kwh=5.0,
    )
    optimized_charge, optimized_shed = schedule_cost_minimizing_charging(
        at_home,
        discharge,
        battery_capacity_kwh=90.0,
        charger_power_kw=7.2,
        initial_soc_kwh=5.0,
        hourly_price_usd_per_kwh=prices,
    )
    _, optimized_underflow = compute_hourly_soc(
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


def test_cost_minimizing_charging_is_feasible_when_trip_exceeds_initial_soc():
    """Shed load keeps the LP feasible when away-hour trip draw exceeds start-of-hour SOC."""
    at_home = np.array([False, True, True, True])
    discharge = np.array([10.0, 0.0, 0.0, 0.0])
    prices = np.array([0.10, 0.10, 0.10, 0.10])

    charge, shed = schedule_cost_minimizing_charging(
        at_home,
        discharge,
        battery_capacity_kwh=90.0,
        charger_power_kw=7.2,
        initial_soc_kwh=5.0,
        hourly_price_usd_per_kwh=prices,
    )

    assert shed[0] == pytest.approx(5.0, rel=1e-6)
    assert charge.sum() == pytest.approx(0.0, abs=0.1)


def test_cost_minimizing_charging_sheds_trip_when_cheaper_than_charging():
    prices = np.array([1.00, 1.00, 0.10, 0.10])
    at_home = np.array([False, True, True, True])
    discharge = np.array([10.0, 0.0, 0.0, 0.0])

    charge, shed = schedule_cost_minimizing_charging(
        at_home,
        discharge,
        battery_capacity_kwh=90.0,
        charger_power_kw=7.2,
        initial_soc_kwh=5.0,
        hourly_price_usd_per_kwh=prices,
        shed_load_penalty_usd_per_kwh=0.50,
    )
    _, underflow = compute_hourly_soc(
        discharge,
        charge,
        initial_soc_kwh=5.0,
    )

    assert underflow[0]
    assert shed[0] == pytest.approx(5.0, rel=1e-6)
    assert charge.sum() == pytest.approx(0.0, abs=1e-6)


def test_generate_soc_schedules_cost_minimizing_not_more_expensive(calculator):
    profile = make_vehicle_profile(
        weekday=make_trip_profile([8], [17], [20.0]),
        weekend=make_trip_profile([10], [19], [25.0]),
    )
    trip_schedules = calculator.trip_schedule_generator.generate_daily_trip_schedule(
        profile, rng=np.random.RandomState(0)
    )
    hours_base = build_hours_base(calculator.start_date, calculator.end_date)
    prices = np.where(hours_base["hour"].to_numpy() < 12, 0.10, 0.20)
    attrs = make_ev_attributes([("b1", 1)])


    immediate = _vehicle_hourly_schedule(
        calculator.generate_soc_schedules(
            trip_schedules,
            vehicle_keys=[("b1", 1)],
            ev_attributes=attrs,
            charging_strategy="immediate",
        ),
        "b1",
        1,
    )
    optimized = _vehicle_hourly_schedule(
        calculator.generate_soc_schedules(
            trip_schedules,
            vehicle_keys=[("b1", 1)],
            ev_attributes=attrs,
            charging_strategy="cost_minimizing",
            hourly_price_usd_per_kwh=prices,
        ),
        "b1",
        1,
    )

    immediate_cost = float((immediate["charge_kwh"].to_numpy() * prices).sum())
    optimized_cost = float((optimized["charge_kwh"].to_numpy() * prices).sum())

    # Trip energy is strategy-independent; cost-min may charge less by ending below full.
    assert optimized["discharge_kwh"].sum() == pytest.approx(immediate["discharge_kwh"].sum(), rel=1e-6)
    assert optimized["charge_kwh"].sum() <= immediate["charge_kwh"].sum() + 1e-6
    assert optimized_cost <= immediate_cost + 1e-6
    assert not optimized["soc_underflow"].any()


def test_immediate_charging_stays_full_without_trips():
    at_home = np.ones(HOURS_PER_YEAR, dtype=bool)
    discharge = np.zeros(HOURS_PER_YEAR, dtype=np.float64)
    charge_kwh = schedule_immediate_charging(
        at_home,
        discharge,
        battery_capacity_kwh=90.0,
        charger_power_kw=7.2,
        initial_soc_kwh=90.0,
    )
    soc_kwh, soc_underflow = compute_hourly_soc(
        discharge,
        charge_kwh,
        initial_soc_kwh=90.0,
    )

    assert soc_kwh[0] == pytest.approx(90.0)
    assert soc_kwh[-1] == pytest.approx(90.0)
    assert charge_kwh.sum() == pytest.approx(0.0)
    assert not soc_underflow.any()


def test_off_peak_charging_never_charges_during_peak():
    from datetime import date

    hours_base = pl.DataFrame({
        "hour_index": list(range(24)),
        "date": [date(2022, 1, 1)] * 24,
        "hour": list(range(24)),
        "timestamp": [datetime(2022, 1, 1, hour) for hour in range(24)],
    })
    is_off_peak = build_is_off_peak(hours_base)
    at_home = np.array([hour < 8 or hour >= 17 for hour in range(24)])
    discharge = np.zeros(24, dtype=np.float64)
    discharge[8:17] = 6.0 / 9.0

    vehicle_trips = pl.DataFrame({
        "bldg_id": ["b1"],
        "vehicle_id": [1],
        "travel_date": [datetime(2022, 1, 1)],
        "trip_departure_date": [datetime(2022, 1, 1)],
        "trip_departure_hour": [8],
        "trip_arrival_date": [datetime(2022, 1, 1)],
        "trip_arrival_hour": [17],
        "trip_miles_driven": [20.0],
    })
    charge_allowed, soc_target_kwh = build_off_peak_charging_params(
        at_home,
        discharge,
        hours_base,
        vehicle_trips,
        battery_capacity_kwh=90.0,
        is_off_peak=is_off_peak,
    )
    charge_kwh = schedule_off_peak_charging(
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


def test_off_peak_charging_blocks_between_trip_arrival_hours():
    from datetime import date

    hours_base = pl.DataFrame({
        "hour_index": list(range(24)),
        "date": [date(2022, 1, 1)] * 24,
        "hour": list(range(24)),
        "timestamp": [datetime(2022, 1, 1, hour) for hour in range(24)],
    })
    is_off_peak = build_is_off_peak(hours_base)
    # Home 0-7, away 8-11, home 12-13 (off-peak lunch), away 14-16, home 17-23.
    at_home = np.array([hour < 8 or hour in (12, 13) or hour >= 17 for hour in range(24)])
    discharge = np.zeros(24, dtype=np.float64)
    discharge[8:12] = 4.0 / 4.0
    discharge[14:17] = 2.0 / 3.0

    vehicle_trips = pl.DataFrame({
        "bldg_id": ["b1", "b1"],
        "vehicle_id": [1, 1],
        "travel_date": [datetime(2022, 1, 1), datetime(2022, 1, 1)],
        "trip_departure_date": [datetime(2022, 1, 1), datetime(2022, 1, 1)],
        "trip_departure_hour": [8, 14],
        "trip_arrival_date": [datetime(2022, 1, 1), datetime(2022, 1, 1)],
        "trip_arrival_hour": [12, 17],
        "trip_miles_driven": [13.33, 6.67],
    })
    charge_allowed, _ = build_off_peak_charging_params(
        at_home,
        discharge,
        hours_base,
        vehicle_trips,
        battery_capacity_kwh=90.0,
        is_off_peak=is_off_peak,
    )

    assert not charge_allowed[12]
    assert not charge_allowed[13]


def test_off_peak_charging_no_emergency_override_after_low_soc_return():
    from datetime import date

    hours_base = pl.DataFrame({
        "hour_index": list(range(24)),
        "date": [date(2022, 1, 1)] * 24,
        "hour": list(range(24)),
        "timestamp": [datetime(2022, 1, 1, hour) for hour in range(24)],
    })
    is_off_peak = build_is_off_peak(hours_base)
    at_home = np.array([hour < 8 or hour >= 17 for hour in range(24)])
    discharge = np.zeros(24, dtype=np.float64)
    discharge[8:17] = 30.0 / 9.0

    vehicle_trips = pl.DataFrame({
        "bldg_id": ["b1"],
        "vehicle_id": [1],
        "travel_date": [datetime(2022, 1, 1)],
        "trip_departure_date": [datetime(2022, 1, 1)],
        "trip_departure_hour": [8],
        "trip_arrival_date": [datetime(2022, 1, 1)],
        "trip_arrival_hour": [17],
        "trip_miles_driven": [100.0],
    })
    charge_allowed, soc_target_kwh = build_off_peak_charging_params(
        at_home,
        discharge,
        hours_base,
        vehicle_trips,
        battery_capacity_kwh=90.0,
        is_off_peak=is_off_peak,
    )
    charge_kwh = schedule_off_peak_charging(
        at_home,
        discharge,
        charge_allowed=charge_allowed,
        soc_target_kwh=soc_target_kwh,
        battery_capacity_kwh=90.0,
        charger_power_kw=7.2,
        initial_soc_kwh=10.0,
    )
    immediate_charge = schedule_immediate_charging(
        at_home,
        discharge,
        battery_capacity_kwh=90.0,
        charger_power_kw=7.2,
        initial_soc_kwh=10.0,
    )
    soc_kwh, _ = compute_hourly_soc(
        discharge,
        charge_kwh,
        initial_soc_kwh=10.0,
    )

    assert charge_kwh[17:22].sum() == pytest.approx(0.0)
    assert immediate_charge[17:22].sum() > 0.0
    assert soc_kwh[17] < soc_target_kwh[17]


def test_off_peak_immediate_never_charges_during_peak_by_default():
    """Pure TOU Immediate: max power off-peak only; no on-peak even with low SOC."""
    hours_base = build_hours_base(datetime(2022, 1, 1, 0), datetime(2022, 1, 1, 23))
    is_off_peak = build_is_off_peak(hours_base, peak_clock_hours=(17, 18, 19, 20, 21))
    # Home overnight; away 8–17; return into peak at 17 with nearly empty pack.
    at_home = np.array([hour < 8 or hour >= 17 for hour in range(24)])
    discharge = np.zeros(24, dtype=np.float64)
    discharge[8:17] = 80.0 / 9.0  # large day trip

    charge_kwh = schedule_off_peak_immediate_charging(
        at_home,
        discharge,
        is_off_peak=is_off_peak,
        battery_capacity_kwh=90.0,
        charger_power_kw=7.2,
        initial_soc_kwh=90.0,
        allow_emergency_peak_charging=False,
    )

    for peak_hour in range(17, 22):
        assert charge_kwh[peak_hour] == pytest.approx(0.0)
    # Rebound: charges at max power once off-peak resumes (hour 22).
    assert charge_kwh[22] == pytest.approx(7.2)
    assert charge_kwh[23] == pytest.approx(7.2)


def test_off_peak_immediate_fills_to_full_not_soc_req():
    """Unlike off_peak, off_peak_immediate keeps charging toward full capacity."""
    hours_base = build_hours_base(datetime(2022, 1, 1, 0), datetime(2022, 1, 1, 23))
    is_off_peak = build_is_off_peak(hours_base, peak_clock_hours=(17, 18, 19, 20, 21))
    at_home = np.ones(24, dtype=bool)
    discharge = np.zeros(24, dtype=np.float64)

    charge_kwh = schedule_off_peak_immediate_charging(
        at_home,
        discharge,
        is_off_peak=is_off_peak,
        battery_capacity_kwh=90.0,
        charger_power_kw=7.2,
        initial_soc_kwh=50.0,
        allow_emergency_peak_charging=False,
    )
    # Off-peak hours before peak: 0..16 → enough to fill 40 kWh headroom.
    assert charge_kwh[0:17].sum() == pytest.approx(40.0)
    assert charge_kwh[17:22].sum() == pytest.approx(0.0)


def test_off_peak_immediate_emergency_allows_peak_when_shortfall():
    """With emergency on, charge on-peak if remaining off-peak supply cannot cover next trip."""
    hours_base = build_hours_base(datetime(2022, 1, 1, 0), datetime(2022, 1, 1, 23))
    # Peak 12–21; only home after 17 (return into peak). Next trip at 22 needs 20 kWh;
    # no off-peak home hours remain before that departure.
    is_off_peak = build_is_off_peak(hours_base, peak_clock_hours=tuple(range(12, 22)))
    at_home = np.array([hour >= 17 for hour in range(24)])
    discharge = np.zeros(24, dtype=np.float64)
    discharge[22:24] = 10.0

    no_emergency = schedule_off_peak_immediate_charging(
        at_home,
        discharge,
        is_off_peak=is_off_peak,
        battery_capacity_kwh=90.0,
        charger_power_kw=7.2,
        initial_soc_kwh=5.0,
        allow_emergency_peak_charging=False,
    )
    with_emergency = schedule_off_peak_immediate_charging(
        at_home,
        discharge,
        is_off_peak=is_off_peak,
        battery_capacity_kwh=90.0,
        charger_power_kw=7.2,
        initial_soc_kwh=5.0,
        allow_emergency_peak_charging=True,
    )

    assert no_emergency[17:22].sum() == pytest.approx(0.0)
    assert with_emergency[17:22].sum() > 0.0


def test_generate_soc_schedules_off_peak(calculator):
    profile = make_vehicle_profile(
        weekday=make_trip_profile([8], [17], [20.0]),
        weekend=make_trip_profile([10], [19], [25.0]),
    )
    trip_schedules = calculator.trip_schedule_generator.generate_daily_trip_schedule(
        profile, rng=np.random.RandomState(0)
    )
    peak_hours = list(range(17, 22))
    attrs = make_ev_attributes([("b1", 1)])


    immediate = _vehicle_hourly_schedule(
        calculator.generate_soc_schedules(
            trip_schedules,
            vehicle_keys=[("b1", 1)],
            ev_attributes=attrs,
            charging_strategy="immediate",
        ),
        "b1",
        1,
    )
    off_peak = _vehicle_hourly_schedule(
        calculator.generate_soc_schedules(
            trip_schedules,
            vehicle_keys=[("b1", 1)],
            ev_attributes=attrs,
            charging_strategy="off_peak",
        ),
        "b1",
        1,
    )

    peak_charge_kwh = off_peak.filter(pl.col("timestamp").dt.hour().is_in(peak_hours))["charge_kwh"].sum()
    assert peak_charge_kwh == pytest.approx(0.0)
    assert off_peak["charge_kwh"].sum() <= immediate["charge_kwh"].sum() + 1e-6


def test_vehicle_hourly_schedules_to_dataframe(calculator):
    profile = make_vehicle_profile(
        weekday=make_trip_profile([8], [17], [20.0]),
        weekend=make_trip_profile([10], [19], [25.0]),
    )
    trip_schedules = calculator.trip_schedule_generator.generate_daily_trip_schedule(
        profile, rng=np.random.RandomState(0)
    )
    soc_df = calculator.generate_soc_schedules(
        trip_schedules,
        vehicle_keys=[("b1", 1)],
        ev_attributes=make_ev_attributes([("b1", 1)]),
    )

    expected_hours = num_hours_for_range(calculator.start_date, calculator.end_date)
    assert soc_df.height == expected_hours
    assert soc_df["bldg_id"].to_list() == ["b1"] * expected_hours
    assert soc_df["vehicle_id"].to_list() == [1] * expected_hours


def test_vehicle_slots_from_building_evs():
    # vehicles=0 buildings are dropped; vehicles=2 expands to vehicle_id 1 and 2.
    buildings = pl.DataFrame({
        "bldg_id": ["a", "b", "c"],
        "vehicles": [1, 0, 2],
    })
    slots = EVDemandCalculator._vehicle_slots_from_building_evs(buildings)
    assert slots.sort(["bldg_id", "vehicle_id"]).to_dicts() == [
        {"bldg_id": "a", "vehicle_id": 1},
        {"bldg_id": "c", "vehicle_id": 1},
        {"bldg_id": "c", "vehicle_id": 2},
    ]


def test_max_daily_miles_from_trip_schedules():
    trips = pl.DataFrame({
        "bldg_id": ["a", "a", "a", "b"],
        "vehicle_id": [1, 1, 1, 1],
        "travel_date": [
            datetime(2022, 1, 1),
            datetime(2022, 1, 1),
            datetime(2022, 1, 2),
            datetime(2022, 1, 1),
        ],
        "trip_miles_driven": [10.0, 15.0, 20.0, 5.0],
    })
    # a day1 = 25, day2 = 20 → max 25; b → 5.
    result = TripScheduleGenerator.max_daily_miles_from_trip_schedules(trips).sort("bldg_id")
    assert result.to_dicts() == [
        {"bldg_id": "a", "vehicle_id": 1, "max_daily_miles": 25.0},
        {"bldg_id": "b", "vehicle_id": 1, "max_daily_miles": 5.0},
    ]


@patch("utils.EVs.ev_demand.TripScheduleGenerator.generate")
@patch("utils.EVs.ev_demand.NHTSProfileSampler.sample")
@patch("utils.EVs.ev_demand.EVAdoptionSampler.sample")
def test_match_and_generate_trip_schedules(sample_evs, sample_profiles, generate_schedule, calculator):
    # Setup expected data
    metadata = calculator.metadata_df
    sample_evs.return_value = metadata.with_columns(
        pl.lit(1).alias("evs"),
        pl.lit(0.05).alias("ev_ownership_probability"),
    )

    profile = make_vehicle_profile(
        weekday=make_trip_profile([8], [17], [20.0]),
        weekend=make_trip_profile([10], [19], [25.0]),
    )
    sample_profiles.return_value = {("b1", 1): profile}

    # Expected schedule data
    schedule_data = {
        "bldg_id": ["b1", "b1"],
        "vehicle_id": [1, 1],
        "travel_date": [datetime(2022, 1, 1), datetime(2022, 1, 2)],
        "trip_departure_date": [datetime(2022, 1, 1), datetime(2022, 1, 2)],
        "trip_departure_hour": [10, 10],
        "trip_arrival_date": [datetime(2022, 1, 1), datetime(2022, 1, 2)],
        "trip_arrival_hour": [19, 19],
        "trip_miles_driven": [25.0, 25.0],
    }
    generate_schedule.return_value = pl.DataFrame(schedule_data)

    # Run the function
    result, ev_attributes = calculator.match_and_generate_trip_schedules()

    # Verify exact expected output
    assert isinstance(result, pl.DataFrame)
    assert result.shape == (2, 8)  # 2 rows, 8 columns

    # Check exact values
    assert result["bldg_id"].to_list() == schedule_data["bldg_id"]
    assert result["vehicle_id"].to_list() == schedule_data["vehicle_id"]
    assert result["trip_departure_hour"].to_list() == schedule_data["trip_departure_hour"]
    assert result["trip_arrival_hour"].to_list() == schedule_data["trip_arrival_hour"]
    assert result["trip_miles_driven"].to_list() == schedule_data["trip_miles_driven"]
    assert [d.strftime("%Y-%m-%d") for d in result["travel_date"]] == [d.strftime("%Y-%m-%d") for d in schedule_data["travel_date"]]
    assert [d.strftime("%Y-%m-%d") for d in result["trip_departure_date"]] == [
        d.strftime("%Y-%m-%d") for d in schedule_data["trip_departure_date"]
    ]
    assert [d.strftime("%Y-%m-%d") for d in result["trip_arrival_date"]] == [
        d.strftime("%Y-%m-%d") for d in schedule_data["trip_arrival_date"]
    ]

    # Battery attributes assigned for the EV slots present after sample
    # (mock sets evs=1 on all 3 metadata buildings → 3 attribute rows).
    assert ev_attributes.height == 3
    assert {"battery_capacity_kwh", "kwh_per_mile", "ev_option_name"} <= set(ev_attributes.columns)

    # Verify mock calls
    sample_evs.assert_called_once()
    sample_profiles.assert_called_once()
    generate_schedule.assert_called_once()


def test_generate_soc_schedules_respects_per_vehicle_battery_attrs(calculator):
    """Heterogeneous ResStock capacities/efficiencies are applied per vehicle."""
    # Same 10-mile, 1-hour trip for two vehicles.
    trips = pl.DataFrame({
        "bldg_id": ["b1", "b2"],
        "vehicle_id": [1, 1],
        "travel_date": [datetime(2022, 1, 1), datetime(2022, 1, 1)],
        "trip_departure_date": [datetime(2022, 1, 1), datetime(2022, 1, 1)],
        "trip_departure_hour": [9, 9],
        "trip_arrival_date": [datetime(2022, 1, 1), datetime(2022, 1, 1)],
        "trip_arrival_hour": [10, 10],
        "trip_miles_driven": [10.0, 10.0],
    })
    # Compact (efficient, smaller pack) vs pickup (thirstier, larger pack).
    attrs = pl.DataFrame({
        "bldg_id": ["b1", "b2"],
        "vehicle_id": [1, 1],
        "ev_option_name": [
            "Compact, Battery Electric Vehicle, 200 mile range",
            "Pickup, Battery Electric Vehicle, 300 mile range",
        ],
        "body_class": ["Compact", "Pickup"],
        "range_miles": [200, 300],
        "battery_capacity_kwh": [40.168, 105.946],
        "kwh_per_mile": [0.209901, 0.373794],
    })

    soc = calculator.generate_soc_schedules(
        trips,
        vehicle_keys=[("b1", 1), ("b2", 1)],
        ev_attributes=attrs,
    )

    b1 = soc.filter(pl.col("bldg_id") == "b1")
    b2 = soc.filter(pl.col("bldg_id") == "b2")
    # SOC ceiling follows each vehicle's usable capacity (not the 90 kWh default).
    assert b1["soc_kwh"].max() == pytest.approx(40.168)
    assert b2["soc_kwh"].max() == pytest.approx(105.946)
    # Same miles, different efficiency -> different discharge totals
    assert b1["discharge_kwh"].sum() == pytest.approx(10.0 * 0.209901)
    assert b2["discharge_kwh"].sum() == pytest.approx(10.0 * 0.373794)
    assert b1["discharge_kwh"].sum() < b2["discharge_kwh"].sum()


def test_generate_soc_schedules_applies_resstock_temp_scale(calculator):
    """Cold outdoor temp increases discharge kWh via ResStock power_mult; charge kW unchanged."""
    from utils.EVs.ev_utils import resstock_temp_power_mult

    trips = pl.DataFrame({
        "bldg_id": ["b1"],
        "vehicle_id": [1],
        "travel_date": [calculator.start_date],
        "trip_departure_date": [calculator.start_date],
        "trip_departure_hour": [9],
        "trip_arrival_date": [calculator.start_date],
        "trip_arrival_hour": [10],
        "trip_miles_driven": [10.0],
    })
    attrs = make_ev_attributes([("b1", 1)], battery_capacity_kwh=90.0, kwh_per_mile=0.30)
    hours_base = build_hours_base(calculator.start_date, calculator.end_date)
    cold_f = 0.0
    hourly_temp = hours_base.select(
        pl.lit("b1").alias("bldg_id"),
        "hour_index",
        pl.lit(cold_f).alias("temp_f"),
    )

    soc = calculator.generate_soc_schedules(
        trips,
        vehicle_keys=[("b1", 1)],
        ev_attributes=attrs,
        hours_base=hours_base,
        hourly_temp_f_by_bldg=hourly_temp,
        charger_power_kw=7.2,
    )
    expected = 10.0 * 0.30 * float(resstock_temp_power_mult(cold_f))
    assert soc["discharge_kwh"].sum() == pytest.approx(expected, rel=1e-6)
    # Immediate charging still refills the (larger) discharge at fixed charger power.
    assert soc["charge_kwh"].sum() == pytest.approx(expected, rel=1e-6)
    assert soc.filter(pl.col("charge_kwh") > 0)["charge_kwh"].max() <= 7.2 + 1e-9


def test_load_ev_demand_config_temperature_section(tmp_path):
    from utils.EVs.ev_demand import load_ev_demand_config

    cfg_path = tmp_path / "ev.yaml"
    cfg_path.write_text(
        """
state: MD
release: res_2024_tmy3_2
start_date: 2024-01-01T04:00:00
end_date: 2024-01-03T03:00:00
sampling:
  ev_assignment: resstock_adoption
temperature:
  temperature_adjustment: resstock
paths:
  weather_dir: /tmp/weather_md
charging:
  charging_strategy: immediate
  charger_power_kw: 7.2
"""
    )
    config = load_ev_demand_config(cfg_path)
    assert config.temperature_adjustment == "resstock"
    assert config.weather_dir == "/tmp/weather_md"


def test_nhts_daily_miles_percentile_filter_noop_by_default(mock_nhts_data):
    """0–100 percentile band leaves the NHTS pool unchanged."""
    filtered = NHTSProfileSampler.filter_by_daily_miles_percentile(
        mock_nhts_data, low=0.0, high=100.0
    )
    assert filtered.height == mock_nhts_data.height


def test_nhts_daily_miles_percentile_filter_drops_extremes():
    """Middle percentile band drops low- and high-mile vehicles."""
    # Four vehicles with distinct max daily miles: 10, 20, 30, 1000.
    nhts = pl.DataFrame({
        "hh_vehicle_id": ["v_lo", "v_mid1", "v_mid2", "v_hi", "v_hi"],
        "weekday": [2, 2, 2, 2, 1],
        "miles_driven": [10.0, 20.0, 30.0, 500.0, 500.0],
        "income_bucket": [1, 1, 1, 1, 1],
        "occupants": [2, 2, 2, 2, 2],
        "vehicles": [1, 1, 1, 1, 1],
        "start_time": [800, 800, 800, 800, 800],
        "end_time": [1700, 1700, 1700, 1700, 1700],
        "trip_weight": [1.0, 1.0, 1.0, 1.0, 1.0],
    })
    sampler = NHTSProfileSampler(
        nhts_df=nhts,
        nhts_daily_miles_percentile_low=25.0,
        nhts_daily_miles_percentile_high=75.0,
    )
    kept = set(sampler.nhts_df["hh_vehicle_id"].unique().to_list())
    assert "v_lo" not in kept
    assert "v_hi" not in kept
    assert kept <= {"v_mid1", "v_mid2"}


def test_load_ev_demand_config_from_yaml(tmp_path):
    from utils.EVs.ev_demand import load_ev_demand_config

    path = tmp_path / "scenario.yml"
    path.write_text(
        """
state: MD
release: res_2024_tmy3_2
start_date: 2024-01-01T04:00:00
end_date: 2024-02-01T03:00:00
sampling:
  nhts_daily_miles_percentile_low: 10
  nhts_daily_miles_percentile_high: 90
  random_state: 7
trips:
  min_trip_away_hours: 1
  miles_noise_std_fraction: 0.1
battery:
  capacity_buffer_fraction: 0.2
pipeline:
  batch_size: 1000
charging:
  charging_strategy: immediate
  charger_power_kw: 7.2
"""
    )
    config = load_ev_demand_config(path)
    assert config.state == "MD"
    assert config.release == "res_2024_tmy3_2"
    assert config.start_date == datetime(2024, 1, 1, 4)
    assert config.end_date == datetime(2024, 2, 1, 3)
    assert config.ev_assignment == "resstock_adoption"
    assert config.match_on_vehicles is False
    assert config.max_vehicles is None
    assert config.nhts_daily_miles_percentile_low == 10
    assert config.nhts_daily_miles_percentile_high == 90
    assert config.random_state == 7
    assert config.batch_size == 1000
    assert config.charger_power_kw == 7.2
    assert config.min_trip_away_hours == 1
    assert config.miles_noise_std_fraction == 0.1
    assert config.capacity_buffer_fraction == 0.2
    assert config.soc_min_fraction is None
    assert config.peak_clock_hours is None
    assert config.flat_price_usd_per_kwh is None
    assert config.num_simulation_hours() == 31 * 24
    assert "ev_data/inputs" in config.nhts_path.replace("\\", "/")
    assert config.ev_ownership_path is not None
    assert config.pums_path is None
    assert config.weather_dir is None
    assert config.temperature_adjustment == "none"


def test_resolve_hourly_prices_flat_and_daily():
    from utils.EVs.ev_demand import EVDemandConfig, resolve_hourly_prices

    flat_cfg = EVDemandConfig(
        state="MD",
        release="res_2024_tmy3_2",
        start_date=datetime(2024, 1, 1, 4),
        end_date=datetime(2024, 1, 3, 3),
        charging_strategy="cost_minimizing",
        flat_price_usd_per_kwh=0.12,
        shed_load_penalty_usd_per_kwh=1000.0,
    )
    flat = resolve_hourly_prices(flat_cfg)
    assert flat is not None
    assert len(flat) == 48
    assert np.allclose(flat, 0.12)

    daily = tuple([0.10] * 12 + [0.20] * 12)
    daily_cfg = EVDemandConfig(
        state="MD",
        release="res_2024_tmy3_2",
        start_date=datetime(2024, 1, 1, 4),
        end_date=datetime(2024, 1, 3, 3),
        charging_strategy="cost_minimizing",
        daily_price_usd_per_kwh=daily,
        shed_load_penalty_usd_per_kwh=1000.0,
    )
    tiled = resolve_hourly_prices(daily_cfg)
    assert tiled is not None
    assert len(tiled) == 48
    assert tiled[0] == 0.10
    assert tiled[12] == 0.20
    assert tiled[24] == 0.10


def test_cost_minimizing_requires_prices_and_shed_penalty():
    from utils.EVs.ev_demand import EVDemandConfig

    with pytest.raises(ValueError, match="cost_minimizing requires one of"):
        EVDemandConfig(
            state="MD",
            release="res_2024_tmy3_2",
            start_date=datetime(2024, 1, 1, 4),
            end_date=datetime(2024, 1, 2, 3),
            charging_strategy="cost_minimizing",
            shed_load_penalty_usd_per_kwh=1000.0,
        )

    with pytest.raises(ValueError, match="shed_load_penalty_usd_per_kwh"):
        EVDemandConfig(
            state="MD",
            release="res_2024_tmy3_2",
            start_date=datetime(2024, 1, 1, 4),
            end_date=datetime(2024, 1, 2, 3),
            charging_strategy="cost_minimizing",
            flat_price_usd_per_kwh=0.12,
        )


def test_off_peak_requires_peak_window_and_soc_targets():
    from utils.EVs.ev_demand import EVDemandConfig

    with pytest.raises(ValueError, match="off_peak requires"):
        EVDemandConfig(
            state="MD",
            release="res_2024_tmy3_2",
            start_date=datetime(2024, 1, 1, 4),
            end_date=datetime(2024, 1, 2, 3),
            charging_strategy="off_peak",
        )

    cfg = EVDemandConfig(
        state="MD",
        release="res_2024_tmy3_2",
        start_date=datetime(2024, 1, 1, 4),
        end_date=datetime(2024, 1, 2, 3),
        charging_strategy="off_peak",
        peak_clock_hours=(17, 18, 19, 20, 21),
        soc_min_fraction=0.2,
        soc_safety_buffer_fraction=0.2,
    )
    assert cfg.peak_clock_hours == (17, 18, 19, 20, 21)


def test_load_md_2024_config_off_peak():
    from utils.EVs.ev_demand import load_ev_demand_config

    config = load_ev_demand_config("utils/EVs/configs/md_2024.yaml")
    assert config.start_date == datetime(2024, 1, 1, 4)
    assert config.end_date == datetime(2025, 1, 1, 3)
    assert config.num_simulation_hours() == 366 * 24  # 2024 leap year travel-day window
    assert config.charging_strategy == "off_peak_immediate"
    assert config.peak_clock_hours == (17, 18, 19, 20, 21)
    assert config.soc_min_fraction is None
    assert config.soc_safety_buffer_fraction is None
    assert config.shed_load_penalty_usd_per_kwh is None
    assert config.flat_price_usd_per_kwh is None
    assert config.allow_emergency_peak_charging is False
    assert config.temperature_adjustment == "resstock"
    assert config.max_departure_hour == 27
    assert config.max_arrival_hour == 28


def test_config_requires_travel_day_aligned_datetimes(tmp_path):
    from utils.EVs.ev_demand import (
        EVDemandConfig,
        InvalidDateFormatError,
        load_ev_demand_config,
    )

    # Date-only strings are rejected at parse time.
    path = tmp_path / "date_only.yml"
    path.write_text(
        """
state: MD
release: res_2024_tmy3_2
start_date: 2024-01-01
end_date: 2024-01-02T03:00:00
charging:
  charging_strategy: immediate
  charger_power_kw: 7.2
"""
    )
    with pytest.raises((InvalidDateFormatError, ValueError)):
        load_ev_demand_config(path)

    # Wrong clock hours rejected in EVDemandConfig.
    with pytest.raises(ValueError, match="start_date must be at 04:00"):
        EVDemandConfig(
            state="MD",
            release="res_2024_tmy3_2",
            start_date=datetime(2024, 1, 1, 0),
            end_date=datetime(2024, 1, 2, 3),
            charging_strategy="immediate",
        )
    with pytest.raises(ValueError, match="end_date must be at 03:00"):
        EVDemandConfig(
            state="MD",
            release="res_2024_tmy3_2",
            start_date=datetime(2024, 1, 1, 4),
            end_date=datetime(2024, 1, 2, 23),
            charging_strategy="immediate",
        )


def test_off_peak_immediate_config_requires_peak_only():
    from utils.EVs.ev_demand import EVDemandConfig

    with pytest.raises(ValueError, match="peak_clock_hours"):
        EVDemandConfig(
            state="MD",
            release="res_2024_tmy3_2",
            start_date=datetime(2024, 1, 1, 4),
            end_date=datetime(2024, 1, 2, 3),
            charging_strategy="off_peak_immediate",
        )

    cfg = EVDemandConfig(
        state="MD",
        release="res_2024_tmy3_2",
        start_date=datetime(2024, 1, 1, 4),
        end_date=datetime(2024, 1, 2, 3),
        charging_strategy="off_peak_immediate",
        peak_clock_hours=(17, 18, 19, 20, 21),
        allow_emergency_peak_charging=True,
    )
    assert cfg.soc_min_fraction is None
    assert cfg.allow_emergency_peak_charging is True


def test_load_ev_demand_config_pums_vehicles_requires_max_vehicles(tmp_path):
    from utils.EVs.ev_demand import load_ev_demand_config

    path = tmp_path / "pums_missing_max.yml"
    path.write_text(
        """
state: MD
release: res_2024_tmy3_2
start_date: 2024-01-01T04:00:00
end_date: 2024-02-01T03:00:00
sampling:
  ev_assignment: pums_vehicles
"""
    )
    with pytest.raises(ValueError, match="max_vehicles is required"):
        load_ev_demand_config(path)


def test_load_ev_demand_config_pums_vehicles_ok(tmp_path):
    from utils.EVs.ev_demand import load_ev_demand_config

    path = tmp_path / "pums_ok.yml"
    path.write_text(
        """
state: MD
release: res_2024_tmy3_2
start_date: 2024-01-01T04:00:00
end_date: 2024-02-01T03:00:00
sampling:
  ev_assignment: pums_vehicles
  max_vehicles: 2
"""
    )
    config = load_ev_demand_config(path)
    assert config.ev_assignment == "pums_vehicles"
    assert config.max_vehicles == 2
    assert config.match_on_vehicles is True
    assert config.pums_path is not None
    assert config.ev_ownership_path is None
    assert config.weather_dir is None


def test_config_path_defaults_by_mode():
    from utils.EVs.ev_demand import EVDemandConfig

    adoption = EVDemandConfig(
        state="MD",
        release="res_2024_tmy3_2",
        start_date=datetime(2024, 1, 1, 4),
        end_date=datetime(2024, 1, 2, 3),
        ev_assignment="resstock_adoption",
    )
    assert adoption.ev_ownership_path is not None
    assert adoption.pums_path is None
    assert adoption.weather_dir is None

    pums = EVDemandConfig(
        state="MD",
        release="res_2024_tmy3_2",
        start_date=datetime(2024, 1, 1, 4),
        end_date=datetime(2024, 1, 2, 3),
        ev_assignment="pums_vehicles",
        max_vehicles=2,
    )
    assert pums.pums_path is not None
    assert pums.ev_ownership_path is None

    with_temp = EVDemandConfig(
        state="MD",
        release="res_2024_tmy3_2",
        start_date=datetime(2024, 1, 1, 4),
        end_date=datetime(2024, 1, 2, 3),
        temperature_adjustment="resstock",
    )
    assert with_temp.weather_dir is not None


def test_calculator_requires_ownership_for_adoption(
    mock_nhts_data, mock_metadata, ev_battery_df, ev_autonomie_df
):
    with pytest.raises(ValueError, match="ev_ownership_df is required"):
        EVDemandCalculator(
            metadata_df=mock_metadata,
            nhts_df=mock_nhts_data,
            ev_battery_df=ev_battery_df,
            ev_autonomie_df=ev_autonomie_df,
            start_date=datetime(2022, 1, 1, 4),
            end_date=datetime(2022, 1, 8, 3),
            ev_assignment="resstock_adoption",
        )


def test_load_ev_demand_config_rejects_match_on_vehicles(tmp_path):
    from utils.EVs.ev_demand import load_ev_demand_config

    path = tmp_path / "legacy.yml"
    path.write_text(
        """
state: MD
release: res_2024_tmy3_2
start_date: 2024-01-01T04:00:00
end_date: 2024-02-01T03:00:00
sampling:
  match_on_vehicles: false
"""
    )
    with pytest.raises(ValueError, match="match_on_vehicles"):
        load_ev_demand_config(path)


def test_assign_ev_slots_resstock_adoption(calculator):
    """Default mode samples 0/1 EVs and aliases to vehicles."""
    # Expand metadata with columns required by EVAdoptionSampler.
    meta = calculator.metadata_df.with_columns(
        pl.lit("0-100%").alias("fpl"),
        pl.lit("Single-Family Detached").alias("building_type"),
        pl.lit("Owner").alias("tenure"),
        pl.lit("G24008500").alias("puma_dependency"),
        pl.lit(False).alias("is_vacant"),
    )
    # Use a lookup that covers the synthetic keys.
    ownership = pl.DataFrame({
        "fpl": ["0-100%"],
        "building_type": ["Single-Family Detached"],
        "tenure": ["Owner"],
        "puma_dependency": ["G24008500"],
        "ev_ownership_probability": [1.0],
    })
    calc = EVDemandCalculator(
        metadata_df=meta,
        nhts_df=calculator.nhts_df,
        ev_ownership_df=ownership,
        ev_battery_df=calculator.ev_battery_df,
        ev_autonomie_df=calculator.ev_autonomie_df,
        start_date=calculator.start_date,
        end_date=calculator.end_date,
        ev_assignment="resstock_adoption",
        random_state=42,
    )
    assert calc.match_on_vehicles is False
    assert calc.nhts_sampler.match_on_vehicles is False
    assigned = calc._assign_ev_slots()
    assert "vehicles" in assigned.columns
    assert set(assigned["vehicles"].unique().to_list()) <= {0, 1}
    assert assigned["vehicles"].sum() == len(meta)  # P(EV)=1 for all


def test_assign_ev_slots_pums_vehicles(calculator, mock_metadata):
    """PUMS mode predicts vehicle counts and enables NHTS vehicle matching."""
    meta = mock_metadata.with_columns(pl.col("income_bucket").alias("income"))
    pums = meta.with_columns(pl.lit(1.0).alias("hh_weight"))
    calc = EVDemandCalculator(
        metadata_df=meta,
        nhts_df=calculator.nhts_df,
        ev_battery_df=calculator.ev_battery_df,
        ev_autonomie_df=calculator.ev_autonomie_df,
        start_date=calculator.start_date,
        end_date=calculator.end_date,
        pums_df=pums,
        ev_assignment="pums_vehicles",
        max_vehicles=2,
        random_state=42,
    )
    assert calc.match_on_vehicles is True
    assert calc.nhts_sampler.match_on_vehicles is True
    assigned = calc._assign_ev_slots()
    assert "vehicles" in assigned.columns
    assert assigned["vehicles"].max() <= 2
