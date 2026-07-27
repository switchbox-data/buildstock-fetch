import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from utils.EVs.NHTSProfileSampler import VehicleProfile

# NHTS travel day is 4:00am through 3:59am the next calendar day.
DEFAULT_TRAVEL_DAY_START_HOUR = 4
# Exclusive end of the travel-day window in extended hours (4am next day = 24 + 4).
DEFAULT_TRAVEL_DAY_END_HOUR = 28

# Defaults for trip perturbation / packing (overridden by EVDemandConfig / YAML).
# Departure/arrival bounds are in travel-day extended hours:
#   4..23 = clock hours on the travel-day start date
#   24..27 = clock hours 0..3 on the following calendar date
DEFAULT_MIN_TRIP_AWAY_HOURS = 1
DEFAULT_MAX_DEPARTURE_HOUR = DEFAULT_TRAVEL_DAY_END_HOUR - 1  # 27 = 3am next day
DEFAULT_MAX_ARRIVAL_HOUR = DEFAULT_TRAVEL_DAY_END_HOUR  # 28 = 4am next day (exclusive)
DEFAULT_TIME_OFFSETS: tuple[int, ...] = (-2, -1, 0, 1, 2)
DEFAULT_TIME_OFFSET_PROBABILITIES: tuple[float, ...] = (0.05, 0.10, 0.70, 0.10, 0.05)
DEFAULT_MILES_NOISE_STD_FRACTION = 0.1


def sample_truncated_normal_nonnegative(
    rng: np.random.RandomState,
    loc: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Sample from ``Normal(loc, scale)`` truncated to ``[0, ∞)``.

    Uses rejection sampling so we stay on ``numpy.random.RandomState`` (no scipy).
    When ``scale <= 0``, returns ``max(loc, 0)`` (degenerate / non-positive noise).

    Args:
        rng: NumPy ``RandomState`` used for draws
        loc: Location (mean of the untruncated normal), same shape as ``scale``
        scale: Scale (std of the untruncated normal), same shape as ``loc``

    Returns:
        Samples with the same shape as ``loc``, all ``>= 0``
    """
    loc_arr = np.asarray(loc, dtype=float)
    scale_arr = np.asarray(scale, dtype=float)
    if loc_arr.shape != scale_arr.shape:
        raise ValueError(
            f"loc and scale must have the same shape; got {loc_arr.shape} vs {scale_arr.shape}"
        )

    out = np.empty(loc_arr.shape, dtype=float)
    degenerate = scale_arr <= 0
    out[degenerate] = np.maximum(loc_arr[degenerate], 0.0)

    active = ~degenerate
    if not np.any(active):
        return out

    out[active] = rng.normal(loc_arr[active], scale_arr[active])
    negative = active & (out < 0.0)
    while np.any(negative):
        out[negative] = rng.normal(loc_arr[negative], scale_arr[negative])
        negative = active & (out < 0.0)
    return out


def clock_hour_to_travel_hour(
    clock_hour: int,
    *,
    travel_day_start_hour: int = DEFAULT_TRAVEL_DAY_START_HOUR,
) -> int:
    """Map a clock hour (0..23) onto NHTS travel-day extended hours.

    Hours at/after ``travel_day_start_hour`` stay on the travel-day start date.
    Hours before ``travel_day_start_hour`` belong to the early morning after midnight
    and are encoded as ``clock_hour + 24``.

    Args:
        clock_hour (int): Clock hour to map (0..23)
        travel_day_start_hour (int): Start hour of the travel day (default 4am)

    Returns:
        int: Travel-day extended hour (0..27)
    """
    hour = int(clock_hour)
    if hour < 0 or hour > 23:
        raise ValueError(f"clock_hour must be in 0..23, got {hour}")
    return hour if hour >= travel_day_start_hour else hour + 24


def travel_hour_to_calendar(
    travel_date: date,
    travel_hour: int,
) -> tuple[date, int]:
    """Convert a travel-day extended hour to a (calendar date, clock hour) pair.
    
    Args:
        travel_date (date): Travel day start date
        travel_hour (int): Travel-day extended hour (0..27)

    Returns:
        tuple[date, int]: (calendar date, clock hour) pair
    """
    hour = int(travel_hour)
    if hour < 24:
        return travel_date, hour
    return travel_date + timedelta(days=1), hour - 24


def iter_travel_day_starts(
    start: datetime,
    end: datetime,
    *,
    travel_day_start_hour: int = DEFAULT_TRAVEL_DAY_START_HOUR,
) -> list[date]:
    """Return travel-day start dates whose [4am, next 4am) window overlaps ``[start, end]``.

    ``end`` is treated as the last instant included in the simulation (typically the
    start of the final hour slot, e.g. 03:00 for a window ending at 03:59).

    Args:
        start (datetime): Start of the simulation window
        end (datetime): End of the simulation window
        travel_day_start_hour (int): Start hour of the travel day (default 4am)

    Returns:
        list[date]: List of travel-day start dates

    Example:
        >>> iter_travel_day_starts(datetime(2024, 1, 1, 4), datetime(2024, 1, 2, 3, 59), travel_day_start_hour=4)
        [datetime.date(2024, 1, 1)]
        >>> iter_travel_day_starts(datetime(2024, 1, 1, 0), datetime(2024, 1, 2, 0, 0), travel_day_start_hour=4)
        [datetime.date(2023, 12, 31), datetime.date(2024, 1, 1)]
    """
    if end < start:
        raise ValueError(f"end {end} must be on or after start {start}")

    # First travel day that could overlap: the travel day containing ``start``.
    if start.hour >= travel_day_start_hour:
        first = start.date()
    else:
        first = start.date() - timedelta(days=1)

    # Last travel day that could overlap: the travel day containing ``end``.
    if end.hour >= travel_day_start_hour:
        last = end.date()
    else:
        last = end.date() - timedelta(days=1)

    if last < first:
        return []

    days: list[date] = []
    current = first
    while current <= last:
        days.append(current)
        current += timedelta(days=1)
    return days


@dataclass
class TripScheduleGenerator:
    """Generate daily trip schedules from sampled NHTS vehicle profiles.

    Each NHTS profile is replayed on an NHTS travel day (4am → 3:59am next day).
    Early-morning clock hours (0–3) are placed on the calendar day after the
    travel-day start date, and overnight trips may span midnight.
    """

    start_date: datetime
    end_date: datetime
    random_state: int = 42
    max_workers: int | None = None
    travel_day_start_hour: int = DEFAULT_TRAVEL_DAY_START_HOUR
    travel_day_end_hour: int = DEFAULT_TRAVEL_DAY_END_HOUR
    min_trip_away_hours: int = DEFAULT_MIN_TRIP_AWAY_HOURS
    max_departure_hour: int = DEFAULT_MAX_DEPARTURE_HOUR
    max_arrival_hour: int = DEFAULT_MAX_ARRIVAL_HOUR
    time_offsets: tuple[int, ...] = DEFAULT_TIME_OFFSETS
    time_offset_probabilities: tuple[float, ...] = DEFAULT_TIME_OFFSET_PROBABILITIES
    miles_noise_std_fraction: float = DEFAULT_MILES_NOISE_STD_FRACTION
    _time_offsets_arr: np.ndarray = field(init=False, repr=False)
    _time_probs_arr: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 <= self.travel_day_start_hour <= 23:
            raise ValueError(
                f"travel_day_start_hour must be in 0..23; got {self.travel_day_start_hour}"
            )
        if self.travel_day_end_hour != self.travel_day_start_hour + 24:
            raise ValueError(
                "travel_day_end_hour must equal travel_day_start_hour + 24 "
                f"(got start={self.travel_day_start_hour}, end={self.travel_day_end_hour})"
            )
        if self.min_trip_away_hours < 1:
            raise ValueError(f"min_trip_away_hours must be >= 1; got {self.min_trip_away_hours}")
        if self.max_departure_hour < self.travel_day_start_hour:
            raise ValueError(
                f"max_departure_hour must be >= travel_day_start_hour; "
                f"got {self.max_departure_hour} < {self.travel_day_start_hour}"
            )
        if self.max_arrival_hour <= self.travel_day_start_hour:
            raise ValueError(
                f"max_arrival_hour must be > travel_day_start_hour; got {self.max_arrival_hour}"
            )
        if self.max_arrival_hour > self.travel_day_end_hour:
            raise ValueError(
                f"max_arrival_hour must be <= travel_day_end_hour ({self.travel_day_end_hour}); "
                f"got {self.max_arrival_hour}"
            )
        if len(self.time_offsets) != len(self.time_offset_probabilities):
            raise ValueError(
                "time_offsets and time_offset_probabilities must have the same length; "
                f"got {len(self.time_offsets)} and {len(self.time_offset_probabilities)}"
            )
        if self.miles_noise_std_fraction < 0:
            raise ValueError(
                f"miles_noise_std_fraction must be >= 0; got {self.miles_noise_std_fraction}"
            )
        probs = np.asarray(self.time_offset_probabilities, dtype=np.float64)
        if probs.sum() <= 0:
            raise ValueError("time_offset_probabilities must sum to a positive value")
        self._time_offsets_arr = np.asarray(self.time_offsets, dtype=np.int64)
        self._time_probs_arr = probs / probs.sum()

    @staticmethod
    def _log_progress(current: int, total: int, description: str, progress_interval: int = 10000) -> None:
        """
        Log progress if at the right interval or at completion.

        Args:
            current: Current number of items processed
            total: Total number of items to process
            description: Description for the progress message
            progress_interval: Interval for logging (calculated if None)
        """

        if current % progress_interval == 0 or current == total:
            percent_complete = (current / total) * 100
            logging.info(f"{description}: {current}/{total} ({percent_complete:.1f}%)")

    def _clock_hours_to_travel_hours(self, clock_hours: np.ndarray) -> np.ndarray:
        """Vectorized clock hour → travel-day extended hour.
        
        Args:
            clock_hours (np.ndarray): Array of clock hours (0..23)

        Returns:
            np.ndarray: Array of travel-day extended hours (0..27)
        """
        hours = clock_hours.astype(int)
        return np.where(
            hours >= self.travel_day_start_hour,
            hours,
            hours + 24,
        )

    def _normalize_day_trip_times(
        self,
        departures: np.ndarray,
        arrival_hours: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Enforce arrival > departure and non-overlapping intervals within a travel day.

        Used for **tour** away windows (leave-home → return-home). Inputs are travel-day
        extended hours. Away hours are ``range(departure_hour, arrival_hour)``.
        Independent random offsets can produce invalid or overlapping intervals.
        Tours are repacked in departure order on a single travel day (4am → next 4am).
        Each tour's away interval ends before the next tour's departure hour begins.
        There is no minimum dwell-at-home between tours — the next departure may equal
        the prior arrival. Returns arrays in the original input order. Tours that no
        longer fit before the travel-day end are dropped (keep_mask=False).

        Args:
            departures (np.ndarray): Array of departure hours (travel-day extended)
            arrival_hours (np.ndarray): Array of first-at-home hours (exclusive end of away interval)

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of normalized departure hours,
                arrival hours, and keep mask
        """
        if len(departures) != len(arrival_hours):
            raise ValueError(
                f"departures and arrival_hours must have the same length, got {len(departures)} and {len(arrival_hours)}"
            )

        n = len(departures)
        keep = np.ones(n, dtype=bool)
        if n == 0:
            return departures.astype(int), arrival_hours.astype(int), keep

        order = np.argsort(departures, kind="stable")
        dep_sorted = departures[order].astype(int)
        arrival_sorted = arrival_hours[order].astype(int)
        keep_sorted = np.ones(n, dtype=bool)

        earliest_next_dep = self.travel_day_start_hour
        for i in range(n):
            if earliest_next_dep > self.max_departure_hour:
                keep_sorted[i:] = False
                break

            dep = min(max(int(dep_sorted[i]), earliest_next_dep), self.max_departure_hour)
            arrival = min(
                max(int(arrival_sorted[i]), dep + self.min_trip_away_hours),
                self.max_arrival_hour,
            )
            if arrival <= dep:
                keep_sorted[i] = False
                continue

            dep_sorted[i] = dep
            arrival_sorted[i] = arrival
            earliest_next_dep = arrival

        dep_out = np.empty(n, dtype=int)
        arrival_out = np.empty(n, dtype=int)
        dep_out[order] = dep_sorted
        arrival_out[order] = arrival_sorted
        keep[order] = keep_sorted
        return dep_out, arrival_out, keep

    def generate_daily_trip_schedule(
        self, profile: VehicleProfile, rng: np.random.RandomState | None = None
    ) -> pl.DataFrame:
        """Generate trip schedules for a vehicle for all travel days in the date range.

        For each travel day (4am → 3:59am next day), NHTS weekday/weekend templates are
        replayed in chronological order. Each output row is one **driving leg** with:

        - ``travel_date`` — start calendar date of the NHTS travel day (4am boundary)
        - ``trip_*`` — drive interval (discharge + temperature)
        - ``tour_*`` — leave-home → return-home window (presence / ``at_home``)
        - ``tour_id`` — links legs that belong to the same away tour

        Time noise is modeled as: independent departure and arrival hour offsets on
        each **drive leg**. Tour leave/return bounds are recomputed from those legs
        (``min`` dep / ``max`` arr). Only **tours** are packed so away windows do not
        overlap; mid-tour drive legs are not forced apart. Tours that end away from
        home in NHTS extend to the travel-day end for presence. Miles noise remains
        per leg.

        Args:
            profile (VehicleProfile): Vehicle profile to generate schedules for
            rng (np.random.RandomState): Random number generator

        Returns:
            pl.DataFrame: trip schedules with ``travel_date`` plus trip/tour calendar bounds
        """
        if rng is None:
            # Prefer the instance seed over the process-global RNG so day replay is
            # reproducible for a given TripScheduleGenerator(random_state=...).
            rng = np.random.RandomState(self.random_state)

        time_offsets = self._time_offsets_arr
        time_probabilities = self._time_probs_arr

        travel_days = iter_travel_day_starts(
            self.start_date,
            self.end_date,
            travel_day_start_hour=self.travel_day_start_hour,
        )

        # Pre-allocate lists for batch DataFrame construction
        bldg_ids: list[str] = []
        vehicle_ids: list[int] = []
        travel_dates: list[datetime] = [] # date of the 4am NHTS travel-day start (not drive date)
        trip_departure_dates: list[datetime] = [] # calendar date of departure times
        trip_arrival_dates: list[datetime] = [] # calendar date of arrival times
        trip_departure_hours: list[int] = [] # trip departure hours
        trip_arrival_hours: list[int] = [] # trip arrival hours 
        trip_miles_driven: list[float] = [] # miles driven for each trip
        tour_ids_out: list[int] = [] # links legs that belong to the same away tour
        tour_departure_dates: list[datetime] = [] # calendar date of tour departure times
        tour_arrival_dates: list[datetime] = [] # calendar date of tour arrival times
        tour_departure_hours: list[int] = [] # tour departure hours
        tour_arrival_hours: list[int] = [] # tour arrival hours (exclusive end of away interval)

        for travel_date in travel_days:
            is_weekday = travel_date.weekday() < 5  # Monday-Friday are weekdays
            day = profile.weekday if is_weekday else profile.weekend
            n_trips = len(day.trip_ids)
            if n_trips == 0:
                continue

            trip_dep = np.array(day.trip_departure_hours, dtype=int)
            trip_arr = np.array(day.trip_arrival_hours, dtype=int)
            base_miles = np.array(day.trip_miles_driven, dtype=float)
            trip_tour_ids = np.array(day.tour_ids, dtype=int)
            tour_ends_away = np.array(day.tour_ends_away, dtype=bool)
            n_tours = len(day.tour_departure_hours)

            # Map clock hours onto the travel-day extended axis (4..27).
            trip_dep = self._clock_hours_to_travel_hours(trip_dep)
            trip_arr = self._clock_hours_to_travel_hours(trip_arr)

            # Drive windows that wrap past midnight on the extended axis.
            trip_wrap = trip_arr <= trip_dep
            trip_arr = trip_arr + trip_wrap.astype(int) * 24

            # Independent dep / arr offsets per drive leg.
            trip_dep = trip_dep + rng.choice(
                time_offsets, size=n_trips, p=time_probabilities
            )
            trip_arr = trip_arr + rng.choice(
                time_offsets, size=n_trips, p=time_probabilities
            )
            # Keep each drive interval valid; do not pack sibling legs against each other.
            trip_arr = np.maximum(trip_arr, trip_dep + self.min_trip_away_hours)

            # Tour leave/return = span of legs (open tours stay away to day end).
            tour_dep = np.zeros(n_tours, dtype=int)
            tour_arr = np.zeros(n_tours, dtype=int)
            keep_tour = np.ones(n_tours, dtype=bool)
            for t_idx in range(n_tours):
                leg_idx = np.flatnonzero(trip_tour_ids == (t_idx + 1))
                if leg_idx.size == 0:
                    keep_tour[t_idx] = False
                    continue
                tour_dep[t_idx] = int(trip_dep[leg_idx].min())
                if tour_ends_away[t_idx]:
                    # tour ends away from home, so the tour arrival hour is the max arrival hour
                    # this is an assumption but might need to be fixed later
                    tour_arr[t_idx] = self.max_arrival_hour
                else:
                    tour_arr[t_idx] = int(trip_arr[leg_idx].max())

            # Pack tours so away windows do not overlap; shift legs with the tour.
            tour_dep_before_pack = tour_dep.copy()
            tour_dep_norm, tour_arr_norm, keep_after_pack = self._normalize_day_trip_times(
                tour_dep, tour_arr
            )
            keep_tour = keep_tour & keep_after_pack

            for t_idx in range(n_tours):
                if not keep_tour[t_idx]:
                    continue
                delta = int(tour_dep_norm[t_idx]) - int(tour_dep_before_pack[t_idx])
                if delta == 0:
                    continue
                leg_mask = trip_tour_ids == (t_idx + 1)
                trip_dep[leg_mask] = trip_dep[leg_mask] + delta
                trip_arr[leg_mask] = trip_arr[leg_mask] + delta

            # Miles ~ truncated Normal(base, base * frac) on [0, ∞).
            miles_variance = sample_truncated_normal_nonnegative(
                rng,
                base_miles,
                base_miles * self.miles_noise_std_fraction,
            )

            # Emit one row per driving leg whose tour survived packing.
            next_tour_id = 1
            old_to_new_tour: dict[int, int] = {}
            for t_idx in range(n_tours):
                if not keep_tour[t_idx]:
                    continue
                old_to_new_tour[t_idx + 1] = next_tour_id
                next_tour_id += 1

            if not old_to_new_tour:
                continue

            for leg in range(n_trips):
                old_tour = int(trip_tour_ids[leg])
                if old_tour not in old_to_new_tour:
                    continue

                # Keep drive intervals inside the (normalized) tour away window.
                t_idx = old_tour - 1
                tour_dep_th = int(tour_dep_norm[t_idx])
                tour_arr_th = int(tour_arr_norm[t_idx])
                dep_th = int(np.clip(trip_dep[leg], tour_dep_th, max(tour_dep_th, tour_arr_th - 1)))
                arr_th = int(np.clip(trip_arr[leg], dep_th + 1, tour_arr_th))
                if arr_th <= dep_th:
                    continue

                dep_cal_date, dep_clock = travel_hour_to_calendar(travel_date, dep_th)
                arr_cal_date, arr_clock = travel_hour_to_calendar(travel_date, arr_th)
                tour_dep_cal, tour_dep_clock = travel_hour_to_calendar(travel_date, tour_dep_th)
                tour_arr_cal, tour_arr_clock = travel_hour_to_calendar(travel_date, tour_arr_th)

                bldg_ids.append(profile.bldg_id)
                vehicle_ids.append(profile.vehicle_id)
                # travel_date = calendar date of the 4am NHTS travel-day start (not drive date).
                travel_dates.append(datetime.combine(travel_date, datetime.min.time()))
                trip_departure_dates.append(datetime.combine(dep_cal_date, datetime.min.time()))
                trip_arrival_dates.append(datetime.combine(arr_cal_date, datetime.min.time()))
                trip_departure_hours.append(dep_clock)
                trip_arrival_hours.append(arr_clock)
                trip_miles_driven.append(float(miles_variance[leg]))
                tour_ids_out.append(old_to_new_tour[old_tour])
                tour_departure_dates.append(datetime.combine(tour_dep_cal, datetime.min.time()))
                tour_arrival_dates.append(datetime.combine(tour_arr_cal, datetime.min.time()))
                tour_departure_hours.append(tour_dep_clock)
                tour_arrival_hours.append(tour_arr_clock)

        schedule_data = {
            "bldg_id": bldg_ids,
            "vehicle_id": vehicle_ids,
            # Start date of the NHTS travel day [4am, next 4am); may differ from trip_* dates
            # when a drive crosses midnight.
            "travel_date": travel_dates,
            # Drive interval (kWh discharge + temperature)
            "trip_departure_date": trip_departure_dates,
            "trip_departure_hour": trip_departure_hours,
            "trip_arrival_date": trip_arrival_dates,
            "trip_arrival_hour": trip_arrival_hours,
            "trip_miles_driven": trip_miles_driven,
            # Home-away tour (presence / charging eligibility)
            "tour_id": tour_ids_out,
            "tour_departure_date": tour_departure_dates,
            "tour_departure_hour": tour_departure_hours,
            "tour_arrival_date": tour_arrival_dates,
            "tour_arrival_hour": tour_arrival_hours,
        }

        return pl.DataFrame(schedule_data).sort(
            [
                "travel_date",
                "tour_id",
                "trip_departure_date",
                "trip_departure_hour",
                "trip_arrival_date",
                "trip_arrival_hour",
            ]
        )

    def generate(
        self,
        profile_params: dict[tuple[str, int], VehicleProfile],
    ) -> pl.DataFrame:
        """
        Generate trip schedules for all given vehicle profiles for all days in the date range as a DataFrame.

        For each day, the departure and arrival times and miles traveled are generated from random
        perturbations of the base values in the vehicle profile. Trips are measured in hourly increments.

        Args:
            profile_params (dict[tuple[str, int], VehicleProfile]): Dict of vehicle profiles and their associated building and vehicle IDs

        Returns:
            pl.DataFrame: daily departure/arrival times and miles driven for each vehicle
        """
        def process_profile_with_seed(profile_and_index: tuple[VehicleProfile, int]) -> pl.DataFrame:
            profile, index = profile_and_index
            # Derive a unique seed for each profile based on the base random_state and index
            profile_seed = self.random_state + index
            rng = np.random.RandomState(profile_seed)
            return self.generate_daily_trip_schedule(profile, rng)

        # Convert to list of (profile, index) tuples for processing
        profiles_with_indices = [(profile, i) for i, profile in enumerate(profile_params.values())]
        total_profiles = len(profiles_with_indices)

        if total_profiles == 0:
            return pl.DataFrame()

        all_schedules = []
        if self.max_workers is None or self.max_workers <= 1 or total_profiles == 1:
            # Serial processing
            for i, profile_and_index in enumerate(profiles_with_indices, 1):
                all_schedules.append(process_profile_with_seed(profile_and_index))
                self._log_progress(i, total_profiles, "Generating trip schedules")
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_index = {
                    executor.submit(process_profile_with_seed, profile_and_index): i
                    for i, profile_and_index in enumerate(profiles_with_indices)
                }

                completed = 0
                for future in as_completed(future_to_index):
                    all_schedules.append(future.result())
                    completed += 1
                    self._log_progress(completed, total_profiles, "Generating trip schedules")

        total_schedules = sum(len(schedule) for schedule in all_schedules)
        logging.info(f"Generated {total_schedules} trip schedules from {total_profiles} vehicle profiles")

        if not all_schedules:
            return pl.DataFrame()
        # Parallel concat order is nondeterministic; sort for stable, chrono-readable output.
        return pl.concat(all_schedules).sort(
            [
                "bldg_id",
                "vehicle_id",
                "travel_date",
                "tour_id",
                "trip_departure_date",
                "trip_departure_hour",
                "trip_arrival_date",
                "trip_arrival_hour",
            ]
        )

    @staticmethod
    def max_daily_miles_from_trip_schedules(trip_schedules: pl.DataFrame) -> pl.DataFrame:
        """
        Per-vehicle peak daily miles over the simulated trip schedule.

        Sums ``trip_miles_driven`` within each (bldg_id, vehicle_id, travel_date), then
        takes the max across travel days. Vehicles absent from ``trip_schedules`` are
        omitted (callers should left-join onto vehicle slots and fill nulls with 0).
        """
        required = {"bldg_id", "vehicle_id", "travel_date", "trip_miles_driven"}
        missing = required - set(trip_schedules.columns)
        if missing:
            raise ValueError(f"trip_schedules missing columns: {sorted(missing)}")

        if trip_schedules.is_empty():
            return pl.DataFrame(
                schema={
                    "bldg_id": trip_schedules.schema.get("bldg_id", pl.Int64),
                    "vehicle_id": pl.Int64,
                    "max_daily_miles": pl.Float64,
                }
            )

        return (
            trip_schedules.group_by(["bldg_id", "vehicle_id", "travel_date"])
            .agg(pl.col("trip_miles_driven").sum().alias("daily_miles"))
            .group_by(["bldg_id", "vehicle_id"])
            .agg(pl.col("daily_miles").max().alias("max_daily_miles"))
        )
