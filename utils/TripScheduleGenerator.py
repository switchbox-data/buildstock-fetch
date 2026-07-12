import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from utils.NHTSProfileSampler import TripProfile, VehicleProfile

MIN_TRIP_AWAY_HOURS = 1  # hourly model: each trip is away for at least one clock hour
MAX_DEPARTURE_HOUR = 23  # latest hour a same-day trip can start (hour 0..23)
MAX_ARRIVAL_HOUR = 24  # latest allowed arrival hour (exclusive end of away interval)


@dataclass
class TripScheduleGenerator:
    """Generate daily trip schedules from sampled NHTS vehicle profiles."""

    start_date: datetime
    end_date: datetime
    random_state: int = 42
    max_workers: int | None = None

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

    @staticmethod
    def _normalize_day_trip_times(
        departures: np.ndarray,
        arrival_hours: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Enforce arrival_hour > departure_hour per trip and non-overlapping away intervals within a day.

        Away hours are ``range(departure_hour, arrival_hour)`` (departure inclusive, arrival exclusive).
        Independent random offsets can produce arrival_hour <= departure_hour or overlapping intervals.
        Trips are repacked in departure order on a single calendar day (no overnight trips).
        Each trip's away interval ends before the next trip's departure hour begins.
        There is no minimum dwell-at-home between trips — the next departure may equal the prior arrival.
        Returns arrays in the original input order. Trips that no longer fit before hour 23
        are dropped (keep_mask=False).

        Args:
            departures: Array of departure hours
            arrival_hours: Array of first-at-home hours (exclusive end of away interval)

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

        earliest_next_dep = 0
        for i in range(n):
            if earliest_next_dep > MAX_DEPARTURE_HOUR:
                keep_sorted[i:] = False
                break

            dep = min(max(int(dep_sorted[i]), earliest_next_dep), MAX_DEPARTURE_HOUR)
            arrival = min(max(int(arrival_sorted[i]), dep + MIN_TRIP_AWAY_HOURS), MAX_ARRIVAL_HOUR)
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

    @staticmethod
    def _preprocess_trip_profile(
        trip_profile: TripProfile,
    ) -> tuple[int, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        available = len(trip_profile.trip_ids)
        if available == 0:
            return 0, None, None, None, None

        weights = np.array(trip_profile.trip_weights)
        weights = weights / weights.sum()
        return (
            available,
            weights,
            np.array(trip_profile.departure_hours),
            np.array(trip_profile.arrival_hours),
            np.array(trip_profile.miles),
        )

    def _generate_vehicle_daily_trip_schedules(  # noqa: C901
        self, profile: VehicleProfile, rng: np.random.RandomState | None = None
    ) -> pl.DataFrame:
        """Generate trip schedules for a vehicle for all days in the date range as a DataFrame.

        For each day, the departure and arrival times and miles traveled are generated from random
        perturbations of the base values in the vehicle profile. Trips are measured in hourly increments.

        Args:
            profile (VehicleProfile): Vehicle profile to generate schedules for
            rng (np.random.RandomState): Random number generator

        Returns:
            pl.DataFrame: DataFrame of hourly trip schedules for the vehicle
        """
        if rng is None:
            rng = np.random.random.__self__  # Use global numpy random if no rng provided

        # Pre-compute constants (move outside loops)
        time_offsets = np.array([-2, -1, 0, 1, 2])
        time_probabilities = np.array([0.05, 0.10, 0.70, 0.10, 0.05])

        # Pre-process weekday data
        weekday_available, weekday_weights, weekday_departures, weekday_arrivals, weekday_miles = (
            self._preprocess_trip_profile(profile.weekday)
        )

        # Pre-process weekend data
        weekend_available, weekend_weights, weekend_departures, weekend_arrivals, weekend_miles = (
            self._preprocess_trip_profile(profile.weekend)
        )

        # Calculate number of days and pre-compute date information
        days = (self.end_date - self.start_date).days + 1

        # Pre-allocate lists for batch operations
        bldg_ids = []
        vehicle_ids = []
        dates = []
        departure_hours = []
        arrival_hours = []
        miles_driven = []

        for day_offset in range(days):
            current_date = self.start_date + timedelta(days=day_offset)
            is_weekday = current_date.weekday() < 5  # Monday-Friday are weekdays

            # Select pre-processed data based on day type
            if is_weekday:
                available_trips = weekday_available
                weights = weekday_weights
                departures = weekday_departures
                arrivals = weekday_arrivals
                base_miles_array = weekday_miles
            else:
                available_trips = weekend_available
                weights = weekend_weights
                departures = weekend_departures
                arrivals = weekend_arrivals
                base_miles_array = weekend_miles

            if available_trips == 0:
                continue  # Skip days where we have no trips

            # Replicate all available trips for this day
            num_trips = available_trips

            # Sample trip indices using the pre-normalized weights
            trip_indices = rng.choice(
                available_trips,
                size=num_trips,
                replace=False,
                p=weights,
            )

            # Vectorized operations for all trips in this day
            if departures is None:
                raise ValueError("departures")
            if arrivals is None:
                raise ValueError("arrivals")
            if base_miles_array is None:
                raise ValueError("base_miles_array")
            selected_departures = departures[trip_indices]
            selected_arrivals = arrivals[trip_indices]
            selected_base_miles = base_miles_array[trip_indices]

            # Vectorized variance calculations
            miles_variance = rng.normal(selected_base_miles, selected_base_miles * 0.1)

            # Vectorized time offset sampling
            departure_offsets = rng.choice(time_offsets, size=num_trips, p=time_probabilities)
            arrival_offsets = rng.choice(time_offsets, size=num_trips, p=time_probabilities)

            # Apply offsets with bounds checking
            departures_with_variance = np.clip(selected_departures + departure_offsets, 0, MAX_DEPARTURE_HOUR)
            arrival_hours_with_variance = np.clip(selected_arrivals + arrival_offsets, 1, MAX_ARRIVAL_HOUR)

            # Offsets are drawn independently per trip, so fix invalid intervals and
            # overlaps before writing the day's schedule (may drop trips that spill past 23:00).
            departures_with_variance, arrival_hours_with_variance, keep_trip = self._normalize_day_trip_times(
                departures_with_variance,
                arrival_hours_with_variance,
            )
            miles_variance = miles_variance[keep_trip]
            num_trips = int(keep_trip.sum())
            if num_trips == 0:
                continue  # e.g. weekday template missing or every trip dropped after packing

            # Batch append to lists
            bldg_ids.extend([profile.bldg_id] * num_trips)
            vehicle_ids.extend([profile.vehicle_id] * num_trips)
            dates.extend([current_date] * num_trips)
            departure_hours.extend(departures_with_variance[keep_trip].tolist())
            arrival_hours.extend(arrival_hours_with_variance[keep_trip].tolist())
            miles_driven.extend(miles_variance.tolist())

        # Create DataFrame directly from lists (vectorized approach)
        schedule_data = {
            "bldg_id": bldg_ids,
            "vehicle_id": vehicle_ids,
            "date": dates,
            "departure_hour": departure_hours,
            "arrival_hour": arrival_hours,
            "miles_driven": miles_driven,
        }

        return pl.DataFrame(schedule_data)

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
            # Create unique seed for this profile
            profile_seed = self.random_state + index
            rng = np.random.RandomState(profile_seed)
            return self._generate_vehicle_daily_trip_schedules(profile, rng=rng)

        profiles_list = list(profile_params.values())
        profiles_with_index = [(profile, i) for i, profile in enumerate(profiles_list)]

        all_schedules: list[pl.DataFrame] = []

        total_profiles = len(profiles_list)
        logging.info(f"Processing {total_profiles} vehicle profiles...")

        # Use parallel processing if we have multiple profiles and max_workers != 1
        if len(profiles_list) > 1 and self.max_workers != 1:
            logging.info(f"Using parallel processing with {self.max_workers or 'all available'} workers")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(process_profile_with_seed, profile_and_index)
                    for profile_and_index in profiles_with_index
                ]

                for completed, future in enumerate(as_completed(futures), 1):
                    schedules = future.result()
                    all_schedules.append(schedules)  # Fixed: append DataFrame, don't extend

                    # Log progress every 5%
                    self._log_progress(completed, total_profiles, "Progress")
        else:
            # Fall back to sequential processing
            logging.info("Using sequential processing")
            for i, profile_and_index in enumerate(profiles_with_index, 1):
                schedules = process_profile_with_seed(profile_and_index)
                all_schedules.append(schedules)  # Fixed: append DataFrame, don't extend

                # Log progress every 5%
                self._log_progress(i, total_profiles, "Progress")

        # Combine all DataFrames (already in DataFrame format)
        total_schedules = sum(len(schedule_df) for schedule_df in all_schedules)
        logging.info(f"Generated {total_schedules} trip schedules from {total_profiles} vehicle profiles")

        if all_schedules:
            return pl.concat(all_schedules)
        else:
            return pl.DataFrame()
