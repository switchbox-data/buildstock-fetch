import logging
from dataclasses import dataclass, field
from typing import Any, Literal, overload

import numpy as np
import polars as pl


class NHTSDataError(Exception):
    """Raised when NHTS data is not loaded."""

    pass


@dataclass
class TripProfile:
    """One NHTS daily trip template (all trips on a weekday or weekend day)."""

    departure_hours: list[int] = field(default_factory=list)  # List of departure hours for each trip
    arrival_hours: list[int] = field(default_factory=list)  # First hour at home (exclusive end of away interval)
    miles: list[float] = field(default_factory=list)  # List of miles for each trip
    trip_weights: list[float] = field(default_factory=list)  # List of trip weights for each trip
    trip_ids: list[int] = field(default_factory=list)  # List of trip IDs

    @property
    def has_trips(self) -> bool:
        return len(self.miles) > 0


@dataclass
class VehicleProfile:
    """Represents a vehicle's driving profile parameters."""

    bldg_id: str
    vehicle_id: int
    weekday: TripProfile = field(default_factory=TripProfile)
    weekend: TripProfile = field(default_factory=TripProfile)


def nhts_departure_hour(start_time: int) -> int:
    """Clock hour when a trip starts (vehicle is away during this hour).

    NHTS ``STRTTIME`` is HHMM; e.g. 830 → hour 8.

    Args:
        start_time (int): NHTS ``STRTTIME`` in HHMM format

    Returns:
        int: Clock hour when a trip starts
    """
    return int(start_time) // 100


def nhts_arrival_hour(end_time: int) -> int:
    """First clock hour at home after a trip ends (exclusive end of the away interval).

    NHTS ``ENDTIME`` is HHMM. If the trip ends exactly on the hour (e.g. 1700),
    the vehicle is home starting that hour. If it ends mid-hour (e.g. 1715),
    it is still away for that full hour and home starting the next hour.

    Away hours are ``range(departure_hour, arrival_hour)``.

    Args:
        end_time (int): NHTS ``ENDTIME`` in HHMM format

    Returns:
        int: First clock hour at home after a trip ends
    """
    end_time = int(end_time)
    hour, minute = divmod(end_time, 100)
    return hour if minute == 0 else hour + 1


def summarize_nhts_match_catalog(catalog: pl.DataFrame) -> pl.DataFrame:
    """Summarize NHTS household/vehicle matching gaps from a ``sample(..., return_catalog=True)`` catalog.

    A number of NHTS vehicle profiles are missing a weekday or weekend trip profile.
    This function summarizes the number of missing profiles and vehicle slots with any gap.

    Args:
        catalog (pl.DataFrame): Catalog of vehicle profiles and NHTS matches

    Returns:
        pl.DataFrame: summary metrics
    """
    if catalog.is_empty():
        return pl.DataFrame({"metric": [], "count": [], "share_of_vehicle_slots": []})

    vehicle_slots = catalog.height
    n_missing_nhts = catalog.filter(~pl.col("nhts_vehicle_matched")).height
    matched = catalog.filter(pl.col("nhts_vehicle_matched"))
    n_missing_weekday = matched.filter(~pl.col("has_weekday_trips")).height
    n_missing_weekend = matched.filter(~pl.col("has_weekend_trips")).height
    n_missing_both = matched.filter(~pl.col("has_weekday_trips") & ~pl.col("has_weekend_trips")).height

    vehicle_slots_with_any_gap = catalog.filter(
        ~pl.col("nhts_vehicle_matched") | ~pl.col("has_weekday_trips") | ~pl.col("has_weekend_trips")
    ).height

    def share(count: int) -> float:
        return count / vehicle_slots if vehicle_slots else 0.0

    return pl.DataFrame({
        "metric": [
            "vehicle_slots",
            "missing_nhts_vehicle_match",
            "missing_weekday_trip_profile",
            "missing_weekend_trip_profile",
            "missing_both_trip_profiles",
            "vehicle_slots_with_any_gap",
        ],
        "count": [
            vehicle_slots,
            n_missing_nhts,
            n_missing_weekday,
            n_missing_weekend,
            n_missing_both,
            vehicle_slots_with_any_gap,
        ],
        "share_of_vehicle_slots": [
            1.0,
            share(n_missing_nhts),
            share(n_missing_weekday),
            share(n_missing_weekend),
            share(n_missing_both),
            share(vehicle_slots_with_any_gap),
        ],
    })


@dataclass
class NHTSProfileSampler:
    """Sample weekday/weekend driving profiles from NHTS trip data."""

    nhts_df: pl.DataFrame | None = None
    max_vehicles: int = 2
    match_on_vehicles: bool = False
    random_state: int = 42
    # Keep NHTS vehicles whose peak survey-day miles fall in [low, high] percentiles.
    # Defaults 0–100 keep the full pool; tighten via YAML (e.g. 10–90) to drop outliers.
    nhts_daily_miles_percentile_low: float = 0.0
    nhts_daily_miles_percentile_high: float = 100.0
    _cache: dict[str, dict] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        np.random.seed(self.random_state)
        if self.nhts_df is not None:
            self.nhts_df = self.filter_by_daily_miles_percentile(
                self.nhts_df,
                low=self.nhts_daily_miles_percentile_low,
                high=self.nhts_daily_miles_percentile_high,
            )

    @staticmethod
    def filter_by_daily_miles_percentile(
        nhts_df: pl.DataFrame,
        *,
        low: float = 0.0,
        high: float = 100.0,
    ) -> pl.DataFrame:
        """Keep NHTS vehicles whose max survey-day miles fall in ``[low, high]`` percentiles.

        For each ``hh_vehicle_id``, daily miles are summed within each day type
        (``weekday``), then the max across day types is used as the vehicle's
        representative daily miles. Vehicles outside the percentile band are dropped
        from the match pool.

        Args:
            nhts_df: NHTS trip rows with ``hh_vehicle_id``, ``weekday``, ``miles_driven``
            low: Inclusive lower percentile (0–100). ``0`` disables the lower cut.
            high: Inclusive upper percentile (0–100). ``100`` disables the upper cut.

        Returns:
            Filtered NHTS DataFrame (unchanged when ``low <= 0`` and ``high >= 100``).
        """
        if low <= 0.0 and high >= 100.0:
            return nhts_df
        if not (0.0 <= low <= high <= 100.0):
            raise ValueError(
                f"Invalid percentile band [{low}, {high}]; require 0 <= low <= high <= 100"
            )
        required = {"hh_vehicle_id", "weekday", "miles_driven"}
        missing = required - set(nhts_df.columns)
        if missing:
            raise ValueError(f"nhts_df missing columns for percentile filter: {sorted(missing)}")
        if nhts_df.is_empty():
            return nhts_df

        vehicle_miles = (
            nhts_df.group_by(["hh_vehicle_id", "weekday"])
            .agg(pl.col("miles_driven").sum().alias("daily_miles"))
            .group_by("hh_vehicle_id")
            .agg(pl.col("daily_miles").max().alias("max_daily_miles"))
        )
        lo = NHTSProfileSampler._scalar_quantile(vehicle_miles["max_daily_miles"], low / 100.0)
        hi = NHTSProfileSampler._scalar_quantile(vehicle_miles["max_daily_miles"], high / 100.0)
        keep_ids = vehicle_miles.filter(
            (pl.col("max_daily_miles") >= lo) & (pl.col("max_daily_miles") <= hi)
        )["hh_vehicle_id"]
        filtered = nhts_df.filter(pl.col("hh_vehicle_id").is_in(keep_ids.to_list()))
        logging.info(
            "NHTS daily-miles percentile filter [%.1f, %.1f]: "
            "kept %s/%s vehicles (miles band [%.1f, %.1f]), %s/%s trip rows",
            low,
            high,
            keep_ids.len(),
            vehicle_miles.height,
            lo,
            hi,
            filtered.height,
            nhts_df.height,
        )
        if filtered.is_empty():
            raise ValueError(
                f"NHTS percentile filter [{low}, {high}] removed all vehicles "
                f"(miles band would be [{lo:.1f}, {hi:.1f}])"
            )
        return filtered

    @staticmethod
    def _scalar_quantile(series: pl.Series, q: float) -> float:
        """Return a single quantile as float (Series.quantile is typed as scalar|list|None)."""
        value = series.quantile(q)
        if value is None:
            raise ValueError(f"quantile({q}) returned None for empty or all-null series")
        if isinstance(value, list):
            raise TypeError(f"quantile({q}) returned a list; expected a scalar")
        return float(value)

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

    def _prepare_cache(self, *, weekday: bool) -> dict:
        """
        Pre-group NHTS data by household demographics for fast matching lookups.

        Weekday and weekend caches only include vehicles with at least one logged trip
        on that day type (weekday=2 Mon–Fri, weekend=1 Sat/Sun).

        Args:
            weekday (bool): If True, prepare weekday cache; otherwise prepare weekend cache

        Returns:
            Dictionary mapping key tuples to sorted hh_vehicle_id lists for deterministic random sampling

        Raises:
            NHTSDataError: If the NHTS data is not loaded
        """
        if self._cache is None:
            self._cache = {}

        cache_key = "weekday" if weekday else "weekend"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.nhts_df is None:
            raise NHTSDataError()

        logging.info("Preparing NHTS %s matching cache...", cache_key)

        # Cap household vehicle count to max_vehicles (default 2).
        # NHTS stores HHVEHCNT on every trip row; some households have 3+ vehicles.
        # Our pipeline also caps PUMS training data and ResStock predictions at max_vehicles,
        # so a building with 2 vehicles should match NHTS households bucketed as 2, not 3+.
        nhts_df = self.nhts_df.with_columns(
            pl.when(pl.col("vehicles") > self.max_vehicles)
            .then(self.max_vehicles)
            .otherwise(pl.col("vehicles"))
            .alias("vehicles")
        )

        # Keep only trips taken on the requested day type (weekday=2 Mon–Fri, weekend=1 Sat/Sun).
        # This ensures match only returns hh_vehicle_ids that actually drove
        # on that day type, so sampled profiles have real trip data to copy.
        day_flag = 2 if weekday else 1
        day_df = nhts_df.filter(pl.col("weekday") == day_flag)

        self._cache[cache_key] = self._build_matching_cache(day_df)

        logging.info("NHTS %s cache prepared successfully", cache_key)
        return self._cache[cache_key]

    def _build_matching_cache(self, df: pl.DataFrame) -> dict:
        """
        Build a lookup dict for match().

        Keys are tuples of (income_bucket, ...) with increasing specificity; values are
        sorted hh_vehicle_id lists for deterministic random sampling.

        Args:
            df (pl.DataFrame): DataFrame with NHTS data including income_bucket, occupants, vehicles, and hh_vehicle_id

        Returns:
            Dictionary mapping key tuples to sorted hh_vehicle_id lists for deterministic random sampling
        """
        cache = {}

        # Tier 1 — exact match on household demographics: (income_bucket, occupants, vehicles).
        # Each NHTS vehicle id appears once per tier it qualifies for.
        exact_groups = df.group_by(["income_bucket", "occupants", "vehicles"]).agg(pl.col("hh_vehicle_id").unique())

        for row in exact_groups.iter_rows(named=True):
            key = (row["income_bucket"], row["occupants"], row["vehicles"])
            # Sort to ensure consistent ordering for deterministic results
            cache[key] = sorted(row["hh_vehicle_id"])

        # Tier 2 — relax vehicle count: (income_bucket, occupants).
        # Used when no exact match has enough vehicles; keys are 2-tuples so they
        # never collide with tier-1's 3-tuple keys.
        income_occ_groups = df.group_by(["income_bucket", "occupants"]).agg(pl.col("hh_vehicle_id").unique())

        for row in income_occ_groups.iter_rows(named=True):
            key = (row["income_bucket"], row["occupants"])
            if key not in cache:  # Don't overwrite exact matches
                # Sort to ensure consistent ordering for deterministic results
                cache[key] = sorted(row["hh_vehicle_id"])

        # Tier 3 — relax occupants too: (income_bucket,).
        # Last structured fallback before match picks the closest income bucket.
        income_groups = df.group_by(["income_bucket"]).agg(pl.col("hh_vehicle_id").unique())

        for row in income_groups.iter_rows(named=True):
            key = (row["income_bucket"],)
            # Sort to ensure consistent ordering for deterministic results
            cache[key] = sorted(row["hh_vehicle_id"])

        return cache

    def match(
        self,
        target_income: int,
        target_occupants: int,
        target_vehicles: int,
        num_samples: int,
        *,
        weekday: bool = True,
        match_on_vehicles: bool | None = None,
    ) -> tuple[str, list[str]]:
        """
        Find the best matching vehicles in NHTS data based on prioritized criteria.
        Will return num_samples different vehicles, falling back to less exact matches if needed.

        Uses pre-built cache to eliminate expensive filtering operations.
        Only considers NHTS vehicles that have at least one trip on the requested day type.

        Args:
            target_income: Target income bucket to match
            target_occupants: Target number of occupants to match
            target_vehicles: Target number of vehicles to match (used only when
                ``match_on_vehicles`` is True)
            num_samples: Number of different vehicles to sample
            weekday: If True, match against weekday trip profiles; otherwise weekend
            match_on_vehicles: If True, try an exact (income, occupants, vehicles) match
                before looser tiers. Defaults to ``self.match_on_vehicles`` (False for
                the max-1-EV model).

        Returns:
            Tuple of (match_type, list of matched_vehicle_ids)
        """
        if match_on_vehicles is None:
            match_on_vehicles = self.match_on_vehicles

        cache = self._prepare_cache(weekday=weekday)

        # Tier 1 — exact match on (income, occupants, vehicles). Skipped when the caller
        # models at most one EV per household and household fleet size is not a match axis.
        if match_on_vehicles:
            exact_key = (target_income, target_occupants, target_vehicles)
            if exact_key in cache and len(cache[exact_key]) >= num_samples:
                return "exact", np.random.choice(cache[exact_key], size=num_samples, replace=False).tolist()

        # Tier 2 — match income and occupants: (income, occupants)
        income_occ_key = (target_income, target_occupants)
        if income_occ_key in cache and len(cache[income_occ_key]) >= num_samples:
            return "income_occupants", np.random.choice(cache[income_occ_key], size=num_samples, replace=False).tolist()

        # Try matching only income: (income,)
        income_key = (target_income,)
        if income_key in cache and len(cache[income_key]) >= num_samples:
            return "income_only", np.random.choice(cache[income_key], size=num_samples, replace=False).tolist()

        # If still no match, find closest income bucket
        available_incomes = [key[0] for key in cache if len(key) == 1]  # Get all single-income keys
        if available_incomes:
            closest_income = min(available_incomes, key=lambda x: abs(x - target_income))
            closest_key = (closest_income,)
            if closest_key in cache and len(cache[closest_key]) >= num_samples:
                return "closest_income", np.random.choice(cache[closest_key], size=num_samples, replace=False).tolist()
            else:
                # If not enough samples, take all available
                return "closest_income", cache[closest_key]

        # Fallback: return empty list if no matches found
        return "no_match", []

    @staticmethod
    def _trip_profile_from_nhts(
        nhts_df: pl.DataFrame,
        matched_vehicle_id: str,
        *,
        weekday: bool,
    ) -> TripProfile:
        """
        Extract trip profile from NHTS data for a given vehicle id and day type.

        Args:
            nhts_df (pl.DataFrame): NHTS trip data DataFrame
            matched_vehicle_id (str): Vehicle id to match
            weekday (bool): If True, extract weekday trip profile; otherwise extract weekend trip profile

        Returns:
            TripProfile for the matched vehicle and day type
        """
        # filter NHTS data for the given vehicle id and day type
        day_flag = 2 if weekday else 1
        trip_data = nhts_df.filter(
            (pl.col("hh_vehicle_id") == matched_vehicle_id) & (pl.col("weekday") == day_flag)
        )
        departures = [nhts_departure_hour(t) for t in trip_data["start_time"]]
        arrivals = [nhts_arrival_hour(t) for t in trip_data["end_time"]]
        miles = trip_data["miles_driven"].to_list()
        weights = trip_data["trip_weight"].to_list()
        trip_ids = list(range(1, len(departures) + 1))
        return TripProfile(
            departure_hours=departures,
            arrival_hours=arrivals,
            miles=miles,
            trip_weights=weights,
            trip_ids=trip_ids,
        )

    @overload
    def sample(
        self,
        bldg_veh_df: pl.DataFrame,
        nhts_df: pl.DataFrame | None = None,
        *,
        return_catalog: Literal[False] = False,
        match_on_vehicles: bool | None = None,
    ) -> dict[tuple[str, int], VehicleProfile]: ...

    @overload
    def sample(
        self,
        bldg_veh_df: pl.DataFrame,
        nhts_df: pl.DataFrame | None = None,
        *,
        return_catalog: Literal[True],
        match_on_vehicles: bool | None = None,
    ) -> tuple[dict[tuple[str, int], VehicleProfile], pl.DataFrame]: ...

    def sample(
        self,
        bldg_veh_df: pl.DataFrame,
        nhts_df: pl.DataFrame | None = None,
        *,
        return_catalog: bool = False,
        match_on_vehicles: bool | None = None,
    ) -> dict[tuple[str, int], VehicleProfile] | tuple[dict[tuple[str, int], VehicleProfile], pl.DataFrame]:
        """
        For each household and vehicle, select separate weekday and weekend trip profiles from NHTS.

        Uses pre-built vehicle trips cache to eliminate expensive per-vehicle filtering.

        Args:
            bldg_veh_df: DataFrame with household and vehicle info, including a ``vehicles`` column.
            nhts_df: NHTS trip data DataFrame with trip weights; defaults to ``self.nhts_df``
            return_catalog: If True, also return a per-vehicle-slot match diagnostics DataFrame.

        Returns:
            Dict mapping (bldg_id, vehicle_id) to sampled trip profile parameters.
            When return_catalog=True, returns (profiles, catalog) where catalog has one row
            per predicted vehicle slot with NHTS match and weekday/weekend trip availability.
        """
        df = bldg_veh_df
        if df is None:
            raise ValueError("No building/vehicle DataFrame provided for profile sampling.")

        if nhts_df is None:
            nhts_df = self.nhts_df
        if nhts_df is None:
            raise NHTSDataError()

        profiles: dict[tuple[str, int], VehicleProfile] = {}
        catalog_records: list[dict[str, Any]] = []
        total_buildings = len(df)
        processed_buildings = 0

        logging.info(f"Sampling vehicle profiles for {total_buildings} buildings...")

        for row in df.iter_rows(named=True):
            bldg_id = row["bldg_id"]
            num_vehicles = row["vehicles"]
            processed_buildings += 1

            if num_vehicles is None:
                raise ValueError(
                    f"bldg_id={bldg_id} has null vehicles; ensure EVAdoptionSampler.sample() completed "
                    "without lookup join misses before sampling profiles."
                )
            if num_vehicles == 0:
                # Log progress for zero-vehicle buildings too
                self._log_progress(processed_buildings, total_buildings, "Building progress")
                continue

            # Match weekday and weekend profiles independently from NHTS vehicles
            # that have at least one logged trip on each day type.
            weekday_match_type, weekday_vehicle_ids = self.match(
                target_income=row["income_bucket"],
                target_occupants=row["occupants"],
                target_vehicles=num_vehicles,
                num_samples=num_vehicles,
                weekday=True,
                match_on_vehicles=match_on_vehicles,
            )
            weekend_match_type, weekend_vehicle_ids = self.match(
                target_income=row["income_bucket"],
                target_occupants=row["occupants"],
                target_vehicles=num_vehicles,
                num_samples=num_vehicles,
                weekday=False,
                match_on_vehicles=match_on_vehicles,
            )

            if len(weekday_vehicle_ids) < num_vehicles:
                raise ValueError(
                    f"bldg_id={bldg_id}: weekday NHTS matching found {len(weekday_vehicle_ids)} "
                    f"profile(s) but {num_vehicles} required (weekday_match_type={weekday_match_type})."
                )
            if len(weekend_vehicle_ids) < num_vehicles:
                raise ValueError(
                    f"bldg_id={bldg_id}: weekend NHTS matching found {len(weekend_vehicle_ids)} "
                    f"profile(s) but {num_vehicles} required (weekend_match_type={weekend_match_type})."
                )

            # Create profiles for each vehicle
            for vehicle_id in range(1, num_vehicles + 1):
                weekday_vehicle_id = weekday_vehicle_ids[vehicle_id - 1]
                weekend_vehicle_id = weekend_vehicle_ids[vehicle_id - 1]

                weekday_profile = self._trip_profile_from_nhts(nhts_df, weekday_vehicle_id, weekday=True)
                weekend_profile = self._trip_profile_from_nhts(nhts_df, weekend_vehicle_id, weekday=False)

                # Create VehicleProfile for this specific vehicle
                profiles[(bldg_id, vehicle_id)] = VehicleProfile(
                    bldg_id=bldg_id,
                    vehicle_id=vehicle_id,
                    weekday=weekday_profile,
                    weekend=weekend_profile,
                )

                if return_catalog:
                    has_weekday_trips = weekday_profile.has_trips
                    has_weekend_trips = weekend_profile.has_trips
                    catalog_records.append({
                        "bldg_id": bldg_id,
                        "vehicle_slot": vehicle_id,
                        "predicted_vehicles": num_vehicles,
                        "weekday_match_type": weekday_match_type,
                        "weekend_match_type": weekend_match_type,
                        "nhts_vehicle_matched": True,
                        "nhts_weekday_matched": has_weekday_trips,
                        "nhts_weekend_matched": has_weekend_trips,
                        "matched_hh_vehicle_id": weekday_vehicle_id,
                        "matched_weekday_hh_vehicle_id": weekday_vehicle_id,
                        "matched_weekend_hh_vehicle_id": weekend_vehicle_id,
                        "has_weekday_trips": has_weekday_trips,
                        "has_weekend_trips": has_weekend_trips,
                        "weekday_trip_count": len(weekday_profile.miles),
                        "weekend_trip_count": len(weekend_profile.miles),
                    })

            # Log progress for buildings with vehicles
            self._log_progress(processed_buildings, total_buildings, "Building progress")

        logging.info(f"Generated {len(profiles)} vehicle profiles from {total_buildings} buildings")
        if return_catalog:
            return profiles, pl.DataFrame(catalog_records)
        return profiles
