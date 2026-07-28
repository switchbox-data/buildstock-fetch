from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, overload

import numpy as np
import polars as pl

from utils.EVs.nhts_tours import (
    TripProfile,
    build_tours_from_legs,
    nhts_arrival_hour,
    nhts_departure_hour,
    trips_as_singleton_tours,
)


class NHTSDataError(Exception):
    """Raised when NHTS data is not loaded."""

    pass


@dataclass
class VehicleProfile:
    """Matched ResStock vehicle slot with weekday/weekend ``TripProfile`` templates."""

    bldg_id: str
    vehicle_id: int
    weekday: TripProfile = field(default_factory=TripProfile)
    weekend: TripProfile = field(default_factory=TripProfile)


# Re-export day-template / hour helpers so existing imports from this module keep working.
__all__ = [
    "NHTSDataError",
    "NHTSProfileSampler",
    "TripProfile",
    "VehicleProfile",
    "nhts_arrival_hour",
    "nhts_departure_hour",
    "summarize_nhts_match_catalog",
]


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

        Each key is a tagged tuple ``(tier_name, *column_values)`` mapping to a sorted
        list of ``hh_vehicle_id``s. The tier name is required because several tiers share
        the same arity of ints; without a tag, e.g. ``(1, 6, 2)`` could mean either
        ``(urban, income, occupants)`` or ``(income, occupants, vehicles)``.

        Args:
            df (pl.DataFrame): NHTS rows with ``urban``, ``income_bucket``, ``occupants``,
                ``vehicles``, and ``hh_vehicle_id``

        Returns:
            Dictionary mapping key tuples to sorted hh_vehicle_id lists
        """
        if "urban" not in df.columns:
            raise ValueError(
                "nhts_df missing 'urban' column (NHTS URBRUR). "
                "Reload with load_nhts_data() or add urban=1/2 to the frame."
            )

        cache: dict[tuple, list[str]] = {}

        def _store(tier: str, group_cols: list[str]) -> None:
            """Index unique vehicle ids under ``(tier, *group_col values)``."""
            groups = df.group_by(group_cols).agg(pl.col("hh_vehicle_id").unique())
            for row in groups.iter_rows(named=True):
                key = (tier, *(row[c] for c in group_cols))
                cache[key] = sorted(row["hh_vehicle_id"])

        # --- Urban-aware and unconditional indexes (match() chooses cascade order) ---
        # Expects coarse bins from load_nhts_data / load_metadata: income 1–3, occupants 1/2/3+.
        # urban=1/2 from NHTS URBRUR; ResStock maps metro status onto the same codes.
        _store("urban_income_occupants_vehicles", ["urban", "income_bucket", "occupants", "vehicles"])
        _store("urban_income_occupants", ["urban", "income_bucket", "occupants"])
        _store("urban_occupants", ["urban", "occupants"])
        _store("urban_income", ["urban", "income_bucket"])
        _store("income_occupants_vehicles", ["income_bucket", "occupants", "vehicles"])
        _store("income_occupants", ["income_bucket", "occupants"])
        _store("income", ["income_bucket"])

        return cache

    def match(
        self,
        target_income: int,
        target_urban: int,
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

        Expects ``income_bucket`` (1–3) and ``occupants`` (1 / 2 / 3+) already coarsened by
        ``load_nhts_data`` / ``load_metadata``. Match order prefers urban/rural, then
        occupants over income when dropping dimensions. Match-type names are the
        dimensions held (``exact`` = all four including urban):

        1. ``exact`` — (urban, income, occupants, vehicles) when ``match_on_vehicles``
        2. ``urban_income_occupants``
        3. ``urban_occupants`` — drop income; keep urban + HH size
        4. ``urban_income`` — drop occupants; keep urban + income
        5. ``income_occupants_vehicles`` — drop urban; when ``match_on_vehicles``
        6. ``income_occupants``
        7. ``income``

        Args:
            target_income: Target coarse income bin (1–3)
            target_urban: Target urbanicity (1=urban, 2=rural), from ResStock metro mapping
            target_occupants: Target household size bin (1 / 2 / 3+)
            target_vehicles: Target number of vehicles to match (used only when
                ``match_on_vehicles`` is True)
            num_samples: Number of different vehicles to sample
            weekday: If True, match against weekday trip profiles; otherwise weekend
            match_on_vehicles: If True, try exact vehicle-count tiers before looser ones.
                Defaults to ``self.match_on_vehicles`` (False for the max-1-EV model).

        Returns:
            Tuple of (match_type, list of matched_vehicle_ids)

        Raises:
            ValueError: If no tier in the cascade contains enough matching profiles
        """
        if match_on_vehicles is None:
            match_on_vehicles = self.match_on_vehicles

        cache = self._prepare_cache(weekday=weekday)

        # Build the fallback cascade. Each attempt is:
        #   (match_type returned to caller, cache tier tag, column values for the key)
        # Vehicle-count tiers are omitted when match_on_vehicles is False (max-1-EV mode).
        attempts: list[tuple[str, str, tuple[int, ...]]] = []

        # 1–2: keep urban/rural + household structure; optionally require vehicle count.
        if match_on_vehicles:
            attempts.append((
                "exact",
                "urban_income_occupants_vehicles",
                (target_urban, target_income, target_occupants, target_vehicles),
            ))
        attempts.append((
            "urban_income_occupants",
            "urban_income_occupants",
            (target_urban, target_income, target_occupants),
        ))

        # 3: drop income but keep urban + occupants (HH size > income for daily miles).
        attempts.append((
            "urban_occupants",
            "urban_occupants",
            (target_urban, target_occupants),
        ))

        # 4: drop occupants but keep urban + income.
        attempts.append(("urban_income", "urban_income", (target_urban, target_income)))

        # 5–6: drop urbanicity; optionally keep vehicle count with HH structure.
        if match_on_vehicles:
            attempts.append((
                "income_occupants_vehicles",
                "income_occupants_vehicles",
                (target_income, target_occupants, target_vehicles),
            ))
        attempts.append((
            "income_occupants",
            "income_occupants",
            (target_income, target_occupants),
        ))

        # 7: income bucket only.
        attempts.append(("income", "income", (target_income,)))

        for match_type, tier, parts in attempts:
            key = (tier, *parts)
            ids = cache.get(key)
            if ids is not None and len(ids) >= num_samples:
                return match_type, np.random.choice(ids, size=num_samples, replace=False).tolist()

        day_type = "weekday" if weekday else "weekend"
        available_incomes = sorted(key[1] for key in cache if key[0] == "income")
        raise ValueError(
            f"No NHTS {day_type} match with at least {num_samples} profile(s) for "
            f"income={target_income}, urban={target_urban}, occupants={target_occupants}, "
            f"vehicles={target_vehicles}; available income bins={available_incomes}."
        )

    @staticmethod
    def _trip_profile_from_nhts(
        nhts_df: pl.DataFrame,
        matched_vehicle_id: str,
        *,
        weekday: bool,
    ) -> TripProfile:
        """
        Extract trip + tour profile from NHTS for a vehicle id and day type.

        When ``why_from`` / ``why_to`` are present, legs are chained into home-based
        tours at **minute** resolution first (``build_tours_from_legs``), then snapped
        to clock hours. Otherwise each leg is treated as its own tour (legacy /
        unit-test fixtures; already hourly).

        Args:
            nhts_df (pl.DataFrame): NHTS trip data DataFrame
            matched_vehicle_id (str): Vehicle id to match
            weekday (bool): If True, extract weekday trip profile; otherwise extract weekend trip profile

        Returns:
            TripProfile for the matched vehicle and day type
        """
        # One NHTS travel day per household: all driver legs for this vehicle + day type.
        day_flag = 2 if weekday else 1
        trip_data = nhts_df.filter(
            (pl.col("hh_vehicle_id") == matched_vehicle_id) & (pl.col("weekday") == day_flag)
        )
        if trip_data.is_empty():
            return TripProfile()

        # Chronological order: prefer SEQ_TRIPID when loaded, else start_time.
        if "seq_trip_id" in trip_data.columns:
            trip_data = trip_data.sort(["start_time", "seq_trip_id"])
        else:
            trip_data = trip_data.sort("start_time")

        start_times = [int(t) for t in trip_data["start_time"].to_list()]
        end_times = [int(t) for t in trip_data["end_time"].to_list()]
        trip_miles_driven = [float(m) for m in trip_data["miles_driven"].to_list()]
        weights = [float(w) for w in trip_data["trip_weight"].to_list()]

        # Purpose columns enable true tour chaining; fixtures without them stay 1:1.
        has_purposes = "why_from" in trip_data.columns and "why_to" in trip_data.columns
        if has_purposes:
            why_from = [int(v) for v in trip_data["why_from"].to_list()]
            why_to = [int(v) for v in trip_data["why_to"].to_list()]
            # Returns a TripProfile: minute-level tours, then hourly snap inside the builder.
            return build_tours_from_legs(
                start_times=start_times,
                end_times=end_times,
                trip_miles_driven=trip_miles_driven,
                trip_weights=weights,
                why_from=why_from,
                why_to=why_to,
            )

        # Fixtures already use clock hours as start_time/end_time stand-ins.
        return trips_as_singleton_tours(
            trip_departure_hours=[nhts_departure_hour(t) for t in start_times],
            trip_arrival_hours=[nhts_arrival_hour(t) for t in end_times],
            trip_miles_driven=trip_miles_driven,
            trip_weights=weights,
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

            if "urban" not in row or row["urban"] is None:
                raise ValueError(
                    f"bldg_id={bldg_id} missing urban (1=urban, 2=rural); "
                    "derive from ResStock metro via load_metadata() / assign_urban_from_metro()."
                )

            # Match weekday and weekend profiles independently from NHTS vehicles
            # that have at least one logged trip on each day type.
            weekday_match_type, weekday_vehicle_ids = self.match(
                target_income=row["income_bucket"],
                target_urban=int(row["urban"]),
                target_occupants=row["occupants"],
                target_vehicles=num_vehicles,
                num_samples=num_vehicles,
                weekday=True,
                match_on_vehicles=match_on_vehicles,
            )
            weekend_match_type, weekend_vehicle_ids = self.match(
                target_income=row["income_bucket"],
                target_urban=int(row["urban"]),
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
                        "weekday_trip_count": len(weekday_profile.trip_miles_driven),
                        "weekend_trip_count": len(weekend_profile.trip_miles_driven),
                    })

            # Log progress for buildings with vehicles
            self._log_progress(processed_buildings, total_buildings, "Building progress")

        logging.info(f"Generated {len(profiles)} vehicle profiles from {total_buildings} buildings")
        if return_catalog:
            return profiles, pl.DataFrame(catalog_records)
        return profiles
