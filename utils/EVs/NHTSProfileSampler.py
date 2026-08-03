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
    """Summarize NHTS household/vehicle matching from a ``sample(..., return_catalog=True)`` catalog.

    Empty weekday/weekend templates are first-class matches (owned but not driven
    that survey day), so they are reported separately from true match failures.

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
    # Intentional empty day templates (idle inventory vehicles), not match failures.
    n_empty_weekday = matched.filter(~pl.col("has_weekday_trips")).height
    n_empty_weekend = matched.filter(~pl.col("has_weekend_trips")).height
    n_empty_both = matched.filter(~pl.col("has_weekday_trips") & ~pl.col("has_weekend_trips")).height

    def share(count: int) -> float:
        return count / vehicle_slots if vehicle_slots else 0.0

    return pl.DataFrame({
        "metric": [
            "vehicle_slots",
            "missing_nhts_vehicle_match",
            "empty_weekday_trip_profile",
            "empty_weekend_trip_profile",
            "empty_both_trip_profiles",
        ],
        "count": [
            vehicle_slots,
            n_missing_nhts,
            n_empty_weekday,
            n_empty_weekend,
            n_empty_both,
        ],
        "share_of_vehicle_slots": [
            1.0,
            share(n_missing_nhts),
            share(n_empty_weekday),
            share(n_empty_weekend),
            share(n_empty_both),
        ],
    })


@dataclass
class NHTSProfileSampler:
    """Sample weekday/weekend driving profiles from NHTS trip data."""

    nhts_df: pl.DataFrame | None = None
    max_vehicles: int = 2
    match_on_vehicles: bool = False
    random_state: int = 42
    _cache: dict[str, dict] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        np.random.seed(self.random_state)

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

        Weekday / weekend caches include every inventory vehicle whose household
        ``TRAVDAY`` falls on that day type — driven trip days and empty (idle)
        vehicle-days alike.

        Cache contents:
        - ``houses``: demographic tier → sorted unique ``house_id`` lists
        - ``vehicles_by_house``: ``house_id`` → sorted ``hh_vehicle_id`` lists

        Args:
            weekday (bool): If True, prepare weekday cache; otherwise weekend

        Returns:
            Dict with ``houses`` and ``vehicles_by_house`` maps

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

        # Cap household vehicle count to max_vehicles (default 2) so PUMS / NHTS
        # vehicle-count tiers align when match_on_vehicles is enabled.
        nhts_df = self.nhts_df.with_columns(
            pl.when(pl.col("vehicles") > self.max_vehicles)
            .then(self.max_vehicles)
            .otherwise(pl.col("vehicles"))
            .alias("vehicles")
        )

        # Fixtures without house_id: treat each vehicle as its own household.
        if "house_id" not in nhts_df.columns:
            nhts_df = nhts_df.with_columns(pl.col("hh_vehicle_id").alias("house_id"))

        # Keep rows for the requested day type (weekday=2 Mon–Fri, weekend=1 Sat/Sun).
        day_flag = 2 if weekday else 1
        day_df = nhts_df.filter(pl.col("weekday") == day_flag)

        self._cache[cache_key] = self._build_matching_cache(day_df)

        logging.info(
            "NHTS %s cache prepared: %s households, %s vehicles",
            cache_key,
            len(self._cache[cache_key]["vehicles_by_house"]),
            sum(len(v) for v in self._cache[cache_key]["vehicles_by_house"].values()),
        )
        return self._cache[cache_key]

    def _build_matching_cache(self, df: pl.DataFrame) -> dict:
        """
        Build household-level lookup structures for ``match()``.

        Matching draws **households** uniformly within a demographic tier, then
        ``match()`` samples vehicles inside those households. That way multi-car
        NHTS homes are not overweighted relative to one-car homes.

        Each house-index key is a tagged tuple ``(tier_name, *column_values)``.

        Args:
            df: NHTS rows for one day type with demographics, ``house_id``,
                ``hh_vehicle_id``

        Returns:
            Dict with ``houses`` (tier → house_id list) and ``vehicles_by_house``
        """
        if "urban" not in df.columns:
            raise ValueError(
                "nhts_df missing 'urban' column (NHTS URBRUR). "
                "Reload with load_nhts_data() or add urban=1/2 to the frame."
            )

        # One row per inventory vehicle (empty markers and trip legs collapse here).
        vehicle_meta = df.select(
            "house_id",
            "hh_vehicle_id",
            "urban",
            "income_bucket",
            "occupants",
            "vehicles",
        ).unique()

        vehicles_by_house: dict[str, list[str]] = {}
        for row in (
            vehicle_meta.group_by("house_id")
            .agg(pl.col("hh_vehicle_id").unique())
            .iter_rows(named=True)
        ):
            vehicles_by_house[str(row["house_id"])] = sorted(str(v) for v in row["hh_vehicle_id"])

        # Household demographics: one row per house (all vehicles share HH attrs).
        house_meta = vehicle_meta.select(
            "house_id", "urban", "income_bucket", "occupants", "vehicles"
        ).unique(subset=["house_id"])

        houses: dict[tuple, list[str]] = {}

        def _store(tier: str, group_cols: list[str]) -> None:
            """Index unique house ids under ``(tier, *group_col values)``."""
            groups = house_meta.group_by(group_cols).agg(pl.col("house_id").unique())
            for row in groups.iter_rows(named=True):
                key = (tier, *(row[c] for c in group_cols))
                houses[key] = sorted(str(h) for h in row["house_id"])

        # Expects coarse bins from load_nhts_data / load_metadata.
        _store("urban_income_occupants_vehicles", ["urban", "income_bucket", "occupants", "vehicles"])
        _store("urban_income_occupants", ["urban", "income_bucket", "occupants"])
        _store("urban_occupants", ["urban", "occupants"])
        _store("urban_income", ["urban", "income_bucket"])
        _store("income_occupants_vehicles", ["income_bucket", "occupants", "vehicles"])
        _store("income_occupants", ["income_bucket", "occupants"])
        _store("income", ["income_bucket"])

        return {"houses": houses, "vehicles_by_house": vehicles_by_house}

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
        Match demographically similar NHTS **households**, then sample vehicles.

        Cascade prefers urban/rural, then occupants over income when dropping
        dimensions. Vehicle-count tiers are used only when ``match_on_vehicles``
        is True (``pums_vehicles`` mode). Empty (idle) vehicle-days are eligible.

        Within a tier, households are shuffled and their vehicles drawn without
        replacement until ``num_samples`` ``hh_vehicle_id``s are collected — so a
        two-car NHTS home can supply two EV slots, but one-car homes are not
        under-weighted relative to three-car homes in the household draw.

        Args:
            target_income: Target coarse income bin (1–3)
            target_urban: Target urbanicity (1=urban, 2=rural)
            target_occupants: Target household size bin (1 / 2 / 3+)
            target_vehicles: Target vehicle count (used only when matching on vehicles)
            num_samples: Number of distinct ``hh_vehicle_id``s to return
            weekday: Match weekday (True) or weekend (False) day-type pool
            match_on_vehicles: Override for ``self.match_on_vehicles``

        Returns:
            Tuple of (match_type, list of matched ``hh_vehicle_id``s)
        """
        if match_on_vehicles is None:
            match_on_vehicles = self.match_on_vehicles

        cache = self._prepare_cache(weekday=weekday)
        houses_index: dict[tuple, list[str]] = cache["houses"]
        vehicles_by_house: dict[str, list[str]] = cache["vehicles_by_house"]

        # Fallback cascade: (match_type, cache tier tag, key column values).
        attempts: list[tuple[str, str, tuple[int, ...]]] = []

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
        attempts.append((
            "urban_occupants",
            "urban_occupants",
            (target_urban, target_occupants),
        ))
        attempts.append(("urban_income", "urban_income", (target_urban, target_income)))
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
        attempts.append(("income", "income", (target_income,)))

        for match_type, tier, parts in attempts:
            key = (tier, *parts)
            house_ids = houses_index.get(key)
            if not house_ids:
                continue

            # How many distinct vehicles sit under these matching households?
            n_available = sum(len(vehicles_by_house[h]) for h in house_ids)
            if n_available < num_samples:
                continue

            # Household-first draw: shuffle houses, then vehicles within each house.
            picked: list[str] = []
            for house_id in np.random.permutation(house_ids):
                for veh_id in np.random.permutation(vehicles_by_house[house_id]):
                    picked.append(str(veh_id))
                    if len(picked) == num_samples:
                        return match_type, picked

        day_type = "weekday" if weekday else "weekend"
        available_incomes = sorted(key[1] for key in houses_index if key[0] == "income")
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

        Empty-day markers (``is_empty_day=True`` or no real trip legs) yield an
        empty ``TripProfile`` — the vehicle stays home all day with 0 miles.

        When purpose columns are present, legs are chained into home-based tours
        at minute resolution then snapped to clock hours. Otherwise each leg is
        its own tour (legacy fixtures).
        """
        day_flag = 2 if weekday else 1
        trip_data = nhts_df.filter(
            (pl.col("hh_vehicle_id") == matched_vehicle_id) & (pl.col("weekday") == day_flag)
        )
        # Drop empty-day markers; only real driver legs build a non-empty profile.
        if "is_empty_day" in trip_data.columns:
            trip_data = trip_data.filter(~pl.col("is_empty_day").fill_null(False))
        trip_data = trip_data.filter(pl.col("start_time").is_not_null())
        if trip_data.is_empty():
            return TripProfile()

        if "seq_trip_id" in trip_data.columns:
            trip_data = trip_data.sort(["start_time", "seq_trip_id"])
        else:
            trip_data = trip_data.sort("start_time")

        start_times = [int(t) for t in trip_data["start_time"].to_list()]
        end_times = [int(t) for t in trip_data["end_time"].to_list()]
        trip_miles_driven = [float(m) for m in trip_data["miles_driven"].to_list()]
        weights = [float(w) for w in trip_data["trip_weight"].to_list()]

        # Purpose columns enable tour chaining; require non-null codes on every leg.
        has_purposes = (
            "why_from" in trip_data.columns
            and "why_to" in trip_data.columns
            and trip_data["why_from"].null_count() == 0
            and trip_data["why_to"].null_count() == 0
        )
        if has_purposes:
            why_from = [int(v) for v in trip_data["why_from"].to_list()]
            why_to = [int(v) for v in trip_data["why_to"].to_list()]
            return build_tours_from_legs(
                start_times=start_times,
                end_times=end_times,
                trip_miles_driven=trip_miles_driven,
                trip_weights=weights,
                why_from=why_from,
                why_to=why_to,
            )

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
        For each ResStock EV slot, match NHTS weekday and weekend vehicle-day templates.

        Matching is household-first (see ``match``); each draw may return an empty
        ``TripProfile`` when the chosen inventory vehicle was idle that survey day.
        Weekday and weekend draws are independent because each NHTS household has
        a single ``TRAVDAY`` (one day type only).

        Args:
            bldg_veh_df: Building frame with ``vehicles`` (EV slot count) + demographics
            nhts_df: NHTS pool; defaults to ``self.nhts_df``
            return_catalog: If True, also return per-slot match diagnostics
            match_on_vehicles: Override for vehicle-count tiers

        Returns:
            Dict ``(bldg_id, vehicle_id) → VehicleProfile``, optionally with catalog
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
                self._log_progress(processed_buildings, total_buildings, "Building progress")
                continue

            if "urban" not in row or row["urban"] is None:
                raise ValueError(
                    f"bldg_id={bldg_id} missing urban (1=urban, 2=rural); "
                    "derive from ResStock metro via load_metadata() / assign_urban_from_metro()."
                )

            # Independent weekday / weekend household→vehicle draws (NHTS is one TRAVDAY).
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

            for vehicle_id in range(1, num_vehicles + 1):
                weekday_vehicle_id = weekday_vehicle_ids[vehicle_id - 1]
                weekend_vehicle_id = weekend_vehicle_ids[vehicle_id - 1]

                # Empty TripProfile is valid: idle inventory vehicle that day.
                weekday_profile = self._trip_profile_from_nhts(nhts_df, weekday_vehicle_id, weekday=True)
                weekend_profile = self._trip_profile_from_nhts(nhts_df, weekend_vehicle_id, weekday=False)

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
                        # Matched a vehicle-day template (possibly empty), not "has trips".
                        "nhts_weekday_matched": True,
                        "nhts_weekend_matched": True,
                        "matched_hh_vehicle_id": weekday_vehicle_id,
                        "matched_weekday_hh_vehicle_id": weekday_vehicle_id,
                        "matched_weekend_hh_vehicle_id": weekend_vehicle_id,
                        "has_weekday_trips": has_weekday_trips,
                        "has_weekend_trips": has_weekend_trips,
                        "weekday_trip_count": len(weekday_profile.trip_miles_driven),
                        "weekend_trip_count": len(weekend_profile.trip_miles_driven),
                    })

            self._log_progress(processed_buildings, total_buildings, "Building progress")

        logging.info(f"Generated {len(profiles)} vehicle profiles from {total_buildings} buildings")
        if return_catalog:
            return profiles, pl.DataFrame(catalog_records)
        return profiles
