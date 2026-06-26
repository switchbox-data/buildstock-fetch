import argparse
import logging
import os
import sys
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final, Literal, cast, overload

import cvxpy as cp
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.preprocessing import LabelEncoder, StandardScaler  # type: ignore[import-untyped]

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import ev_utils

BASEPATH: Final[Path] = Path(__file__).resolve().parent  # just one level up

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MetadataPathError(Exception):
    """Raised when no metadata path is provided."""

    pass


class MetadataDataFrameError(Exception):
    """Raised when no metadata DataFrame is available."""

    pass


class VehicleOwnershipModelError(Exception):
    """Raised when vehicle ownership model is not fitted."""

    pass


class NHTSDataError(Exception):
    """Raised when NHTS data is not loaded."""

    pass


class InsufficientVehiclesError(Exception):
    """Raised when there are not enough matching vehicles in NHTS data."""

    def __init__(self, bldg_id: int, vehicle_id: int, count: int):
        self.message = f"Building {bldg_id}, vehicle {vehicle_id}: {count} matching vehicles"
        super().__init__(self.message)


class NoDateRangeError(Exception):
    """Raised when no start_date or end_date is provided."""

    pass


class InvalidDateFormatError(ValueError):
    """Raised when date string is not in YYYY-MM-DD format."""

    def __init__(self, date_str: str):
        super().__init__(f"Invalid date format: {date_str}. Use YYYY-MM-DD format.")


@dataclass
class EVDemandConfig:
    state: str
    release: str
    metadata_path: str | None = None
    pums_path: str | None = None
    nhts_path: str = f"{BASEPATH}/ev_data/inputs/NHTS_v2_1_trip_surveys.csv"
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.metadata_path is None:
            self.metadata_path = f"{BASEPATH}/ev_data/inputs/{self.release}/metadata/{self.state}/metadata.parquet"
        if self.pums_path is None:
            self.pums_path = f"{BASEPATH}/ev_data/inputs/{self.state}_2021_pums_PUMA_HINCP_VEH_NP.csv"
        if self.output_dir is None:
            self.output_dir = Path(f"{BASEPATH}/ev_data/outputs/{self.state}_{self.release}")


@dataclass
class VehicleProfile:
    """Represents a vehicle's driving profile parameters."""

    bldg_id: str
    vehicle_id: int
    weekday_departure_hour: list[int] = field(default_factory=list)  # List of departure hours for each weekday trip
    weekday_arrival_hour: list[int] = field(default_factory=list)  # First hour at home (exclusive end of away interval)
    weekday_miles: list[float] = field(default_factory=list)  # List of miles for each weekday trip
    weekday_trip_weights: list[float] = field(default_factory=list)  # List of trip weights for each weekday trip
    weekend_departure_hour: list[int] = field(default_factory=list)  # List of departure hours for each weekend trip
    weekend_arrival_hour: list[int] = field(default_factory=list)  # First hour at home (exclusive end of away interval)
    weekend_miles: list[float] = field(default_factory=list)  # List of miles for each weekend trip
    weekend_trip_weights: list[float] = field(default_factory=list)  # List of trip weights for each weekend trip
    weekday_trip_ids: list[int] = field(default_factory=list)  # List of trip IDs for weekdays
    weekend_trip_ids: list[int] = field(default_factory=list)  # List of trip IDs for weekends


@dataclass
class TripSchedule:
    """Represents a daily trip schedule for a vehicle."""

    bldg_id: int  # Changed from int to str to match VehicleProfile
    vehicle_id: int
    date: datetime
    departure_hour: int  # first away hour (inclusive)
    arrival_hour: int  # first home hour (exclusive end of away interval; same as nhts_home_hour)
    miles_driven: float


MIN_TRIP_AWAY_HOURS = 1  # hourly model: each trip is away for at least one clock hour
MAX_DEPARTURE_HOUR = 23  # latest hour a same-day trip can start (hour 0..23)
MAX_HOME_HOUR = 24  # exclusive end of away interval; 24 = home after hour 23 ends
HOURS_PER_YEAR = 8760  # standard hourly load-curve length (365 days x 24 hours)
DEFAULT_BATTERY_CAPACITY_KWH = 120.0  # uniform fleet battery assumption for SOC modeling
DEFAULT_KWH_PER_MILE = 0.30  # simple-model assumption
DEFAULT_LEVEL2_CHARGER_KW = 7.2  # typical 32 A @ 240 V residential Level 2 charger
# How home charging is scheduled before SOC is derived from discharge + charge.
ChargingStrategy = Literal["immediate", "cost_minimizing"]


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

    Away hours are ``range(departure_hour, home_hour)``.

    Args:
        end_time (int): NHTS ``ENDTIME`` in HHMM format

    Returns:
        int: First clock hour at home after a trip ends
    """
    end_time = int(end_time)
    hour, minute = divmod(end_time, 100)
    return hour if minute == 0 else hour + 1


def summarize_nhts_match_catalog(catalog: pl.DataFrame) -> pl.DataFrame:
    """Summarize NHTS household/vehicle matching gaps from sample_vehicle_profiles catalog.

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


class EVDemandCalculator:
    """
    Calculator for EV demand based on ResStock metadata, PUMS data, and NHTS trip data.

    This class implements the workflow described in the methodology:
    1. Load ResStock metadata
    2. Fit vehicle ownership model using PUMS data
    3. Predict number of vehicles per household
    4. Sample vehicle driving profiles from NHTS
    5. Generate annual trip schedules
    """

    def __init__(
        self,
        metadata_df: pl.DataFrame,
        nhts_df: pl.DataFrame,
        pums_df: pl.DataFrame,
        start_date: datetime,
        end_date: datetime,
        max_vehicles: int = 2,
        random_state: int = 42,
        max_workers: int | None = None,
    ):
        """
        Initialize the EV demand calculator.

        Args:
            metadata_df: ResStock metadata DataFrame
            nhts_df: NHTS trip data DataFrame
            pums_df: PUMS data DataFrame
            start_date: Start date for trip generation
            end_date: End date for trip generation
            max_vehicles: Maximum number of vehicles per household
            random_state: Random seed for reproducible results
            max_workers: Maximum number of worker threads for parallel execution (None = use all cores)
        """
        # Set random seed for reproducible results
        np.random.seed(random_state)

        self.max_vehicles = max_vehicles

        self.metadata_df = metadata_df
        self.nhts_df = nhts_df
        self.pums_df = pums_df

        self.start_date = start_date
        self.end_date = end_date
        self.num_days = (self.end_date - self.start_date).days + 1
        self.num_hours = self.num_days * 24

        self.vehicle_ownership_model: Any | None = None
        self.random_state = random_state
        self.max_workers = max_workers

        # Features used for vehicle assignment
        self.veh_assign_features = ["occupants", "income", "metro"]

        # Cache for NHTS data to avoid repeated filtering
        self._nhts_cache: dict | None = None

        # Yuksel and Michalek (2015) polynomial coefficients for energy consumption
        # c(T) = sum(a_n * T^n) for n=0 to 5, units: kWh/mi/°F^n
        # self.efficiency_coefficients = np.array([
        #     0.3950,  # a_0 (constant term)
        #     -0.0022,  # a_1 (linear term)
        #     9.1978e-5,  # a_2 (quadratic term)
        #     -3.9249e-6,  # a_3 (cubic term)
        #     5.2918e-8,  # a_4 (quartic term)
        #     -2.0659e-10,  # a_5 (quintic term)
        # ])

    def _log_progress(self, current: int, total: int, description: str, progress_interval: int = 10000) -> None:
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

    def fit_vehicle_ownership_model(self, pums_df: pl.DataFrame) -> Any:
        """
        Fit a multinomial logistic regression model to predict number of vehicles per household using PUMS data.
        Results limited to 0, 1, or 2 vehicles.

        Args:
            pums_df: DataFrame with PUMS household data, including 'occupants', 'income', 'metro', 'vehicles'

        Returns:
            Trained model object
        """
        # Preprocess data: replace vehicles > 2 with 2, drop nulls, encode categorical
        pums_df = pums_df.with_columns(
            pl.when(pl.col("vehicles") > self.max_vehicles)
            .then(self.max_vehicles)
            .otherwise(pl.col("vehicles"))
            .alias("vehicles")
        )

        # Drop rows with missing values in required features
        feature_columns = self.veh_assign_features

        # Drop rows with missing values in required features and target variable
        initial_count = len(pums_df)
        pums_df = pums_df.drop_nulls(subset=[*feature_columns, "vehicles"])
        dropped_count = initial_count - len(pums_df)

        if dropped_count > 0:
            logging.warning(f"Dropped {dropped_count} records with missing values in vehicle ownership model features")

        # Prepare features and encode categorical
        X = pums_df.select(feature_columns)
        y = pums_df.select("vehicles")

        # Encode metro (only categorical feature)
        self.label_encoders = {}
        le = LabelEncoder()
        metro_encoded = le.fit_transform(X.get_column("metro").to_numpy())
        X_encoded = X.with_columns(pl.Series(metro_encoded).alias("metro"))
        self.label_encoders["metro"] = le

        # Encode target and scale features
        self.target_encoder = LabelEncoder()
        y_encoded = self.target_encoder.fit_transform(y.get_column("vehicles").to_numpy())

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_encoded.to_numpy())

        # Fit model with sample weights
        self.vehicle_ownership_model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=self.random_state)
        self.vehicle_ownership_model.fit(X_scaled, y_encoded, sample_weight=pums_df.get_column("hh_weight").to_numpy())

        return self.vehicle_ownership_model

    def _prepare_nhts_cache(self) -> dict:
        """
        Pre-group NHTS data by key attributes to avoid repeated filtering.
        This eliminates the need for collect() calls in find_best_matches().
        """
        if self._nhts_cache is not None:
            return self._nhts_cache  # Already cached

        logging.info("Preparing NHTS data cache to optimize matching...")

        # Cap the number of vehicles at the max number of vehicles per household
        nhts_df = self.nhts_df.with_columns(
            pl.when(pl.col("vehicles") > self.max_vehicles)
            .then(self.max_vehicles)
            .otherwise(pl.col("vehicles"))
            .alias("vehicles")
        )

        self._nhts_cache = self._build_matching_cache(nhts_df)

        logging.info("NHTS cache prepared successfully")

        return self._nhts_cache

    def _build_matching_cache(self, df: pl.DataFrame) -> dict:
        """Build cache dictionary for fast matching lookups."""
        cache = {}

        # Group by different combinations for iterative matching
        # 1. Exact match: (income_bucket, occupants, vehicles)
        exact_groups = df.group_by(["income_bucket", "occupants", "vehicles"]).agg(pl.col("hh_vehicle_id").unique())

        for row in exact_groups.iter_rows(named=True):
            key = (row["income_bucket"], row["occupants"], row["vehicles"])
            # Sort to ensure consistent ordering for deterministic results
            cache[key] = sorted(row["hh_vehicle_id"])

        # 2. Income + occupants match: (income_bucket, occupants)
        income_occ_groups = df.group_by(["income_bucket", "occupants"]).agg(pl.col("hh_vehicle_id").unique())

        for row in income_occ_groups.iter_rows(named=True):
            key = (row["income_bucket"], row["occupants"])
            if key not in cache:  # Don't overwrite exact matches
                # Sort to ensure consistent ordering for deterministic results
                cache[key] = sorted(row["hh_vehicle_id"])

        # 3. Income only match: (income_bucket,)
        income_groups = df.group_by(["income_bucket"]).agg(pl.col("hh_vehicle_id").unique())

        for row in income_groups.iter_rows(named=True):
            key = (row["income_bucket"],)
            # Sort to ensure consistent ordering for deterministic results
            cache[key] = sorted(row["hh_vehicle_id"])

        return cache

    def predict_num_vehicles(self, metadata_df: pl.DataFrame | None = None) -> pl.DataFrame:
        """
        Predict number of vehicles for each household in the metadata using the fitted model.
        If the model hasn't been fitted yet, it will be fitted automatically using the PUMS data.

        Args:
            metadata_df: DataFrame with ResStock metadata.

        Returns:
            DataFrame with an added 'vehicles' column
        """
        df = self.metadata_df if metadata_df is None else metadata_df
        if df is None:
            raise MetadataDataFrameError()

        # Automatically fit the model if it hasn't been fitted yet
        if self.vehicle_ownership_model is None:
            logging.info("Vehicle ownership model not fitted yet. Fitting model...")
            self.fit_vehicle_ownership_model(self.pums_df)

        # Step 1: Prepare features from metadata
        feature_columns = self.veh_assign_features
        X = df.select(["bldg_id", *feature_columns])

        # Validate no missing values in required features
        if X.select(feature_columns).null_count().sum_horizontal().item() > 0:
            # Find building IDs with missing values
            missing_data = X.filter(
                pl.col("occupants").is_null() | pl.col("income").is_null() | pl.col("metro").is_null()
            )
            missing_bldg_ids = missing_data.get_column("bldg_id").to_list()
            raise ValueError("Missing vehicle ownership model input data for building IDs: " + str(missing_bldg_ids))

        # Separate features and building IDs
        features = X.select(feature_columns)

        # Step 2: Encode categorical variables
        features_encoded = features.clone()
        metro_encoded = self.label_encoders["metro"].transform(features.get_column("metro").to_numpy())
        features_encoded = features_encoded.with_columns(pl.Series(metro_encoded).alias("metro"))

        # Step 3: Scale features
        features_scaled = self.scaler.transform(features_encoded.to_numpy())

        if self.vehicle_ownership_model is None:
            raise ValueError("vehicle_ownership_model")

        # Step 4: Make predictions
        predictions_encoded = self.vehicle_ownership_model.predict(features_scaled)

        # Step 5: Decode predictions and add to DataFrame
        predictions_decoded = self.target_encoder.inverse_transform(predictions_encoded)

        # Add predictions to original DataFrame
        bldg_veh_df = df.with_columns(pl.Series(predictions_decoded).alias("vehicles"))

        return bldg_veh_df

    def find_best_matches(
        self, target_income: int, target_occupants: int, target_vehicles: int, num_samples: int, weekday: bool = True
    ) -> tuple[str, list[str]]:
        """
        Find the best matching vehicles in NHTS data based on prioritized criteria.
        Will return num_samples different vehicles, falling back to less exact matches if needed.

        Uses pre-built cache to eliminate expensive filtering operations.
        Matches based on household characteristics only (weekday parameter is ignored).

        Args:
            target_income: Target income bucket to match
            target_occupants: Target number of occupants to match
            target_vehicles: Target number of vehicles to match
            num_samples: Number of different vehicles to sample
            weekday: Ignored - kept for compatibility

        Returns:
            Tuple of (match_type, list of matched_vehicle_ids)
        """
        # Ensure cache is prepared
        cache = self._prepare_nhts_cache()

        # Try exact match first: (income, occupants, vehicles)
        exact_key = (target_income, target_occupants, target_vehicles)
        if exact_key in cache and len(cache[exact_key]) >= num_samples:
            return "exact", np.random.choice(cache[exact_key], size=num_samples, replace=False).tolist()

        # Try matching only income and occupants: (income, occupants)
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

    @overload
    def sample_vehicle_profiles(
        self,
        bldg_veh_df: pl.DataFrame,
        nhts_df: pl.DataFrame,
        *,
        return_catalog: Literal[False] = False,
    ) -> dict[tuple[str, int], VehicleProfile]: ...

    @overload
    def sample_vehicle_profiles(
        self,
        bldg_veh_df: pl.DataFrame,
        nhts_df: pl.DataFrame,
        *,
        return_catalog: Literal[True],
    ) -> tuple[dict[tuple[str, int], VehicleProfile], pl.DataFrame]: ...

    def sample_vehicle_profiles(
        self,
        bldg_veh_df: pl.DataFrame,
        nhts_df: pl.DataFrame,
        *,
        return_catalog: bool = False,
    ) -> dict[tuple[str, int], VehicleProfile] | tuple[dict[tuple[str, int], VehicleProfile], pl.DataFrame]:
        """
        For each household and vehicle, select a weekday and weekend trip profiles from NHTS.

        Uses pre-built vehicle trips cache to eliminate expensive per-vehicle filtering.

        Args:
            bldg_veh_df: DataFrame with household and vehicle info.
            nhts_df: NHTS trip data DataFrame with trip weights
            return_catalog: If True, also return a per-vehicle-slot match diagnostics DataFrame.

        Returns:
            Dict mapping (bldg_id, vehicle_id) to sampled trip profile parameters.
            When return_catalog=True, returns (profiles, catalog) where catalog has one row
            per predicted vehicle slot with NHTS match and weekday/weekend trip availability.
        """
        df = bldg_veh_df
        if df is None:
            raise MetadataDataFrameError()

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

            if num_vehicles == 0:
                # Log progress for zero-vehicle buildings too
                self._log_progress(processed_buildings, total_buildings, "Building progress")
                continue

            # Find best vehicle matches for all cars at this building
            match_type, matched_vehicle_ids = self.find_best_matches(
                target_income=row["income_bucket"],
                target_occupants=row["occupants"],
                target_vehicles=num_vehicles,
                num_samples=num_vehicles,
            )

            # Create profiles for each vehicle
            for vehicle_id in range(1, num_vehicles + 1):
                if vehicle_id > len(matched_vehicle_ids):
                    if return_catalog:
                        catalog_records.append({
                            "bldg_id": bldg_id,
                            "vehicle_slot": vehicle_id,
                            "predicted_vehicles": num_vehicles,
                            "match_type": match_type,
                            "nhts_vehicle_matched": False,
                            "matched_hh_vehicle_id": None,
                            "has_weekday_trips": False,
                            "has_weekend_trips": False,
                            "weekday_trip_count": 0,
                            "weekend_trip_count": 0,
                        })
                    continue

                matched_vehicle_id = matched_vehicle_ids[vehicle_id - 1]

                # Get weekday trips for this vehicle (simple filtering)
                weekday_data = self.nhts_df.filter(
                    (pl.col("hh_vehicle_id") == matched_vehicle_id) & (pl.col("weekday") == 2)
                )

                # Get weekend trips for this vehicle (simple filtering)
                weekend_data = self.nhts_df.filter(
                    (pl.col("hh_vehicle_id") == matched_vehicle_id) & (pl.col("weekday") == 1)
                )

                # Process weekday trips (extract from filtered data)
                weekday_departures = [nhts_departure_hour(t) for t in weekday_data["start_time"]]
                weekday_arrivals = [nhts_arrival_hour(t) for t in weekday_data["end_time"]]
                weekday_miles = weekday_data["miles_driven"].to_list()
                weekday_weights = weekday_data["trip_weight"].to_list()
                weekday_trip_ids = list(range(1, len(weekday_departures) + 1))

                # Process weekend trips (extract from filtered data)
                weekend_departures = [nhts_departure_hour(t) for t in weekend_data["start_time"]]
                weekend_arrivals = [nhts_arrival_hour(t) for t in weekend_data["end_time"]]
                weekend_miles = weekend_data["miles_driven"].to_list()
                weekend_weights = weekend_data["trip_weight"].to_list()
                weekend_trip_ids = list(range(1, len(weekend_departures) + 1))

                # Create VehicleProfile for this specific vehicle
                profiles[(bldg_id, vehicle_id)] = VehicleProfile(
                    bldg_id=bldg_id,
                    vehicle_id=vehicle_id,  # Now vehicle_id is already 1-based
                    weekday_departure_hour=weekday_departures,
                    weekday_arrival_hour=weekday_arrivals,
                    weekday_miles=weekday_miles,
                    weekday_trip_weights=weekday_weights,
                    weekend_departure_hour=weekend_departures,
                    weekend_arrival_hour=weekend_arrivals,
                    weekend_miles=weekend_miles,
                    weekend_trip_weights=weekend_weights,
                    weekday_trip_ids=weekday_trip_ids,
                    weekend_trip_ids=weekend_trip_ids,
                )

                if return_catalog:
                    catalog_records.append({
                        "bldg_id": bldg_id,
                        "vehicle_slot": vehicle_id,
                        "predicted_vehicles": num_vehicles,
                        "match_type": match_type,
                        "nhts_vehicle_matched": True,
                        "matched_hh_vehicle_id": matched_vehicle_id,
                        "has_weekday_trips": len(weekday_miles) > 0,
                        "has_weekend_trips": len(weekend_miles) > 0,
                        "weekday_trip_count": len(weekday_miles),
                        "weekend_trip_count": len(weekend_miles),
                    })

            # Log progress for buildings with vehicles
            self._log_progress(processed_buildings, total_buildings, "Building progress")

        logging.info(f"Generated {len(profiles)} vehicle profiles from {total_buildings} buildings")
        if return_catalog:
            return profiles, pl.DataFrame(catalog_records)
        return profiles

    @staticmethod
    def _normalize_day_trip_times(
        departures: np.ndarray,
        home_hours: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Enforce home_hour > departure_hour per trip and non-overlapping away intervals within a day.

        Away hours are ``range(departure_hour, home_hour)`` (departure inclusive, home exclusive).
        Independent random offsets can produce home_hour <= departure_hour or overlapping intervals.
        Trips are repacked in departure order on a single calendar day (no overnight trips).
        Each trip's away interval ends before the next trip's departure hour begins.
        Returns arrays in the original input order. Trips that no longer fit before hour 23
        are dropped (keep_mask=False).

        Args:
            departures: Array of departure hours
            home_hours: Array of first-at-home hours (exclusive end of away interval)

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of normalized departure hours,
                home hours, and keep mask
        """
        if len(departures) != len(home_hours):
            raise ValueError(
                f"departures and home_hours must have the same length, got {len(departures)} and {len(home_hours)}"
            )

        n = len(departures)
        keep = np.ones(n, dtype=bool)
        if n == 0:
            return departures.astype(int), home_hours.astype(int), keep

        order = np.argsort(departures, kind="stable")
        dep_sorted = departures[order].astype(int)
        home_sorted = home_hours[order].astype(int)
        keep_sorted = np.ones(n, dtype=bool)

        earliest_next_dep = 0
        for i in range(n):
            if earliest_next_dep > MAX_DEPARTURE_HOUR:
                keep_sorted[i:] = False
                break

            dep = min(max(int(dep_sorted[i]), earliest_next_dep), MAX_DEPARTURE_HOUR)
            home = min(max(int(home_sorted[i]), dep + MIN_TRIP_AWAY_HOURS), MAX_HOME_HOUR)
            if home <= dep:
                keep_sorted[i] = False
                continue

            dep_sorted[i] = dep
            home_sorted[i] = home
            earliest_next_dep = home

        dep_out = np.empty(n, dtype=int)
        home_out = np.empty(n, dtype=int)
        dep_out[order] = dep_sorted
        home_out[order] = home_sorted
        keep[order] = keep_sorted
        return dep_out, home_out, keep

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
        weekday_available = len(profile.weekday_trip_ids)
        weekday_weights = None
        weekday_departures = None
        weekday_arrivals = None
        weekday_miles = None

        if weekday_available > 0:
            weekday_weights = np.array(profile.weekday_trip_weights)
            weekday_weights = weekday_weights / weekday_weights.sum()  # Pre-normalize
            weekday_departures = np.array(profile.weekday_departure_hour)
            weekday_arrivals = np.array(profile.weekday_arrival_hour)
            weekday_miles = np.array(profile.weekday_miles)

        # Pre-process weekend data
        weekend_available = len(profile.weekend_trip_ids)
        weekend_weights = None
        weekend_departures = None
        weekend_arrivals = None
        weekend_miles = None

        if weekend_available > 0:
            weekend_weights = np.array(profile.weekend_trip_weights)
            weekend_weights = weekend_weights / weekend_weights.sum()  # Pre-normalize
            weekend_departures = np.array(profile.weekend_departure_hour)
            weekend_arrivals = np.array(profile.weekend_arrival_hour)
            weekend_miles = np.array(profile.weekend_miles)

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
            home_hours_with_variance = np.clip(selected_arrivals + arrival_offsets, 1, MAX_HOME_HOUR)

            # Offsets are drawn independently per trip, so fix invalid intervals and
            # overlaps before writing the day's schedule (may drop trips that spill past 23:00).
            departures_with_variance, home_hours_with_variance, keep_trip = self._normalize_day_trip_times(
                departures_with_variance,
                home_hours_with_variance,
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
            arrival_hours.extend(home_hours_with_variance[keep_trip].tolist())
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

    def generate_daily_trip_schedules(
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
        def process_profile_with_seed(profile_and_index):
            profile, index = profile_and_index
            # Create unique seed for this profile
            profile_seed = self.random_state + index
            rng = np.random.RandomState(profile_seed)
            return self._generate_vehicle_daily_trip_schedules(profile, rng=rng)

        profiles_list = list(profile_params.values())
        profiles_with_index = [(profile, i) for i, profile in enumerate(profiles_list)]

        all_schedules = []

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

    def match_and_generate_trip_schedules(self) -> pl.DataFrame:
        """
        Generate trip schedules for all buildings in the metadata.

        Uses the vehicle ownership model to assign vehicles to buildings and then generates trip schedules for each vehicle.

        Args:
            None

        Returns:
            pl.DataFrame: DataFrame of trip schedules for all buildings
        """
        # Assign cars to metadata buildings
        logging.info("Assigning cars to metadata buildings")
        bldg_veh_df = self.predict_num_vehicles()
        # Get all vehicle profiles
        logging.info("Assigning vehicle profiles")
        vehicle_profiles = cast(
            dict[tuple[str, int], VehicleProfile],
            self.sample_vehicle_profiles(bldg_veh_df, self.nhts_df),
        )

        # Generate trip schedules for each vehicle
        logging.info("Generating trip schedules")
        trip_schedules = self.generate_daily_trip_schedules(vehicle_profiles)

        return trip_schedules

    def _build_hourly_timestamps(self) -> pl.DataFrame:
        """Build hourly timestamps for the instance date range (inclusive, aligned to whole hours).

        Returns:
            pl.DataFrame: hourly timestamps from ``self.start_date`` 00:00 through ``self.end_date`` 23:00

        Raises:
            ValueError: If ``self.end_date`` is before ``self.start_date``
        """
        start_hour = self.start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_hour = self.end_date.replace(hour=23, minute=0, second=0, microsecond=0)
        if end_hour < start_hour:
            raise ValueError(
                f"end_date {self.end_date} must be on or after start_date {self.start_date}"
            )

        timestamps: list[datetime] = []
        current = start_hour
        while current <= end_hour:
            timestamps.append(current)
            current += timedelta(hours=1)

        return pl.DataFrame({"timestamp": timestamps})

    def _build_hours_base(self) -> pl.DataFrame:
        """Build the hourly calendar for the instance date range, used for trip-to-hour joins.

        Returns:
            pl.DataFrame: hourly calendar for the instance date range
        """
        if self.start_date is None or self.end_date is None:
            raise NoDateRangeError()
        return (
            self._build_hourly_timestamps()
            .with_row_index("hour_index")  # stable 0..num_hours-1 index in chronological order
            .with_columns(
                pl.col("timestamp").dt.date().alias("date"),  # calendar date for joining daily trip rows
                pl.col("timestamp").dt.hour().alias("hour"),  # clock hour (0..23) for joining trip rows
            )
        )
    
    def generate_vehicle_presence_schedules(
        self,
        trip_schedules: pl.DataFrame,
        *,
        hours_base: pl.DataFrame | None = None,
        vehicle_keys: Iterable[tuple[str | int, int]] | None = None,
    ) -> dict[tuple[str | int, int], pl.DataFrame]:
        """
        Map each vehicle's trip schedule to an hourly schedule of home/away status for the instance date range.

        Uses the same away-hour model as generate_daily_trip_schedules: a vehicle is
        away from home for hours ``range(departure_hour, arrival_hour)`` on each trip day,
        where ``arrival_hour`` is the first hour at home. It is at home (and available to
        charge) in all other hours.

        Args:
            trip_schedules (pl.DataFrame): DataFrame of trip schedules
            hours_base (pl.DataFrame): hourly calendar for the instance date range
            vehicle_keys (Iterable[tuple[str | int, int]]): Iterable of vehicle keys to generate presence schedules for

        Returns:
            dict[tuple[str | int, int], pl.DataFrame]: Dict of vehicle keys to hourly presence schedules
        """
        if hours_base is None:
            hours_base = self._build_hours_base()  # shared calendar with join keys for matching trip rows to hours

        if vehicle_keys is None:
            if trip_schedules.is_empty():
                return {}  # no vehicles requested and no trips to infer vehicle ids from
            vehicle_keys_df = trip_schedules.select("bldg_id", "vehicle_id").unique()  # infer vehicles from trips
        else:
            vehicle_keys_df = pl.DataFrame({
                "bldg_id": [key[0] for key in vehicle_keys],  # building id for each requested vehicle slot
                "vehicle_id": [key[1] for key in vehicle_keys],  # 1-based vehicle index within the building
            }).unique()  # caller may pass vehicle_profiles.keys(); dedupe just in case

        if trip_schedules.is_empty():
            # no trips means the vehicle never leaves home; every hour is chargeable
            hourly_presence = (
                vehicle_keys_df.join(hours_base, how="cross")  # one hourly schedule per requested vehicle
                .with_columns(
                    pl.lit(True).alias("at_home"),  # default presence state without trip evidence
                    pl.lit(False).alias("away_from_home"),  # explicit complement of at_home
                    pl.lit(True).alias("can_charge"),  # home charging assumed available whenever at home
                )
                .select("bldg_id", "vehicle_id", "hour_index", "timestamp", "at_home", "away_from_home", "can_charge")
            )
        else:
            away_hours = (
                trip_schedules.with_columns(pl.col("date").cast(pl.Date).alias("date"))  # normalize to date-only key
                .with_columns(
                    # away from departure_hour (inclusive) through arrival_hour (exclusive)
                    pl.int_ranges(pl.col("departure_hour"), pl.col("arrival_hour")).alias("hour"),
                )
                .explode("hour")  # one row per away hour instead of one row per trip interval
                .select("bldg_id", "vehicle_id", "date", "hour")  # minimal join keys for marking away hours
                .unique()  # overlapping trips on the same day collapse to a single away marker
            )

            hourly_presence = (
                vehicle_keys_df.join(hours_base, how="cross")  # hourly rows x number of vehicles
                .join(
                    away_hours.with_columns(pl.lit(False).alias("at_home")),  # away rows carry at_home=False
                    on=["bldg_id", "vehicle_id", "date", "hour"],  # match a specific vehicle-hour to a trip hour
                    how="left",  # keep all hours; hours without trips remain null until filled below
                )
                .with_columns(
                    pl.col("at_home").fill_null(True).alias("at_home"),
                )
                .with_columns(
                    (~pl.col("at_home")).alias("away_from_home"),
                    pl.col("at_home").alias("can_charge"),
                )
                .select("bldg_id", "vehicle_id", "hour_index", "timestamp", "at_home", "away_from_home", "can_charge")
            )

        presence_by_vehicle: dict[tuple[str | int, int], pl.DataFrame] = {}  # output container keyed by vehicle
        for vehicle_frame in hourly_presence.partition_by(["bldg_id", "vehicle_id"], as_dict=False):
            bldg_id = vehicle_frame["bldg_id"][0]  # building id is constant within each partition
            vehicle_id = int(vehicle_frame["vehicle_id"][0])  # vehicle index is constant within each partition
            presence_by_vehicle[(bldg_id, vehicle_id)] = vehicle_frame.drop("bldg_id", "vehicle_id").sort(
                "hour_index"  # return a clean per-vehicle hourly frame sorted chronologically
            )

        return presence_by_vehicle  # dict[(bldg_id, vehicle_id)] -> hourly presence DataFrame
    
    @staticmethod
    def _build_hourly_discharge_kwh(
        trip_schedules: pl.DataFrame,
        hours_base: pl.DataFrame,
        *,
        kwh_per_mile: float,
        ev_adoption_rate: float,
    ) -> pl.DataFrame:
        """
        Map each vehicle's trip schedule to an hourly schedule of discharge kWh for the instance date range.
        Spread each trip's driving energy uniformly over its away-from-home hours.

        Away hours are ``range(departure_hour, arrival_hour)`` where ``arrival_hour`` is the
        first hour at home (exclusive end of the away interval).
        
        Args:
            trip_schedules (pl.DataFrame): DataFrame of trip schedules
            hours_base (pl.DataFrame): hourly calendar for the instance date range
            kwh_per_mile (float): kWh per mile
            ev_adoption_rate (float): EV adoption rate

        Returns:
            pl.DataFrame: hourly discharge kWh for each trip
        """
        if trip_schedules.is_empty():
            # no trips -> no driving discharge; return typed empty frame for downstream joins
            return pl.DataFrame(
                schema={
                    "bldg_id": pl.Int64,
                    "vehicle_id": pl.Int64,
                    "hour_index": pl.UInt32,
                    "discharge_kwh": pl.Float64,
                }
            )

        away_hour_discharge = (
            trip_schedules.with_columns(pl.col("date").cast(pl.Date).alias("date"))  # normalize to date-only key
            .with_columns(
                # away from departure_hour (inclusive) through arrival_hour (exclusive)
                pl.int_ranges(pl.col("departure_hour"), pl.col("arrival_hour")).alias("hour"),
                (pl.col("miles_driven") * kwh_per_mile * ev_adoption_rate).alias("trip_kwh"),
            )
            .with_columns(
                # divide total trip energy evenly across all away hours for that trip
                (pl.col("trip_kwh") / pl.col("hour").list.len()).alias("discharge_kwh_per_away_hour")
            )
            .explode("hour")  # one row per away hour instead of one row per trip
            .join(
                hours_base.select("date", "hour", "hour_index"),  # map calendar (date, hour) -> hour_index index
                on=["date", "hour"],
                how="inner",
            )
            .group_by("bldg_id", "vehicle_id", "hour_index")
            # overlapping trips contributing to the same hour add their discharge shares together
            .agg(pl.col("discharge_kwh_per_away_hour").sum().alias("discharge_kwh"))
        )
        return away_hour_discharge

    @staticmethod
    def _schedule_immediate_charging(
        at_home: np.ndarray,
        discharge_kwh: np.ndarray,
        *,
        battery_capacity_kwh: float,
        charger_power_kw: float,
        initial_soc_kwh: float,
    ) -> np.ndarray:
        """
        Given a vehicle's hourly presence and discharge schedule, build an hourly charge schedule 
        by plugging in at max power whenever home and not full.

        This is the naive "charge as soon as you get home" policy. It forward-simulates SOC
        hour-by-hour because each hour's charge limit depends on energy remaining after that
        hour's trip draw. Returns only ``charge_kwh``; pair with ``_compute_hourly_soc`` for
        beginning-of-hour SOC and underflow flags.

        Args:
            at_home: Whether the vehicle is home at the start of each hour
            discharge_kwh: Fixed trip draw ``x_t^DB`` each hour (kWh)
            battery_capacity_kwh: Battery capacity ``K^B`` (kWh)
            charger_power_kw: Max charge rate ``C^B`` when home (kW = kWh/hour)
            initial_soc_kwh: Start-of-hour-0 SOC ``s_0`` (kWh)

        Returns:
            Hourly charge energy ``x_t^CB`` (kWh), same length as ``at_home``
        """
        if len(at_home) != len(discharge_kwh):
            raise ValueError(
                f"at_home and discharge_kwh must have the same length, got {len(at_home)} and {len(discharge_kwh)}"
            )

        num_hours = len(at_home)
        charge_kwh = np.zeros(num_hours, dtype=np.float64)
        current_soc = initial_soc_kwh

        for hour_idx in range(num_hours):
            # Discharge first (same order as _compute_hourly_soc).
            trip_draw = discharge_kwh[hour_idx]
            if trip_draw > current_soc:
                current_soc = 0.0  # no public charging; battery empty until next charge
            else:
                current_soc -= trip_draw

            # Charge at Level 2 whenever home and below full capacity.
            if at_home[hour_idx] and current_soc < battery_capacity_kwh:
                added = min(charger_power_kw, battery_capacity_kwh - current_soc)
                charge_kwh[hour_idx] = added
                current_soc += added

        return charge_kwh

    @staticmethod
    def _schedule_cost_minimizing_charging(
        at_home: np.ndarray,
        discharge_kwh: np.ndarray,
        *,
        battery_capacity_kwh: float,
        charger_power_kw: float,
        initial_soc_kwh: float,
        hourly_price_usd_per_kwh: np.ndarray,
    ) -> np.ndarray:
        """
        Given a vehicle's hourly presence and discharge schedule, build a perfect-foresight hourly 
        charge schedule that minimizes electricity cost (cvxpy LP).

        Returns only ``charge_kwh``; pair with ``_compute_hourly_soc`` for beginning-of-hour
        SOC and underflow flags. This is a theoretical lower bound on charging cost when the
        driver knows all future trips and can shift charging to the cheapest home hours.

        LP formulation (``T = num_hours`` clock hours indexed ``0..T-1``):

        - Given: ``s_0`` (``initial_soc_kwh``), fixed discharge ``x_t^DB = discharge_kwh[t]``
        - Decide: charge ``x_t^CB`` for ``t = 0..T-1`` and start-of-hour SOC ``s_t`` for ``t = 1..T``
        - Minimize ``sum_{t=0}^{T-1} p_t x_t^CB``
        - Subject to:
            ``s_1 = s_0 + x_0^CB - x_0^DB``
            ``s_{t+1} = s_t + x_t^CB - x_t^DB`` for ``t = 1..T-1``
            ``0 <= x_t^CB <= C^B`` (charger limit when home, else 0)
            ``0 <= s_t <= K^B`` for ``t = 1..T``
            ``s_T = s_0`` (return to initial SOC at horizon end)

        Discharging and charging are mutually exclusive in this model: trip draw is
        assigned to away hours where ``x_t^CB = 0``, so ``s_t >= x_t^DB`` follows from
        ``s_{t+1} >= 0`` and the transition equalities without an extra constraint.

        Args:
            at_home: Whether the vehicle is home at the start of each hour
            discharge_kwh: Fixed trip draw ``x_t^DB`` each hour (kWh)
            battery_capacity_kwh: Battery capacity ``K^B`` (kWh)
            charger_power_kw: Max charge rate ``C^B`` when home (kW = kWh/hour)
            initial_soc_kwh: Start-of-hour-0 SOC ``s_0`` (kWh)
            hourly_price_usd_per_kwh: Marginal price ``p_t`` each hour ($/kWh)

        Returns:
            Hourly charge energy ``x_t^CB`` (kWh), same length as ``at_home``

        Raises:
            ValueError: If input arrays differ in length or prices are negative
            RuntimeError: If the LP solver fails to find a feasible schedule
        """
        num_hours = len(at_home)
        if len(discharge_kwh) != num_hours:
            raise ValueError(
                f"at_home and discharge_kwh must have the same length, got {len(at_home)} and {len(discharge_kwh)}"
            )
        if len(hourly_price_usd_per_kwh) != num_hours:
            raise ValueError(
                "hourly_price_usd_per_kwh must match schedule length, "
                f"got {len(hourly_price_usd_per_kwh)} and expected {num_hours}"
            )
        if np.any(hourly_price_usd_per_kwh < 0):
            raise ValueError("hourly_price_usd_per_kwh must be non-negative")

        discharge = np.asarray(discharge_kwh, dtype=np.float64)
        prices = np.asarray(hourly_price_usd_per_kwh, dtype=np.float64)
        max_charge = np.where(at_home, charger_power_kw, 0.0)
        s_0 = float(initial_soc_kwh)

        # Decision variables: x_t^CB (charge) and s_t for t = 1..T (s_0 is fixed).
        charge = cp.Variable(num_hours, name="charge")
        soc = cp.Variable(num_hours, name="soc")  # s_1..s_T

        constraints: list[cp.Constraint] = [
            soc[0] == s_0 + charge[0] - discharge[0],  # s_1 = s_0 + x_0^CB - x_0^DB
            # soc[-1] == s_0,  # terminal SOC: must replenish any energy drawn from s_0
            charge >= 0,
            charge <= max_charge,  # zero when away (discharge and charge never overlap)
            soc >= 0,
            soc <= battery_capacity_kwh,
        ]
        if num_hours > 1:
            # s_{t+1} = s_t + x_t^CB - x_t^DB for t = 1..T-1
            constraints.append(soc[1:] == soc[:-1] + charge[1:] - discharge[1:])

        problem = cp.Problem(cp.Minimize(prices @ charge), constraints)
        problem.solve()

        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(f"Cost-minimizing charging LP failed: {problem.status}")

        return np.asarray(charge.value, dtype=np.float64).reshape(-1)

    @staticmethod
    def _compute_hourly_soc(
        discharge_kwh: np.ndarray,
        charge_kwh: np.ndarray,
        *,
        initial_soc_kwh: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Given a vehicle's hourly discharge and charge schedule, derive the beginning-of-hour SOC 
        and underflow flags.

        Shared by both charging strategies: once ``charge_kwh`` is chosen, SOC is computed with
        the same hour-by-hour rules. ``soc_kwh[t]`` is the battery level at the **start** of
        hour *t*; within the hour, discharge is applied first, then charge.

        Args:
            discharge_kwh: Fixed trip draw each hour (kWh)
            charge_kwh: Scheduled charge each hour (kWh), from immediate or cost-minimizing policy
            initial_soc_kwh: Start-of-hour-0 SOC (kWh)

        Returns:
            Tuple of beginning-of-hour SOC (kWh) and per-hour underflow flag (True when trip
            draw exceeds available SOC at the start of that hour; SOC is clamped to zero)
        """
        if len(discharge_kwh) != len(charge_kwh):
            raise ValueError(
                f"discharge_kwh and charge_kwh must have the same length, got {len(discharge_kwh)} and {len(charge_kwh)}"
            )

        num_hours = len(discharge_kwh)
        soc_kwh = np.empty(num_hours, dtype=np.float64)
        soc_underflow = np.zeros(num_hours, dtype=bool)

        current_soc = initial_soc_kwh
        for hour_idx in range(num_hours):
            soc_kwh[hour_idx] = current_soc  # record beginning-of-hour SOC

            trip_draw = discharge_kwh[hour_idx]
            if trip_draw > current_soc + 1e-9:
                soc_underflow[hour_idx] = True
                current_soc = 0.0
            else:
                current_soc -= trip_draw

            current_soc += charge_kwh[hour_idx]  # charge after discharge within the hour

        return soc_kwh, soc_underflow

    def generate_vehicle_soc_schedules(
        self,
        trip_schedules: pl.DataFrame,
        *,
        vehicle_keys: Iterable[tuple[str | int, int]] | None = None,
        hours_base: pl.DataFrame | None = None,
        presence_by_vehicle: dict[tuple[str | int, int], pl.DataFrame] | None = None,
        battery_capacity_kwh: float = DEFAULT_BATTERY_CAPACITY_KWH,
        kwh_per_mile: float = DEFAULT_KWH_PER_MILE,
        charger_power_kw: float = DEFAULT_LEVEL2_CHARGER_KW,
        ev_adoption_rate: float = 1.0,
        initial_soc_kwh: float | None = None,
        charging_strategy: ChargingStrategy = "immediate",
        hourly_price_usd_per_kwh: np.ndarray | None = None,
    ) -> dict[tuple[str | int, int], pl.DataFrame]:
        """
        Map each vehicle to an hourly SOC, charging, and discharge schedule for the instance date range.

        Pipeline per vehicle:
        1. Spread trip miles into hourly ``discharge_kwh`` (away hours only)
        2. Build ``charge_kwh`` via ``charging_strategy`` (immediate or cost-minimizing LP)
        3. Derive ``soc_kwh`` and ``soc_underflow`` from discharge + charge

        ``soc_kwh`` is the battery level at the beginning of each hour (aligned with ``timestamp``).

        Charging strategies:
        - ``immediate``: charge at full power whenever home and not full (default).
        - ``cost_minimizing``: perfect-foresight LP that shifts charging to the cheapest
          home hours while meeting all trip energy needs. Requires ``hourly_price_usd_per_kwh``.

        Args:
            trip_schedules: DataFrame of trip schedules
            vehicle_keys: Vehicle keys to include when building presence schedules internally
            presence_by_vehicle: Pre-built hourly presence schedules per vehicle; when provided,
                presence is not recomputed and ``vehicle_keys`` is ignored
            hours_base: Hourly calendar for trip-to-hour joins; built from the instance date range if None
            battery_capacity_kwh: Battery capacity in kWh
            kwh_per_mile: kWh per mile
            charger_power_kw: Charger power in kW
            ev_adoption_rate: EV adoption rate
            initial_soc_kwh: Initial SOC in kWh at the start of hour 0
            charging_strategy: ``immediate`` or ``cost_minimizing``
            hourly_price_usd_per_kwh: Length-``num_hours`` marginal price array for optimized charging

        Returns:
            Dict of vehicle keys to hourly SOC schedules

        Raises:
            ValueError: If ``battery_capacity_kwh`` is not positive
            ValueError: If ``charger_power_kw`` is negative
            ValueError: If ``kwh_per_mile`` is negative
            ValueError: If ``initial_soc_kwh`` is not within [0, ``battery_capacity_kwh``]
            ValueError: If a pre-built presence schedule does not match the hourly calendar length
            ValueError: If ``charging_strategy`` is ``cost_minimizing`` without hourly prices
        """
        if battery_capacity_kwh <= 0:
            raise ValueError(f"battery_capacity_kwh must be positive, got {battery_capacity_kwh}")
        if charger_power_kw < 0:
            raise ValueError(f"charger_power_kw must be non-negative, got {charger_power_kw}")
        if kwh_per_mile < 0:
            raise ValueError(f"kwh_per_mile must be non-negative, got {kwh_per_mile}")

        start_soc = battery_capacity_kwh if initial_soc_kwh is None else initial_soc_kwh
        if not 0.0 <= start_soc <= battery_capacity_kwh:
            raise ValueError(f"initial_soc_kwh must be within [0, {battery_capacity_kwh}], got {start_soc}")

        if hours_base is None:
            hours_base = self._build_hours_base()
        num_hours = hours_base.height

        if charging_strategy == "cost_minimizing":
            if hourly_price_usd_per_kwh is None:
                raise ValueError("hourly_price_usd_per_kwh is required when charging_strategy='cost_minimizing'")
            if len(hourly_price_usd_per_kwh) != num_hours:
                raise ValueError(
                    f"hourly_price_usd_per_kwh must have length {num_hours}, got {len(hourly_price_usd_per_kwh)}"
                )

        if presence_by_vehicle is None:
            presence_by_vehicle = self.generate_vehicle_presence_schedules(
                trip_schedules,
                hours_base=hours_base,
                vehicle_keys=vehicle_keys,
            )
        else:
            for vehicle_key, presence in presence_by_vehicle.items():
                if presence.height != num_hours:
                    raise ValueError(
                        f"presence schedule for {vehicle_key} has {presence.height} rows, "
                        f"expected {num_hours} for the instance date range"
                    )

        # Build hourly discharge kWh
        discharge_by_hour = self._build_hourly_discharge_kwh(
            trip_schedules,
            hours_base,
            kwh_per_mile=kwh_per_mile,
            ev_adoption_rate=ev_adoption_rate,
        )

        discharge_lookup: dict[tuple[str | int, int], dict[int, float]] = {}
        for discharge_frame in discharge_by_hour.partition_by(["bldg_id", "vehicle_id"], as_dict=False):
            vehicle_key = (discharge_frame["bldg_id"][0], int(discharge_frame["vehicle_id"][0]))
            discharge_lookup[vehicle_key] = dict(
                zip(
                    discharge_frame["hour_index"].to_list(),
                    discharge_frame["discharge_kwh"].to_list(),
                    strict=True,
                )
            )

        # Generate hourly SOC schedules
        soc_by_vehicle: dict[tuple[str | int, int], pl.DataFrame] = {}
        for vehicle_key, presence in presence_by_vehicle.items():
            discharge_arr = np.zeros(num_hours, dtype=np.float64)  # default: no driving this hour
            # Look up discharge kWh for each hour of the day
            for hour_index, discharge in discharge_lookup.get(vehicle_key, {}).items():
                discharge_arr[int(hour_index)] = discharge

            at_home = presence["at_home"].to_numpy()
            # Step 2: choose hourly charge schedule (strategy-specific).
            if charging_strategy == "immediate":
                charge_kwh = self._schedule_immediate_charging(
                    at_home,
                    discharge_arr,
                    battery_capacity_kwh=battery_capacity_kwh,
                    charger_power_kw=charger_power_kw,
                    initial_soc_kwh=start_soc,
                )
            else:
                charge_kwh = self._schedule_cost_minimizing_charging(
                    at_home,
                    discharge_arr,
                    battery_capacity_kwh=battery_capacity_kwh,
                    charger_power_kw=charger_power_kw,
                    initial_soc_kwh=start_soc,
                    hourly_price_usd_per_kwh=np.asarray(hourly_price_usd_per_kwh, dtype=np.float64),
                )
            # Step 3: derive beginning-of-hour SOC from discharge + charge (shared logic).
            soc_kwh, soc_underflow = self._compute_hourly_soc(
                discharge_arr,
                charge_kwh,
                initial_soc_kwh=start_soc,
            )

            soc_by_vehicle[vehicle_key] = presence.with_columns(
                pl.Series("discharge_kwh", discharge_arr),
                pl.Series("charge_kwh", charge_kwh),
                pl.Series("soc_kwh", soc_kwh),
                pl.Series("soc_underflow", soc_underflow),
            )

        return soc_by_vehicle  # dict[(bldg_id, vehicle_id)] -> hourly SOC DataFrame

    @staticmethod
    def vehicle_hourly_schedules_to_dataframe(
        schedules_by_vehicle: dict[tuple[str | int, int], pl.DataFrame],
    ) -> pl.DataFrame:
        """Flatten per-vehicle hourly schedule dicts into a single long-form DataFrame.

        Stacks one row per vehicle-hour so building-level aggregations (e.g. TOU cost) can
        use ``group_by("bldg_id")`` instead of looping over the input dict.

        Args:
            schedules_by_vehicle: Dict keyed by ``(bldg_id, vehicle_id)`` with hourly frames
                from ``generate_vehicle_presence_schedules`` or ``generate_vehicle_soc_schedules``

        Returns:
            DataFrame with the following columns:
            - bldg_id: Building ID
            - vehicle_id: Vehicle ID
            - hour_index: Hour index
            - timestamp: Timestamp
            - at_home: Whether the vehicle is at home
            - away_from_home: Whether the vehicle is away from home
            - can_charge: Whether the vehicle can charge
            - discharge_kwh: Discharge kWh
            - charge_kwh: Charge kWh
            - soc_kwh: SOC kWh
            - soc_underflow: Whether the vehicle has a SOC underflow
        """
        if not schedules_by_vehicle:
            return pl.DataFrame()

        # Tag each per-vehicle frame with its building/vehicle keys, then stack.
        frames = [
            frame.with_columns(
                pl.lit(bldg_id).alias("bldg_id"),
                pl.lit(vehicle_id).alias("vehicle_id"),
            )
            for (bldg_id, vehicle_id), frame in schedules_by_vehicle.items()
        ]
        return pl.concat(frames).select(
            "bldg_id",
            "vehicle_id",
            "hour_index",
            "timestamp",
            "at_home",
            "away_from_home",
            "can_charge",
            "discharge_kwh",
            "charge_kwh",
            "soc_kwh",
            "soc_underflow",
        )


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate EV demand trip schedules from ResStock metadata",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("--state", required=True, help="State abbreviation (e.g., NY, CA, TX)")
    parser.add_argument("--release", required=True, help="BuildStock release version (e.g., res_2022_tmy3_1.1)")
    parser.add_argument("--start-date", required=True, help="Start date for simulation (YYYY-MM-DD format)")
    parser.add_argument("--end-date", required=True, help="End date for simulation (YYYY-MM-DD format)")

    # Optional S3 upload
    parser.add_argument(
        "--upload-s3", action="store_true", default=False, help="Upload results to S3 bucket instead of saving locally"
    )

    return parser.parse_args()


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as e:
        raise InvalidDateFormatError(date_str) from e


def upload_batch_to_s3(batch_trip_schedules, config, file_name, batch_number):
    """Upload a single batch of trip schedules to S3 with partitioning."""
    import io

    if len(batch_trip_schedules) == 0:
        return True

    # Add metadata columns
    batch_with_metadata = batch_trip_schedules.with_columns([
        pl.lit(config.release).alias("release"),
        pl.lit(config.state).alias("state"),
    ])

    # Partition the batch
    partitions = batch_with_metadata.partition_by(["release", "state"])

    upload_success = True
    for partition in partitions:
        # Get partition values for file naming
        release_val = partition["release"][0] if len(partition["release"]) > 0 else "unknown"
        state_val = partition["state"][0] if len(partition["state"]) > 0 else "unknown"

        # Create partitioned file name with batch number
        partition_file_name = f"{file_name}/release={release_val}/state={state_val}/batch_{batch_number:03d}.parquet"

        # Write partition to memory buffer
        buffer = io.BytesIO()
        partition.write_parquet(buffer)
        file_content = buffer.getvalue()

        # Upload partition to S3
        partition_upload_success = ev_utils.upload_object_to_s3(file_content, partition_file_name)
        if not partition_upload_success:
            print(
                f"Error: S3 upload failed for partition release={release_val}, state={state_val}, batch={batch_number}"
            )
            upload_success = False
            break

    return upload_success


def main():
    """Main function to run EV demand calculation with command-line arguments."""
    args = parse_arguments()

    # Parse dates
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)

    # Validate date range
    if start_date >= end_date:
        print("Error: Start date must be before end date")
        return 1

    # Step 1: Create configuration
    config = EVDemandConfig(state=args.state, release=args.release)

    # Step 2: Load all data
    metadata_df, nhts_df, pums_df = ev_utils.load_all_input_data(config)
    print(f"Loaded metadata: {len(metadata_df)} rows")
    print(f"Loaded NHTS data: {len(nhts_df)} rows")
    print(f"Loaded PUMS data: {len(pums_df)} rows")

    # Process metadata in batches of 20,000 rows
    batch_size = 20000
    total_rows = len(metadata_df)
    all_trip_schedules = []
    all_soc_schedules = []
    trip_file_name = "trip_schedules"
    soc_file_name = "vehicle_soc_schedules"

    for i in range(0, total_rows, batch_size):
        batch_end = min(i + batch_size, total_rows)
        batch_metadata = metadata_df[i:batch_end]
        batch_number = i // batch_size + 1

        print(f"Processing batch {batch_number}: rows {i + 1} to {batch_end} ({len(batch_metadata)} rows)")

        calculator = EVDemandCalculator(
            metadata_df=batch_metadata,
            nhts_df=nhts_df,
            pums_df=pums_df,
            start_date=start_date,
            end_date=end_date,
            max_workers=8,  # Use worker threads for parallel processing
        )

        batch_trip_schedules = calculator.match_and_generate_trip_schedules()
        batch_soc_schedules = EVDemandCalculator.vehicle_hourly_schedules_to_dataframe(
            calculator.generate_vehicle_soc_schedules(batch_trip_schedules),
        )

        print(f"Completed batch {batch_number}: generated {len(batch_trip_schedules)} trip schedules")
        print(f"Completed batch {batch_number}: generated {len(batch_soc_schedules)} hourly SOC rows")

        if args.upload_s3:
            # Upload batch directly to S3
            print(f"Uploading batch {batch_number} to S3...")
            trip_upload_success = upload_batch_to_s3(batch_trip_schedules, config, trip_file_name, batch_number)
            soc_upload_success = upload_batch_to_s3(batch_soc_schedules, config, soc_file_name, batch_number)

            if not trip_upload_success or not soc_upload_success:
                print(f"Error: S3 upload failed for batch {batch_number}")
                return 1

            print(f"Successfully uploaded batch {batch_number} to S3")
            # Clear batch data to free memory
            del batch_trip_schedules
            del batch_soc_schedules
        else:
            # Keep batch for local saving
            all_trip_schedules.append(batch_trip_schedules)
            all_soc_schedules.append(batch_soc_schedules)

    if args.upload_s3:
        print(f"All batches uploaded to S3 with partitioning: {trip_file_name}/ and {soc_file_name}/")
        logging.info(
            f"Uploaded all batches to S3 with partitioning: {trip_file_name}/ and {soc_file_name}/"
        )
    else:
        # Combine all batches for local saving
        print("Combining all batches...")
        if config.output_dir is None:
            raise ValueError("config.output_dir")
        os.makedirs(config.output_dir, exist_ok=True)

        if all_trip_schedules:
            combined_trip_schedules = pl.concat(all_trip_schedules)
            logging.info(f"Combined all batches: {len(combined_trip_schedules)} total trip schedules")

            final_trip_schedules = combined_trip_schedules.with_columns([
                pl.lit(config.release).alias("release"),
                pl.lit(config.state).alias("state"),
            ]).sort(["bldg_id", "vehicle_id", "date"])

            local_trip_path = f"{config.output_dir}/{trip_file_name}"
            final_trip_schedules.write_parquet(local_trip_path, partition_by=["release", "state"])

            print(f"Trip schedules written to: {local_trip_path}")
            logging.info(f"Written trip schedules to {local_trip_path}")
        else:
            logging.warning("No trip schedules generated")

        if all_soc_schedules:
            combined_soc_schedules = pl.concat(all_soc_schedules)
            logging.info(f"Combined all batches: {len(combined_soc_schedules)} total hourly SOC rows")

            final_soc_schedules = combined_soc_schedules.with_columns([
                pl.lit(config.release).alias("release"),
                pl.lit(config.state).alias("state"),
            ]).sort(["bldg_id", "vehicle_id", "hour_index"])

            local_soc_path = f"{config.output_dir}/{soc_file_name}"
            final_soc_schedules.write_parquet(local_soc_path, partition_by=["release", "state"])

            print(f"Vehicle SOC schedules written to: {local_soc_path}")
            logging.info(f"Written vehicle SOC schedules to {local_soc_path}")
        else:
            logging.warning("No vehicle SOC schedules generated")

    return 0


# # Example usage
# Example usage
if __name__ == "__main__":
    exit(main())
