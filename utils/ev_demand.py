import argparse
import calendar
import logging
import os
import sys
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final, Literal, cast, overload

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
    weekday_arrival_hour: list[int] = field(default_factory=list)  # List of arrival hours for each weekday trip
    weekday_miles: list[float] = field(default_factory=list)  # List of miles for each weekday trip
    weekday_trip_weights: list[float] = field(default_factory=list)  # List of trip weights for each weekday trip
    weekend_departure_hour: list[int] = field(default_factory=list)  # List of departure hours for each weekend trip
    weekend_arrival_hour: list[int] = field(default_factory=list)  # List of arrival hours for each weekend trip
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
    departure_hour: int
    arrival_hour: int
    miles_driven: float


MIN_TRIP_DURATION_HOURS = 1  # hourly model: each trip spans at least one away-from-home hour
HOURS_PER_YEAR = 8760  # standard hourly load-curve length (365 days x 24 hours)
DEFAULT_BATTERY_CAPACITY_KWH = 90.0  # uniform fleet battery assumption for SOC modeling
DEFAULT_KWH_PER_MILE = 0.30  # simple-model assumption
DEFAULT_LEVEL2_CHARGER_KW = 7.2  # typical 32 A @ 240 V residential Level 2 charger


def build_8760_hour_timestamps(year: int) -> pl.DataFrame:
    """Build 8760 naive hourly timestamps for a calendar year (drops Dec 31 in leap years)."""
    start = datetime(year, 1, 1, 0, 0, 0)  # first hour of the target year
    end = datetime(year, 12, 31, 23, 0, 0)  # last hour of the target year
    timestamps: list[datetime] = []  # collect one timestamp per hour before building the DataFrame
    current = start  # walk forward one hour at a time from start to end
    while current <= end:
        timestamps.append(current)  # record this hour
        current += timedelta(hours=1)  # advance to the next hour

    if calendar.isleap(year):
        # leap years have 8784 raw hours; drop Dec 31 to match ResStock/CAIRO 8760 convention
        timestamps = [timestamp for timestamp in timestamps if not (timestamp.month == 12 and timestamp.day == 31)]

    if len(timestamps) != HOURS_PER_YEAR:
        raise ValueError(f"Expected {HOURS_PER_YEAR} timestamps, got {len(timestamps)} for year {year}")
    return pl.DataFrame({"timestamp": timestamps})  # one-column reference grid shared by all vehicles


def generate_vehicle_presence_schedules(
    trip_schedules: pl.DataFrame,
    start_date: datetime,
    *,
    vehicle_keys: Iterable[tuple[str | int, int]] | None = None,
) -> dict[tuple[str | int, int], pl.DataFrame]:
    """
    Build an 8760-row hourly home/away schedule for each vehicle.

    Uses the same inclusive-hour trip model as generate_daily_schedules: a vehicle is
    away from home from departure_hour through arrival_hour on each trip day. It is at
    home (and available to charge) in all other hours.
    """
    # shared 8760-hour calendar with join keys for matching trip rows to hours
    hours_base = (
        build_8760_hour_timestamps(start_date.year)  # Jan 1 00:00 through Dec 31 23:00 (8760 rows)
        .with_row_index("hour_of_year")  # stable 0..8759 index aligned with load-curve hour order
        .with_columns(
            pl.col("timestamp").dt.date().alias("date"),  # calendar date for joining daily trip rows
            pl.col("timestamp").dt.hour().alias("hour"),  # clock hour (0..23) for joining trip rows
        )
    )

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
            vehicle_keys_df.join(hours_base, how="cross")  # one 8760-row schedule per requested vehicle
            .with_columns(
                pl.lit(True).alias("at_home"),  # default presence state without trip evidence
                pl.lit(False).alias("away_from_home"),  # explicit complement of at_home
                pl.lit(True).alias("can_charge"),  # home charging assumed available whenever at home
            )
            .select("bldg_id", "vehicle_id", "hour_of_year", "timestamp", "at_home", "away_from_home", "can_charge")
        )
    else:
        away_hours = (
            trip_schedules.with_columns(pl.col("date").cast(pl.Date).alias("date"))  # normalize to date-only key
            .with_columns(
                # inclusive hour model: away from departure_hour through arrival_hour
                # +1 to include the arrival hour itself
                pl.int_ranges(pl.col("departure_hour"), pl.col("arrival_hour") + 1).alias("hour"),
            )
            .explode("hour")  # one row per away hour instead of one row per trip interval
            .select("bldg_id", "vehicle_id", "date", "hour")  # minimal join keys for marking away hours
            .unique()  # overlapping trips on the same day collapse to a single away marker
        )

        hourly_presence = (
            vehicle_keys_df.join(hours_base, how="cross")  # 8760 rows x number of vehicles
            .join(
                away_hours.with_columns(pl.lit(False).alias("at_home")),  # away rows carry at_home=False
                on=["bldg_id", "vehicle_id", "date", "hour"],  # match a specific vehicle-hour to a trip hour
                how="left",  # keep all 8760 hours; hours without trips remain null until filled below
            )
            .with_columns(
                pl.col("at_home").fill_null(True).alias("at_home"),
            )
            .with_columns(
                (~pl.col("at_home")).alias("away_from_home"),
                pl.col("at_home").alias("can_charge"),
            )
            .select("bldg_id", "vehicle_id", "hour_of_year", "timestamp", "at_home", "away_from_home", "can_charge")
        )

    presence_by_vehicle: dict[tuple[str | int, int], pl.DataFrame] = {}  # output container keyed by vehicle
    for vehicle_frame in hourly_presence.partition_by(["bldg_id", "vehicle_id"], as_dict=False):
        bldg_id = vehicle_frame["bldg_id"][0]  # building id is constant within each partition
        vehicle_id = int(vehicle_frame["vehicle_id"][0])  # vehicle index is constant within each partition
        presence_by_vehicle[(bldg_id, vehicle_id)] = vehicle_frame.drop("bldg_id", "vehicle_id").sort(
            "hour_of_year"  # return a clean per-vehicle 8760-row frame sorted chronologically
        )

    return presence_by_vehicle  # dict[(bldg_id, vehicle_id)] -> hourly presence DataFrame


def _build_hourly_discharge_kwh(
    trip_schedules: pl.DataFrame,
    hours_base: pl.DataFrame,
    *,
    kwh_per_mile: float,
    ev_adoption_rate: float,
) -> pl.DataFrame:
    """Spread each trip's driving energy uniformly over its inclusive away-from-home hours."""
    if trip_schedules.is_empty():
        # no trips -> no driving discharge; return typed empty frame for downstream joins
        return pl.DataFrame(
            schema={
                "bldg_id": pl.Int64,
                "vehicle_id": pl.Int64,
                "hour_of_year": pl.UInt32,
                "discharge_kwh": pl.Float64,
            }
        )

    away_hour_discharge = (
        trip_schedules.with_columns(pl.col("date").cast(pl.Date).alias("date"))  # normalize to date-only key
        .with_columns(
            # same inclusive away-hour model as presence: dep through arr (+1 for exclusive end)
            pl.int_ranges(pl.col("departure_hour"), pl.col("arrival_hour") + 1).alias("hour"),
            (pl.col("miles_driven") * kwh_per_mile * ev_adoption_rate).alias("trip_kwh"),
        )
        .with_columns(
            # divide total trip energy evenly across all away hours for that trip
            (pl.col("trip_kwh") / pl.col("hour").list.len()).alias("discharge_kwh_per_away_hour")
        )
        .explode("hour")  # one row per away hour instead of one row per trip
        .join(
            hours_base.select("date", "hour", "hour_of_year"),  # map calendar (date, hour) -> hour_of_year index
            on=["date", "hour"],
            how="inner",
        )
        .group_by("bldg_id", "vehicle_id", "hour_of_year")
        # overlapping trips contributing to the same hour add their discharge shares together
        .agg(pl.col("discharge_kwh_per_away_hour").sum().alias("discharge_kwh"))
    )
    return away_hour_discharge


def _simulate_hourly_soc(
    at_home: np.ndarray,
    discharge_kwh: np.ndarray,
    *,
    battery_capacity_kwh: float,
    charger_power_kw: float,
    initial_soc_kwh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hour-by-hour SOC balance: discharge while away, then charge at Level 2 when home and not full.

    Returns end-of-hour SOC, hourly charge energy, and a per-hour underflow flag when discharge
    exceeds available SOC (SOC is clamped to zero).
    """
    if len(at_home) != HOURS_PER_YEAR or len(discharge_kwh) != HOURS_PER_YEAR:
        raise ValueError(f"Expected length {HOURS_PER_YEAR}, got {len(at_home)} and {len(discharge_kwh)}")

    soc_kwh = np.empty(HOURS_PER_YEAR, dtype=np.float64)  # end-of-hour battery level
    charge_kwh = np.zeros(HOURS_PER_YEAR, dtype=np.float64)  # energy charged each hour
    soc_underflow = np.zeros(HOURS_PER_YEAR, dtype=bool)  # True when trip draw exceeds available SOC

    current_soc = initial_soc_kwh  # carry SOC forward; only reset at year start unless caller overrides
    for hour_idx in range(HOURS_PER_YEAR):
        discharge = discharge_kwh[hour_idx]
        if discharge > 0.0:
            if discharge > current_soc:
                # trip requires more energy than remains in the battery; clamp and flag (no public charging)
                soc_underflow[hour_idx] = True
                current_soc = 0.0
            else:
                current_soc -= discharge

        if at_home[hour_idx] and current_soc < battery_capacity_kwh:
            # plug in immediately at Level 2 rate whenever home and not full (no smart scheduling)
            added = min(charger_power_kw, battery_capacity_kwh - current_soc)
            charge_kwh[hour_idx] = added
            current_soc += added

        soc_kwh[hour_idx] = current_soc  # record end-of-hour SOC after discharge then charge

    return soc_kwh, charge_kwh, soc_underflow


def generate_vehicle_soc_schedules(
    trip_schedules: pl.DataFrame,
    start_date: datetime,
    *,
    vehicle_keys: Iterable[tuple[str | int, int]] | None = None,
    battery_capacity_kwh: float = DEFAULT_BATTERY_CAPACITY_KWH,
    kwh_per_mile: float = DEFAULT_KWH_PER_MILE,
    charger_power_kw: float = DEFAULT_LEVEL2_CHARGER_KW,
    ev_adoption_rate: float = 1.0,
    initial_soc_kwh: float | None = None,
) -> dict[tuple[str | int, int], pl.DataFrame]:
    """
    Build an 8760-row hourly SOC schedule for each vehicle.

    Assumes each vehicle starts the year at full battery (or ``initial_soc_kwh``). Trip energy is
    spread uniformly over inclusive away hours (departure through arrival). Whenever the vehicle is
    home and below full capacity, it charges at ``charger_power_kw`` until full. SOC is clamped at
    zero when discharge exceeds available energy; those hours are flagged with ``soc_underflow``.
    """
    # validate physical/model parameters before building 8760-hour arrays
    if battery_capacity_kwh <= 0:
        raise ValueError(f"battery_capacity_kwh must be positive, got {battery_capacity_kwh}")
    if charger_power_kw < 0:
        raise ValueError(f"charger_power_kw must be non-negative, got {charger_power_kw}")
    if kwh_per_mile < 0:
        raise ValueError(f"kwh_per_mile must be non-negative, got {kwh_per_mile}")

    start_soc = battery_capacity_kwh if initial_soc_kwh is None else initial_soc_kwh
    if not 0.0 <= start_soc <= battery_capacity_kwh:
        raise ValueError(f"initial_soc_kwh must be within [0, {battery_capacity_kwh}], got {start_soc}")

    # shared calendar used to translate trip (date, clock hour) rows into hour_of_year indices
    hours_base = (
        build_8760_hour_timestamps(start_date.year)
        .with_row_index("hour_of_year")
        .with_columns(
            pl.col("timestamp").dt.date().alias("date"),
            pl.col("timestamp").dt.hour().alias("hour"),
        )
    )

    # where the vehicle is home vs away (determines when charging is allowed)
    presence_by_vehicle = generate_vehicle_presence_schedules(
        trip_schedules,
        start_date,
        vehicle_keys=vehicle_keys,
    )

    # how much driving energy is consumed in each hour (vectorized from trip schedules)
    discharge_by_hour = _build_hourly_discharge_kwh(
        trip_schedules,
        hours_base,
        kwh_per_mile=kwh_per_mile,
        ev_adoption_rate=ev_adoption_rate,
    )

    # index discharge by vehicle and hour_of_year for fast numpy scatter in the sequential SOC loop
    discharge_lookup: dict[tuple[str | int, int], dict[int, float]] = {}
    for discharge_frame in discharge_by_hour.partition_by(["bldg_id", "vehicle_id"], as_dict=False):
        vehicle_key = (discharge_frame["bldg_id"][0], int(discharge_frame["vehicle_id"][0]))
        discharge_lookup[vehicle_key] = dict(
            zip(
                discharge_frame["hour_of_year"].to_list(),
                discharge_frame["discharge_kwh"].to_list(),
                strict=True,
            )
        )

    # per vehicle, run sequential hour-by-hour SOC balance (cannot be fully vectorized)
    soc_by_vehicle: dict[tuple[str | int, int], pl.DataFrame] = {}
    for vehicle_key, presence in presence_by_vehicle.items():
        discharge_arr = np.zeros(HOURS_PER_YEAR, dtype=np.float64)  # default: no driving this hour
        for hour_of_year, discharge in discharge_lookup.get(vehicle_key, {}).items():
            discharge_arr[int(hour_of_year)] = discharge

        at_home = presence["at_home"].to_numpy()
        soc_kwh, charge_kwh, soc_underflow = _simulate_hourly_soc(
            at_home,
            discharge_arr,
            battery_capacity_kwh=battery_capacity_kwh,
            charger_power_kw=charger_power_kw,
            initial_soc_kwh=start_soc,
        )

        soc_by_vehicle[vehicle_key] = presence.with_columns(
            pl.Series("discharge_kwh", discharge_arr),
            pl.Series("charge_kwh", charge_kwh),
            pl.Series("soc_kwh", soc_kwh),
            pl.Series("soc_underflow", soc_underflow),
        )

    return soc_by_vehicle  # dict[(bldg_id, vehicle_id)] -> 8760-row hourly SOC DataFrame


def normalize_day_trip_times(
    departures: np.ndarray,
    arrivals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Enforce arrival > departure per trip and non-overlapping trips within a day.

    Independent random offsets can produce arrival <= departure or overlapping intervals.
    Trips are repacked in departure order on a single calendar day (no overnight trips).
    Each trip starts strictly after the previous trip ends (next dep >= prev arr + 1).
    Returns arrays in the original input order. Trips that no longer fit before hour 23
    are dropped (keep_mask=False).
    """
    n = len(departures)
    keep = np.ones(n, dtype=bool)
    if n == 0:
        return departures.astype(int), arrivals.astype(int), keep

    # Pack in chronological order, then restore the caller's trip index order below.
    order = np.argsort(departures, kind="stable")
    dep_sorted = departures[order].astype(int)
    arr_sorted = arrivals[order].astype(int)
    keep_sorted = np.ones(n, dtype=bool)

    earliest_next_dep = 0
    for i in range(n):
        # No room left today for another trip of minimum duration.
        if earliest_next_dep > 23 - MIN_TRIP_DURATION_HOURS:
            keep_sorted[i:] = False
            break

        # Push forward if this trip overlaps a prior one; enforce arr > dep.
        dep = min(max(int(dep_sorted[i]), earliest_next_dep), 23 - MIN_TRIP_DURATION_HOURS)
        arr = min(max(int(arr_sorted[i]), dep + MIN_TRIP_DURATION_HOURS), 23)

        dep_sorted[i] = dep
        arr_sorted[i] = arr
        # Inclusive hour model: away through arrival hour, so next dep is at least arr + 1.
        earliest_next_dep = arr + 1

    # Map packed times back to the original trip-index order for downstream joins.
    dep_out = np.empty(n, dtype=int)
    arr_out = np.empty(n, dtype=int)
    dep_out[order] = dep_sorted
    arr_out[order] = arr_sorted
    keep[order] = keep_sorted
    return dep_out, arr_out, keep


def summarize_nhts_match_catalog(catalog: pl.DataFrame) -> pl.DataFrame:
    """Summarize NHTS household/vehicle matching gaps from sample_vehicle_profiles catalog."""
    if catalog.is_empty():
        return pl.DataFrame({"metric": [], "count": [], "share_of_vehicle_slots": []})

    vehicle_slots = catalog.height
    n_missing_nhts = catalog.filter(~pl.col("nhts_vehicle_matched")).height
    matched = catalog.filter(pl.col("nhts_vehicle_matched"))
    n_missing_weekday = matched.filter(~pl.col("has_weekday_trips")).height
    n_missing_weekend = matched.filter(~pl.col("has_weekend_trips")).height
    n_missing_both = matched.filter(~pl.col("has_weekday_trips") & ~pl.col("has_weekend_trips")).height

    buildings_with_gaps = (
        catalog.filter(~pl.col("nhts_vehicle_matched") | ~pl.col("has_weekday_trips") | ~pl.col("has_weekend_trips"))
        .select("bldg_id")
        .unique()
        .height
    )
    buildings_with_vehicles = catalog.select("bldg_id").unique().height

    def share(count: int) -> float:
        return count / vehicle_slots if vehicle_slots else 0.0

    return pl.DataFrame({
        "metric": [
            "vehicle_slots",
            "missing_nhts_vehicle_match",
            "missing_weekday_trip_profile",
            "missing_weekend_trip_profile",
            "missing_both_trip_profiles",
            "buildings_with_any_gap",
        ],
        "count": [
            vehicle_slots,
            n_missing_nhts,
            n_missing_weekday,
            n_missing_weekend,
            n_missing_both,
            buildings_with_gaps,
        ],
        "share_of_vehicle_slots": [
            1.0,
            share(n_missing_nhts),
            share(n_missing_weekday),
            share(n_missing_weekend),
            share(n_missing_both),
            buildings_with_gaps / buildings_with_vehicles if buildings_with_vehicles else 0.0,
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
                weekday_departures = [t // 100 for t in weekday_data["start_time"]]  # Convert HHMM to HH
                weekday_arrivals = [t // 100 for t in weekday_data["end_time"]]
                weekday_miles = weekday_data["miles_driven"].to_list()
                weekday_weights = weekday_data["trip_weight"].to_list()
                weekday_trip_ids = list(range(1, len(weekday_departures) + 1))

                # Process weekend trips (extract from filtered data)
                weekend_departures = [t // 100 for t in weekend_data["start_time"]]
                weekend_arrivals = [t // 100 for t in weekend_data["end_time"]]
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

    def generate_daily_schedules(  # noqa: C901
        self, profile: VehicleProfile, rng: np.random.RandomState | None = None
    ) -> pl.DataFrame:
        """Generate trip schedules for all days in the date range as a DataFrame."""
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
            departures_with_variance = np.clip(selected_departures + departure_offsets, 0, 23)
            arrivals_with_variance = np.clip(selected_arrivals + arrival_offsets, 0, 23)

            # Offsets are drawn independently per trip, so fix invalid intervals and
            # overlaps before writing the day's schedule (may drop trips that spill past 23:00).
            departures_with_variance, arrivals_with_variance, keep_trip = normalize_day_trip_times(
                departures_with_variance,
                arrivals_with_variance,
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
            arrival_hours.extend(arrivals_with_variance[keep_trip].tolist())
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

    def _generate_annual_trip_schedule(
        self,
        profile_params: dict[tuple[str, int], VehicleProfile],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pl.DataFrame:
        """
        Generate an annual trip schedule for each vehicle based on sampled parameters.

        Args:
            profile_params: Dict of sampled trip profile parameters
            start_date: Start date for the schedule (overrides instance start_date)
            end_date: End date for the schedule (overrides instance end_date)

        Returns:
            DataFrame with daily departure/arrival times and miles driven for each vehicle
        """
        # Use instance dates if none provided
        start = start_date if start_date is not None else self.start_date
        end = end_date if end_date is not None else self.end_date

        if start is None or end is None:
            raise NoDateRangeError()

        def process_profile_with_seed(profile_and_index):
            profile, index = profile_and_index
            # Create unique seed for this profile
            profile_seed = self.random_state + index
            rng = np.random.RandomState(profile_seed)
            return self.generate_daily_schedules(profile, rng=rng)

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

    def generate_trip_schedules(self) -> pl.DataFrame:
        """
        Generate trip schedules for all vehicles in the metadata.
        """
        # assign cars to metadata buildings
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
        trip_schedules = self._generate_annual_trip_schedule(vehicle_profiles)

        return trip_schedules

    def generate_vehicle_presence_schedules(
        self,
        trip_schedules: pl.DataFrame,
        *,
        vehicle_keys: Iterable[tuple[str | int, int]] | None = None,
    ) -> dict[tuple[str | int, int], pl.DataFrame]:
        """
        Map each vehicle to an 8760-row hourly schedule of home/away status.

        Vehicles are available to charge whenever they are at home.
        """
        if self.start_date is None:
            raise NoDateRangeError()  # year anchor is required to build the 8760-hour calendar
        return generate_vehicle_presence_schedules(
            trip_schedules,
            self.start_date,  # use the same schedule year as trip generation
            vehicle_keys=vehicle_keys,  # optional explicit vehicle list, e.g. vehicle_profiles.keys()
        )

    def generate_vehicle_soc_schedules(
        self,
        trip_schedules: pl.DataFrame,
        *,
        vehicle_keys: Iterable[tuple[str | int, int]] | None = None,
        battery_capacity_kwh: float = DEFAULT_BATTERY_CAPACITY_KWH,
        kwh_per_mile: float = DEFAULT_KWH_PER_MILE,
        charger_power_kw: float = DEFAULT_LEVEL2_CHARGER_KW,
        ev_adoption_rate: float = 1.0,
        initial_soc_kwh: float | None = None,
    ) -> dict[tuple[str | int, int], pl.DataFrame]:
        """
        Map each vehicle to an 8760-row hourly SOC, charging, and discharge schedule.

        Vehicles start the year fully charged and recharge at ``charger_power_kw`` whenever they
        are home and below full capacity.
        """
        if self.start_date is None:
            raise NoDateRangeError()  # year anchor is required to build the 8760-hour calendar
        return generate_vehicle_soc_schedules(
            trip_schedules,
            self.start_date,  # use the same schedule year as trip generation
            vehicle_keys=vehicle_keys,  # optional explicit vehicle list, e.g. vehicle_profiles.keys()
            battery_capacity_kwh=battery_capacity_kwh,
            kwh_per_mile=kwh_per_mile,
            charger_power_kw=charger_power_kw,
            ev_adoption_rate=ev_adoption_rate,
            initial_soc_kwh=initial_soc_kwh,
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
    file_name = "trip_schedules"

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

        batch_trip_schedules = calculator.generate_trip_schedules()

        print(f"Completed batch {batch_number}: generated {len(batch_trip_schedules)} trip schedules")

        if args.upload_s3:
            # Upload batch directly to S3
            print(f"Uploading batch {batch_number} to S3...")
            upload_success = upload_batch_to_s3(batch_trip_schedules, config, file_name, batch_number)

            if not upload_success:
                print(f"Error: S3 upload failed for batch {batch_number}")
                return 1

            print(f"Successfully uploaded batch {batch_number} to S3")
            # Clear batch data to free memory
            del batch_trip_schedules
        else:
            # Keep batch for local saving
            all_trip_schedules.append(batch_trip_schedules)

    if args.upload_s3:
        print(f"All batches uploaded to S3 with partitioning: {file_name}/")
        logging.info(f"Uploaded all batches to S3 with partitioning: {file_name}/")
    else:
        # Combine all batches for local saving
        print("Combining all batches...")
        if all_trip_schedules:
            combined_trip_schedules = pl.concat(all_trip_schedules)
            logging.info(f"Combined all batches: {len(combined_trip_schedules)} total trip schedules")

            # Add metadata and sort in one chain
            final_trip_schedules = combined_trip_schedules.with_columns([
                pl.lit(config.release).alias("release"),
                pl.lit(config.state).alias("state"),
            ]).sort(["bldg_id", "vehicle_id", "date"])

            # Save locally (default behavior)
            if config.output_dir is None:
                raise ValueError("config.output_dir")
            os.makedirs(config.output_dir, exist_ok=True)
            local_file_path = f"{config.output_dir}/{file_name}"
            final_trip_schedules.write_parquet(local_file_path, partition_by=["release", "state"])

            print(f"Results written to: {local_file_path}")
            logging.info(f"Written results to {local_file_path}")
        else:
            logging.warning("No trip schedules generated")

    return 0


# # Example usage
# Example usage
if __name__ == "__main__":
    exit(main())
