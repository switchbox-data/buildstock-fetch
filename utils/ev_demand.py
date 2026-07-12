import argparse
import logging
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Literal, cast, overload

import numpy as np
import polars as pl

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import ev_utils
from utils.ChargingSimulator import ChargingSimulator
from utils.NHTSProfileSampler import (
    NHTSDataError,
    NHTSProfileSampler,
    TripProfile,
    VehicleProfile,
    nhts_arrival_hour,
    nhts_departure_hour,
    summarize_nhts_match_catalog,
)
from utils.EVAdoptionSampler import EVAdoptionSampler
from utils.TripScheduleGenerator import (
    MAX_ARRIVAL_HOUR,
    MAX_DEPARTURE_HOUR,
    MIN_TRIP_AWAY_HOURS,
    TripScheduleGenerator,
)
from utils.VehicleOwnershipModel import VehicleOwnershipModel
from utils.charging import (
    ChargingStrategy,
    DEFAULT_PEAK_CLOCK_HOURS,
    DEFAULT_SHED_LOAD_PENALTY_USD_PER_KWH,
    DEFAULT_SOC_MIN_FRACTION,
    DEFAULT_SOC_SAFETY_BUFFER_FRACTION,
    build_hours_base,
    build_hourly_timestamps,
    build_is_off_peak,
    build_off_peak_charging_params,
    compute_hourly_soc,
    schedule_cost_minimizing_charging,
    schedule_immediate_charging,
    schedule_off_peak_charging,
)

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
    ev_ownership_path: str = (
        f"{BASEPATH}/ev_data/inputs/resstock_ev_reference/Electric_Vehicle_Ownership.tsv"
    )
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.metadata_path is None:
            self.metadata_path = f"{BASEPATH}/ev_data/inputs/{self.release}/metadata/{self.state}/metadata.parquet"
        if self.pums_path is None:
            self.pums_path = f"{BASEPATH}/ev_data/inputs/{self.state}_2021_pums_PUMA_HINCP_VEH_NP.csv"
        if self.output_dir is None:
            self.output_dir = Path(f"{BASEPATH}/ev_data/outputs/{self.state}_{self.release}")


HOURS_PER_YEAR = 8760  # standard hourly load-curve length (365 days x 24 hours)
DEFAULT_BATTERY_CAPACITY_KWH = 90.0  # uniform fleet battery assumption for SOC modeling
DEFAULT_KWH_PER_MILE = 0.30  # simple-model assumption
DEFAULT_LEVEL2_CHARGER_KW = 7.2  # typical 32 A @ 240 V residential Level 2 charger


class EVDemandCalculator:
    """
    Facade over the EV demand pipeline components.

    Constructs ``VehicleOwnershipModel``, ``EVAdoptionSampler``, ``NHTSProfileSampler``,
    ``TripScheduleGenerator``, and ``ChargingSimulator``, then exposes the original
    ``EVDemandCalculator`` public API as thin delegations.

    ``match_and_generate_trip_schedules()`` remains the end-to-end orchestrator:
    predict EV adoption → sample NHTS profiles → generate daily trip schedules.
    """

    def __init__(
        self,
        metadata_df: pl.DataFrame,
        nhts_df: pl.DataFrame,
        ev_ownership_df: pl.DataFrame,
        start_date: datetime,
        end_date: datetime,
        pums_df: pl.DataFrame | None = None,
        max_vehicles: int = 2,
        random_state: int = 42,
        max_workers: int | None = None,
    ):
        """
        Initialize the EV demand calculator facade and its pipeline components.

        Args:
            metadata_df: ResStock metadata DataFrame
            nhts_df: NHTS trip data DataFrame
            ev_ownership_df: NREL EV ownership lookup (from load_ev_ownership_lookup)
            start_date: Start date for trip generation
            end_date: End date for trip generation
            pums_df: PUMS data DataFrame; required only for ``predict_num_vehicles()``
            max_vehicles: Maximum number of vehicles per household when fitting the PUMS model
            random_state: Random seed for reproducible results
            max_workers: Maximum number of worker threads for parallel execution (None = use all cores)
        """
        np.random.seed(random_state)

        # Shared inputs retained for orchestration and backward-compatible attribute access.
        self.metadata_df = metadata_df
        self.nhts_df = nhts_df
        self.pums_df = pums_df
        self.ev_ownership_df = ev_ownership_df
        self.start_date = start_date
        self.end_date = end_date
        self.max_vehicles = max_vehicles
        self.random_state = random_state
        self.max_workers = max_workers
        self.num_days = (self.end_date - self.start_date).days + 1
        self.num_hours = self.num_days * 24

        # Pipeline components.
        self.vehicle_ownership = VehicleOwnershipModel(
            max_vehicles=max_vehicles,
            random_state=random_state,
        )
        self.ev_adoption_sampler = EVAdoptionSampler(
            ev_ownership_df=ev_ownership_df,
            random_state=random_state,
        )
        self.nhts_sampler = NHTSProfileSampler(
            nhts_df=nhts_df,
            max_vehicles=max_vehicles,
            random_state=random_state,
        )
        self.trip_schedule_generator = TripScheduleGenerator(
            start_date=start_date,
            end_date=end_date,
            random_state=random_state,
            max_workers=max_workers,
        )
        self.charging_simulator = ChargingSimulator(
            start_date=start_date,
            end_date=end_date,
        )

    # --- Vehicle ownership (delegates to VehicleOwnershipModel) ---

    @property
    def veh_assign_features(self) -> list[str]:
        return self.vehicle_ownership.feature_columns

    @property
    def vehicle_ownership_model(self) -> Any | None:
        return self.vehicle_ownership.vehicle_ownership_model

    @vehicle_ownership_model.setter
    def vehicle_ownership_model(self, value: Any | None) -> None:
        self.vehicle_ownership.vehicle_ownership_model = value

    @property
    def label_encoders(self) -> dict[str, Any]:
        return self.vehicle_ownership.label_encoders

    @property
    def target_encoder(self) -> Any | None:
        return self.vehicle_ownership.target_encoder

    @property
    def scaler(self) -> Any | None:
        return self.vehicle_ownership.scaler

    def fit_vehicle_ownership_model(self, pums_df: pl.DataFrame) -> Any:
        return self.vehicle_ownership.fit(pums_df)

    def predict_num_vehicles(self, metadata_df: pl.DataFrame | None = None) -> pl.DataFrame:
        df = self.metadata_df if metadata_df is None else metadata_df
        if df is None:
            raise MetadataDataFrameError()

        if self.vehicle_ownership.vehicle_ownership_model is None:
            if self.pums_df is None:
                raise ValueError(
                    "pums_df is required to fit the vehicle ownership model. "
                    "Pass pums_df to EVDemandCalculator or call fit_vehicle_ownership_model() first."
                )
            logging.info("Vehicle ownership model not fitted yet. Fitting model...")
            self.vehicle_ownership.fit(self.pums_df)

        return self.vehicle_ownership.predict(df)

    # --- EV adoption (delegates to EVAdoptionSampler) ---

    def predict_num_EVs(self, metadata_df: pl.DataFrame | None = None) -> pl.DataFrame:
        df = self.metadata_df if metadata_df is None else metadata_df
        if df is None:
            raise MetadataDataFrameError()
        return self.ev_adoption_sampler.sample(df)

    # --- NHTS profile sampling (delegates to NHTSProfileSampler) ---

    def find_best_matches(
        self, target_income: int, target_occupants: int, target_vehicles: int, num_samples: int, *, weekday: bool = True
    ) -> tuple[str, list[str]]:
        return self.nhts_sampler.find_best_matches(
            target_income=target_income,
            target_occupants=target_occupants,
            target_vehicles=target_vehicles,
            num_samples=num_samples,
            weekday=weekday,
        )

    @overload
    def sample_vehicle_profiles(
        self,
        bldg_veh_df: pl.DataFrame,
        nhts_df: pl.DataFrame | None = None,
        *,
        return_catalog: Literal[False] = False,
    ) -> dict[tuple[str, int], VehicleProfile]: ...

    @overload
    def sample_vehicle_profiles(
        self,
        bldg_veh_df: pl.DataFrame,
        nhts_df: pl.DataFrame | None = None,
        *,
        return_catalog: Literal[True],
    ) -> tuple[dict[tuple[str, int], VehicleProfile], pl.DataFrame]: ...

    def sample_vehicle_profiles(
        self,
        bldg_veh_df: pl.DataFrame,
        nhts_df: pl.DataFrame | None = None,
        *,
        return_catalog: bool = False,
    ) -> dict[tuple[str, int], VehicleProfile] | tuple[dict[tuple[str, int], VehicleProfile], pl.DataFrame]:
        if bldg_veh_df is None:
            raise MetadataDataFrameError()

        if nhts_df is None:
            nhts_df = self.nhts_df

        return self.nhts_sampler.sample_vehicle_profiles(
            bldg_veh_df,
            nhts_df,
            return_catalog=return_catalog,
        )

    # --- Trip schedule generation (delegates to TripScheduleGenerator) ---

    @staticmethod
    def _normalize_day_trip_times(
        departures: np.ndarray,
        arrival_hours: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return TripScheduleGenerator._normalize_day_trip_times(departures, arrival_hours)

    def _generate_vehicle_daily_trip_schedules(
        self, profile: VehicleProfile, rng: np.random.RandomState | None = None
    ) -> pl.DataFrame:
        return self.trip_schedule_generator._generate_vehicle_daily_trip_schedules(profile, rng=rng)

    def generate_daily_trip_schedules(
        self,
        profile_params: dict[tuple[str, int], VehicleProfile],
    ) -> pl.DataFrame:
        return self.trip_schedule_generator.generate(profile_params)

    def match_and_generate_trip_schedules(self) -> pl.DataFrame:
        """
        Generate trip schedules for all buildings in the metadata.

        Uses the vehicle ownership model to assign vehicles to buildings and then generates trip schedules for each vehicle.

        Args:
            None

        Returns:
            pl.DataFrame: DataFrame of trip schedules for all buildings
        """
        # EV adoption (0/1 per household) drives which buildings get trip profiles.
        # predict_num_vehicles() is kept separately for total-vehicle-count analysis.
        logging.info("Predicting EV ownership for metadata buildings")
        bldg_ev_df = self.predict_num_EVs()
        bldg_veh_df = bldg_ev_df.with_columns(pl.col("evs").alias("vehicles"))
        logging.info("Assigning vehicle profiles")
        vehicle_profiles = cast(
            dict[tuple[str, int], VehicleProfile],
            self.sample_vehicle_profiles(bldg_veh_df),
        )

        # Generate trip schedules for each vehicle
        logging.info("Generating trip schedules")
        trip_schedules = self.generate_daily_trip_schedules(vehicle_profiles)

        return trip_schedules

    # --- Charging / SOC (delegates to ChargingSimulator and charging.py) ---

    def _build_hourly_timestamps(self) -> pl.DataFrame:
        return build_hourly_timestamps(self.start_date, self.end_date)

    def _build_hours_base(self) -> pl.DataFrame:
        if self.start_date is None or self.end_date is None:
            raise NoDateRangeError()
        return build_hours_base(self.start_date, self.end_date)

    def generate_vehicle_presence_schedules(
        self,
        trip_schedules: pl.DataFrame,
        *,
        hours_base: pl.DataFrame | None = None,
        vehicle_keys: Iterable[tuple[str | int, int]] | None = None,
    ) -> dict[tuple[str | int, int], pl.DataFrame]:
        return self.charging_simulator.generate_presence(
            trip_schedules,
            hours_base=hours_base,
            vehicle_keys=vehicle_keys,
        )

    @staticmethod
    def _build_hourly_discharge_kwh(
        trip_schedules: pl.DataFrame,
        hours_base: pl.DataFrame,
        *,
        kwh_per_mile: float,
        ev_adoption_rate: float,
    ) -> pl.DataFrame:
        return ChargingSimulator._build_hourly_discharge_kwh(
            trip_schedules,
            hours_base,
            kwh_per_mile=kwh_per_mile,
            ev_adoption_rate=ev_adoption_rate,
        )

    @staticmethod
    def _schedule_immediate_charging(
        at_home: np.ndarray,
        discharge_kwh: np.ndarray,
        *,
        battery_capacity_kwh: float,
        charger_power_kw: float,
        initial_soc_kwh: float,
    ) -> np.ndarray:
        return schedule_immediate_charging(
            at_home,
            discharge_kwh,
            battery_capacity_kwh=battery_capacity_kwh,
            charger_power_kw=charger_power_kw,
            initial_soc_kwh=initial_soc_kwh,
        )

    @staticmethod
    def _schedule_cost_minimizing_charging(
        at_home: np.ndarray,
        discharge_kwh: np.ndarray,
        *,
        battery_capacity_kwh: float,
        charger_power_kw: float,
        initial_soc_kwh: float,
        hourly_price_usd_per_kwh: np.ndarray,
        shed_load_penalty_usd_per_kwh: float | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return schedule_cost_minimizing_charging(
            at_home,
            discharge_kwh,
            battery_capacity_kwh=battery_capacity_kwh,
            charger_power_kw=charger_power_kw,
            initial_soc_kwh=initial_soc_kwh,
            hourly_price_usd_per_kwh=hourly_price_usd_per_kwh,
            shed_load_penalty_usd_per_kwh=shed_load_penalty_usd_per_kwh,
        )

    @staticmethod
    def _build_is_off_peak(
        hours_base: pl.DataFrame,
        *,
        peak_clock_hours: Iterable[int] = DEFAULT_PEAK_CLOCK_HOURS,
    ) -> np.ndarray:
        return build_is_off_peak(hours_base, peak_clock_hours=peak_clock_hours)

    @staticmethod
    def _build_off_peak_charging_params(
        at_home: np.ndarray,
        discharge_kwh: np.ndarray,
        hours_base: pl.DataFrame,
        vehicle_trips: pl.DataFrame,
        *,
        battery_capacity_kwh: float,
        is_off_peak: np.ndarray,
        soc_min_fraction: float = DEFAULT_SOC_MIN_FRACTION,
        soc_safety_buffer_fraction: float = DEFAULT_SOC_SAFETY_BUFFER_FRACTION,
    ) -> tuple[np.ndarray, np.ndarray]:
        return build_off_peak_charging_params(
            at_home,
            discharge_kwh,
            hours_base,
            vehicle_trips,
            battery_capacity_kwh=battery_capacity_kwh,
            is_off_peak=is_off_peak,
            soc_min_fraction=soc_min_fraction,
            soc_safety_buffer_fraction=soc_safety_buffer_fraction,
        )

    @staticmethod
    def _schedule_off_peak_charging(
        at_home: np.ndarray,
        discharge_kwh: np.ndarray,
        *,
        charge_allowed: np.ndarray,
        soc_target_kwh: np.ndarray,
        battery_capacity_kwh: float,
        charger_power_kw: float,
        initial_soc_kwh: float,
    ) -> np.ndarray:
        return schedule_off_peak_charging(
            at_home,
            discharge_kwh,
            charge_allowed=charge_allowed,
            soc_target_kwh=soc_target_kwh,
            battery_capacity_kwh=battery_capacity_kwh,
            charger_power_kw=charger_power_kw,
            initial_soc_kwh=initial_soc_kwh,
        )

    @staticmethod
    def _compute_hourly_soc(
        discharge_kwh: np.ndarray,
        charge_kwh: np.ndarray,
        *,
        initial_soc_kwh: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        return compute_hourly_soc(discharge_kwh, charge_kwh, initial_soc_kwh=initial_soc_kwh)

    def generate_soc_schedules(
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
        shed_load_penalty_usd_per_kwh: float | np.ndarray | None = None,
        peak_clock_hours: Iterable[int] = DEFAULT_PEAK_CLOCK_HOURS,
        soc_min_fraction: float = DEFAULT_SOC_MIN_FRACTION,
        soc_safety_buffer_fraction: float = DEFAULT_SOC_SAFETY_BUFFER_FRACTION,
    ) -> pl.DataFrame:
        return self.charging_simulator.generate_soc(
            trip_schedules,
            vehicle_keys=vehicle_keys,
            hours_base=hours_base,
            presence_by_vehicle=presence_by_vehicle,
            battery_capacity_kwh=battery_capacity_kwh,
            kwh_per_mile=kwh_per_mile,
            charger_power_kw=charger_power_kw,
            ev_adoption_rate=ev_adoption_rate,
            initial_soc_kwh=initial_soc_kwh,
            charging_strategy=charging_strategy,
            hourly_price_usd_per_kwh=hourly_price_usd_per_kwh,
            shed_load_penalty_usd_per_kwh=shed_load_penalty_usd_per_kwh,
            peak_clock_hours=peak_clock_hours,
            soc_min_fraction=soc_min_fraction,
            soc_safety_buffer_fraction=soc_safety_buffer_fraction,
        )

    def generate_vehicle_soc_schedules(
        self,
        trip_schedules: pl.DataFrame,
        **kwargs: Any,
    ) -> pl.DataFrame:
        return self.generate_soc_schedules(trip_schedules, **kwargs)

    @staticmethod
    def vehicle_hourly_schedules_to_dataframe(
        schedules_by_vehicle: dict[tuple[str | int, int], pl.DataFrame],
    ) -> pl.DataFrame:
        return ChargingSimulator.to_dataframe(schedules_by_vehicle)


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

    # Step 2: Load all data (lookup loaded once and shared across batches)
    metadata_df, nhts_df, pums_df, ev_ownership_df = ev_utils.load_all_input_data(config)
    print(f"Loaded metadata: {len(metadata_df)} rows")
    print(f"Loaded NHTS data: {len(nhts_df)} rows")
    print(f"Loaded PUMS data: {len(pums_df)} rows")
    print(f"Loaded EV ownership lookup: {ev_ownership_df.height:,} rows")
    state_ev_rate = ev_utils.state_ev_ownership_rate(ev_ownership_df, config.state)
    print(
        f"PUMS-weighted mean P(EV) over occupied lookup segments ({config.state}): {state_ev_rate:.4f}"
    )

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
            ev_ownership_df=ev_ownership_df,
            start_date=start_date,
            end_date=end_date,
            # pums_df=pums_df,
            max_workers=8,  # Use worker threads for parallel processing
        )

        batch_trip_schedules = calculator.match_and_generate_trip_schedules()
        batch_soc_schedules = calculator.generate_soc_schedules(batch_trip_schedules)

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
