import argparse
import logging
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, cast

import numpy as np
import polars as pl

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import ev_utils
from utils.ChargingSimulator import DEFAULT_LEVEL2_CHARGER_KW, ChargingSimulator
from utils.EVAdoptionSampler import EVAdoptionSampler
from utils.EVBatteryAssigner import EVBatteryAssigner
from utils.NHTSProfileSampler import NHTSProfileSampler, VehicleProfile
from utils.TripScheduleGenerator import TripScheduleGenerator
from utils.VehicleOwnershipModel import VehicleOwnershipModel
from utils.charging import (
    ChargingStrategy,
    DEFAULT_PEAK_CLOCK_HOURS,
    DEFAULT_SOC_MIN_FRACTION,
    DEFAULT_SOC_SAFETY_BUFFER_FRACTION,
)

BASEPATH: Final[Path] = Path(__file__).resolve().parent  # just one level up

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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
    # ResStock national BEV class × range shares (housing-characteristic TSV).
    ev_battery_path: str = (
        f"{BASEPATH}/ev_data/inputs/resstock_ev_reference/Electric_Vehicle_Battery.tsv"
    )
    # Autonomie usable capacity (kWh) + efficiency (kWh/mi) keyed by the same option names.
    ev_autonomie_path: str = (
        f"{BASEPATH}/ev_data/inputs/resstock_ev_reference/resstock_autonomie_2022_vehicle_params.csv"
    )
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.metadata_path is None:
            self.metadata_path = f"{BASEPATH}/ev_data/inputs/{self.release}/metadata/{self.state}/metadata.parquet"
        if self.pums_path is None:
            self.pums_path = f"{BASEPATH}/ev_data/inputs/{self.state}_2021_pums_PUMA_HINCP_VEH_NP.csv"
        if self.output_dir is None:
            self.output_dir = Path(f"{BASEPATH}/ev_data/outputs/{self.state}_{self.release}")


class EVDemandCalculator:
    """
    Orchestrator for the EV demand pipeline.

    Constructs ``VehicleOwnershipModel``, ``EVAdoptionSampler``, ``EVBatteryAssigner``,
    ``NHTSProfileSampler``, ``TripScheduleGenerator``, and ``ChargingSimulator``.

    Public API:
    - ``match_and_generate_trip_schedules()`` — EV adoption → NHTS profiles →
      daily trip schedules → ResStock battery attrs
    - ``generate_soc_schedules()`` — hourly SOC / charge / discharge from trips
    """

    def __init__(
        self,
        metadata_df: pl.DataFrame,
        nhts_df: pl.DataFrame,
        ev_ownership_df: pl.DataFrame,
        ev_battery_df: pl.DataFrame,
        ev_autonomie_df: pl.DataFrame,
        start_date: datetime,
        end_date: datetime,
        pums_df: pl.DataFrame | None = None,
        max_vehicles: int = 2,
        match_on_vehicles: bool = False,
        random_state: int = 42,
        max_workers: int | None = None,
    ):
        """
        Initialize the EV demand calculator and its pipeline components.

        Args:
            metadata_df: ResStock metadata DataFrame
            nhts_df: NHTS trip data DataFrame
            ev_ownership_df: NREL EV ownership lookup (from load_ev_ownership_lookup)
            ev_battery_df: ResStock EV battery option shares (from load_ev_battery_lookup)
            ev_autonomie_df: Autonomie capacity / efficiency params (from load_ev_autonomie_params)
            start_date: Start date for trip generation
            end_date: End date for trip generation
            pums_df: PUMS data DataFrame (optional; used with ``vehicle_ownership``)
            max_vehicles: Maximum number of vehicles per household when fitting the PUMS model
            match_on_vehicles: If True, include household vehicle count in NHTS profile matching.
                Defaults to False for the max-1-EV model.
            random_state: Random seed for reproducible results
            max_workers: Maximum number of worker threads for parallel execution (None = use all cores)
        """
        np.random.seed(random_state)

        self.metadata_df = metadata_df
        self.nhts_df = nhts_df
        self.pums_df = pums_df
        self.ev_ownership_df = ev_ownership_df
        self.ev_battery_df = ev_battery_df
        self.ev_autonomie_df = ev_autonomie_df
        self.start_date = start_date
        self.end_date = end_date
        self.max_vehicles = max_vehicles
        self.random_state = random_state
        self.max_workers = max_workers
        # Filled by match_and_generate_trip_schedules(); consumed by generate_soc_schedules().
        self.ev_attributes: pl.DataFrame | None = None

        # Pipeline components.
        self.vehicle_ownership = VehicleOwnershipModel(
            max_vehicles=max_vehicles,
            random_state=random_state,
        )
        self.ev_adoption_sampler = EVAdoptionSampler(
            ev_ownership_df=ev_ownership_df,
            random_state=random_state,
        )
        self.battery_assigner = EVBatteryAssigner(
            option_probabilities=ev_battery_df,
            autonomie_params=ev_autonomie_df,
            random_state=random_state,
        )
        self.nhts_sampler = NHTSProfileSampler(
            nhts_df=nhts_df,
            max_vehicles=max_vehicles,
            match_on_vehicles=match_on_vehicles,
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

    @staticmethod
    def _vehicle_slots_from_building_evs(bldg_veh_df: pl.DataFrame) -> pl.DataFrame:
        """Expand buildings with ``vehicles`` > 0 into one row per ``(bldg_id, vehicle_id)``.

        ``vehicle_id`` is 1-based within each building, matching NHTS / trip schedule slots.
        In the current max-1-EV adoption model, ``vehicles`` is usually 0 or 1.

        Args:
            bldg_veh_df (pl.DataFrame): The building vehicle DataFrame

        Returns:
            pl.DataFrame: The vehicle slots DataFrame expanded from the building vehicle DataFrame
        """
        if "bldg_id" not in bldg_veh_df.columns or "vehicles" not in bldg_veh_df.columns:
            raise ValueError("bldg_veh_df must include bldg_id and vehicles columns")

        occupied = bldg_veh_df.filter(pl.col("vehicles") > 0)
        if occupied.is_empty():
            return pl.DataFrame(
                schema={
                    "bldg_id": bldg_veh_df.schema.get("bldg_id", pl.Int64),
                    "vehicle_id": pl.Int64,
                }
            )

        return (
            occupied.select("bldg_id", "vehicles")
            .with_columns(
                pl.int_ranges(1, pl.col("vehicles") + 1).alias("vehicle_id"),
            )
            .explode("vehicle_id")
            .select("bldg_id", pl.col("vehicle_id").cast(pl.Int64))
        )

    def match_and_generate_trip_schedules(self) -> pl.DataFrame:
        """
        Generate trip schedules for all buildings in the metadata.

        Uses EV adoption sampling to assign EVs, NHTS profiles for travel behavior,
        generates trip schedules, then assigns ResStock battery attributes
        conditioned on each vehicle's peak daily miles.

        Side effect:
            Sets ``self.ev_attributes`` to the per-vehicle ResStock battery assignment table.

        Returns:
            pl.DataFrame: DataFrame of trip schedules for all buildings
        """
        logging.info("Predicting EV ownership for metadata buildings")
        bldg_ev_df = self.ev_adoption_sampler.sample(self.metadata_df)
        # NHTS sampler still expects a ``vehicles`` column (count of EV slots to fill).
        bldg_veh_df = bldg_ev_df.with_columns(pl.col("evs").alias("vehicles"))

        logging.info("Assigning vehicle profiles")
        vehicle_profiles = cast(
            dict[tuple[str, int], VehicleProfile],
            self.nhts_sampler.sample(bldg_veh_df),
        )

        logging.info("Generating trip schedules")
        trip_schedules = self.trip_schedule_generator.generate(vehicle_profiles)

        logging.info("Assigning ResStock EV battery attributes (stock-conditional)")
        vehicle_slots = self._vehicle_slots_from_building_evs(bldg_veh_df)
        max_miles = TripScheduleGenerator.max_daily_miles_from_trip_schedules(trip_schedules)
        vehicle_duty = (
            vehicle_slots.join(max_miles, on=["bldg_id", "vehicle_id"], how="left")
            .with_columns(pl.col("max_daily_miles").fill_null(0.0))
        )
        self.ev_attributes = self.battery_assigner.assign(vehicle_duty)
        logging.info(
            "Assigned battery attributes for %s EV vehicle slot(s)",
            self.ev_attributes.height,
        )

        return trip_schedules

    def generate_soc_schedules(
        self,
        trip_schedules: pl.DataFrame,
        *,
        vehicle_keys: Iterable[tuple[str | int, int]] | None = None,
        hours_base: pl.DataFrame | None = None,
        presence_by_vehicle: dict[tuple[str | int, int], pl.DataFrame] | None = None,
        ev_attributes: pl.DataFrame | None = None,
        charger_power_kw: float = DEFAULT_LEVEL2_CHARGER_KW,
        initial_soc_kwh: float | None = None,
        charging_strategy: ChargingStrategy = "immediate",
        hourly_price_usd_per_kwh: np.ndarray | None = None,
        shed_load_penalty_usd_per_kwh: float | np.ndarray | None = None,
        peak_clock_hours: Iterable[int] = DEFAULT_PEAK_CLOCK_HOURS,
        soc_min_fraction: float = DEFAULT_SOC_MIN_FRACTION,
        soc_safety_buffer_fraction: float = DEFAULT_SOC_SAFETY_BUFFER_FRACTION,
    ) -> pl.DataFrame:
        """Generate hourly SOC / charge / discharge schedules from trip schedules.

        Prefers explicitly passed ``ev_attributes``; otherwise uses attributes stored by
        ``match_and_generate_trip_schedules()``.
        """
        attrs = self.ev_attributes if ev_attributes is None else ev_attributes
        if attrs is None or attrs.is_empty():
            raise ValueError(
                "ev_attributes is required for SOC schedules. "
                "Call match_and_generate_trip_schedules() first or pass ev_attributes=..."
            )
        return self.charging_simulator.generate_soc(
            trip_schedules,
            vehicle_keys=vehicle_keys,
            hours_base=hours_base,
            presence_by_vehicle=presence_by_vehicle,
            ev_attributes=attrs,
            charger_power_kw=charger_power_kw,
            initial_soc_kwh=initial_soc_kwh,
            charging_strategy=charging_strategy,
            hourly_price_usd_per_kwh=hourly_price_usd_per_kwh,
            shed_load_penalty_usd_per_kwh=shed_load_penalty_usd_per_kwh,
            peak_clock_hours=peak_clock_hours,
            soc_min_fraction=soc_min_fraction,
            soc_safety_buffer_fraction=soc_safety_buffer_fraction,
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

    # Step 2: Load all data (lookups loaded once and shared across batches)
    (
        metadata_df,
        nhts_df,
        pums_df,
        ev_ownership_df,
        ev_battery_df,
        ev_autonomie_df,
    ) = ev_utils.load_all_input_data(config)
    print(f"Loaded metadata: {len(metadata_df)} rows")
    print(f"Loaded NHTS data: {len(nhts_df)} rows")
    print(f"Loaded PUMS data: {len(pums_df)} rows")
    print(f"Loaded EV ownership lookup: {ev_ownership_df.height:,} rows")
    print(f"Loaded EV battery options: {ev_battery_df.height:,} rows")
    print(f"Loaded Autonomie vehicle params: {ev_autonomie_df.height:,} rows")
    state_ev_rate = ev_utils.state_ev_ownership_rate(ev_ownership_df, config.state)
    print(
        f"PUMS-weighted mean P(EV) over occupied lookup segments ({config.state}): {state_ev_rate:.4f}"
    )

    # Process metadata in batches of 20,000 rows
    batch_size = 20000
    total_rows = len(metadata_df)
    all_trip_schedules = []
    all_soc_schedules = []
    all_ev_attributes = []  # per-batch ResStock battery assignment tables
    trip_file_name = "trip_schedules"
    soc_file_name = "vehicle_soc_schedules"
    attrs_file_name = "ev_attributes"  # written alongside trips / SOC

    for i in range(0, total_rows, batch_size):
        batch_end = min(i + batch_size, total_rows)
        batch_metadata = metadata_df[i:batch_end]
        batch_number = i // batch_size + 1

        print(f"Processing batch {batch_number}: rows {i + 1} to {batch_end} ({len(batch_metadata)} rows)")

        calculator = EVDemandCalculator(
            metadata_df=batch_metadata,
            nhts_df=nhts_df,
            ev_ownership_df=ev_ownership_df,
            ev_battery_df=ev_battery_df,
            ev_autonomie_df=ev_autonomie_df,
            start_date=start_date,
            end_date=end_date,
            # pums_df=pums_df,
            max_workers=8,  # Use worker threads for parallel processing
        )

        # Side effect: calculator.ev_attributes is set during this call.
        batch_trip_schedules = calculator.match_and_generate_trip_schedules()
        batch_ev_attributes = calculator.ev_attributes
        # SOC uses the attributes just assigned (per-vehicle capacity + kWh/mi).
        batch_soc_schedules = calculator.generate_soc_schedules(batch_trip_schedules)

        print(f"Completed batch {batch_number}: generated {len(batch_trip_schedules)} trip schedules")
        print(
            f"Completed batch {batch_number}: assigned "
            f"{0 if batch_ev_attributes is None else len(batch_ev_attributes)} EV battery attributes"
        )
        print(f"Completed batch {batch_number}: generated {len(batch_soc_schedules)} hourly SOC rows")

        if args.upload_s3:
            # Upload batch directly to S3
            print(f"Uploading batch {batch_number} to S3...")
            trip_upload_success = upload_batch_to_s3(batch_trip_schedules, config, trip_file_name, batch_number)
            soc_upload_success = upload_batch_to_s3(batch_soc_schedules, config, soc_file_name, batch_number)
            attrs_upload_success = True
            # Battery attrs are sparse (one row per EV); skip empty batches.
            if batch_ev_attributes is not None and batch_ev_attributes.height > 0:
                attrs_upload_success = upload_batch_to_s3(
                    batch_ev_attributes, config, attrs_file_name, batch_number
                )

            if not trip_upload_success or not soc_upload_success or not attrs_upload_success:
                print(f"Error: S3 upload failed for batch {batch_number}")
                return 1

            print(f"Successfully uploaded batch {batch_number} to S3")
            # Clear batch data to free memory
            del batch_trip_schedules
            del batch_soc_schedules
            del batch_ev_attributes
        else:
            # Keep batch for local saving
            all_trip_schedules.append(batch_trip_schedules)
            all_soc_schedules.append(batch_soc_schedules)
            if batch_ev_attributes is not None and batch_ev_attributes.height > 0:
                all_ev_attributes.append(batch_ev_attributes)

    if args.upload_s3:
        print(
            f"All batches uploaded to S3 with partitioning: "
            f"{trip_file_name}/, {soc_file_name}/, and {attrs_file_name}/"
        )
        logging.info(
            f"Uploaded all batches to S3 with partitioning: "
            f"{trip_file_name}/, {soc_file_name}/, and {attrs_file_name}/"
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

        # Persist sampled battery class / capacity / efficiency for QA and later type-matching.
        if all_ev_attributes:
            combined_ev_attributes = pl.concat(all_ev_attributes)
            logging.info(f"Combined all batches: {len(combined_ev_attributes)} total EV attribute rows")

            final_ev_attributes = combined_ev_attributes.with_columns([
                pl.lit(config.release).alias("release"),
                pl.lit(config.state).alias("state"),
            ]).sort(["bldg_id", "vehicle_id"])

            local_attrs_path = f"{config.output_dir}/{attrs_file_name}"
            final_ev_attributes.write_parquet(local_attrs_path, partition_by=["release", "state"])

            print(f"EV attributes written to: {local_attrs_path}")
            logging.info(f"Written EV attributes to {local_attrs_path}")
        else:
            logging.warning("No EV attributes generated")
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


if __name__ == "__main__":
    exit(main())
