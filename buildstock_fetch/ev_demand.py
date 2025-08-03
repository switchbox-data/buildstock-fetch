from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Optional

import numpy as np
import polars as pl

from buildstock_fetch.utils import ev_utils

BASEPATH: Final[Path] = Path(__file__).resolve().parents[1]


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


class WeatherDataError(Exception):
    """Raised when weather data is not loaded."""

    pass


@dataclass
class EVDemandConfig:
    state: str
    release: str
    metadata_path: Optional[str] = None
    pums_path: Optional[str] = None
    nhts_path: str = f"{BASEPATH}/utils/ev_data/inputs/NHTS_v2_1_trip_surveys.csv"
    weather_path: Optional[str] = None
    output_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.metadata_path is None:
            self.metadata_path = f"{Path(__file__).parent}/data/{self.release}/metadata/{self.state}/metadata.parquet"
        if self.pums_path is None:
            self.pums_path = f"{BASEPATH}/utils/ev_data/inputs/{self.state}_2021_pums_PUMA_HINCP_VEH_NP.csv"
        if self.weather_path is None:
            self.weather_path = f"{BASEPATH}//data/{self.release}/weather.csv"


@dataclass
class VehicleProfile:
    """Represents a vehicle's driving profile parameters."""

    building_id: int
    vehicle_id: int
    weekday_departure_hour: int
    weekday_arrival_hour: int
    weekday_miles: float
    weekend_departure_hour: int
    weekend_arrival_hour: int
    weekend_miles: float


@dataclass
class TripSchedule:
    """Represents a daily trip schedule for a vehicle."""

    building_id: int
    vehicle_id: int
    date: datetime
    departure_hour: int
    arrival_hour: int
    miles_driven: float
    avg_temp_while_away: float
    kwh_consumed: float


class EVDemandCalculator:
    """
    Calculator for EV demand based on ResStock metadata, PUMS data, and NHTS trip data.

    This class implements the workflow described in the methodology:
    1. Load ResStock metadata
    2. Fit vehicle ownership model using PUMS data
    3. Predict number of vehicles per household
    4. Sample vehicle driving profiles from NHTS
    5. Generate annual trip schedules
    6. Convert miles to kWh using temperature-dependent efficiency
    7. Assign battery capacities
    """

    def __init__(
        self,
        metadata_df: pl.DataFrame,
        nhts_df: Optional[pl.DataFrame] = None,
        weather_df: Optional[pl.DataFrame] = None,
    ):
        """
        Initialize the EV demand calculator.

        Args:
            metadata_df: ResStock metadata DataFrame
            nhts_df: NHTS trip data DataFrame (optional)
            weather_df: Weather data DataFrame (optional)
        """
        self.metadata_df = metadata_df
        self.nhts_df = nhts_df
        self.weather_df = weather_df
        self.vehicle_ownership_model: Optional[object] = None

        # Available battery capacities in kWh
        self.battery_capacities = [12, 40, 60, 90, 120]

        # Yuksel and Michalek (2015) polynomial coefficients for energy consumption
        # c(T) = sum(a_n * T^n) for n=0 to 5, units: kWh/mi/°F^n
        self.efficiency_coefficients = np.array([
            0.3950,  # a_0 (constant term)
            -0.0022,  # a_1 (linear term)
            9.1978e-5,  # a_2 (quadratic term)
            -3.9249e-6,  # a_3 (cubic term)
            5.2918e-8,  # a_4 (quartic term)
            -2.0659e-10,  # a_5 (quintic term)
        ])

    def fit_vehicle_ownership_model(self, pums_df: pl.DataFrame) -> object:
        """
        Fit a model to predict number of vehicles per household using PUMS data.

        Args:
            pums_df: DataFrame with PUMS household data, including 'occupants', 'inc', 'metro_area', 'num_vehicles'

        Returns:
            Trained model object
        """
        # TODO: Implement multinomial logistic regression or similar
        # This is a placeholder - replace with actual model fitting
        self.vehicle_ownership_model = "placeholder_model"
        return self.vehicle_ownership_model

    def predict_num_vehicles(self, metadata_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        Predict number of vehicles for each household in the metadata using the fitted model.

        Args:
            metadata_df: DataFrame with ResStock metadata. If None, uses self.metadata_df

        Returns:
            DataFrame with an added 'num_vehicles' column
        """
        df = metadata_df or self.metadata_df
        if df is None:
            raise MetadataDataFrameError()

        if self.vehicle_ownership_model is None:
            raise VehicleOwnershipModelError()

        # TODO: Implement actual prediction logic
        # This is a placeholder - replace with actual model prediction
        df = df.with_columns(pl.lit(1).alias("num_vehicles"))  # Placeholder: assume 1 vehicle per household
        return df

    def sample_vehicle_profiles(
        self, metadata_df: Optional[pl.DataFrame] = None
    ) -> dict[tuple[int, int], VehicleProfile]:
        """
        For each household and vehicle, sample weekday and weekend trip profiles from NHTS.

        Args:
            metadata_df: DataFrame with household and vehicle info. If None, uses self.metadata_df

        Returns:
            Dict mapping (building_id, vehicle_id) to sampled trip profile parameters
        """
        df = self.metadata_df
        if df is None:
            raise MetadataDataFrameError()

        if self.nhts_df is None:
            raise NHTSDataError()

        # TODO: Implement NHTS sampling logic
        # This is a placeholder - replace with actual sampling logic
        profiles = {}
        for row in df.iter_rows(named=True):
            building_id = row["building_id"]
            num_vehicles = row.get("num_vehicles", 1)

            for vehicle_id in range(num_vehicles):
                profiles[(building_id, vehicle_id)] = VehicleProfile(
                    building_id=building_id,
                    vehicle_id=vehicle_id,
                    weekday_departure_hour=8,
                    weekday_arrival_hour=18,
                    weekday_miles=30.0,
                    weekend_departure_hour=10,
                    weekend_arrival_hour=16,
                    weekend_miles=20.0,
                )

        return profiles

    def get_avg_temp_while_away(self, departure_hour: int, arrival_hour: int, date: datetime) -> float:
        """
        Calculate the average outdoor temperature during the hours the vehicle is away from home.

        Args:
            departure_hour: Hour vehicle leaves home
            arrival_hour: Hour vehicle returns home
            date: The date for which to compute the average

        Returns:
            Average temperature during vehicle absence
        """
        if self.weather_df is None:
            raise WeatherDataError()

        # TODO: Implement temperature calculation logic
        # This is a placeholder - replace with actual temperature calculation
        return 70.0  # Placeholder: assume 70°F average

    def miles_to_kwh(self, daily_miles: float, avg_temp: float) -> float:
        """
        Calculate daily electricity consumption for electric vehicles based on
        temperature and daily miles driven using the Yuksel and Michalek (2015) regression. @yuksel_EffectsRegionalTemperature_2015

        Args:
            daily_miles: Number of miles driven in a day
            avg_temp: Average outdoor temperature during driving hours (in °F)

        Returns:
            Daily electricity consumption in kWh
        """
        # Convert inputs to numpy arrays for vectorized operations
        temp = np.asarray(avg_temp)
        miles = np.asarray(daily_miles)

        # Apply temperature bounds as described in the paper
        temp_bounded = np.clip(temp, -15, 110)

        # Calculate energy consumption per mile using polynomial regression
        # c(T) = a_0 + a_1*T + a_2*T^2 + a_3*T^3 + a_4*T^4 + a_5*T^5
        # polyval expects coefficients in reverse order
        consumption_per_mile = np.polyval(self.efficiency_coefficients[::-1], temp_bounded)

        # Calculate total daily energy consumption
        daily_consumption_kwh = consumption_per_mile * miles

        # Return scalar if input was scalar
        if np.isscalar(daily_miles) and np.isscalar(avg_temp):
            return float(daily_consumption_kwh)
        return float(daily_consumption_kwh)

    def generate_annual_trip_schedule(
        self,
        profile_params: dict[tuple[int, int], VehicleProfile],
        start_date: str = "2022-01-01",
        end_date: str = "2022-12-31",
    ) -> pl.DataFrame:
        """
        Generate an annual trip schedule for each vehicle based on sampled parameters.

        Args:
            profile_params: Dict of sampled trip profile parameters
            start_date: Start date for the annual schedule
            end_date: End date for the annual schedule

        Returns:
            DataFrame with daily departure/arrival times and miles driven for each vehicle
        """
        # TODO: Implement annual schedule generation
        # This is a placeholder - replace with actual schedule generation logic
        schedules = []

        # Parse start date once
        start_datetime = datetime(2022, 1, 1)  # Fixed date for placeholder

        for (building_id, vehicle_id), profile in profile_params.items():
            # Placeholder: generate one day of data
            schedule = TripSchedule(
                building_id=building_id,
                vehicle_id=vehicle_id,
                date=start_datetime,
                departure_hour=profile.weekday_departure_hour,
                arrival_hour=profile.weekday_arrival_hour,
                miles_driven=profile.weekday_miles,
                avg_temp_while_away=self.get_avg_temp_while_away(
                    profile.weekday_departure_hour, profile.weekday_arrival_hour, start_datetime
                ),
                kwh_consumed=0.0,  # Will be calculated below
            )

            # Calculate kWh consumed
            schedule.kwh_consumed = self.miles_to_kwh(schedule.miles_driven, schedule.avg_temp_while_away)

            schedules.append(schedule)

        # Convert to DataFrame
        return pl.DataFrame([vars(schedule) for schedule in schedules])

    def assign_battery_capacity(self, daily_kwh: pl.Series) -> pl.Series:
        """
        Assign the minimum EV battery capacity that covers the max daily kWh plus a 20% buffer.

        Args:
            daily_kwh: Series of max daily kWh for each vehicle

        Returns:
            Series of assigned battery capacities (12, 40, 60, 90, 120 kWh)
        """
        # Calculate required capacity with 20% buffer
        required_capacity = daily_kwh * 1.2

        # Find the minimum battery capacity that meets the requirement
        battery_capacities = pl.Series(self.battery_capacities)
        assigned_capacities: list[int] = []

        for required in required_capacity:
            # Find the smallest battery that can handle the required capacity
            suitable_batteries = battery_capacities.filter(battery_capacities >= required)
            if len(suitable_batteries) > 0:
                assigned_capacities.append(int(suitable_batteries[0]))
            else:
                # If no battery is large enough, assign the largest available
                assigned_capacities.append(int(battery_capacities[-1]))

        return pl.Series(assigned_capacities)

    # def run_complete_workflow(
    #     self, pums_path: str, nhts_path: str, weather_path: str, output_dir: Optional[Path] = None
    # ) -> dict[str, Union[pl.DataFrame, pl.Series]]:
    #     """
    #     Run the complete EV demand workflow.

    #     Args:
    #         pums_path: Path to PUMS data file
    #         nhts_path: Path to NHTS data file
    #         weather_path: Path to weather data file
    #         output_dir: Directory to save output files

    #     Returns:
    #         Dictionary containing all output DataFrames and Series
    #     """
    #     # Step 1: Load metadata
    #     metadata_df = self.load_metadata()

    #     # Step 2: Load PUMS data and fit vehicle ownership model
    #     pums_df = pl.read_csv(pums_path)  # TODO: Adjust based on actual PUMS format
    #     self.fit_vehicle_ownership_model(pums_df)

    #     # Step 3: Predict number of vehicles
    #     metadata_with_vehicles = self.predict_num_vehicles(metadata_df)

    #     # Step 4: Load NHTS data
    #     self.load_nhts_data(nhts_path)

    #     # Step 5: Load weather data
    #     self.load_weather_data(weather_path)

    #     # Step 6: Sample vehicle profiles
    #     vehicle_profiles = self.sample_vehicle_profiles(metadata_with_vehicles)

    #     # Step 7: Generate annual trip schedules
    #     trip_schedules = self.generate_annual_trip_schedule(vehicle_profiles)

    #     # Step 8: Assign battery capacities
    #     max_daily_kwh = trip_schedules.group_by(["building_id", "vehicle_id"]).agg(pl.col("kwh_consumed").max())
    #     battery_capacities = self.assign_battery_capacity(max_daily_kwh.get_column("kwh_consumed"))

    #     # Save outputs if output_dir is provided
    #     if output_dir:
    #         output_dir = Path(output_dir)
    #         output_dir.mkdir(exist_ok=True)

    #         metadata_with_vehicles.write_csv(output_dir / "metadata_with_vehicles.csv")
    #         trip_schedules.write_csv(output_dir / "trip_schedules.csv")
    #         battery_capacities.to_frame().write_csv(output_dir / "battery_capacities.csv")

    #     return {
    #         "metadata_with_vehicles": metadata_with_vehicles,
    #         "trip_schedules": trip_schedules,
    #         "battery_capacities": battery_capacities,
    #     }


# Example usage
if __name__ == "__main__":
    # Step 1: Create configuration
    config = EVDemandConfig(state="NY", release="resstock_tmy3_release_1")

    # Step 2: Load all data
    metadata_df, nhts_df, pums_df, weather_df = ev_utils.load_all_input_data(config)
    print(f"✓ Loaded metadata: {len(metadata_df)} rows")
    print(f"✓ Loaded NHTS data: {len(nhts_df)} rows")
    print(f"✓ Loaded weather data: {len(weather_df)} rows")

    # # Step 3: Initialize calculator with data
    # calculator = EVDemandCalculator(
    #     metadata_df=metadata_df,
    #     nhts_df=nhts_df,
    #     weather_df=weather_df
    # )

    # print(f"✓ Initialized calculator")
    # print(f"  - Battery capacities: {calculator.battery_capacities}")
    # print(
    #     f"  - Efficiency coefficients: {len(calculator.efficiency_coefficients)} terms")

    # # Step 4: Run calculations (placeholder for now)
    # print("Ready to run EV demand calculations!")
