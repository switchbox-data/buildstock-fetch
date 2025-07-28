from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import polars as pl


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
    date: pl.Datetime
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

    def __init__(self, metadata_path: Optional[str] = None):
        """
        Initialize the EV demand calculator.

        Args:
            metadata_path: Path to the ResStock metadata parquet file
        """
        self.metadata_path = metadata_path
        self.metadata_df: Optional[pl.DataFrame] = None
        self.vehicle_ownership_model: Optional[object] = None
        self.nhts_df: Optional[pl.DataFrame] = None
        self.weather_df: Optional[pl.DataFrame] = None

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

    def load_metadata(self, metadata_path: Optional[str] = None) -> pl.DataFrame:
        """
        Load and parse the ResStock metadata parquet file.

        Args:
            metadata_path: Path to the metadata parquet file. If None, uses self.metadata_path

        Returns:
            DataFrame with columns including 'building_id', 'occupants', 'inc', 'metro_area' (PUMA), etc.
        """
        path = metadata_path or self.metadata_path
        if path is None:
            raise MetadataPathError()

        self.metadata_df = pl.read_parquet(path)
        return self.metadata_df

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

    def load_nhts_data(self, nhts_path: str) -> pl.DataFrame:
        """
        Load and preprocess the NHTS trip data.

        Args:
            nhts_path: Path to the NHTS trip data file

        Returns:
            DataFrame with trip records, including trip weights, household features, and trip parameters
        """
        # TODO: Implement NHTS data loading and preprocessing
        # This is a placeholder - replace with actual NHTS loading logic
        self.nhts_df = pl.DataFrame()  # Placeholder
        return self.nhts_df

    def load_weather_data(self, weather_path: str) -> pl.DataFrame:
        """
        Load hourly weather data (e.g., temperature) for a given location.

        Args:
            weather_path: Path to the weather data file (e.g., TMY3 CSV or EPW)

        Returns:
            DataFrame with at least columns ['datetime', 'temperature']
        """
        # TODO: Implement weather data loading
        # This is a placeholder - replace with actual weather loading logic
        self.weather_df = pl.DataFrame()  # Placeholder
        return self.weather_df

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
        df = metadata_df or self.metadata_df
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

    def get_avg_temp_while_away(self, departure_hour: int, arrival_hour: int, date: pl.Datetime) -> float:
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

        for (building_id, vehicle_id), profile in profile_params.items():
            # Placeholder: generate one day of data
            schedule = TripSchedule(
                building_id=building_id,
                vehicle_id=vehicle_id,
                date=pl.datetime(start_date),
                departure_hour=profile.weekday_departure_hour,
                arrival_hour=profile.weekday_arrival_hour,
                miles_driven=profile.weekday_miles,
                avg_temp_while_away=self.get_avg_temp_while_away(
                    profile.weekday_departure_hour, profile.weekday_arrival_hour, pl.datetime(start_date)
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

        return pl.Series(assigned_capacities, name=daily_kwh.name)

    def run_complete_workflow(
        self, pums_path: str, nhts_path: str, weather_path: str, output_dir: Optional[Path] = None
    ) -> dict[str, Union[pl.DataFrame, pl.Series]]:
        """
        Run the complete EV demand workflow.

        Args:
            pums_path: Path to PUMS data file
            nhts_path: Path to NHTS data file
            weather_path: Path to weather data file
            output_dir: Directory to save output files

        Returns:
            Dictionary containing all output DataFrames and Series
        """
        # Step 1: Load metadata
        metadata_df = self.load_metadata()

        # Step 2: Load PUMS data and fit vehicle ownership model
        pums_df = pl.read_csv(pums_path)  # TODO: Adjust based on actual PUMS format
        self.fit_vehicle_ownership_model(pums_df)

        # Step 3: Predict number of vehicles
        metadata_with_vehicles = self.predict_num_vehicles(metadata_df)

        # Step 4: Load NHTS data
        self.load_nhts_data(nhts_path)

        # Step 5: Load weather data
        self.load_weather_data(weather_path)

        # Step 6: Sample vehicle profiles
        vehicle_profiles = self.sample_vehicle_profiles(metadata_with_vehicles)

        # Step 7: Generate annual trip schedules
        trip_schedules = self.generate_annual_trip_schedule(vehicle_profiles)

        # Step 8: Assign battery capacities
        max_daily_kwh = trip_schedules.group_by(["building_id", "vehicle_id"]).agg(pl.col("kwh_consumed").max())
        battery_capacities = self.assign_battery_capacity(max_daily_kwh.get_column("kwh_consumed"))

        # Save outputs if output_dir is provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)

            metadata_with_vehicles.write_csv(output_dir / "metadata_with_vehicles.csv")
            trip_schedules.write_csv(output_dir / "trip_schedules.csv")
            battery_capacities.to_frame().write_csv(output_dir / "battery_capacities.csv")

        return {
            "metadata_with_vehicles": metadata_with_vehicles,
            "trip_schedules": trip_schedules,
            "battery_capacities": battery_capacities,
        }


# Example usage
if __name__ == "__main__":
    from pathlib import Path

    from main import BuildingID, fetch_bldg_data  # type: ignore[import-not-found]

    print("=== EV Demand Calculator Example ===")

    # Step 1: Download metadata using the main module
    print("1. Downloading ResStock metadata...")
    bldg_ids = [BuildingID(bldg_id=7), BuildingID(bldg_id=8), BuildingID(bldg_id=9)]
    output_dir = Path(__file__).parent / "data"

    try:
        downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, ("metadata",), output_dir)

        # Find the metadata file
        metadata_file = None
        for path in downloaded_paths:
            if "metadata" in str(path):
                metadata_file = path
                break

        if metadata_file is None:
            print("No metadata file found in downloaded paths")
            metadata_file = "path/to/metadata.parquet"  # Placeholder for example
    except Exception as e:
        print(f"Error downloading metadata: {e}")
        metadata_file = "path/to/metadata.parquet"  # Placeholder for example

    # Step 2: Initialize the EV demand calculator
    print("2. Initializing EV demand calculator...")
    calculator = EVDemandCalculator(str(metadata_file))

    # Step 3: Test the miles_to_kwh function
    print("3. Testing energy consumption calculation...")
    print("Based on Yuksel & Michalek (2015) Nissan Leaf regression\n")

    # Single temperature and mileage
    temp_f = 70  # 70°F
    daily_miles = 40  # 40 miles per day
    consumption = calculator.miles_to_kwh(temp_f, daily_miles)
    print(f"Temperature: {temp_f}°F")
    print(f"Daily miles: {daily_miles}")
    print(f"Daily consumption: {consumption:.2f} kWh")
    print(f"Consumption per mile: {consumption / daily_miles:.3f} kWh/mile\n")

    # Compare different temperatures
    temperatures = [20, 40, 60, 70, 80, 100]  # Various temperatures
    miles = 50  # Fixed daily mileage

    print("Temperature comparison (50 miles/day):")
    print("Temp (°F) | Consumption (kWh) | Per Mile (kWh/mi)")
    print("-" * 50)

    for temp in temperatures:
        consumption = calculator.miles_to_kwh(temp, miles)
        per_mile = consumption / miles
        print(f"{temp:8} | {consumption:13.2f} | {per_mile:11.3f}")

    # Step 4: Demonstrate complete workflow (with placeholder data paths)
    print("\n4. Complete workflow demonstration...")
    print("Note: This requires actual data files to run fully")

    try:
        # Example of how to run the complete workflow
        results = calculator.run_complete_workflow(
            pums_path="path/to/pums_data.csv",
            nhts_path="path/to/nhts_data.csv",
            weather_path="path/to/weather_data.csv",
            output_dir=output_dir / "ev_results",
        )

        print("Workflow completed successfully!")
        print(f"Generated {len(results['metadata_with_vehicles'])} household records")
        print(f"Generated {len(results['trip_schedules'])} trip schedule records")
        print(f"Assigned battery capacities to {len(results['battery_capacities'])} vehicles")

    except Exception as e:
        print(f"Workflow demonstration failed (expected with placeholder paths): {e}")
        print("To run the complete workflow, provide actual paths to:")
        print("- PUMS data file")
        print("- NHTS data file")
        print("- Weather data file")

    print("\n" + "=" * 60)
    print("Workflow Summary:")
    print("1. Download ResStock metadata using main.py")
    print("2. Load PUMS data and fit vehicle ownership model")
    print("3. Predict number of vehicles per household")
    print("4. Load NHTS trip data and sample driving profiles")
    print("5. Load weather data for temperature calculations")
    print("6. Generate annual trip schedules")
    print("7. Convert miles to kWh using temperature-dependent efficiency")
    print("8. Assign appropriate battery capacities")
    print("=" * 60)
