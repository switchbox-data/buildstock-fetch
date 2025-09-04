import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final, Optional

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.preprocessing import LabelEncoder, StandardScaler  # type: ignore[import-untyped]

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import ev_utils

BASEPATH: Final[Path] = Path(__file__).resolve().parent  # just one level up


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


class InsufficientVehiclesError(Exception):
    """Raised when there are not enough matching vehicles in NHTS data."""

    def __init__(self, bldg_id: str, vehicle_id: int, count: int):
        self.message = f"Building {bldg_id}, vehicle {vehicle_id}: {count} matching vehicles"
        super().__init__(self.message)


class NoDateRangeError(Exception):
    """Raised when no start_date or end_date is provided."""

    pass


@dataclass
class EVDemandConfig:
    state: str
    release: str
    metadata_path: Optional[str] = None
    pums_path: Optional[str] = None
    nhts_path: str = f"{BASEPATH}/ev_data/inputs/NHTS_v2_1_trip_surveys.csv"
    weather_path: Optional[str] = None
    output_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.metadata_path is None:
            self.metadata_path = f"{BASEPATH}/ev_data/inputs/{self.release}/metadata/{self.state}/metadata.parquet"
        if self.pums_path is None:
            self.pums_path = f"{BASEPATH}/ev_data/inputs/{self.state}_2021_pums_PUMA_HINCP_VEH_NP.csv"
        if self.weather_path is None:
            self.weather_path = f"{BASEPATH}/ev_data/inputs/weather.csv"  # move weather data here too


@dataclass
class VehicleProfile:
    """Represents a vehicle's driving profile parameters."""

    building_id: str
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

    building_id: str  # Changed from int to str to match VehicleProfile
    vehicle_id: int
    date: datetime
    departure_hour: int
    arrival_hour: int
    miles_driven: float
    # avg_temp_while_away: float #TBD if this will be added here or in rdp
    # kwh_consumed: float


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
        nhts_df: pl.DataFrame,
        pums_df: pl.DataFrame,
        start_date: datetime,
        end_date: datetime,
        max_vehicles: int = 2,
        random_state: int = 42,
        weather_df: Optional[pl.DataFrame] = None,
    ):
        """
        Initialize the EV demand calculator.

        Args:
            metadata_df: ResStock metadata DataFrame
            nhts_df: NHTS trip data DataFrame
            pums_df: PUMS data DataFrame
            weather_df: Weather data DataFrame (optional)
            start_date: Start date for trip generation
            end_date: End date for trip generation
            random_state: Random seed for reproducible results
        """
        # Set random seed for reproducible results
        np.random.seed(random_state)

        self.max_vehicles = max_vehicles

        self.metadata_df = metadata_df
        self.nhts_df = nhts_df
        self.pums_df = pums_df
        self.weather_df = weather_df

        self.start_date = start_date
        self.end_date = end_date
        self.num_days = (self.end_date - self.start_date).days + 1

        self.vehicle_ownership_model: Optional[Any] = None
        self.random_state = random_state

        # Features used for vehicle assignment
        self.veh_assign_features = ["occupants", "income", "metro"]

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

    def predict_num_vehicles(self, metadata_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
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

        Args:
            target_income: Target income bucket to match
            target_occupants: Target number of occupants to match
            target_vehicles: Target number of vehicles to match
            num_samples: Number of different vehicles to sample
            weekday: Whether to match against weekday (True) or weekend (False) trips

        Returns:
            Tuple of (match_type, list of matched_vehicle_ids)
        """
        # Select weekday or weekend trips
        trips = self.nhts_df.filter(pl.col("weekday") == (2 if weekday else 1))

        # Try exact match first
        exact_matches = (
            trips.filter(
                (pl.col("income_bucket") == target_income)
                & (pl.col("occupants") == target_occupants)
                & (pl.col("vehicles") == target_vehicles)
            )
            .select(["hh_vehicle_id"])
            .unique()
        )

        if exact_matches.height >= num_samples:
            return "exact", exact_matches.sample(num_samples).get_column("hh_vehicle_id").to_list()

        # Try matching only income and occupants
        income_occ_matches = (
            trips.filter((pl.col("income_bucket") == target_income) & (pl.col("occupants") == target_occupants))
            .select(["hh_vehicle_id"])
            .unique()
        )

        if income_occ_matches.height >= num_samples:
            return "income_occupants", income_occ_matches.sample(num_samples).get_column("hh_vehicle_id").to_list()

        # Try matching only income
        income_matches = trips.filter(pl.col("income_bucket") == target_income).select(["hh_vehicle_id"]).unique()

        if income_matches.height >= num_samples:
            return "income_only", income_matches.sample(num_samples).get_column("hh_vehicle_id").to_list()

        # If still no match, find closest income bucket
        all_incomes = trips.select("income_bucket").unique().sort("income_bucket")
        closest_income = min(all_incomes.get_column("income_bucket"), key=lambda x: abs(x - target_income))

        fallback_matches = trips.filter(pl.col("income_bucket") == closest_income).select(["hh_vehicle_id"]).unique()

        return "closest_income", fallback_matches.sample(num_samples).get_column("hh_vehicle_id").to_list()

    def sample_vehicle_profiles(
        self, bldg_veh_df: pl.DataFrame, nhts_df: pl.DataFrame
    ) -> dict[tuple[str, int], VehicleProfile]:
        """
        For each household and vehicle, select a weekday and weekend trip profiles from NHTS.

        Args:
            bldg_veh_df: DataFrame with household and vehicle info.
            nhts_df: NHTS trip data DataFrame with trip weights

        Returns:
            Dict mapping (building_id, vehicle_id) to sampled trip profile parameters
        """
        df = bldg_veh_df
        if df is None:
            raise MetadataDataFrameError()

        if nhts_df is None:
            raise NHTSDataError()

        # Cap the number of vehicles at the max number of vehicles per household
        nhts_df = nhts_df.with_columns(
            pl.when(pl.col("vehicles") > self.max_vehicles)
            .then(self.max_vehicles)
            .otherwise(pl.col("vehicles"))
            .alias("vehicles")
        )

        # Group NHTS data by weekday/weekend
        weekday_trips = nhts_df.filter(pl.col("weekday") == 2)
        weekend_trips = nhts_df.filter(pl.col("weekday") == 1)

        profiles = {}

        for row in df.iter_rows(named=True):
            bldg_id = row["bldg_id"]
            num_vehicles = row["vehicles"]

            if num_vehicles == 0:
                continue

            # Get all vehicle matches at once
            match_type, matched_vehicle_ids = self.find_best_matches(
                target_income=row["income_bucket"],
                target_occupants=row["occupants"],
                target_vehicles=num_vehicles,
                num_samples=num_vehicles,
            )

            # Create profiles for each vehicle
            for vehicle_id, matched_vehicle_id in enumerate(matched_vehicle_ids, start=1):
                # Find best matching vehicle based on our criteria
                # match_type, matched_vehicle_id = self.find_best_match(
                #     target_income=row["income_bucket"], target_occupants=row["occupants"], target_vehicles=num_vehicles
                # )

                # Get all weekday trips for this vehicle
                weekday_samples = weekday_trips.filter(pl.col("hh_vehicle_id") == matched_vehicle_id).select([
                    "start_time",
                    "end_time",
                    "miles_driven",
                    "trip_weight",
                ])

                # Get all weekend trips for the same vehicle
                weekend_samples = weekend_trips.filter(pl.col("hh_vehicle_id") == matched_vehicle_id).select([
                    "start_time",
                    "end_time",
                    "miles_driven",
                    "trip_weight",
                ])

                # Process weekday trips
                weekday_rows = weekday_samples.rows()
                weekday_departures = [t[0] // 100 for t in weekday_rows]  # Convert HHMM to HH
                weekday_arrivals = [t[1] // 100 for t in weekday_rows]
                weekday_miles = [t[2] for t in weekday_rows]
                weekday_weights = [t[3] for t in weekday_rows]
                weekday_trip_ids = list(range(1, len(weekday_rows) + 1))

                # Process weekend trips
                weekend_rows = weekend_samples.rows()
                weekend_departures = [t[0] // 100 for t in weekend_rows]
                weekend_arrivals = [t[1] // 100 for t in weekend_rows]
                weekend_miles = [t[2] for t in weekend_rows]
                weekend_weights = [t[3] for t in weekend_rows]
                weekend_trip_ids = list(range(1, len(weekend_rows) + 1))

                # Create VehicleProfile for this specific vehicle
                profiles[(bldg_id, vehicle_id)] = VehicleProfile(
                    building_id=bldg_id,
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

        return profiles

    def generate_daily_schedules(self, profile: VehicleProfile) -> list[TripSchedule]:
        """Generate trip schedules for all days in the date range."""
        schedules = []

        # Calculate number of days between start and end dates
        days = (self.end_date - self.start_date).days + 1

        for day_offset in range(days):
            current_date = self.start_date + timedelta(days=day_offset)
            is_weekday = current_date.weekday() < 5  # Monday-Friday are weekdays

            # Get available trips and weights for this day type
            if is_weekday:
                available_trips = len(profile.weekday_trip_ids)
                weights = np.array(profile.weekday_trip_weights)
            else:
                available_trips = len(profile.weekend_trip_ids)
                weights = np.array(profile.weekend_trip_weights)

            if available_trips == 0:
                continue  # Skip days where we have no trips

            # Randomly determine number of trips for this day (1 to max available)
            num_trips = np.random.randint(1, available_trips + 1)

            # Normalize weights for numpy's choice function
            weights = weights / weights.sum()

            # Sample trip indices using the weights
            trip_indices = np.random.choice(
                available_trips,
                size=num_trips,
                replace=False,  # Don't use the same trip twice
                p=weights,
            )

            for idx in trip_indices:
                if is_weekday:
                    departure = profile.weekday_departure_hour[idx]
                    arrival = profile.weekday_arrival_hour[idx]
                    base_miles = profile.weekday_miles[idx]
                else:
                    departure = profile.weekend_departure_hour[idx]
                    arrival = profile.weekend_arrival_hour[idx]
                    base_miles = profile.weekend_miles[idx]

                # Add variance to miles only (keep original departure/arrival times)
                miles = np.random.normal(base_miles, base_miles * 0.1)

                # # Get temperature and calculate energy consumption - TODO: decide if we do this here or in rdp
                # avg_temp = self.get_avg_temp_while_away(departure, arrival, current_date)
                # kwh = self.miles_to_kwh(miles, avg_temp)

                schedules.append(
                    TripSchedule(
                        building_id=profile.building_id,
                        vehicle_id=profile.vehicle_id,
                        date=current_date,
                        departure_hour=departure,
                        arrival_hour=arrival,
                        miles_driven=miles,
                        # avg_temp_while_away=avg_temp,
                        # kwh_consumed=kwh,
                    )
                )

        return schedules

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

    def _generate_annual_trip_schedule(
        self,
        profile_params: dict[tuple[str, int], VehicleProfile],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
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

        all_schedules = []

        # Generate schedules for each vehicle
        for profile in profile_params.values():
            schedules = self.generate_daily_schedules(profile)
            all_schedules.extend(schedules)

        # Convert to DataFrame
        return pl.DataFrame([vars(schedule) for schedule in all_schedules])

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

    def generate_trip_schedules(self) -> pl.DataFrame:
        """
        Generate trip schedules for all vehicles in the metadata.
        """
        # assign cars to metadata buildings
        bldg_veh_df = self.predict_num_vehicles()
        # Get all vehicle profiles
        vehicle_profiles = self.sample_vehicle_profiles(bldg_veh_df, self.nhts_df)

        # Generate trip schedules for each vehicle
        trip_schedules = self._generate_annual_trip_schedule(vehicle_profiles)

        return trip_schedules


# # Example usage
if __name__ == "__main__":
    # Step 1: Create configuration
    config = EVDemandConfig(state="NY", release="res_2022_tmy3_1.1")

    # Step 2: Load all data
    metadata_df, nhts_df, pums_df, weather_df = ev_utils.load_all_input_data(config)
    print(f"✓ Loaded metadata: {len(metadata_df)} rows")
    print(f"✓ Loaded NHTS data: {len(nhts_df)} rows")
    print(f"✓ Loaded weather data: {len(weather_df)} rows")

    # Test the multinomial model
    calculator = EVDemandCalculator(
        metadata_df=metadata_df[0:2, :],
        nhts_df=nhts_df,
        pums_df=pums_df,
        weather_df=weather_df,
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 1, 7),
    )

    # TODO: Add a step to update the buildstock_releases.json file. There will be a function in resolve_bldgid_sets.py that will do this.

    trip_schedules = calculator.generate_trip_schedules()
