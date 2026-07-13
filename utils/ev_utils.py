"""
Utility functions for EV demand calculations.

This module contains:
- Census division mapping functions
- Data loading functions for metadata, NHTS, and PUMS data
"""

import logging
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import polars as pl

__all__ = [
    "get_census_division_for_state",
    "load_all_input_data",
    "load_ev_autonomie_params",
    "load_ev_battery_lookup",
    "load_ev_ownership_lookup",
    "load_metadata",
    "load_metro_puma_map",
    "load_nhts_data",
    "load_pums_data",
    "resstock_puma_dependency",
    "state_ev_ownership_rate",
]

STATE_TO_CENSUS_DIVISION: dict[str, int] = {
    # New England (1)
    "CT": 1,
    "ME": 1,
    "MA": 1,
    "NH": 1,
    "RI": 1,
    "VT": 1,
    # Middle Atlantic (2)
    "NJ": 2,
    "NY": 2,
    "PA": 2,
    # East North Central (3)
    "IL": 3,
    "IN": 3,
    "MI": 3,
    "OH": 3,
    "WI": 3,
    # West North Central (4)
    "IA": 4,
    "KS": 4,
    "MN": 4,
    "MO": 4,
    "NE": 4,
    "ND": 4,
    "SD": 4,
    # South Atlantic (5)
    "DE": 5,
    "DC": 5,
    "FL": 5,
    "GA": 5,
    "MD": 5,
    "NC": 5,
    "SC": 5,
    "VA": 5,
    "WV": 5,
    # East South Central (6)
    "AL": 6,
    "KY": 6,
    "MS": 6,
    "TN": 6,
    # West South Central (7)
    "AR": 7,
    "LA": 7,
    "OK": 7,
    "TX": 7,
    # Mountain (8)
    "AZ": 8,
    "CO": 8,
    "ID": 8,
    "MT": 8,
    "NV": 8,
    "NM": 8,
    "UT": 8,
    "WY": 8,
    # Pacific (9)
    "AK": 9,
    "CA": 9,
    "HI": 9,
    "OR": 9,
    "WA": 9,
}


def get_census_division_for_state(state: str) -> int:
    """
    Get the census division number for a given state.

    Args:
        state: State abbreviation (e.g., "NY", "CA")

    Returns:
        Census division number (1-9)

    Raises:
        ValueError: If the state is not found in any census division
    """
    try:
        return STATE_TO_CENSUS_DIVISION[state]
    except KeyError as e:
        msg = f"State {state} not found in any census division"
        raise ValueError(msg) from e


def assign_income_midpoints(income_str: str | None) -> int | None:
    """
    Convert income range strings to numeric midpoints using string parsing.

    Args:
        income_str: Income range string (e.g., "60000-69999", "0-10000")

    Returns:
        Numeric midpoint of the income range
    """
    if income_str is None:
        return None

    # Handle ranges like "60000-69999"
    if "-" in income_str:
        parts = income_str.split("-")
        min_val = int(parts[0])
        max_val = int(parts[1])
        return (min_val + max_val) // 2

    # If it's not a range (<10,000, 200,000+), return the value as-is
    return int(income_str)


def assign_nhts_income_bucket(income: int) -> int:
    """
    Assign an income bucket to an NHTS income value.

    Args:
        income: Annual household income in dollars

    Returns:
        NHTS income bucket number (1-11)

    NHTS Income Buckets:
        01 = Less than $10,000
        02 = $10,000 to $14,999
        03 = $15,000 to $24,999
        04 = $25,000 to $34,999
        05 = $35,000 to $49,999
        06 = $50,000 to $74,999
        07 = $75,000 to $99,999
        08 = $100,000 to $124,999
        09 = $125,000 to $149,999
        10 = $150,000 to $199,999
        11 = $200,000 or more
    """
    # List of (threshold, bucket) pairs
    income_buckets = [
        (10000, 1),
        (15000, 2),
        (25000, 3),
        (35000, 4),
        (50000, 5),
        (75000, 6),
        (100000, 7),
        (125000, 8),
        (150000, 9),
        (200000, 10),
        (float("inf"), 11),  # Catch all for $200,000 or more
    ]

    for threshold, bucket in income_buckets:
        if income < threshold:
            return bucket

    return 11  # Should never reach here due to inf threshold, but makes mypy happy


def resstock_puma_dependency(state: str, puma_geoid: str) -> str:
    """Convert a ResStock PUMA GEOID to the NREL EV lookup key, e.g. G24000901 → 'MD, 00901'.
    
    Args:
        state: State abbreviation (e.g., "NY", "CA")
        puma_geoid: ResStock PUMA GEOID (e.g., "G24000901")

    Returns:
        NREL EV lookup key (e.g., "MD, 00901")
    """
    # ResStock stores full Census GEOIDs (e.g. G24000901); NREL's lookup uses "ST, NNNNN".
    return f"{state}, {puma_geoid[-5:].zfill(5)}"


def load_ev_ownership_lookup(ev_ownership_path: str | Path, state: str) -> pl.DataFrame:
    """
    Load NREL's ResStock EV ownership lookup table for a single state.

    Args:
        ev_ownership_path: Path to the EV ownership lookup file
        state: State abbreviation to filter lookup rows for (e.g., "MD", "NY")

    Returns:
        DataFrame with columns including 'fpl', 'building_type', 'puma_dependency', 'tenure', 'ev_ownership_probability', 'source_weight'
    """
    path = Path(ev_ownership_path)
    if not path.exists():
        msg = (
            f"EV ownership lookup not found: {path}. "
            "Run `just download-resstock-ev-reference` to download the data."
        )
        raise FileNotFoundError(msg)

    # Column names follow ResStock's housing_characteristics TSV convention
    # (Dependency=… / Option=Yes). We rename to short join keys used in metadata.
    return (
        pl.read_csv(path, separator="\t")
        .rename({
            "Dependency=Federal Poverty Level": "fpl",
            "Dependency=Geometry Building Type RECS": "building_type",
            "Dependency=PUMA": "puma_dependency",
            "Dependency=Tenure": "tenure",
            "Option=Yes": "ev_ownership_probability",
        })
        .filter(pl.col("puma_dependency").str.starts_with(f"{state},"))
        .select(
            "fpl",
            "building_type",
            "puma_dependency",
            "tenure",
            "ev_ownership_probability",
            "source_weight",  # PUMS weight for this segment; used in state_ev_ownership_rate()
        )
    )


def state_ev_ownership_rate(ev_lookup: pl.DataFrame, state: str) -> float:
    """
    PUMS-weighted mean P(EV) over occupied lookup segments for a state.

    Expects a state-filtered lookup from load_ev_ownership_lookup().

    ResStock documents vacant units as ``Tenure = 'Not Available'``.
    Those lookup rows carry vacant-unit PUMS weight and must be excluded when computing
    an occupied-segment fallback rate.

    Args:
        ev_lookup: NREL EV ownership lookup from load_ev_ownership_lookup()
        state: State abbreviation (e.g., "MD", "NY")

    Returns:
        PUMS-weighted mean P(EV) over occupied lookup segments for the state
    """
    # Vacant stock in the EV lookup: tenure="Not Available" 
    state_lookup = ev_lookup.filter(
        pl.col("puma_dependency").str.starts_with(f"{state},"),
        pl.col("tenure") != "Not Available",
    )
    total_weight = state_lookup["source_weight"].sum()
    if total_weight == 0:
        msg = f"No occupied EV lookup rows found for state {state}"
        raise ValueError(msg)
    return float((state_lookup["ev_ownership_probability"] * state_lookup["source_weight"]).sum() / total_weight)


def load_metadata(metadata_path: str, state: str) -> pl.DataFrame:
    """
    Load and parse the ResStock metadata parquet file.

    Args:
        metadata_path: Path to the metadata parquet file

    Returns:
        DataFrame with columns including 'bldg_id', 'occupants', 'income', 'metro', 'puma'.

    Raises:
        FileNotFoundError: If the metadata file doesn't exist
    """
    if not Path(metadata_path).exists():
        msg = f"Metadata file not found: {metadata_path}"
        raise FileNotFoundError(msg)
    # Scan parquet file and ensure bldg_id is properly formatted with leading zeros
    metadata_df = (
        pl.scan_parquet(metadata_path)
        .filter(pl.col("in.state") == state)
        # Select and rename columns
        .select([
            pl.col("bldg_id"),
            pl.col("weight"),
            pl.col("in.puma_metro_status").alias("metro"),
            pl.col("in.puma").alias("puma"),
            pl.col("in.income").alias("income"),
            pl.col("in.occupants").alias("occupants"),
            # EV adoption join keys — must match NREL Electric_Vehicle_Ownership.tsv exactly
            pl.col("in.federal_poverty_level").alias("fpl"),
            pl.col("in.geometry_building_type_recs").alias("building_type"),
            pl.col("in.tenure").alias("tenure"),
            # ResStock: vacant units have Tenure/FPL = "Not Available"
            # we key off in.vacancy_status directly for is_vacant.
            (pl.col("in.vacancy_status") == "Vacant").alias("is_vacant"),
        ])
        # Process household size - replace "10+" with "10" and cast to numeric
        .with_columns([
            pl.when(pl.col("occupants") == "10+")
            .then(pl.lit("10"))
            .otherwise(pl.col("occupants"))
            .cast(pl.Int64)
            .alias("occupants")
        ])
        .filter(pl.col("occupants") > 0)
        # Process income categories - convert to standard ranges
        .with_columns([
            pl.when(pl.col("income") == "<10000")
            .then(pl.lit("0-10000"))
            .when(pl.col("income") == "200000+")
            .then(pl.lit("200000-400000"))
            .when(pl.col("income") == "Not Available")
            .then(pl.lit(None))
            .otherwise(pl.col("income"))
            .alias("income")
        ])
        # Convert income ranges to numeric midpoints
        .with_columns([pl.col("income").map_elements(assign_income_midpoints, return_dtype=pl.Int64).alias("income")])
        .with_columns([
            pl.col("income").map_elements(assign_nhts_income_bucket, return_dtype=pl.Int64).alias("income_bucket")
        ])
        # Extract last 5 characters from PUMA (used for puma_dependency keys)
        .with_columns([pl.col("puma").str.slice(-5).alias("puma")])
        # Build NREL lookup key: "MD, 00805" (same format as Dependency=PUMA in the TSV)
        .with_columns(
            pl.concat_str([pl.lit(f"{state}, "), pl.col("puma").str.zfill(5)]).alias("puma_dependency"),
        )
        .with_columns([
            pl.col("occupants").cast(pl.UInt8),  # Instead of Int64
            pl.col("income_bucket").cast(pl.UInt8),  # 1-11 fits in UInt8
            pl.col("puma").cast(pl.Utf8),
        ])
        .collect()
    )

    return metadata_df


def load_nhts_data(nhts_path: str, state: str) -> pl.DataFrame:
    """
    Load and preprocess the NHTS trip data for a specific state.

    Args:
        nhts_path: Path to the NHTS trip data file
        state: State abbreviation to filter data for

    Returns:
        DataFrame with trip records filtered for the specified state's census division
    """
    if not Path(nhts_path).exists():
        msg = f"NHTS file not found: {nhts_path}. Please run `just download-nhts` to download the data."
        raise FileNotFoundError(msg)

    # Define the columns we need
    needed_columns = [
        "CENSUS_D",  # census division (needed for filtering)
        "VEHCASEID",  # unique hh/vehicle id
        "VEHID",  # vehicle id
        "STRTTIME",  # start time
        "ENDTIME",  # end time
        "TRPMILES",  # miles driven
        "TRAVDAY",  # day of week the trip was taken (1=Sun ... 7=Sat)
        "HHSIZE",  # occupants
        "HHFAMINC",  # household income
        "HHVEHCNT",  # total number of vehicles
        "URBRUR",  # urban/rural status urban(1)/rural(2)
        "WTTRDFIN",  # trip weight
    ]

    # Load only the needed columns
    nhts_df = pl.read_csv(nhts_path, columns=needed_columns, schema_overrides={"VEHCASEID": pl.Utf8})

    # Get the census division for this state
    state_division = get_census_division_for_state(state)

    # Filter to only keep census division for this state
    nhts_df = nhts_df.filter(
        pl.col("CENSUS_D") == state_division,
        pl.col("HHVEHCNT") > 0,
        pl.col("VEHCASEID") != "-1",
        pl.col("HHFAMINC") > 0,
    )  # -7, -8 are not valid income bucket values

    # Remove the CENSUS_D column since we don't need it anymore
    nhts_df = nhts_df.drop("CENSUS_D")

    # Derive the weekday/weekend flag from TRAVDAY (day the trip was taken) rather than
    # NHTS's TDWKND, which reclassifies Friday trips starting at/after 18:00 as weekend.
    # Using TRAVDAY keeps a full travel day together (weekday=2 for Mon-Fri, 1 for Sat/Sun).
    nhts_df = nhts_df.with_columns(
        pl.when(pl.col("TRAVDAY").is_in([1, 7]))
        .then(1)  # Sunday(1) or Saturday(7) -> weekend
        .otherwise(2)  # Monday-Friday -> weekday
        .alias("TRAVDAY")
    )

    nhts_df = nhts_df.rename({
        "HHSIZE": "occupants",
        "HHFAMINC": "income_bucket",
        "HHVEHCNT": "vehicles",
        "URBRUR": "urban",
        "TRAVDAY": "weekday",
        "VEHCASEID": "hh_vehicle_id",
        "VEHID": "vehicle_id",
        "STRTTIME": "start_time",
        "ENDTIME": "end_time",
        "TRPMILES": "miles_driven",
        "WTTRDFIN": "trip_weight",
    })

    return nhts_df


def load_pums_data(pums_path: str, metadata_path: str) -> pl.DataFrame:
    """
    Load and preprocess the PUMS data for a specific state.

    Args:
        pums_path: Path to the PUMS data file
        metadata_path: Path to metadata file for metro-PUMA mapping
    """
    if not Path(pums_path).exists():
        msg = f"PUMS file not found: {pums_path}. Please run `just download-pums` to download the data."
        raise FileNotFoundError(msg)

    # Read CSV with "b" as null value and PUMA as string
    pums_df = pl.read_csv(pums_path, null_values=["b"], schema_overrides={"PUMA": pl.Utf8})

    pums_df = pums_df.rename({
        "HINCP": "income",
        "NP": "occupants",
        "VEH": "vehicles",
        "PUMA": "puma",
        "WGTP": "hh_weight",
    })

    pums_df = pums_df.filter(pl.col("income") > 0)

    # Convert vehicles to numeric (Int64)
    pums_df = pums_df.with_columns([pl.col("vehicles").cast(pl.Int64)])

    # join with metro-puma mapping
    metro_puma_df = load_metro_puma_map(metadata_path)
    pums_df = pums_df.join(metro_puma_df, on="puma", how="left")

    return pums_df


def load_metro_puma_map(metadata_path: str) -> pl.DataFrame:
    """
    Load the metro-puma mapping file. We need to assign the metro variable to the PUMS data based on a lookup of the puma code.
    """
    if not Path(metadata_path).exists():
        msg = f"Metadata file not found: {metadata_path}"
        raise FileNotFoundError(msg)

    # Load metadata file lazily and process
    metro_lookup_df = (
        pl.scan_parquet(metadata_path)
        # Select and rename only needed columns
        .select([
            pl.col("in.puma_metro_status").alias("metro"),
            pl.col("in.puma").alias("puma"),
        ])
        # Extract last 5 characters from PUMA
        .with_columns([pl.col("puma").str.slice(-5).alias("puma")])
        # Drop duplicates
        .unique()
        .collect()
    )

    return metro_lookup_df


def load_ev_battery_lookup(ev_battery_path: str | Path) -> pl.DataFrame:
    """
    Load ResStock national EV battery option shares (Electric_Vehicle_Battery.tsv).

    Args:
        ev_battery_path: Path to the EV battery housing-characteristic TSV

    Returns:
        DataFrame with columns ``ev_option_name``, ``probability``
    """
    # Import here to avoid circular imports with EVBatteryAssigner helpers.
    from utils.EVBatteryAssigner import load_ev_battery_option_probabilities

    path = Path(ev_battery_path)
    if not path.exists():
        msg = (
            f"EV battery options file not found: {path}. "
            "Run `just download-resstock-ev-reference` to download the data."
        )
        raise FileNotFoundError(msg)
    return load_ev_battery_option_probabilities(path)


def load_ev_autonomie_params(ev_autonomie_path: str | Path) -> pl.DataFrame:
    """
    Load Autonomie usable capacity and efficiency keyed by EV battery option name.

    Args:
        ev_autonomie_path: Path to ``resstock_autonomie_2022_vehicle_params.csv``

    Returns:
        DataFrame with ``ev_option_name``, ``battery_capacity_kwh``, ``kwh_per_mile``,
        ``body_class``, and ``range_miles``
    """
    from utils.EVBatteryAssigner import load_autonomie_vehicle_params

    path = Path(ev_autonomie_path)
    if not path.exists():
        msg = (
            f"Autonomie vehicle params not found: {path}. "
            "Run `just download-resstock-ev-reference` to download the data."
        )
        raise FileNotFoundError(msg)
    return load_autonomie_vehicle_params(path)


def load_all_input_data(
    ev_demand_config: Any,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Load all input data for the EV demand calculator.

    Returns:
        Tuple of (metadata_df, nhts_df, pums_df, ev_ownership_df, ev_battery_df, ev_autonomie_df)
    """
    metadata_df = load_metadata(ev_demand_config.metadata_path, ev_demand_config.state)
    nhts_df = load_nhts_data(ev_demand_config.nhts_path, ev_demand_config.state)
    pums_df = load_pums_data(ev_demand_config.pums_path, ev_demand_config.metadata_path)
    # Conditional P(EV) by FPL / building type / tenure / PUMA (state-filtered).
    ev_ownership_df = load_ev_ownership_lookup(
        ev_demand_config.ev_ownership_path,
        ev_demand_config.state,
    )
    # National BEV class × range shares + Autonomie physical parameters.
    ev_battery_df = load_ev_battery_lookup(ev_demand_config.ev_battery_path)
    ev_autonomie_df = load_ev_autonomie_params(ev_demand_config.ev_autonomie_path)

    return metadata_df, nhts_df, pums_df, ev_ownership_df, ev_battery_df, ev_autonomie_df


def assign_battery_capacity(battery_capacities, daily_kwh: pl.Series) -> pl.Series:
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
    battery_capacities = pl.Series(battery_capacities)
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
    efficiency_coefficients = np.array([
        0.3950,  # a_0 (constant term)
        -0.0022,  # a_1 (linear term)
        9.1978e-5,  # a_2 (quadratic term)
        -3.9249e-6,  # a_3 (cubic term)
        5.2918e-8,  # a_4 (quartic term)
        -2.0659e-10,  # a_5 (quintic term)
    ])
    consumption_per_mile = np.polyval(efficiency_coefficients[::-1], temp_bounded)

    # Calculate total daily energy consumption
    daily_consumption_kwh = consumption_per_mile * miles

    # Return scalar if input was scalar
    if np.isscalar(daily_miles) and np.isscalar(avg_temp):
        return float(daily_consumption_kwh)
    return float(daily_consumption_kwh)


def upload_object_to_s3(file_content: bytes, file_name: str) -> bool:
    """Upload file content directly to S3 bucket from memory."""
    bucket_name = "buildstock-fetch"
    s3_key = f"ev_demand/{file_name}"

    try:
        s3_client: Any = boto3.client("s3")
        print(f"Uploading {file_name} to s3://{bucket_name}/{s3_key}...")

        # Upload directly from memory
        s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=file_content)

        logging.info(f"Successfully uploaded file to S3: s3://{bucket_name}/{s3_key}")
    except Exception:
        logging.exception("Failed to upload to S3")
        return False
    else:
        return True
