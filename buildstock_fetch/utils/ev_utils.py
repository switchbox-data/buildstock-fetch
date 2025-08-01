"""
Utility functions for EV demand calculations.

This module contains:
- Census division mapping functions
- Data loading functions for metadata, NHTS, and weather data
"""

from pathlib import Path
from typing import Any

import polars as pl

__all__ = [
    "get_census_division_for_state",
    "load_all_input_data",
    "load_metadata",
    "load_metro_puma_map",
    "load_nhts_data",
    "load_pums_data",
    "load_weather_data",
]


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
    census_divisions = {
        1: ["CT", "ME", "MA", "NH", "RI", "VT"],  # New England
        2: ["NJ", "NY", "PA"],  # Middle Atlantic
        3: ["IL", "IN", "MI", "OH", "WI"],  # East North Central
        4: ["IA", "KS", "MN", "MO", "NE", "ND", "SD"],  # West North Central
        5: ["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV"],  # South Atlantic
        6: ["AL", "KY", "MS", "TN"],  # East South Central
        7: ["AR", "LA", "OK", "TX"],  # West South Central
        8: ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY"],  # Mountain
        9: ["AK", "CA", "HI", "OR", "WA"],  # Pacific
    }

    for division, states in census_divisions.items():
        if state in states:
            return division

    msg = f"State {state} not found in any census division"
    raise ValueError(msg)


def assign_income_midpoints(income_str: str) -> int:
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

    # If it's not a range, return the value as-is
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
    if income < 10000:
        return 1
    elif income < 15000:
        return 2
    elif income < 25000:
        return 3
    elif income < 35000:
        return 4
    elif income < 50000:
        return 5
    elif income < 75000:
        return 6
    elif income < 100000:
        return 7
    elif income < 125000:
        return 8
    elif income < 150000:
        return 9
    elif income < 200000:
        return 10
    else:  # $200,000 or more
        return 11

def load_metadata(metadata_path: str) -> pl.DataFrame:
    """
    Load and parse the ResStock metadata parquet file.

    Args:
        metadata_path: Path to the metadata parquet file

    Returns:
        DataFrame with columns including 'bldg_id', 'occupants', 'income', 'metro', 'puma.

    Raises:
        FileNotFoundError: If the metadata file doesn't exist
    """
    if not Path(metadata_path).exists():
        msg = f"Metadata file not found: {metadata_path}"
        raise FileNotFoundError(msg)

    metadata_df = pl.read_parquet(metadata_path)

    # Select and rename columns
    metadata_df = metadata_df.select([
        pl.col("bldg_id"),
        pl.col("weight"),
        pl.col("in.puma_metro_status").alias("metro"),
        pl.col("in.puma").alias("puma"),
        pl.col("in.income").alias("income"),
        pl.col("in.occupants").alias("occupants"),
    ])

    # Process household size - replace "10+" with "10" and convert to numeric
    metadata_df = metadata_df.with_columns([
        pl.when(pl.col("occupants") == "10+").then(pl.lit("10")).otherwise(pl.col("occupants")).alias("occupants_temp")
    ])

    # Now cast to numeric
    metadata_df = metadata_df.with_columns([pl.col("occupants_temp").cast(pl.Int64).alias("occupants")]).drop(
        "occupants_temp"
    )

    # # Create household size bins
    # metadata_df = metadata_df.with_columns([
    #     pl.when(pl.col("occupants") > 1)
    #     .then(pl.lit("2+ ppl"))
    #     .when(pl.col("occupants") == 1)
    #     .then(pl.lit("1 person"))
    #     .otherwise(pl.lit(None))
    #     .alias("hhsize_bins")
    # ])

    # Process income categories - convert to standard ranges first
    metadata_df = metadata_df.with_columns([
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
    metadata_df = metadata_df.with_columns([pl.col("income").map_elements(assign_income_midpoints).alias("income")])

    # Convert income ranges to numeric midpoints
    metadata_df = metadata_df.with_columns(
        [pl.col("income").map_elements(assign_nhts_income_bucket).alias("income_bucket")])

    # Extract last 5 characters from PUMA
    metadata_df = metadata_df.with_columns([pl.col("puma").str.slice(-5).alias("puma")])

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
        "TDWKND",  # weekday(2)/weekend(1) flag
        "HHSIZE",  # occupants
        "HHFAMINC",  # household income
        "HHVEHCNT",  # total number of vehicles
        "URBRUR",  # urban/rural status urban(1)/rural(2)
    ]

    # Load only the needed columns
    nhts_df = pl.read_csv(nhts_path, columns=needed_columns, schema_overrides={"VEHCASEID": pl.Utf8})

    # Get the census division for this state
    state_division = get_census_division_for_state(state)

    # Filter to only keep census division for this state
    nhts_df = nhts_df.filter(pl.col("CENSUS_D") == state_division,
    pl.col("HHVEHCNT") > 0,
    pl.col("VEHCASEID")!="-1",
    pl.col("HHFAMINC")>0) # -7, -8 are not valid income bucket values

    # Remove the CENSUS_D column since we don't need it anymore
    nhts_df = nhts_df.drop("CENSUS_D")

    nhts_df = nhts_df.rename({
        "HHSIZE": "occupants",
        "HHFAMINC": "income_bucket",
        "HHVEHCNT": "vehicles",
        "URBRUR": "urban",
        "TDWKND": "weekday",
        "VEHCASEID": "hh_vehicle_id",
        "VEHID": "vehicle_id",
        "STRTTIME": "start_time",
        "ENDTIME": "end_time",
        "TRPMILES": "miles_driven",
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

    # If metadata_path is provided, join with metro-PUMA mapping
    if metadata_path and Path(metadata_path).exists():
        metro_puma_df = load_metro_puma_map(metadata_path)
        pums_df = pums_df.join(metro_puma_df, on="puma", how="left")

    return pums_df


def load_weather_data(weather_path: str) -> pl.DataFrame:
    """
    Load hourly weather data (e.g., temperature) for a given location.

    Args:
        weather_path: Path to the weather data file (e.g., TMY3 CSV or EPW)

    Returns:
        DataFrame with at least columns ['datetime', 'temperature']
    """
    # if not Path(weather_path).exists():
    #     msg = f"Weather file not found: {weather_path}"
    #     raise FileNotFoundError(msg)

    # TODO: Implement weather data loading
    # This is a placeholder - replace with actual weather loading logic
    return pl.DataFrame()  # Placeholder


def load_metro_puma_map(metadata_path: str) -> pl.DataFrame:
    """
    Load the metro-puma mapping file. We need to assign the metro variable to the PUMS data based on a lookup of the puma code.
    """
    if not Path(metadata_path).exists():
        msg = f"Metadata file not found: {metadata_path}"
        raise FileNotFoundError(msg)

    metro_lookup_df = pl.read_parquet(metadata_path, columns=["in.puma_metro_status", "in.puma"])

    # Select and rename columns
    metro_lookup_df = metro_lookup_df.select([
        pl.col("in.puma_metro_status").alias("metro"),
        pl.col("in.puma").alias("puma"),
    ])

    # Extract last 5 characters from PUMA and get distinct values
    metro_lookup_df = metro_lookup_df.with_columns([pl.col("puma").str.slice(-5).alias("puma")]).unique()

    return metro_lookup_df


def load_all_input_data(ev_demand_config: Any) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Load all input data for the EV demand calculator.

    Returns:
        Tuple of (metadata_df, nhts_df, pums_df, weather_df)
    """
    metadata_df = load_metadata(ev_demand_config.metadata_path)
    nhts_df = load_nhts_data(ev_demand_config.nhts_path, ev_demand_config.state)
    pums_df = load_pums_data(ev_demand_config.pums_path, ev_demand_config.metadata_path)
    weather_df = load_weather_data(ev_demand_config.weather_path)

    return metadata_df, nhts_df, pums_df, weather_df
