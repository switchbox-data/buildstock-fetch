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


STATE_TO_CENSUS_DIVISION: dict[str, int] = {
    # New England (1)
    "CT": 1, "ME": 1, "MA": 1, "NH": 1, "RI": 1, "VT": 1,
    # Middle Atlantic (2)
    "NJ": 2, "NY": 2, "PA": 2,
    # East North Central (3)
    "IL": 3, "IN": 3, "MI": 3, "OH": 3, "WI": 3,
    # West North Central (4)
    "IA": 4, "KS": 4, "MN": 4, "MO": 4, "NE": 4, "ND": 4, "SD": 4,
    # South Atlantic (5)
    "DE": 5, "DC": 5, "FL": 5, "GA": 5, "MD": 5, "NC": 5, "SC": 5, "VA": 5, "WV": 5,
    # East South Central (6)
    "AL": 6, "KY": 6, "MS": 6, "TN": 6,
    # West South Central (7)
    "AR": 7, "LA": 7, "OK": 7, "TX": 7,
    # Mountain (8)
    "AZ": 8, "CO": 8, "ID": 8, "MT": 8, "NV": 8, "NM": 8, "UT": 8, "WY": 8,
    # Pacific (9)
    "AK": 9, "CA": 9, "HI": 9, "OR": 9, "WA": 9,
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

    # If it's not a range (<10,000, 200,000+), return the value as-is
    return int(income_str)


def load_metadata(metadata_path: str) -> pl.DataFrame:
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

    # Scan parquet file lazily with bldg_id as string to preserve leading zeros
    metadata_df = (
        pl.scan_parquet(metadata_path, columns_to_dtypes={"bldg_id": pl.Utf8})
        # Select and rename columns
        .select([
            pl.col("bldg_id"),
            pl.col("weight"),
            pl.col("in.puma_metro_status").alias("metro"),
            pl.col("in.puma").alias("puma"),
            pl.col("in.income").alias("income"),
            pl.col("in.occupants").alias("occupants"),
        ])
        # Process household size - replace "10+" with "10" and cast to numeric
        .with_columns([
            pl.when(pl.col("occupants") == "10+")
            .then(pl.lit("10"))
            .otherwise(pl.col("occupants"))
            .cast(pl.Int64)
            .alias("occupants")
        ])
        # Process income categories - convert to standard ranges first
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
        .with_columns([pl.col("income").map_elements(assign_income_midpoints).alias("income")])
        # Extract last 5 characters from PUMA
        .with_columns([pl.col("puma").str.slice(-5).alias("puma")])
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
    nhts_df = pl.read_csv(nhts_path, columns=needed_columns)

    # Get the census division for this state
    state_division = get_census_division_for_state(state)

    # Filter to only keep census division for this state
    nhts_df = nhts_df.filter(pl.col("CENSUS_D") == state_division)

    # Remove the CENSUS_D column since we don't need it anymore
    nhts_df = nhts_df.drop("CENSUS_D")

    nhts_df = nhts_df.rename({
        "HHSIZE": "occupants",
        "HHFAMINC": "income",
        "HHVEHCNT": "vehicles",
        "URBRUR": "urban",
        "TDWKND": "weekday",
        "VEHCASEID": "vehicle_id",
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

    # join with metro-puma mapping
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
