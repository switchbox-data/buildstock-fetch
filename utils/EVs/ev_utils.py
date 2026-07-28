"""
Utility functions for EV demand calculations.

This module contains:
- Census division mapping functions
- Data loading functions for metadata, NHTS, PUMS, and ResStock EV reference data
"""

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import polars as pl

from utils.EVs.nhts_tours import NHTS_HOME_PURPOSES

__all__ = [
    "EVDemandInputs",
    "assign_urban_from_metro",
    "build_bldg_hourly_temp_f",
    "get_census_division_for_state",
    "load_all_input_data",
    "load_ev_autonomie_params",
    "load_ev_battery_lookup",
    "load_ev_ownership_lookup",
    "load_hourly_temp_f_for_buildings",
    "load_metadata",
    "load_metro_puma_map",
    "load_nhts_data",
    "load_pums_data",
    "load_resstock_weather_station_temps",
    "load_station_temps_for_buildings",
    "load_weather_station_map",
    "parse_ev_option_name",
    "parse_release_for_weather_map",
    "resstock_puma_dependency",
    "resstock_temp_power_mult",
    "resolve_bldg_weather_stations",
    "state_ev_ownership_rate",
    "yuksel_michalek_miles_to_kwh",
]

# ResStock ``in.puma_metro_status`` → NHTS-like urban (1) / rural (2), matching URBRUR.
_METRO_TO_URBAN: dict[str, int] = {
    "In metro area, principal city": 1,
    "In metro area, not/partially in principal city": 1,
    "Not/partially in metro area": 2,
}

# Housing-characteristic TSV headers look like: Option=Compact, Battery Electric Vehicle, 200 mile range
_OPTION_HEADER_RE = re.compile(r"^Option=(.+)$")
# Canonical option names used by both the battery TSV and Autonomie CSV.
_OPTION_PARSE_RE = re.compile(
    r"^(?P<body_class>Compact|Midsize|Pickup|SUV), "
    r"Battery Electric Vehicle, "
    r"(?P<range_miles>\d+) mile range$"
)

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


def assign_urban_from_metro(metro: str) -> int:
    """Map ResStock ``in.puma_metro_status`` to NHTS-like urban (1) / rural (2).

    Principal-city and non-principal metro both map to urban; non-metro maps to rural.
    Aligns with NHTS ``URBRUR`` (1=Urban, 2=Rural).

    Args:
        metro: ResStock PUMA metro status string

    Returns:
        1 for urban, 2 for rural

    Raises:
        ValueError: If ``metro`` is not a known ResStock metro status
    """
    try:
        return _METRO_TO_URBAN[metro]
    except KeyError as exc:
        raise ValueError(
            f"Unknown ResStock metro status {metro!r}; "
            f"expected one of {sorted(_METRO_TO_URBAN)}"
        ) from exc


def assign_nhts_income_bucket(income: int) -> int:
    """
    Assign a coarse income bin for NHTS profile matching.

    Collapses the NHTS HHFAMINC 1–11 scale into three stable pools:

        1 = less than $50,000          (HHFAMINC 01–05)
        2 = $50,000 to $149,999        (HHFAMINC 06–09)
        3 = $150,000 or more           (HHFAMINC 10–11)

    Args:
        income: Annual household income in dollars

    Returns:
        Coarse income bin (1–3)
    """
    if income < 50000:
        return 1
    if income < 150000:
        return 2
    return 3


def coarsen_nhts_hhfaminc(hhfaminc: int) -> int:
    """Map raw NHTS HHFAMINC codes (1–11) to the same 1–3 bins as ``assign_nhts_income_bucket``."""
    if hhfaminc <= 5:
        return 1
    if hhfaminc <= 9:
        return 2
    return 3


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
        DataFrame with columns including 'bldg_id', 'occupants', 'income', 'metro',
        'urban', 'puma'.

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
        # Process household size - replace "10+" with "10", cast numeric, cap at 3+ for matching
        .with_columns([
            pl.when(pl.col("occupants") == "10+")
            .then(pl.lit("10"))
            .otherwise(pl.col("occupants"))
            .cast(pl.Int64)
            .alias("occupants")
        ])
        .filter(pl.col("occupants") > 0)
        .with_columns([
            pl.when(pl.col("occupants") >= 3)
            .then(3)
            .otherwise(pl.col("occupants"))
            .alias("occupants")
        ])
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
        # Binary urban/rural for NHTS profile matching (aligned with NHTS URBRUR)
        .with_columns([
            pl.col("metro").map_elements(assign_urban_from_metro, return_dtype=pl.Int64).alias("urban")
        ])
        # Extract last 5 characters from PUMA (used for puma_dependency keys)
        .with_columns([pl.col("puma").str.slice(-5).alias("puma")])
        # Build NREL lookup key: "MD, 00805" (same format as Dependency=PUMA in the TSV)
        .with_columns(
            pl.concat_str([pl.lit(f"{state}, "), pl.col("puma").str.zfill(5)]).alias("puma_dependency"),
        )
        .with_columns([
            pl.col("occupants").cast(pl.UInt8),  # 1 / 2 / 3+
            pl.col("income_bucket").cast(pl.UInt8),  # 1–3 coarse bins
            pl.col("urban").cast(pl.UInt8),  # 1=urban, 2=rural
            pl.col("puma").cast(pl.Utf8),
        ])
        .collect()
    )

    return metadata_df


def _nhts_persons_path(nhts_trip_path: str | Path) -> Path:
    """Resolve the sibling NHTS persons CSV used for ``OUTOFTWN``.

    Trip surveys live at ``NHTS_v2_1_trip_surveys.csv``; the person file is
    expected beside it as ``NHTS_v2_1_persons.csv`` (same ORNL NextGen release).

    Args:
        nhts_trip_path: Path to the NHTS trip data file

    Returns:
        Path to the NHTS persons data file
    """
    trip_path = Path(nhts_trip_path)
    return trip_path.with_name("NHTS_v2_1_persons.csv")


def load_nhts_data(nhts_path: str, state: str) -> pl.DataFrame:
    """
    Load and preprocess the NHTS trip data for a specific state.

    Keeps purpose / sequencing columns so vehicle-day legs can be chained into
    home-based tours (``why_from`` / ``why_to`` / ``seq_trip_id``). Restricts to
    household-vehicle *driver* trips (``DRVR_FLG=01``) so passenger duplicates do
    not double-count vehicle movement.

    Home-charging-only pool: keep vehicle-days that start and end at home
    (first-leg ``WHYFROM`` and last-leg ``WHYTO`` in ``NHTS_HOME_PURPOSES``) and
    whose driver was not away for the entire travel day (person-file
    ``OUTOFTWN=02``).

    Args:
        nhts_path: Path to the NHTS trip data file
        state: State abbreviation to filter data for

    Returns:
        DataFrame with trip records filtered for the specified state's census division
    """
    if not Path(nhts_path).exists():
        msg = f"NHTS file not found: {nhts_path}. Please run `just download-nhts` to download the data."
        raise FileNotFoundError(msg)

    # Person-file flag ``OUTOFTWN`` is not on the trip CSV; join from the sibling persons file.
    persons_path = _nhts_persons_path(nhts_path)
    if not persons_path.exists():
        msg = (
            f"NHTS persons file not found: {persons_path}. "
            "Required for OUTOFTWN (away-entire-day) filtering; place "
            "NHTS_v2_1_persons.csv next to the trip surveys CSV."
        )
        raise FileNotFoundError(msg)

    # Define the columns we need from the trip file.
    # HOUSEID/PERSONID are join keys for OUTOFTWN only (dropped before return).
    needed_columns = [
        "CENSUS_D",  # census division (needed for filtering)
        "HOUSEID",  # join to persons for OUTOFTWN
        "PERSONID",  # join to persons for OUTOFTWN
        "VEHCASEID",  # unique hh/vehicle id
        "VEHID",  # vehicle id
        "VEHTYPE",  # household vehicle type (needed for light-duty filter)
        "STRTTIME",  # start time
        "ENDTIME",  # end time
        "TRPMILES",  # miles driven
        "TRAVDAY",  # day of week the trip was taken (1=Sun ... 7=Sat)
        "HHSIZE",  # occupants
        "HHFAMINC",  # household income
        "HHVEHCNT",  # total number of vehicles
        "URBRUR",  # urban/rural status urban(1)/rural(2)
        "WTTRDFIN",  # trip weight
        # Tour chaining / vehicle timeline
        "WHYFROM",  # origin purpose (home = 01/02)
        "WHYTO",  # destination purpose (home = 01/02)
        "SEQ_TRIPID",  # order within person travel day
        "DRVR_FLG",  # 01=driver (moves the vehicle), 02=passenger
    ]

    # Load only the needed trip columns (HOUSEID/PERSONID as strings for a stable join).
    nhts_df = pl.read_csv(
        nhts_path,
        columns=needed_columns,
        schema_overrides={
            "VEHCASEID": pl.Utf8,
            "HOUSEID": pl.Utf8,
            "PERSONID": pl.Utf8,
        },
    )

    # Get the census division for this state
    state_division = get_census_division_for_state(state)

    # Light-duty passenger vehicles only (NHTS VEHTYPE):
    # 1=car, 2=van, 3=SUV, 4=pickup. Excludes other truck (5), RV (6), motorcycle/moped (7).
    light_duty_veh_types = [1, 2, 3, 4]

    # Filter to only keep census division for this state
    nhts_df = nhts_df.filter(
        pl.col("CENSUS_D") == state_division,
        pl.col("HHVEHCNT") > 0,
        # Household-vehicle trips only (excludes transit, walk, bike, non-HH vehicles, etc.)
        pl.col("VEHCASEID") != "-1",
        pl.col("HHFAMINC") > 0,  # -7, -8 are not valid income bucket values
        pl.col("VEHTYPE").cast(pl.Int64).is_in(light_duty_veh_types),
        # Driver only: passenger rows share VEHCASEID but do not move the car again.
        pl.col("DRVR_FLG").cast(pl.Int64) == 1,
    )

    # Drop filter-only columns (keep HOUSEID/PERSONID until after OUTOFTWN join).
    nhts_df = nhts_df.drop("CENSUS_D", "VEHTYPE", "DRVR_FLG")

    # --- OUTOFTWN: drop drivers who were away from home the entire travel day ---
    # Codebook (person file): 01=Yes (away entire day), 02=No. Home-charging models
    # assume the vehicle can be at the residence that day, so keep OUTOFTWN=02 only.
    persons_df = pl.read_csv(
        persons_path,
        columns=["HOUSEID", "PERSONID", "OUTOFTWN"],
        schema_overrides={"HOUSEID": pl.Utf8, "PERSONID": pl.Utf8},
    ).with_columns(pl.col("OUTOFTWN").cast(pl.Int64))
    n_before_outoftwn = nhts_df.height
    nhts_df = (
        nhts_df.join(persons_df, on=["HOUSEID", "PERSONID"], how="inner")
        .filter(pl.col("OUTOFTWN") == 2)
        .drop("HOUSEID", "PERSONID", "OUTOFTWN")
    )
    logging.info(
        "NHTS OUTOFTWN filter (keep 02=not away entire day): %s/%s trip rows retained",
        nhts_df.height,
        n_before_outoftwn,
    )

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
        "WHYFROM": "why_from",
        "WHYTO": "why_to",
        "SEQ_TRIPID": "seq_trip_id",
    })

    # NHTS URBRUR is 1=Urban, 2=Rural (may arrive as 01/02 strings from CSV).
    # Purpose / sequence codes arrive as zero-padded strings; cast for tour logic.
    # Income: collapse HHFAMINC 1–11 → 3 bins; occupants: cap at 3+ (match ResStock load).
    nhts_df = nhts_df.with_columns(
        pl.col("urban").cast(pl.Int64),
        pl.col("why_from").cast(pl.Int64),
        pl.col("why_to").cast(pl.Int64),
        pl.col("seq_trip_id").cast(pl.Int64),
        pl.col("occupants").cast(pl.Int64),
        pl.col("income_bucket").cast(pl.Int64),
    ).with_columns(
        pl.when(pl.col("income_bucket") <= 5)
        .then(1)
        .when(pl.col("income_bucket") <= 9)
        .then(2)
        .otherwise(3)
        .alias("income_bucket"),
        pl.when(pl.col("occupants") >= 3)
        .then(3)
        .otherwise(pl.col("occupants"))
        .alias("occupants"),
    )

    # --- Closed home-based vehicle-days only (start and end at home) ---
    # Home purposes: 01=regular activities at home, 02=work from home (paid).
    # A vehicle-day is one (hh_vehicle_id, weekday) group. Require the chronologically
    # first leg to leave home and the last leg to return home so presence/charging
    # templates are closed leave-home → return-home loops (no open overnight tours).
    home_purposes = list(NHTS_HOME_PURPOSES)
    n_before_home = nhts_df.height
    n_vehicles_before = nhts_df["hh_vehicle_id"].n_unique()
    closed_home_days = (
        nhts_df.sort(["hh_vehicle_id", "weekday", "start_time", "seq_trip_id"])
        .group_by(["hh_vehicle_id", "weekday"])
        .agg(
            pl.col("why_from").first().alias("_day_start_why_from"),
            pl.col("why_to").last().alias("_day_end_why_to"),
        )
        .filter(
            pl.col("_day_start_why_from").is_in(home_purposes),
            pl.col("_day_end_why_to").is_in(home_purposes),
        )
        .select(["hh_vehicle_id", "weekday"])
    )
    nhts_df = nhts_df.join(closed_home_days, on=["hh_vehicle_id", "weekday"], how="inner")
    logging.info(
        "NHTS closed-home filter (start WHYFROM and end WHYTO in %s): "
        "%s/%s trip rows, %s/%s vehicles retained",
        sorted(home_purposes),
        nhts_df.height,
        n_before_home,
        nhts_df["hh_vehicle_id"].n_unique(),
        n_vehicles_before,
    )

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


def parse_ev_option_name(option_name: str) -> tuple[str, int]:
    """Extract body class and range miles from a ResStock EV battery option name.

    Examples:
        >>> parse_ev_option_name("Compact, Battery Electric Vehicle, 200 mile range")
        ('Compact', 200)
    """
    match = _OPTION_PARSE_RE.match(option_name.strip())
    if match is None:
        raise ValueError(f"Unrecognized EV battery option name: {option_name!r}")
    return match.group("body_class"), int(match.group("range_miles"))


def load_ev_battery_lookup(ev_battery_path: str | Path) -> pl.DataFrame:
    """
    Load ResStock national EV battery option shares (Electric_Vehicle_Battery.tsv).

    The housing-characteristic file has one data row of option shares and a
    trailing ``sampling_probability`` column that should equal 1.

    Args:
        ev_battery_path: Path to the EV battery housing-characteristic TSV

    Returns:
        DataFrame with columns ``ev_option_name``, ``probability``
    """
    path = Path(ev_battery_path)
    if not path.exists():
        raise FileNotFoundError(
            f"EV battery options file not found: {path}. "
            "Run `just download-resstock-ev-reference` to download the data."
        )

    # Skip blank lines and ResStock comment lines that start with '#'.
    data_rows: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            data_rows.append(stripped)

    # Expect exactly: header row with Option=... columns, then one probability row.
    if len(data_rows) < 2:
        raise ValueError(f"EV battery TSV must have a header and data row: {path}")

    header_cells = data_rows[0].split("\t")
    value_cells = data_rows[1].split("\t")
    if len(header_cells) != len(value_cells):
        raise ValueError(
            f"EV battery TSV header/value length mismatch in {path}: "
            f"{len(header_cells)} vs {len(value_cells)}"
        )

    # Build (option_name, probability) pairs; ignore the trailing sampling_probability=1 column.
    option_names: list[str] = []
    probabilities: list[float] = []
    for header, value in zip(header_cells, value_cells, strict=True):
        option_match = _OPTION_HEADER_RE.match(header)
        if option_match is None:
            if header == "sampling_probability":
                continue  # ResStock QA column; not an EV option
            raise ValueError(f"Unexpected EV battery TSV column: {header!r}")
        option_names.append(option_match.group(1))
        probabilities.append(float(value))

    probs = pl.DataFrame({
        "ev_option_name": option_names,
        "probability": probabilities,
    })
    # Soft check before multinomial sampling (numpy also requires sum≈1).
    total = float(probs["probability"].sum())
    if not np.isclose(total, 1.0, atol=1e-5):
        raise ValueError(f"EV battery option probabilities sum to {total}, expected 1.0")
    return probs


def load_ev_autonomie_params(ev_autonomie_path: str | Path) -> pl.DataFrame:
    """
    Load Autonomie usable capacity and efficiency keyed by EV battery option name.

    These are the physical model inputs that ResStock maps into HPXML / EnergyPlus
    (usable kWh pack size and combined kWh per mile).

    Args:
        ev_autonomie_path: Path to ``resstock_autonomie_2022_vehicle_params.csv``

    Returns:
        DataFrame with ``ev_option_name``, ``battery_capacity_kwh``, ``kwh_per_mile``,
        ``body_class``, and ``range_miles``
    """
    path = Path(ev_autonomie_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Autonomie vehicle params not found: {path}. "
            "Run `just download-resstock-ev-reference` to download the data."
        )

    params = (
        pl.read_csv(path)
        .select(
            pl.col("option_name").alias("ev_option_name"),
            # Usable capacity is what SOC / charging constraints should use (not nominal).
            pl.col("vehicle_battery_usable_capacity_kwh").cast(pl.Float64).alias("battery_capacity_kwh"),
            # Base vehicle efficiency before any temperature adjustment.
            pl.col("vehicle_fuel_economy_combined_kwh_per_mile")
            .cast(pl.Float64)
            .alias("kwh_per_mile"),
        )
        # Derive body_class / range_miles from the option string for matching / diagnostics.
        .with_columns(
            pl.col("ev_option_name")
            .map_elements(lambda name: parse_ev_option_name(name)[0], return_dtype=pl.Utf8)
            .alias("body_class"),
            pl.col("ev_option_name")
            .map_elements(lambda name: parse_ev_option_name(name)[1], return_dtype=pl.Int64)
            .alias("range_miles"),
        )
    )
    if params.is_empty():
        raise ValueError(f"Autonomie vehicle params file is empty: {path}")
    return params


@dataclass
class EVDemandInputs:
    """Tables loaded for an EV demand run (mode-dependent fields may be ``None``)."""

    metadata_df: pl.DataFrame
    nhts_df: pl.DataFrame
    ev_battery_df: pl.DataFrame
    ev_autonomie_df: pl.DataFrame
    # Present when ev_assignment=pums_vehicles.
    pums_df: pl.DataFrame | None = None
    # Present when ev_assignment=resstock_adoption.
    ev_ownership_df: pl.DataFrame | None = None
    # Present when temperature_adjustment=resstock.
    weather_map: pl.DataFrame | None = None
    # Station name → month/day/hour temps; shared across batches when preloaded.
    station_temps: dict[str, pl.DataFrame] | None = field(default=None)


def load_all_input_data(ev_demand_config: Any) -> EVDemandInputs:
    """
    Load input tables for the EV demand calculator.

    Mode-gated loads:
    - ``pums_df`` only when ``ev_assignment=pums_vehicles``
    - ``ev_ownership_df`` only when ``ev_assignment=resstock_adoption``
    - ``weather_map`` / ``station_temps`` only when ``temperature_adjustment=resstock``
      (station CSVs are preloaded for all metadata buildings so batches share the cache)
    """
    metadata_df = load_metadata(ev_demand_config.metadata_path, ev_demand_config.state)
    nhts_df = load_nhts_data(ev_demand_config.nhts_path, ev_demand_config.state)
    ev_battery_df = load_ev_battery_lookup(ev_demand_config.ev_battery_path)
    ev_autonomie_df = load_ev_autonomie_params(ev_demand_config.ev_autonomie_path)

    pums_df: pl.DataFrame | None = None
    if ev_demand_config.ev_assignment == "pums_vehicles":
        if not ev_demand_config.pums_path:
            raise ValueError("pums_path is required when ev_assignment=pums_vehicles")
        pums_df = load_pums_data(ev_demand_config.pums_path, ev_demand_config.metadata_path)

    ev_ownership_df: pl.DataFrame | None = None
    if ev_demand_config.ev_assignment == "resstock_adoption":
        if not ev_demand_config.ev_ownership_path:
            raise ValueError(
                "ev_ownership_path is required when ev_assignment=resstock_adoption"
            )
        ev_ownership_df = load_ev_ownership_lookup(
            ev_demand_config.ev_ownership_path,
            ev_demand_config.state,
        )

    weather_map: pl.DataFrame | None = None
    station_temps: dict[str, pl.DataFrame] | None = None
    if ev_demand_config.temperature_adjustment == "resstock":
        if not ev_demand_config.weather_dir:
            raise ValueError(
                "weather_dir is required when temperature_adjustment=resstock"
            )
        weather_map = load_weather_station_map()
        station_temps = load_station_temps_for_buildings(
            metadata_df["bldg_id"],
            state=ev_demand_config.state,
            release=ev_demand_config.release,
            weather_dir=ev_demand_config.weather_dir,
            weather_map=weather_map,
        )
        logging.info(
            "Preloaded %s weather station CSV(s) from %s",
            len(station_temps),
            ev_demand_config.weather_dir,
        )

    return EVDemandInputs(
        metadata_df=metadata_df,
        nhts_df=nhts_df,
        ev_battery_df=ev_battery_df,
        ev_autonomie_df=ev_autonomie_df,
        pums_df=pums_df,
        ev_ownership_df=ev_ownership_df,
        weather_map=weather_map,
        station_temps=station_temps,
    )


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


# OpenStudio-HPXML / ResStock 2025 EV discharge temperature curve (°C powers).
# Source: HPXMLtoOpenStudio/resources/vehicle.rb (Geotab + Recurrent update of Yip et al. 2023 Fig. 9).
_RESSTOCK_TEMP_POWER_COEFS_C: tuple[float, ...] = (
    1.412768,
    -3.910397e-02,
    9.408235e-04,
    8.971560e-06,
    -7.699244e-07,
    1.265614e-08,
)

# ResStock weather CSVs use EnergyPlus-style hour-ending timestamps.
_WEATHER_DRYBULB_COL = "Dry Bulb Temperature [°C]"
_WEATHER_DATETIME_COL = "date_time"


def resstock_temp_power_mult(temp_f: float | np.ndarray) -> float | np.ndarray:
    """ResStock / OpenStudio-HPXML EV discharge power multiplier vs outdoor dry-bulb °F.

    Clips temperature to 0–100°F (OpenStudio-HPXML), converts to °C, then evaluates the
    5th-order polynomial used by the EnergyPlus EMS EV discharge program. Apply only to
    driving discharge energy (not charger kW).

    Args:
        temp_f: Outdoor dry-bulb temperature in °F (scalar or array)

    Returns:
        Unitless power multiplier (same shape as input)
    """
    temp = np.asarray(temp_f, dtype=np.float64)
    scalar_input = temp.ndim == 0
    temp = np.atleast_1d(temp)
    temp_clipped = np.clip(temp, 0.0, 100.0) # clip temperature to 0-100°F
    temp_c = (temp_clipped - 32.0) * (5.0 / 9.0) # convert to °C
    power_mult = np.zeros_like(temp_c)
    for i, coef in enumerate(_RESSTOCK_TEMP_POWER_COEFS_C):
        power_mult = power_mult + coef * np.power(temp_c, i) # get power multiplier for each coefficient
    if scalar_input:
        return float(power_mult[0])
    return power_mult


# originally miles_to_kwh
def yuksel_michalek_miles_to_kwh(daily_miles: float, avg_temp: float) -> float:
    """
    Absolute kWh from miles and temp via Yuksel and Michalek (2015) Nissan Leaf regression.

    Legacy helper (formerly ``miles_to_kwh``). Prefer ``resstock_temp_power_mult`` scaled onto
    Autonomie ``kwh_per_mile`` for ResStock-aligned modeling.

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


def parse_release_for_weather_map(release: str) -> tuple[str, str, str, str]:
    """Parse an EV-demand release key into weather-station-map lookup fields.

    Args:
        release: Release key string (e.g. "res_2024_tmy3_2")

    Returns:
        Tuple of (product, year, weather, version)
    
    Examples:
        >>> parse_release_for_weather_map("res_2024_tmy3_2")
        ('resstock', '2024', 'tmy3', '2')
    """
    match = re.fullmatch(
        r"res(?:stock)?_(?P<year>\d{4})_(?P<weather>tmy3|amy2018|amy2012)_(?P<version>\d+)",
        release,
    )
    if match is None:
        match = re.fullmatch(
            r"resstock_(?P<weather>tmy3|amy2018|amy2012)_release_(?P<version>\d+)",
            release,
        )
        if match is None:
            raise ValueError(
                f"Unrecognized release key for weather map lookup: {release!r}. "
                "Expected e.g. 'res_2024_tmy3_2'."
            )
        # Legacy test-style keys omit year; weather map rows are year-scoped.
        raise ValueError(
            f"Release key {release!r} is missing a release year; "
            "use e.g. 'res_2024_tmy3_2' for weather station lookup."
        )
    return "resstock", match.group("year"), match.group("weather"), match.group("version")


def load_weather_station_map(weather_map_path: str | Path | None = None) -> pl.DataFrame:
    """Load the buildstock-fetch building → weather-station parquet map.
    
    Args:
        weather_map_path: Path to the weather station map parquet file

    Returns:
        DataFrame with ``product``, ``release_year``, ``weather_file``, ``release_version``, ``state``, ``bldg_id``, and ``weather_station_name``
    """
    if weather_map_path is None:
        from buildstock_fetch.constants import WEATHER_FILE_DIR

        path = Path(WEATHER_FILE_DIR) / "weather_station_map.parquet"
    else:
        path = Path(weather_map_path)
    if not path.exists():
        raise FileNotFoundError(f"Weather station map not found: {path}")
    return pl.read_parquet(path)


def resolve_bldg_weather_stations(
    bldg_ids: pl.Series | list[Any],
    *,
    state: str,
    release: str,
    weather_map: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Map building IDs to ResStock weather station names for a release/state.

    Args:
        bldg_ids: Series or list of building IDs
        state: State code
        release: Release key string (e.g. "res_2024_tmy3_2")
        weather_map: DataFrame with ``product``, ``release_year``, ``weather_file``, ``release_version``, ``state``, ``bldg_id``, and ``weather_station_name``

    Returns:
        DataFrame with ``bldg_id`` (same dtype as input where possible) and
        ``weather_station_name``.
    """
    product, year, weather_file, version = parse_release_for_weather_map(release)
    weather_map = load_weather_station_map() if weather_map is None else weather_map

    bldg_id_series = pl.Series("bldg_id", bldg_ids).unique()
    # Weather map stores numeric bldg_id; metadata may be zero-padded strings.
    bldg_keys = (
        pl.DataFrame({"bldg_id_raw": bldg_id_series})
        .with_columns(
            pl.col("bldg_id_raw").cast(pl.Utf8).str.replace(r"^0+", "").alias("bldg_id_str")
        )
        .with_columns(
            pl.when(pl.col("bldg_id_str") == "")
            .then(pl.lit("0"))
            .otherwise(pl.col("bldg_id_str"))
            .cast(pl.Int64)
            .alias("bldg_id_int")
        )
    )

    # weather stations
    stations = weather_map.filter(
        pl.col("product") == product,
        pl.col("release_year") == year,
        pl.col("weather_file") == weather_file,
        pl.col("release_version") == version,
        pl.col("state") == state,
    ).select(
        pl.col("bldg_id").alias("bldg_id_int"),
        "weather_station_name",
    )

    # map building ids to weather stations
    joined = bldg_keys.join(stations, on="bldg_id_int", how="left")
    missing = joined.filter(pl.col("weather_station_name").is_null())
    if missing.height > 0:
        sample = missing["bldg_id_raw"].head(5).to_list()
        raise ValueError(
            f"No weather station mapping for {missing.height} building(s) in "
            f"release={release!r} state={state!r}; examples: {sample}"
        )
    return joined.select(
        pl.col("bldg_id_raw").alias("bldg_id"),
        "weather_station_name",
    )


def load_resstock_weather_station_temps(weather_csv_path: str | Path) -> pl.DataFrame:
    """Load a ResStock OEDI weather CSV into month/day/hour dry-bulb °F.

    ResStock weather timestamps are hour-ending (01:00 is the first hour of the day).
    Returned ``hour`` is start-of-hour clock hour 0–23 for joining to ``hours_base``.

    Args:
        weather_csv_path: Path to the weather CSV file

    Returns:
        DataFrame with ``ts``, ``temp_c``, ``temp_f``, ``month``, ``day``, and ``hour``
    """
    path = Path(weather_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Weather file not found: {path}")

    raw = pl.read_csv(path)
    if _WEATHER_DATETIME_COL not in raw.columns or _WEATHER_DRYBULB_COL not in raw.columns:
        raise ValueError(
            f"Weather file {path} missing required columns "
            f"{_WEATHER_DATETIME_COL!r} / {_WEATHER_DRYBULB_COL!r}; got {raw.columns}"
        )

    parsed = raw.select(
        pl.col(_WEATHER_DATETIME_COL).str.to_datetime("%Y-%m-%d %H:%M:%S").alias("ts"),
        pl.col(_WEATHER_DRYBULB_COL).cast(pl.Float64).alias("temp_c"),
    ).with_columns(
        # Hour-ending → start-of-hour: 01:00 → hour 0 same day; 00:00 → hour 23 prior day.
        (pl.col("ts") - pl.duration(hours=1)).alias("hour_start"),
    ).with_columns(
        (pl.col("temp_c") * 9.0 / 5.0 + 32.0).alias("temp_f"),
        pl.col("hour_start").dt.month().alias("month"),
        pl.col("hour_start").dt.day().alias("day"),
        pl.col("hour_start").dt.hour().alias("hour"),
    )

    # TMY composites mix years; keep one row per (month, day, hour). Prefer first occurrence.
    return (
        parsed.select("month", "day", "hour", "temp_f")
        .unique(subset=["month", "day", "hour"], keep="first")
        .sort(["month", "day", "hour"])
    )


def build_bldg_hourly_temp_f(
    *,
    hours_base: pl.DataFrame,
    bldg_stations: pl.DataFrame,
    station_temps: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """Align per-station typical-year temps onto a simulation ``hours_base`` calendar.

    Joins on (month, day, hour). Leap-day Feb 29 uses Feb 28 temperatures when the weather
    file is a non-leap typical year.

    Args:
        hours_base: From ``build_hours_base`` (needs ``hour_index``, ``date``, ``hour``)
        bldg_stations: ``bldg_id``, ``weather_station_name``
        station_temps: Map station name → frame from ``load_resstock_weather_station_temps``

    Returns:
        ``bldg_id``, ``hour_index``, ``temp_f``
    """
    required_hours = {"hour_index", "date", "hour"}
    missing_hours = required_hours - set(hours_base.columns)
    if missing_hours:
        raise ValueError(f"hours_base missing columns: {sorted(missing_hours)}")
    required_stations = {"bldg_id", "weather_station_name"}
    missing_stations = required_stations - set(bldg_stations.columns)
    if missing_stations:
        raise ValueError(f"bldg_stations missing columns: {sorted(missing_stations)}")

    calendar = hours_base.select(
        "hour_index",
        "date",
        "hour",
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.day().alias("day"),
    ).with_columns(
        # Non-leap TMY: map Feb 29 → Feb 28.
        pl.when((pl.col("month") == 2) & (pl.col("day") == 29))
        .then(pl.lit(28))
        .otherwise(pl.col("day"))
        .alias("day"),
    )

    frames: list[pl.DataFrame] = []
    for row in (
        bldg_stations.group_by("weather_station_name")
        .agg(pl.col("bldg_id"))
        .iter_rows(named=True)
    ):
        station_name = row["weather_station_name"]
        bldg_ids = row["bldg_id"]
        if station_name not in station_temps:
            raise KeyError(f"Missing weather temps for station {station_name!r}")
        temps = station_temps[station_name]
        aligned = calendar.join(temps, on=["month", "day", "hour"], how="left")
        if aligned["temp_f"].null_count() > 0:
            raise ValueError(
                f"Weather station {station_name!r} missing temps for "
                f"{aligned.filter(pl.col('temp_f').is_null()).height} hour(s) in the date range"
            )
        station_bldgs = pl.DataFrame({"bldg_id": bldg_ids})
        frames.append(
            station_bldgs.join(aligned.select("hour_index", "temp_f"), how="cross")
        )

    if not frames:
        return pl.DataFrame(
            schema={"bldg_id": pl.Utf8, "hour_index": pl.UInt32, "temp_f": pl.Float64}
        )
    return pl.concat(frames)


def _fill_station_temps_cache(
    station_names: Iterable[str],
    *,
    weather_dir: str | Path,
    station_temps: dict[str, pl.DataFrame],
) -> None:
    """Read missing station CSVs into ``station_temps`` (mutates in place)."""
    weather_dir_path = Path(weather_dir)
    for station_name in station_names:
        if station_name in station_temps:
            continue
        csv_path = weather_dir_path / f"{station_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Weather file not found for station {station_name}: {csv_path}. "
                "Download ResStock weather CSVs into weather_dir "
                "(OEDI .../weather/state=XX/{station}_TMY3.csv)."
            )
        station_temps[station_name] = load_resstock_weather_station_temps(csv_path)


def load_station_temps_for_buildings(
    bldg_ids: pl.Series | list[Any],
    *,
    state: str,
    release: str,
    weather_dir: str | Path,
    weather_map: pl.DataFrame | None = None,
    station_temps: dict[str, pl.DataFrame] | None = None,
) -> dict[str, pl.DataFrame]:
    """Load ResStock weather CSVs for stations used by ``bldg_ids``.

    Existing entries in ``station_temps`` are reused; missing stations are read from
    ``weather_dir`` and inserted into the returned (and mutated) cache dict.
    """
    cache = station_temps if station_temps is not None else {}
    bldg_stations = resolve_bldg_weather_stations(
        bldg_ids, state=state, release=release, weather_map=weather_map
    )
    _fill_station_temps_cache(
        bldg_stations["weather_station_name"].unique().to_list(),
        weather_dir=weather_dir,
        station_temps=cache,
    )
    return cache


def load_hourly_temp_f_for_buildings(
    bldg_ids: pl.Series | list[Any],
    *,
    hours_base: pl.DataFrame,
    state: str,
    release: str,
    weather_dir: str | Path,
    weather_map: pl.DataFrame | None = None,
    station_temps: dict[str, pl.DataFrame] | None = None,
) -> pl.DataFrame:
    """Resolve stations, load ResStock weather CSVs, and return per-building hourly °F.

    Expects files named ``{weather_station_name}.csv`` under ``weather_dir`` (ResStock OEDI
    weather CSV schema with ``date_time`` and ``Dry Bulb Temperature [°C]``).

    When ``station_temps`` is provided it is used as a mutable cache (stations already
    present are not re-read). Pass the preloaded cache from ``load_all_input_data`` so
    batches share station CSVs.

    Args:
        bldg_ids: Series or list of building IDs
        hours_base: From ``build_hours_base`` (needs ``hour_index``, ``date``, ``hour``)
        state: State code
        release: Release key string (e.g. "res_2024_tmy3_2")
        weather_dir: Path to the weather directory
        weather_map: DataFrame with ``product``, ``release_year``, ``weather_file``, ``release_version``, ``state``, ``bldg_id``, and ``weather_station_name``
        station_temps: Optional mutable cache of station name → month/day/hour temps

    Returns:
        DataFrame with ``bldg_id``, ``hour_index``, and ``temp_f``
    """
    cache = station_temps if station_temps is not None else {}
    bldg_stations = resolve_bldg_weather_stations(
        bldg_ids, state=state, release=release, weather_map=weather_map
    )
    station_names = bldg_stations["weather_station_name"].unique().to_list()
    _fill_station_temps_cache(
        station_names,
        weather_dir=weather_dir,
        station_temps=cache,
    )
    needed = {name: cache[name] for name in station_names}
    return build_bldg_hourly_temp_f(
        hours_base=hours_base,
        bldg_stations=bldg_stations,
        station_temps=needed,
    )


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
