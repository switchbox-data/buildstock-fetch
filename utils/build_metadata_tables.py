import contextlib
import json
import logging
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.resources import files
from pathlib import Path
from typing import Any, Union

import boto3
import polars as pl
import pyarrow.fs as fs  # type: ignore[import-untyped]
from botocore import UNSIGNED
from botocore.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("build_metadata_tables.log")],
)
logger = logging.getLogger(__name__)

# Global column definitions - organized by category with priority
PUMA_COLUMNS_TO_KEEP = [
    "in.puma",
    "in.nhgis_puma_gisjoin",
    "in.resstock_puma_id",
]

STATE_COLUMNS_TO_KEEP = [
    "in.state",
    "in.state_name",
]

COUNTY_COLUMNS_TO_KEEP = [
    "in.resstock_county_id",
    "in.nhgis_county_gisjoin",
    "in.county_name",
    "in.county",
]

BUILDING_COLUMNS_TO_KEEP = [
    "bldg_id",
]


def select_priority_columns(available_columns: list) -> list:
    """
    Select one column from each category based on priority (first available).

    Args:
        available_columns (list): List of available columns in the parquet file

    Returns:
        list: Selected columns with one from each category
    """
    selected_columns = []

    # Select one column from each category
    for col_list in [PUMA_COLUMNS_TO_KEEP, STATE_COLUMNS_TO_KEEP, COUNTY_COLUMNS_TO_KEEP, BUILDING_COLUMNS_TO_KEEP]:
        for col in col_list:
            if col in available_columns:
                selected_columns.append(col)
                break  # Only take the first available column from this category

    return selected_columns


def rename_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Rename columns to standardized names based on their category.

    Args:
        df (pl.DataFrame): DataFrame with columns to rename

    Returns:
        pl.DataFrame: DataFrame with renamed columns
    """
    column_mapping = {}
    for col in df.columns:
        if col in PUMA_COLUMNS_TO_KEEP:
            column_mapping[col] = "puma"
        elif col in STATE_COLUMNS_TO_KEEP:
            column_mapping[col] = "state"
        elif col in COUNTY_COLUMNS_TO_KEEP:
            column_mapping[col] = "county"
        elif col in BUILDING_COLUMNS_TO_KEEP:
            column_mapping[col] = "bldg_id"
    logger.info(f"Column mapping: {column_mapping}")

    return df.rename(column_mapping)


def _read_parquet_with_retry(read_func, file_key: str, max_retries: int = 3, *args, **kwargs) -> pl.DataFrame:
    """
    Helper function to read parquet files with retry logic for transient S3 errors.

    Args:
        read_func: Function to call for reading (e.g., pl.read_parquet)
        file_key: S3 key for logging purposes
        max_retries: Maximum number of retry attempts (default: 3)
        *args, **kwargs: Arguments to pass to read_func

    Returns:
        pl.DataFrame: The read DataFrame

    Raises:
        Exception: If all retry attempts fail
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            return read_func(*args, **kwargs)
        except (OSError, Exception) as e:
            last_exception = e
            error_msg = str(e).lower()
            # Check if it's a checksum or validation error that might be transient
            is_transient = (
                "checksum" in error_msg
                or "validation" in error_msg
                or "timeout" in error_msg
                or "connection" in error_msg
            )

            if is_transient and attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                logger.warning(
                    f"Transient error reading {file_key} (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                # Not a transient error or out of retries
                if attempt < max_retries - 1:
                    logger.warning(f"Error reading {file_key} (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                    time.sleep(1)
                else:
                    logger.exception(f"Failed to read {file_key} after {max_retries} attempts")
                    raise

    # Should never reach here, but just in case
    raise last_exception or Exception(f"Failed to read {file_key} after {max_retries} attempts")


def _read_parquet_with_columns(
    file_key: str, s3: Any, bucket_name: str, max_retries: int = 3
) -> tuple[pl.DataFrame, list]:
    """
    Read parquet file and return Polars DataFrame with selected columns.
    Downloads file to temp location first to avoid PyArrow S3FileSystem checksum validation issues.

    Args:
        file_key (str): The S3 key of the parquet file
        s3 (fs.S3FileSystem): S3 filesystem instance (not used, kept for compatibility)
        bucket_name (str): Name of the S3 bucket
        max_retries (int): Maximum number of retry attempts (default: 3)

    Returns:
        tuple: (Polars DataFrame, list of available columns)

    Raises:
        Exception: If all retry attempts fail
    """
    # Create boto3 client for reliable file download (bypasses PyArrow checksum validation)
    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    def _download_and_read():
        # Download file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Download using boto3 (handles checksums more reliably)
            s3_client.download_file(bucket_name, file_key, tmp_path)
            # Read from local file (no checksum validation issues)
            return pl.read_parquet(tmp_path)
        finally:
            # Clean up temp file
            with contextlib.suppress(Exception):
                Path(tmp_path).unlink(missing_ok=True)

    # Read the file with retry logic - get all columns first
    df_full = _read_parquet_with_retry(_download_and_read, file_key, max_retries)

    # Get available columns from the DataFrame
    available_columns = df_full.columns

    # Select columns with priority
    selected_columns = select_priority_columns(available_columns)

    # If we need to filter columns, do it from the already-read DataFrame
    if selected_columns and set(selected_columns) != set(available_columns):
        df = df_full.select(selected_columns)
    else:
        df = df_full

    return df, available_columns


def _handle_bldg_id_column(df: pl.DataFrame, file_key: str, s3: Any, bucket_name: str) -> pl.DataFrame:
    """
    Handle bldg_id column extraction and validation.

    Args:
        df (pl.DataFrame): DataFrame to process
        file_key (str): The S3 key of the parquet file
        s3 (fs.S3FileSystem): S3 filesystem instance
        bucket_name (str): Name of the S3 bucket

    Returns:
        pl.DataFrame: DataFrame with bldg_id properly handled
    """
    # Check if bldg_id is missing from columns
    if "bldg_id" not in df.columns:
        # Create boto3 client for reliable file download (bypasses PyArrow checksum validation)
        s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

        def _download_and_read_full():
            # Download file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp_file:
                tmp_path = tmp_file.name

            try:
                # Download using boto3 (handles checksums more reliably)
                s3_client.download_file(bucket_name, file_key, tmp_path)
                # Read from local file (no checksum validation issues)
                return pl.read_parquet(tmp_path)
            finally:
                # Clean up temp file
                with contextlib.suppress(Exception):
                    Path(tmp_path).unlink(missing_ok=True)

        # Try reading the full parquet file to check all columns with retry logic
        df_full = _read_parquet_with_retry(_download_and_read_full, file_key, max_retries=3)

        if "bldg_id" in df_full.columns:
            # If bldg_id exists in the full file, use it
            df = df_full
            logger.info("Found bldg_id in full parquet file")
        else:
            logger.warning("Warning: bldg_id column not found in parquet file")

    return df


def _validate_bldg_id(df: pl.DataFrame, release_name: Union[str, None]) -> bool:
    """
    Validate bldg_id column and return True if valid.

    Args:
        df (pl.DataFrame): DataFrame to validate
        release_name (str | None): Name of the release for error reporting

    Returns:
        bool: True if bldg_id is valid, False otherwise
    """
    if "bldg_id" not in df.columns:
        logger.exception(f"ERROR: bldg_id column not found in DataFrame after processing in {release_name}")
        logger.exception("This indicates the column was not properly extracted from the parquet file")
        return False

    # Ensure bldg_id is integer if it exists
    df = df.with_columns(pl.col("bldg_id").cast(pl.Int64))
    logger.info("Converted bldg_id to integer type")

    # Verify that bldg_id contains proper integer values
    nan_count = df.filter(pl.col("bldg_id").is_null()).height
    total_rows = df.height

    if nan_count == total_rows:
        logger.exception(f"ERROR: bldg_id column contains only null values after conversion in {release_name}")
        logger.exception("This indicates the column was not properly extracted from the parquet file")
        return False
    elif nan_count > total_rows * 0.5:  # More than 50% null
        logger.exception(
            f"ERROR: bldg_id column has {nan_count} null values out of {total_rows} total rows in {release_name}"
        )
        logger.exception("This indicates the column was not properly extracted from the parquet file")
        return False
    else:
        # Remove rows where bldg_id is null
        df = df.filter(pl.col("bldg_id").is_not_null())
        logger.info(f"Removed {nan_count} rows with null bldg_id values in {release_name}")
        logger.info(f"bldg_id column verified: {total_rows - nan_count} valid integer values in {release_name}")
        return True


def _add_metadata_columns(
    df: pl.DataFrame,
    res_com: Union[str, None],
    weather: Union[str, None],
    release_version: Union[str, None],
    release_year: Union[str, None],
) -> pl.DataFrame:
    """
    Add metadata columns to DataFrame.

    Args:
        df (pl.DataFrame): DataFrame to add columns to
        res_com (str | None): The res_com value
        weather (str | None): The weather value
        release_version (str | None): The release number
        release_year (str | None): The release year

    Returns:
        pl.DataFrame: DataFrame with metadata columns added
    """
    columns_to_add = []

    if res_com is not None:
        columns_to_add.append(pl.lit(res_com).alias("product"))
    if weather is not None:
        columns_to_add.append(pl.lit(weather).alias("weather_file"))
    if release_version is not None:
        columns_to_add.append(pl.lit(release_version).alias("release_version"))
    if release_year is not None:
        columns_to_add.append(pl.lit(release_year).alias("release_year"))

    if columns_to_add:
        df = df.with_columns(columns_to_add)

    return df


def process_parquet_file(
    file_key: str,
    s3: Any,
    bucket_name: str,
    res_com: Union[str, None] = None,
    weather: Union[str, None] = None,
    release_version: Union[str, None] = None,
    release_year: Union[str, None] = None,
    release_name: Union[str, None] = None,
) -> Union[pl.DataFrame, None]:
    """
    Process a parquet file from S3 and return a polars DataFrame.

    Args:
        file_key (str): The S3 key of the parquet file
        s3 (fs.S3FileSystem): S3 filesystem instance
        bucket_name (str): Name of the S3 bucket
        res_com (str): The res_com value (resstock or comstock)
        weather (str): The weather value (tmy3, amy2012, or amy2018)
        release_version (str): The release number
        release_year (str): The release year
        release_name (str): The release name

    Returns:
        pl.DataFrame: Processed DataFrame
    """
    try:
        # Read parquet file with selected columns
        df, available_columns = _read_parquet_with_columns(file_key, s3, bucket_name)

        # Handle bldg_id column extraction
        df = _handle_bldg_id_column(df, file_key, s3, bucket_name)

        # Rename columns to standardized names
        df = rename_columns(df)
        logger.info(f"Columns after renaming: {df.columns}")

        # Validate bldg_id column
        if not _validate_bldg_id(df, release_name):
            return None

        # Add metadata columns
        df = _add_metadata_columns(df, res_com, weather, release_version, release_year)

        # Keep only unique rows based on bldg_id
        initial_rows = df.height
        df = df.unique(subset=["bldg_id"], keep="first")
        removed_rows = initial_rows - df.height
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} duplicate rows based on bldg_id in {release_name}")

        logger.info(f"Successfully loaded {df.height} rows with {len(df.columns)} columns in {release_name}")

    except Exception:
        logger.exception("Error processing file")
        return None
    else:
        return df


def _extract_upgrade_number(filename: str) -> Union[str, None]:
    """
    Extract upgrade number from filename like 'AK_G0200130_upgrade01.parquet'.

    Args:
        filename (str): The filename to parse

    Returns:
        str | None: The upgrade number (e.g., '01') or None if not found
    """
    if "upgrade" not in filename:
        return None

    # Find the upgrade number after "upgrade" in the filename
    upgrade_index = filename.find("upgrade")
    if upgrade_index == -1:
        return None

    # Extract the number after "upgrade"
    upgrade_part = filename[upgrade_index + 7 :]  # 7 is length of "upgrade"
    # Find where the number ends (before .parquet)
    dot_index = upgrade_part.find(".parquet")
    if dot_index == -1:
        return None

    return upgrade_part[:dot_index]


def convert_state_names_to_abbreviations(data_dir: Path, skip_sorting: bool = False) -> None:
    """
    Convert full state names to abbreviations in partitioned parquet files.

    ONLY renames state directory names from full names to abbreviations.
    Does NOT change any data or partition structure.

    Args:
        data_dir (Path): Directory containing the building_data folder
        skip_sorting (bool): Ignored, kept for compatibility
    """
    import os

    # State name to abbreviation mapping
    state_mapping = {
        "Alabama": "AL",
        "Alaska": "AK",
        "Arizona": "AZ",
        "Arkansas": "AR",
        "California": "CA",
        "Colorado": "CO",
        "Connecticut": "CT",
        "Delaware": "DE",
        "Florida": "FL",
        "Georgia": "GA",
        "Hawaii": "HI",
        "Idaho": "ID",
        "Illinois": "IL",
        "Indiana": "IN",
        "Iowa": "IA",
        "Kansas": "KS",
        "Kentucky": "KY",
        "Louisiana": "LA",
        "Maine": "ME",
        "Maryland": "MD",
        "Massachusetts": "MA",
        "Michigan": "MI",
        "Minnesota": "MN",
        "Mississippi": "MS",
        "Missouri": "MO",
        "Montana": "MT",
        "Nebraska": "NE",
        "Nevada": "NV",
        "New Hampshire": "NH",
        "New Jersey": "NJ",
        "New Mexico": "NM",
        "New York": "NY",
        "North Carolina": "NC",
        "North Dakota": "ND",
        "Ohio": "OH",
        "Oklahoma": "OK",
        "Oregon": "OR",
        "Pennsylvania": "PA",
        "Rhode Island": "RI",
        "South Carolina": "SC",
        "South Dakota": "SD",
        "Tennessee": "TN",
        "Texas": "TX",
        "Utah": "UT",
        "Vermont": "VT",
        "Virginia": "VA",
        "Washington": "WA",
        "West Virginia": "WV",
        "Wisconsin": "WI",
        "Wyoming": "WY",
        "District of Columbia": "DC",
        # Handle URL-encoded versions
        "New%20Hampshire": "NH",
        "New%20Jersey": "NJ",
        "New%20Mexico": "NM",
        "New%20York": "NY",
        "North%20Carolina": "NC",
        "North%20Dakota": "ND",
        "Rhode%20Island": "RI",
        "South%20Carolina": "SC",
        "South%20Dakota": "SD",
        "West%20Virginia": "WV",
        "District%20of%20Columbia": "DC",
    }

    parquet_dir = data_dir / "building_data" / "combined_metadata.parquet"

    if not parquet_dir.exists():
        logger.error(f"Parquet directory not found: {parquet_dir}")
        return

    logger.info("Starting state directory name conversion")

    # Walk through all directories and rename state directories
    for root, dirs, _files in os.walk(parquet_dir):
        for dir_name in dirs:
            if dir_name.startswith("state="):
                state_name = dir_name[6:]  # Remove "state=" prefix
                if state_name in state_mapping:
                    old_path = os.path.join(root, dir_name)
                    new_dir_name = f"state={state_mapping[state_name]}"
                    new_path = os.path.join(root, new_dir_name)

                    if old_path != new_path:
                        logger.info(f"Renaming: {old_path} -> {new_path}")
                        os.rename(old_path, new_path)

    logger.info("Successfully completed state directory name conversion")


def find_all_parquet_files(base_file_key: str, s3_client: Any, bucket_name: str) -> dict:
    """
    Find all parquet files in the S3 bucket and organize them by upgrade type.

    Returns:
        dict: Dictionary with keys 'baseline' and 'upgrade_XX' containing lists of file paths
    """
    all_files: dict[str, list[str]] = {}
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=base_file_key)

    for page in pages:
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]

                # Check if file ends with .parquet
                if not key.endswith(".parquet"):
                    continue

                if key.endswith("baseline.parquet"):
                    upgrade_num = 0
                    upgrade_key = f"upgrade_{upgrade_num}"
                    if upgrade_key not in all_files:
                        all_files[upgrade_key] = []
                    all_files[upgrade_key].append(key)
                elif "upgrade" in key:
                    # Extract upgrade number from filename
                    parts = key.split("/")
                    filename = parts[-1]
                    upgrade_num = _extract_upgrade_number(filename)
                    upgrade_num = int(upgrade_num)

                    if upgrade_num is not None:
                        upgrade_key = f"upgrade_{upgrade_num}"
                        if upgrade_key not in all_files:
                            all_files[upgrade_key] = []
                        all_files[upgrade_key].append(key)

    return all_files


def _process_single_file(
    file_key: str,
    bucket_name: str,
    res_com: Union[str, None],
    weather: Union[str, None],
    release_version: Union[str, None],
    release_year: Union[str, None],
    release_name: Union[str, None],
) -> Union[pl.DataFrame, None]:
    """
    Process a single parquet file. This function is designed to be called in parallel.

    Args:
        file_key: S3 key of the parquet file
        bucket_name: S3 bucket name
        res_com: The res_com value
        weather: The weather value
        release_version: The release version
        release_year: The release year
        release_name: The release name

    Returns:
        DataFrame if successful, None otherwise
    """
    # Create S3 filesystem for this thread (thread-safe)
    s3 = fs.S3FileSystem(anonymous=True, region="us-west-2")
    try:
        df = process_parquet_file(
            file_key, s3, bucket_name, res_com, weather, release_version, release_year, release_name
        )
        if df is not None and df.height > 0:
            return df
        else:
            logger.error(f"Failed to process {file_key} for {release_name}")
            return None
    except Exception:
        logger.exception(f"Error processing file {file_key} for {release_name}")
        return None


def _process_release_2021(
    release_name: str,
    release: dict[str, Any],
    bucket_name: str,
) -> Union[pl.DataFrame, None]:
    """Process a 2021 release."""
    release_year = release["release_year"]
    res_com = release["res_com"]
    weather = release["weather"]
    release_version = release["release_number"]

    file_key = (
        f"nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
        f"{release_year}/{res_com}_{weather}_release_{release_version}/metadata/metadata.parquet"
    )
    return _process_single_file(file_key, bucket_name, res_com, weather, release_version, release_year, release_name)


def _process_release_2022_2024(
    release_name: str,
    release: dict[str, Any],
    bucket_name: str,
) -> Union[pl.DataFrame, None]:
    """Process a 2022-2024 release (excluding special 2024 releases)."""
    release_year = release["release_year"]
    res_com = release["res_com"]
    weather = release["weather"]
    release_version = release["release_number"]
    upgrade_ids = release.get("upgrade_ids", [])

    if not upgrade_ids:
        return None

    upgrade_id = int(upgrade_ids[0])
    if upgrade_id == 0:
        file_key = (
            f"nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
            f"{release_year}/{res_com}_{weather}_release_{release_version}/metadata/baseline.parquet"
        )
    else:
        file_key = (
            f"nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
            f"{release_year}/{res_com}_{weather}_release_{release_version}/metadata/"
            f"upgrade{upgrade_id:02d}.parquet"
        )

    return _process_single_file(file_key, bucket_name, res_com, weather, release_version, release_year, release_name)


def _process_release_2024_2025(
    release_name: str,
    release: dict[str, Any],
    bucket_name: str,
    s3_client: Any,
) -> Union[pl.DataFrame, None]:
    """Process a 2024/2025 release with multiple files."""
    release_year = release["release_year"]
    res_com = release["res_com"]
    weather = release["weather"]
    release_version = release["release_number"]
    upgrade_ids = release.get("upgrade_ids", [])

    if not upgrade_ids:
        return None

    if release_name == "res_2025_amy2018_1":
        base_file_key = (
            f"nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
            f"{release_year}/{res_com}_{weather}_release_{release_version}/metadata_and_annual_results/"
            f"by_state/full/parquet/"
        )
    else:
        base_file_key = (
            f"nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
            f"{release_year}/{res_com}_{weather}_release_{release_version}/"
            f"metadata_and_annual_results/by_state_and_county/full/parquet/"
        )

    # Find all available parquet files
    all_parquet_files = find_all_parquet_files(base_file_key, s3_client, bucket_name)
    logger.info(f"Found parquet files for {release_name}: {list(all_parquet_files.keys())}")

    upgrade_id = int(upgrade_ids[0])
    if upgrade_id == 0 and release_name == "com_2024_amy2018_2":
        file_key_suffix = "baseline.parquet"
        available_files = all_parquet_files.get("upgrade_0", [])
    elif (
        release_name == "com_2025_amy2018_1"
        or release_name == "com_2025_amy2018_2"
        or release_name == "com_2025_amy2012_2"
        or release_name == "res_2025_amy2018_1"
    ):
        file_key_suffix = f"upgrade_{upgrade_id}.parquet"
        available_files = all_parquet_files.get(f"upgrade_{upgrade_id}", [])
    else:
        return None

    logger.info(f"Found {len(available_files)} {file_key_suffix} files for {release_name}")

    if not available_files:
        logger.warning(f"No files found for {file_key_suffix} in {release_name}")
        return None

    # Process all files in parallel
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_file = {
            executor.submit(
                _process_single_file,
                file_key,
                bucket_name,
                res_com,
                weather,
                release_version,
                release_year,
                release_name,
            ): file_key
            for file_key in available_files
        }

        combined_dfs = []
        for i, future in enumerate(as_completed(future_to_file), 1):
            file_key = future_to_file[future]
            try:
                df = future.result()
                if df is not None and df.height > 0:
                    combined_dfs.append(df)
                    logger.info(f"Processed file {i}/{len(available_files)}: {file_key}")
                else:
                    logger.error(f"Failed to process {file_key} for {release_name}")
            except Exception:
                logger.exception(f"Error processing {file_key} for {release_name}")

    if combined_dfs:
        # Concatenate all DataFrames for this release
        combined_df = pl.concat(combined_dfs)
        logger.info(f"Combined {len(combined_dfs)} files into {combined_df.height} total rows in {release_name}")
        return combined_df
    else:
        logger.warning(f"No valid data found for {file_key_suffix} in {release_name}")
        return None


if __name__ == "__main__":
    # Load the release data
    releases_file = files("buildstock_fetch").joinpath("data").joinpath("buildstock_releases.json")
    data = json.loads(Path(str(releases_file)).read_text(encoding="utf-8"))

    # Directory to save the data
    data_dir = files("buildstock_fetch").joinpath("data")
    downloaded_paths: list[str] = []
    data_dir.mkdir(parents=True, exist_ok=True)

    # S3 bucket and filesystem
    bucket_name = "oedi-data-lake"
    s3_client = boto3.client("s3", region_name="us-west-2", config=Config(signature_version=UNSIGNED))

    # Process all releases in parallel
    all_dataframes: list[pl.DataFrame] = []
    failed_releases: list[str] = []

    def _process_release_wrapper(release_name: str, release: dict[str, Any]) -> tuple[str, Union[pl.DataFrame, None]]:
        """Wrapper function to process a release and return (release_name, df)."""
        release_year = release["release_year"]

        try:
            if release_year == "2021":
                df = _process_release_2021(release_name, release, bucket_name)
            elif (
                release_year == "2022"
                or release_year == "2023"
                or (release_year == "2024" and release_name != "com_2024_amy2018_2")
            ):
                df = _process_release_2022_2024(release_name, release, bucket_name)
            elif (
                release_name == "com_2024_amy2018_2"
                or release_name == "com_2025_amy2018_1"
                or release_name == "com_2025_amy2018_2"
                or release_name == "com_2025_amy2012_2"
                or release_name == "res_2025_amy2018_1"
            ):
                df = _process_release_2024_2025(release_name, release, bucket_name, s3_client)
            else:
                logger.warning(f"Unknown release type for {release_name}")
                return (release_name, None)
        except Exception:
            logger.exception(f"Error processing release {release_name}")
            return (release_name, None)
        else:
            return (release_name, df)

    # Process releases in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_release = {
            executor.submit(_process_release_wrapper, release_name, release): release_name
            for release_name, release in data.items()
        }

        for future in as_completed(future_to_release):
            release_name = future_to_release[future]
            try:
                processed_name, df = future.result()
                if df is not None and df.height > 0:
                    all_dataframes.append(df)
                    logger.info(f"Successfully processed release: {processed_name}")
                else:
                    failed_releases.append(processed_name)
                    logger.error(
                        f"ERROR: Failed to process {processed_name}. bldg_id column is either missing or contains non-integer values"
                    )
            except Exception:
                failed_releases.append(release_name)
                logger.exception(f"Exception processing release {release_name}")

    # Check if any releases failed
    if failed_releases:
        logger.error(f"Failed to process {len(failed_releases)} release(s): {failed_releases}")
        logger.error("Exiting script...")
        sys.exit(1)

    # Combine all DataFrames and save as partitioned parquet
    if all_dataframes:
        logger.info(f"Combining {len(all_dataframes)} DataFrames...")

        # Ensure consistent schema across all DataFrames
        logger.info("Ensuring consistent schema across DataFrames...")
        normalized_dataframes = []
        for i, df in enumerate(all_dataframes):
            # Convert categorical columns to string to ensure compatibility
            df_normalized = df.with_columns([
                pl.col(col).cast(pl.Utf8) for col in df.columns if df.schema[col] == pl.Categorical
            ])
            normalized_dataframes.append(df_normalized)
            logger.info(f"Normalized DataFrame {i + 1} schema: {df_normalized.schema}")

        final_df = pl.concat(normalized_dataframes)
        logger.info(f"Final DataFrame has {final_df.height} rows and {len(final_df.columns)} columns")

        # Sort by county
        if "county" in final_df.columns:
            final_df = final_df.sort("county")

        # Save as partitioned parquet file
        output_file = data_dir / "building_data" / "combined_metadata.parquet"
        final_df.write_parquet(
            str(output_file),  # Convert Path to string for Polars
            use_pyarrow=True,
            partition_by=["product", "release_year", "weather_file", "release_version", "state"],
        )
        logger.info(f"Successfully saved combined DataFrame to {output_file}")

        # Convert state names to abbreviations
        logger.info("Starting state name abbreviation conversion...")
        convert_state_names_to_abbreviations(data_dir, skip_sorting=True)
    else:
        logger.warning("No data to save")
