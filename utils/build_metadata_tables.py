import contextlib
import gc
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Union

import boto3
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
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
    Uses lazy evaluation for memory efficiency on large files.
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

    # Download file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp_file:
        tmp_path = tmp_file.name

    try:

        def _download_file():
            # Download using boto3 (handles checksums more reliably)
            s3_client.download_file(bucket_name, file_key, tmp_path)

        # Download with retry logic
        logger.info(f"Downloading {file_key} to temporary location...")
        _read_parquet_with_retry(_download_file, file_key, max_retries)

        # Get file size for logging
        file_size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
        logger.info(f"Downloaded {file_key} ({file_size_mb:.2f} MB), reading with lazy evaluation...")

        # Use lazy evaluation to get schema without loading all data (memory efficient)
        lazy_df = pl.scan_parquet(tmp_path)

        # Get available columns from schema (doesn't load data)
        available_columns = lazy_df.collect_schema().names()

        # Select columns with priority
        selected_columns = select_priority_columns(available_columns)

        # Read only selected columns using lazy evaluation (memory efficient)
        # Use streaming for large files to reduce memory footprint
        if selected_columns and set(selected_columns) != set(available_columns):
            df = lazy_df.select(selected_columns).collect(streaming=True)
        else:
            df = lazy_df.collect(streaming=True)
    finally:
        # Clean up temp file immediately
        with contextlib.suppress(Exception):
            Path(tmp_path).unlink(missing_ok=True)
        # Force garbage collection after file operations
        gc.collect()

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
            # Free the full dataframe reference if we're using it
            del df_full
            gc.collect()
        else:
            logger.warning("Warning: bldg_id column not found in parquet file")
            del df_full
            gc.collect()

    return df


def _validate_bldg_id(df: pl.DataFrame, release_name: Union[str, None]) -> tuple[pl.DataFrame, bool]:
    """
    Validate bldg_id column and return the modified DataFrame and validation status.

    Args:
        df (pl.DataFrame): DataFrame to validate
        release_name (str | None): Name of the release for error reporting

    Returns:
        tuple: (modified DataFrame, True if valid, False otherwise)
    """
    if "bldg_id" not in df.columns:
        logger.exception(f"ERROR: bldg_id column not found in DataFrame after processing in {release_name}")
        logger.exception("This indicates the column was not properly extracted from the parquet file")
        return df, False

    # Ensure bldg_id is integer if it exists
    df = df.with_columns(pl.col("bldg_id").cast(pl.Int64))
    logger.info("Converted bldg_id to integer type")

    # Verify that bldg_id contains proper integer values
    nan_count = df.filter(pl.col("bldg_id").is_null()).height
    total_rows = df.height

    if nan_count == total_rows:
        logger.exception(f"ERROR: bldg_id column contains only null values after conversion in {release_name}")
        logger.exception("This indicates the column was not properly extracted from the parquet file")
        return df, False
    elif nan_count > total_rows * 0.5:  # More than 50% null
        logger.exception(
            f"ERROR: bldg_id column has {nan_count} null values out of {total_rows} total rows in {release_name}"
        )
        logger.exception("This indicates the column was not properly extracted from the parquet file")
        return df, False
    else:
        # Remove rows where bldg_id is null
        df = df.filter(pl.col("bldg_id").is_not_null())
        logger.info(f"Removed {nan_count} rows with null bldg_id values in {release_name}")
        logger.info(f"bldg_id column verified: {total_rows - nan_count} valid integer values in {release_name}")
        return df, True


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
        df, _available_columns = _read_parquet_with_columns(file_key, s3, bucket_name)

        # Handle bldg_id column extraction
        df = _handle_bldg_id_column(df, file_key, s3, bucket_name)

        # Rename columns to standardized names
        df = rename_columns(df)
        logger.info(f"Columns after renaming: {df.columns}")

        # Validate bldg_id column and get the modified dataframe
        df, is_valid = _validate_bldg_id(df, release_name)
        if not is_valid:
            del df  # Free memory if validation failed
            gc.collect()
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

        # Force garbage collection after processing
        gc.collect()

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

    print("TEST RELEASE NAME: ", release_name)
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


def _build_base_file_key_2024_2025(
    release_name: str, release_year: str, res_com: str, weather: str, release_version: str
) -> str:
    """Build the base file key for 2024/2025 releases."""
    if release_name in ("res_2025_amy2018_1", "res_2025_amy2012_1"):
        return (
            f"nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
            f"{release_year}/{res_com}_{weather}_release_{release_version}/metadata_and_annual_results/"
            f"by_state/full/parquet/"
        )
    return (
        f"nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
        f"{release_year}/{res_com}_{weather}_release_{release_version}/"
        f"metadata_and_annual_results/by_state_and_county/full/parquet/"
    )


def _get_file_key_and_available_files(
    release_name: str, upgrade_id: int, all_parquet_files: dict[str, list[str]]
) -> tuple[str, list[str]] | None:
    """Get file key suffix and available files for a release."""
    if upgrade_id == 0 and release_name == "com_2024_amy2018_2":
        file_key_suffix = "baseline.parquet"
        available_files = all_parquet_files.get("upgrade_0", [])
    elif release_name in (
        "com_2025_amy2018_1",
        "com_2025_amy2018_2",
        "com_2025_amy2012_2",
        "com_2025_amy2018_3",
        "res_2025_amy2018_1",
        "res_2025_amy2012_1",
    ):
        file_key_suffix = f"upgrade_{upgrade_id}.parquet"
        available_files = all_parquet_files.get(f"upgrade_{upgrade_id}", [])
        print("TEST AVAILABLE FILES: ", available_files)
    else:
        return None
    return file_key_suffix, available_files


def _process_files_sequentially(
    available_files: list[str],
    release_name: str,
    bucket_name: str,
    res_com: str,
    weather: str,
    release_version: str,
    release_year: str,
    temp_dir: Path,
) -> list[Path]:
    """Process files sequentially and write to disk."""
    temp_files = []
    for i, file_key in enumerate(available_files, 1):
        try:
            df = _process_single_file(
                file_key,
                bucket_name,
                res_com,
                weather,
                release_version,
                release_year,
                release_name,
            )
            if df is not None and df.height > 0:
                temp_file = temp_dir / f"file_{i:06d}.parquet"
                df.write_parquet(temp_file)
                temp_files.append(temp_file)
                del df
                gc.collect()
                logger.info(f"Processed file {i}/{len(available_files)}: {file_key}")
            else:
                logger.error(f"Failed to process {file_key} for {release_name}")
        except Exception:
            logger.exception(f"Error processing {file_key} for {release_name}")
        finally:
            gc.collect()
            if i % 10 == 0:
                gc.collect()
    return temp_files


def _combine_files_chunked(temp_files: list[Path], temp_dir: Path, chunk_size: int = 100) -> pl.DataFrame:
    """Combine files in chunks for large releases."""
    chunk_files = []
    for chunk_start in range(0, len(temp_files), chunk_size):
        try:
            chunk_file_list = temp_files[chunk_start : chunk_start + chunk_size]
            lazy_frames = [pl.scan_parquet(str(f)) for f in chunk_file_list]
            try:
                chunk_df = pl.concat(lazy_frames).collect(engine="streaming")
            except TypeError:
                chunk_df = pl.concat(lazy_frames).collect(streaming=True)
            chunk_file = temp_dir / f"chunk_{chunk_start // chunk_size:04d}.parquet"
            chunk_df.write_parquet(chunk_file)
            chunk_files.append(chunk_file)
            del lazy_frames
            del chunk_df
            gc.collect()
            logger.info(
                f"Processed chunk {chunk_start // chunk_size + 1}/{(len(temp_files) + chunk_size - 1) // chunk_size}"
            )
        except Exception:
            logger.exception(f"Error processing chunk starting at {chunk_start}")
            raise

    # Final concatenation from chunk files
    lazy_frames = [pl.scan_parquet(str(f)) for f in chunk_files]
    try:
        combined_df = pl.concat(lazy_frames).collect(engine="streaming")
    except TypeError:
        combined_df = pl.concat(lazy_frames).collect(streaming=True)
    del lazy_frames
    gc.collect()
    return combined_df


def _combine_files_direct(temp_files: list[Path]) -> pl.DataFrame:
    """Combine files directly for smaller releases."""
    lazy_frames = [pl.scan_parquet(str(f)) for f in temp_files]
    try:
        combined_df = pl.concat(lazy_frames).collect(engine="streaming")
    except TypeError:
        combined_df = pl.concat(lazy_frames).collect(streaming=True)
    del lazy_frames
    gc.collect()
    return combined_df


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

    base_file_key = _build_base_file_key_2024_2025(release_name, release_year, res_com, weather, release_version)
    all_parquet_files = find_all_parquet_files(base_file_key, s3_client, bucket_name)
    logger.info(f"Found parquet files for {release_name}: {list(all_parquet_files.keys())}")

    upgrade_id = int(upgrade_ids[0])
    result = _get_file_key_and_available_files(release_name, upgrade_id, all_parquet_files)
    if result is None:
        return None
    file_key_suffix, available_files = result

    logger.info(f"Found {len(available_files)} {file_key_suffix} files for {release_name}")

    if not available_files:
        logger.warning(f"No files found for {file_key_suffix} in {release_name}")
        return None

    temp_dir = Path(tempfile.mkdtemp(prefix=f"release_{release_name}_"))
    try:
        temp_files = _process_files_sequentially(
            available_files,
            release_name,
            bucket_name,
            res_com,
            weather,
            release_version,
            release_year,
            temp_dir,
        )

        if not temp_files:
            logger.warning(f"No valid data found for {file_key_suffix} in {release_name}")
            return None

        logger.info(f"Combining {len(temp_files)} files from disk for {release_name}...")
        try:
            if len(temp_files) > 200:
                combined_df = _combine_files_chunked(temp_files, temp_dir)
            else:
                combined_df = _combine_files_direct(temp_files)

            logger.info(f"Combined {len(temp_files)} files into {combined_df.height} total rows in {release_name}")
        except Exception:
            logger.exception(f"Error combining files for {release_name}")
            raise
        else:
            return combined_df
    finally:
        with contextlib.suppress(Exception):
            shutil.rmtree(temp_dir)
        gc.collect()


def _process_release_2024_res_tmy3_1(
    release_name: str,
    release: dict[str, Any],
    bucket_name: str,
    s3_client: Any,
) -> Union[pl.DataFrame, None]:
    """Process a 2024 ResStock TMY3 release 1."""
    release_year = "2024"
    res_com = "resstock"
    weather = "tmy3"
    release_version = "1"
    upgrade_ids = release.get("upgrade_ids", [])
    if not upgrade_ids:
        return None
    upgrade_id = int(upgrade_ids[0])
    if upgrade_id == 0:
        file_key = (
            f"nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
            f"{release_year}/resstock_dataset_2024.1/resstock_tmy3/metadata/"
            f"baseline.parquet"
        )
    else:
        file_key = (
            f"nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
            f"{release_year}/resstock_dataset_2024.1/resstock_tmy3/metadata/"
            f"upgrade{upgrade_id:02d}.parquet"
        )
    return _process_single_file(file_key, bucket_name, res_com, weather, release_version, release_year, release_name)


if __name__ == "__main__":
    # Get the project root (parent of utils directory where this script is located)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "buildstock_fetch" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load the release data from source directory, not installed package
    releases_file = data_dir / "buildstock_releases.json"
    if not releases_file.exists():
        logger.error(f"Releases file not found at {releases_file}")
        sys.exit(1)

    data = json.loads(releases_file.read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(data)} releases from {releases_file}")
    logger.info(f"Using data directory: {data_dir}")

    # S3 bucket and filesystem
    bucket_name = "oedi-data-lake"
    s3_client = boto3.client("s3", region_name="us-west-2", config=Config(signature_version=UNSIGNED))

    # Process releases sequentially and write to disk
    failed_releases: list[str] = []

    logger.info(f"Releases to process: {list(data.keys())}")

    def _process_release_wrapper(release_name: str, release: dict[str, Any]) -> tuple[str, Union[pl.DataFrame, None]]:
        """Wrapper function to process a release and return (release_name, df)."""
        release_year = release["release_year"]

        try:
            if release_year == "2021":
                df = _process_release_2021(release_name, release, bucket_name)
            elif (
                release_year == "2022"
                or release_year == "2023"
                or (
                    release_year == "2024"
                    and (release_name != "com_2024_amy2018_2" and release_name != "res_2024_tmy3_1")
                )
            ):
                df = _process_release_2022_2024(release_name, release, bucket_name)
            elif release_name == "res_2024_tmy3_1":
                df = _process_release_2024_res_tmy3_1(release_name, release, bucket_name, s3_client)

            elif (
                release_name == "com_2024_amy2018_2"
                or release_name == "com_2025_amy2018_1"
                or release_name == "com_2025_amy2018_2"
                or release_name == "com_2025_amy2018_3"
                or release_name == "com_2025_amy2012_2"
                or release_name == "res_2025_amy2018_1"
                or release_name == "res_2025_amy2012_1"
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

    # Process releases SEQUENTIALLY to prevent memory overload
    # Write each release to disk immediately after processing
    # Use utils directory for temp_releases instead of data/building_data
    temp_releases_dir = script_dir / "temp_releases"
    temp_releases_dir.mkdir(parents=True, exist_ok=True)

    # Migrate existing temp files from old location (data/building_data/temp_releases) if they exist
    old_temp_releases_dir = data_dir / "building_data" / "temp_releases"
    if old_temp_releases_dir.exists() and old_temp_releases_dir.is_dir():
        old_temp_files = list(old_temp_releases_dir.glob("release_*.parquet"))
        if old_temp_files:
            logger.info(f"Found {len(old_temp_files)} temp files in old location, migrating to new location...")
            for old_file in old_temp_files:
                new_file = temp_releases_dir / old_file.name
                if not new_file.exists():
                    try:
                        shutil.move(str(old_file), str(new_file))
                        logger.info(f"Migrated {old_file.name} to new location")
                    except Exception as e:
                        logger.warning(f"Could not migrate {old_file.name}: {e}")
                else:
                    logger.info(f"Skipping {old_file.name} (already exists in new location)")
                    with contextlib.suppress(Exception):
                        old_file.unlink()
            # Try to remove old directory if empty
            with contextlib.suppress(Exception):
                if not any(old_temp_releases_dir.iterdir()):
                    old_temp_releases_dir.rmdir()
                    logger.info("Removed empty old temp_releases directory")

    # Check which temp_releases files already exist
    existing_temp_files = {f.stem.replace("release_", "") for f in temp_releases_dir.glob("release_*.parquet")}
    logger.info(f"Found {len(existing_temp_files)} existing temp release files in {temp_releases_dir}")

    # Only process releases that don't have existing temp files
    releases_to_process = {name: release for name, release in data.items() if name not in existing_temp_files}
    skipped_releases = {name: release for name, release in data.items() if name in existing_temp_files}

    if skipped_releases:
        logger.info(
            f"Skipping {len(skipped_releases)} releases that already have temp files: {list(skipped_releases.keys())}"
        )

    if releases_to_process:
        logger.info(f"Processing {len(releases_to_process)} new releases: {list(releases_to_process.keys())}")

        # Process releases one at a time
        for release_name, release in releases_to_process.items():
            try:
                logger.info(f"Processing release: {release_name}")
                processed_name, df = _process_release_wrapper(release_name, release)

                if df is not None and df.height > 0:
                    # Write release to disk immediately to free memory
                    temp_file = temp_releases_dir / f"release_{processed_name}.parquet"
                    df.write_parquet(temp_file)
                    logger.info(f"Successfully processed and saved release: {processed_name} ({df.height} rows)")
                    del df  # Free memory immediately
                    gc.collect()
                else:
                    failed_releases.append(processed_name)
                    logger.error(
                        f"ERROR: Failed to process {processed_name}. bldg_id column is either missing or contains non-integer values"
                    )
            except Exception:
                failed_releases.append(release_name)
                logger.exception(f"Exception processing release {release_name}")
            finally:
                # Force garbage collection after each release
                gc.collect()
    else:
        logger.info("All releases already have temp files, skipping download phase")

    # Collect all temp release files (including existing ones)
    temp_release_files = sorted(temp_releases_dir.glob("release_*.parquet"))
    logger.info(f"Found {len(temp_release_files)} total temp release files (including existing ones)")

    # Check if any releases failed
    if failed_releases:
        logger.error(f"Failed to process {len(failed_releases)} release(s): {failed_releases}")
        logger.error("Exiting script...")
        sys.exit(1)

    # Combine all releases from disk using memory-efficient chunked approach
    # This prevents OOM by never loading all data into memory at once
    if temp_release_files:
        logger.info(
            f"Combining {len(temp_release_files)} releases from disk using memory-efficient chunked approach..."
        )

        # Save as partitioned parquet file
        # When using partition_by, the output is a directory, not a file
        output_dir = data_dir / "building_data" / "combined_metadata.parquet"

        # Ensure parent directory exists
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        # Remove existing output directory if it exists (from previous runs)
        if output_dir.exists():
            if output_dir.is_dir():
                logger.info(f"Removing existing output directory: {output_dir}")
                shutil.rmtree(output_dir)
            else:
                # If it's a file, remove it (shouldn't happen with partition_by, but handle it)
                logger.warning(f"Removing existing file at output location: {output_dir}")
                output_dir.unlink()

        partition_cols = ["product", "release_year", "weather_file", "release_version", "state"]
        logger.info(f"Partitioning by: {partition_cols}")

        # Process files in chunks to avoid memory issues
        # Use PyArrow dataset writing directly from lazy frames
        chunk_size = 5  # Process 5 files at a time
        total_chunks = (len(temp_release_files) + chunk_size - 1) // chunk_size

        try:
            # Process files in chunks and write directly to partitioned output
            for chunk_idx in range(total_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min(chunk_start + chunk_size, len(temp_release_files))
                chunk_files = temp_release_files[chunk_start:chunk_end]

                logger.info(f"Processing chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_files)} files)...")

                # Read chunk files using lazy evaluation
                chunk_lazy_frames = []
                for temp_file in chunk_files:
                    try:
                        lazy_df = pl.scan_parquet(str(temp_file))
                        # Normalize schema: convert categorical to string for each file individually
                        # This ensures all files have consistent schemas when concatenated
                        file_schema = lazy_df.collect_schema()
                        categorical_cols = [col for col in file_schema.names() if file_schema[col] == pl.Categorical]
                        if categorical_cols:
                            casts = {col: pl.col(col).cast(pl.Utf8) for col in categorical_cols}
                            lazy_df = lazy_df.with_columns([casts[col] for col in casts])
                        chunk_lazy_frames.append(lazy_df)
                    except Exception:
                        logger.exception(f"Error loading lazy frame from {temp_file.name}")
                        raise

                # Concatenate chunk using lazy evaluation
                chunk_combined = pl.concat(chunk_lazy_frames) if len(chunk_lazy_frames) > 1 else chunk_lazy_frames[0]

                # Sort in lazy evaluation (memory efficient)
                try:
                    schema = chunk_combined.collect_schema()
                    sort_cols = [col for col in partition_cols if col in schema.names()]
                    if "county" in schema.names():
                        sort_cols.append("county")

                    if sort_cols:
                        chunk_combined = chunk_combined.sort(sort_cols)
                except Exception as e:
                    logger.warning(f"Could not sort chunk in lazy evaluation: {e}. Continuing without sort...")

                # Convert to PyArrow table and write directly to partitioned dataset
                # This writes incrementally without loading all data into memory
                try:
                    # Collect chunk with streaming
                    try:
                        chunk_df = chunk_combined.collect(engine="streaming")
                    except TypeError:
                        chunk_df = chunk_combined.collect(streaming=True)

                    # Convert to PyArrow table
                    chunk_table = chunk_df.to_arrow()
                    del chunk_df
                    gc.collect()

                    # Write chunk to partitioned dataset
                    # Use append mode for chunks after the first
                    if chunk_idx == 0:
                        # First chunk: create the dataset
                        partition_schema = pa.schema([
                            chunk_table.schema.field(col) for col in partition_cols if col in chunk_table.column_names
                        ])
                        ds.write_dataset(
                            chunk_table,
                            base_dir=str(output_dir),
                            format="parquet",
                            partitioning=ds.partitioning(schema=partition_schema, flavor="hive"),
                            basename_template=f"chunk-{chunk_idx:04d}-{{i}}.parquet",
                            existing_data_behavior="delete_matching",
                        )
                    else:
                        # Subsequent chunks: append to existing dataset
                        partition_schema = pa.schema([
                            chunk_table.schema.field(col) for col in partition_cols if col in chunk_table.column_names
                        ])
                        ds.write_dataset(
                            chunk_table,
                            base_dir=str(output_dir),
                            format="parquet",
                            partitioning=ds.partitioning(schema=partition_schema, flavor="hive"),
                            basename_template=f"chunk-{chunk_idx:04d}-{{i}}.parquet",
                            existing_data_behavior="overwrite_or_ignore",
                        )

                    del chunk_table
                    del chunk_lazy_frames
                    del chunk_combined
                    gc.collect()

                    logger.info(f"Completed chunk {chunk_idx + 1}/{total_chunks}")

                except Exception:
                    logger.exception(f"Error processing chunk {chunk_idx + 1}")
                    raise

            logger.info("Successfully combined all releases using chunked approach")

        except Exception:
            logger.exception("Error combining releases")
            gc.collect()
            raise

        # Verify output directory was created and log partition information
        # Ensure parent directory exists (in case it wasn't created)
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        # Verify files were actually written
        if not output_dir.exists():
            msg = f"Output directory {output_dir} was not created"
            logger.error(f"ERROR: {msg}")
            raise FileNotFoundError(msg)

        if not output_dir.is_dir():
            msg = f"Output path {output_dir} is not a directory"
            logger.error(f"ERROR: {msg}")
            raise NotADirectoryError(msg)

        # Log partition values for debugging (read from output directory)
        logger.info("Reading partition information from output directory...")
        try:
            # Read a sample of the output to get partition info
            output_dataset = ds.dataset(str(output_dir), format="parquet")
            for col in partition_cols:
                try:
                    # Get unique values from the partition
                    # This is memory-efficient as it only reads metadata
                    unique_vals = set()
                    for fragment in output_dataset.get_fragments():
                        partition_values = fragment.partition_expression
                        if partition_values:
                            # Extract partition values from expression
                            for part in partition_values:
                                if col in str(part):
                                    # Extract value from partition expression
                                    val_str = str(part)
                                    if "=" in val_str:
                                        val = val_str.split("=")[-1].strip("'\"")
                                        unique_vals.add(val)
                    unique_vals_list = sorted(unique_vals)
                    logger.info(
                        f"  {col}: {unique_vals_list[:10]}{'...' if len(unique_vals_list) > 10 else ''} ({len(unique_vals_list)} unique values)"
                    )
                except Exception as e:
                    logger.warning(f"  Could not get unique values for {col}: {e}")
        except Exception as e:
            logger.warning(f"Could not read partition information: {e}")

        # Note: We do NOT clean up temporary release files anymore
        # They are kept in utils/temp_releases for future runs to skip already-downloaded releases
        logger.info(f"Keeping temp release files in {temp_releases_dir} for future runs")

        # Verify parquet files exist
        parquet_files = list(output_dir.rglob("*.parquet"))
        logger.info(f"Successfully saved combined DataFrame to {output_dir}")
        logger.info(f"Created {len(parquet_files)} parquet files in partition structure")
        if parquet_files:
            # Show the directory structure
            sample_dirs = sorted({f.parent.relative_to(output_dir) for f in parquet_files[:10]})
            logger.info(f"Sample partition directories: {[str(d) for d in sample_dirs[:5]]}")
            logger.info(f"Sample file locations: {[str(f.relative_to(output_dir)) for f in parquet_files[:5]]}")
        else:
            logger.warning(f"WARNING: No parquet files found in output directory {output_dir}")

        # Convert state names to abbreviations
        logger.info("Starting state name abbreviation conversion...")
        convert_state_names_to_abbreviations(data_dir, skip_sorting=True)

        # Final garbage collection
        gc.collect()
    else:
        logger.warning("No data to save")
