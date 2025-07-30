import json
import logging
import os
import sys
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


def _read_parquet_with_columns(file_key: str, s3: Any, bucket_name: str) -> tuple[pl.DataFrame, list]:
    """
    Read parquet file and return Polars DataFrame with selected columns.

    Args:
        file_key (str): The S3 key of the parquet file
        s3 (fs.S3FileSystem): S3 filesystem instance
        bucket_name (str): Name of the S3 bucket

    Returns:
        tuple: (Polars DataFrame, list of available columns)
    """
    # Use PyArrow's S3FileSystem to read the file
    s3_file_path = f"{bucket_name}/{file_key}"

    # Read the file using PyArrow's S3FileSystem, then convert to Polars
    with s3.open_input_file(s3_file_path) as f:
        # Get available columns by reading the file
        lazy_df = pl.scan_parquet(f)
        available_columns = lazy_df.collect_schema().names()

        # Select columns with priority
        selected_columns = select_priority_columns(available_columns)

        # Reset file pointer and read with selected columns
        f.seek(0)
        df = pl.read_parquet(f, columns=selected_columns) if selected_columns else pl.read_parquet(f)

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
        # Construct S3 path
        s3_path = f"s3://{bucket_name}/{file_key}"

        # Extract storage options from s3fs filesystem
        storage_options = {
            "aws_access_key_id": getattr(s3, "key", None),
            "aws_secret_access_key": getattr(s3, "secret", None),
            "aws_region": getattr(s3, "client_kwargs", {}).get("region_name", "us-west-2"),
        }

        # Remove None values from storage_options
        storage_options = {k: v for k, v in storage_options.items() if v is not None}

        # Try reading the full parquet file to check all columns
        df_full = pl.read_parquet(s3_path, storage_options=storage_options)

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

        logger.info(f"Successfully loaded {df.height} rows with {len(df.columns)} columns in {release_name}")

    except Exception:
        logger.exception("Error processing file")
        return None
    else:
        return df


def find_metadata_files(base_file_key: str, file_key_suffix: str, s3_client: Any, bucket_name: str) -> list:
    """
    Find all metadata files in the S3 bucket.
    """
    download_files = []
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=base_file_key)
    for page in pages:
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]

                # Check if file ends with 'baseline.parquet'
                if key.endswith(file_key_suffix):
                    download_files.append(key)
    return download_files


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
                    if "baseline" not in all_files:
                        all_files["baseline"] = []
                    all_files["baseline"].append(key)
                elif "upgrade" in key:
                    # Extract upgrade number from filename
                    parts = key.split("/")
                    filename = parts[-1]
                    upgrade_num = _extract_upgrade_number(filename)

                    if upgrade_num is not None:
                        upgrade_key = f"upgrade_{upgrade_num}"
                        if upgrade_key not in all_files:
                            all_files[upgrade_key] = []
                        all_files[upgrade_key].append(key)

    return all_files


if __name__ == "__main__":
    # Load the release data
    releases_file = files("buildstock_fetch.utils").joinpath("buildstock_releases.json")
    with open(str(releases_file)) as file:
        data = json.load(file)

    # Directory to save the data
    data_dir = Path(__file__).parent.parent / "data"
    downloaded_paths: list[str] = []
    os.makedirs(data_dir, exist_ok=True)

    # List to collect all DataFrames
    all_dataframes = []

    # S3 bucket and filesystem
    bucket_name = "oedi-data-lake"
    s3 = fs.S3FileSystem(anonymous=True, region="us-west-2")
    s3_client = boto3.client("s3", region_name="us-west-2", config=Config(signature_version=UNSIGNED))

    for release_name, release in data.items():
        release_year = release["release_year"]
        res_com = release["res_com"]
        weather = release["weather"]
        release_version = release["release_number"]
        upgrade_ids = release.get("upgrade_ids", [])

        if release_year == "2021":
            file_key = (
                f"nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
                f"{release_year}/{res_com}_{weather}_release_{release_version}/metadata/metadata.parquet"
            )
            df = process_parquet_file(
                file_key, s3, bucket_name, res_com, weather, release_version, release_year, release_name
            )
            if df is not None and df.height > 0:
                all_dataframes.append(df)
            else:
                logger.exception(
                    f"ERROR: Failed to process {release_name}. bldg_id column is either missing or contains non-integer values"
                )
                logger.exception("Exiting script...")
                sys.exit(1)

        elif (
            release_year == "2022"
            or release_year == "2023"
            or (release_year == "2024" and release_name != "com_2024_amy2018_2")
        ):
            # Process only the first upgrade_id
            if upgrade_ids:
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

                df = process_parquet_file(
                    file_key, s3, bucket_name, res_com, weather, release_version, release_year, release_name
                )
                if df is not None and df.height > 0:
                    all_dataframes.append(df)
                else:
                    logger.exception(
                        f"ERROR: Failed to process {release_name}. bldg_id column is either missing or contains non-integer values"
                    )
                    logger.exception("Exiting script...")
                    sys.exit(1)

        elif release_name == "com_2024_amy2018_2" or release_name == "com_2025_amy2018_1":
            base_file_key = (
                f"nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
                f"{release_year}/{res_com}_{weather}_release_{release_version}/"
                f"metadata_and_annual_results/by_state_and_county/full/parquet/"
            )

            # Find all available parquet files at once
            all_parquet_files = find_all_parquet_files(base_file_key, s3_client, bucket_name)
            logger.info(f"Found parquet files: {list(all_parquet_files.keys())}")

            # Process only the first upgrade_id
            if upgrade_ids:
                upgrade_id = int(upgrade_ids[0])
                if upgrade_id == 0 and release_name == "com_2024_amy2018_2":
                    file_key_suffix = "baseline.parquet"
                    available_files = all_parquet_files.get("baseline", [])
                elif release_name == "com_2025_amy2018_1":
                    file_key_suffix = f"upgrade{upgrade_id}.parquet"
                    available_files = all_parquet_files.get(f"upgrade_{upgrade_id}", [])
                else:
                    file_key_suffix = f"upgrade{upgrade_id:02d}.parquet"
                    available_files = all_parquet_files.get(f"upgrade_{upgrade_id:02d}", [])

                logger.info(f"Found {len(available_files)} {file_key_suffix} files:")

                if available_files:
                    # Combine all parquet files for this upgrade
                    combined_dfs = []
                    files_to_process = available_files  # Process all available files

                    for i, file_key in enumerate(files_to_process, 1):
                        logger.info(f"Processing file {i} out of {len(files_to_process)}")
                        df = process_parquet_file(
                            file_key, s3, bucket_name, res_com, weather, release_version, release_year, release_name
                        )
                        if df is not None and df.height > 0:
                            combined_dfs.append(df)
                        else:
                            logger.exception(
                                f"ERROR: Failed to process {release_name}. bldg_id column is either missing or contains non-integer values"
                            )
                            logger.exception("Exiting script...")
                            sys.exit(1)

                    if combined_dfs:
                        # Concatenate all DataFrames for this release
                        combined_df = pl.concat(combined_dfs)
                        logger.info(
                            f"Combined {len(combined_dfs)} files into {combined_df.height} total rows in {release_name}"
                        )
                        all_dataframes.append(combined_df)
                    else:
                        logger.warning(f"No valid data found for {file_key_suffix} in {release_name}")
                else:
                    logger.warning(f"No files found for {file_key_suffix} in {release_name}")

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
        output_file = f"{data_dir}/building_data/combined_metadata.parquet"
        final_df.write_parquet(
            output_file,
            use_pyarrow=True,
            partition_by=["product", "release_year", "weather_file", "release_version", "state"],
        )
        logger.info(f"Successfully saved combined DataFrame to {output_file}")
    else:
        logger.warning("No data to save")
