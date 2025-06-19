import json
import os
from pathlib import Path

import boto3
import pandas as pd
import pyarrow.fs as fs
import pyarrow.parquet as pq
from botocore import UNSIGNED
from botocore.config import Config


def process_parquet_file(
    file_key: str, columns_to_keep: list, s3: fs.S3FileSystem, bucket_name: str
) -> pd.DataFrame | None:
    """
    Process a parquet file from S3 and return a pandas DataFrame.

    Args:
        file_key (str): The S3 key of the parquet file
        columns_to_keep (list): List of column names to keep
        s3 (fs.S3FileSystem): S3 filesystem instance
        bucket_name (str): Name of the S3 bucket

    Returns:
        pd.DataFrame: Processed DataFrame
    """
    try:
        # Check available columns
        parquet_file = pq.ParquetFile(f"{bucket_name}/{file_key}", filesystem=s3)
        available_columns = parquet_file.schema_arrow.names
        existing_columns = [col for col in columns_to_keep if col in available_columns]
        missing_columns = [col for col in columns_to_keep if col not in available_columns]

        # Report missing columns
        if missing_columns:
            print(f"Warning: These columns don't exist and will be skipped: {missing_columns}")

        # Read with existing columns (will read all if existing_columns is empty)
        table = pq.read_table(f"{bucket_name}/{file_key}", columns=existing_columns or None, filesystem=s3)

        # Convert to DataFrame
        df = table.to_pandas()
        print(f"Successfully loaded {len(df)} rows with {len(df.columns)} columns")

    except Exception as e:
        print(f"Error processing file: {e}")
        return None
    else:
        return df


def find_metadata_files(base_file_key: str, file_key_suffix: str, s3_client, bucket_name: str) -> list:
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


def _extract_upgrade_number(filename: str) -> str | None:
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


def find_all_2024_comstock_amy2018_2_parquet(base_file_key: str, s3_client, bucket_name: str) -> dict:
    """
    Find all parquet files in the S3 bucket and organize them by upgrade type.

    Returns:
        dict: Dictionary with keys 'baseline' and 'upgrade_XX' containing lists of file paths
    """
    all_files = {}
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
    with open("/workspaces/buildstock-fetcher/utils/buildstock_releases.json") as file:
        data = json.load(file)

    # Directory to save the data
    data_dir = Path(__file__).parent.parent / "data"
    downloaded_paths = []
    os.makedirs(data_dir, exist_ok=True)

    # Columns to keep from the metadata table
    columns_to_keep = [
        "in.puma",
        "in.nhgis_puma_gisjoin",
        "in.resstock_puma_id",
        "in.state",
        "in.state_name",
        "in.resstock_county_id",
        "in.county",
        "bldg_id",
        "upgrade",
    ]

    # S3 bucket and filesystem
    bucket_name = "oedi-data-lake"
    s3 = fs.S3FileSystem(anonymous=True, region="us-west-2")
    s3_client = boto3.client("s3", region_name="us-west-2", config=Config(signature_version=UNSIGNED))

    for release_name, release in data.items():
        release_year = release["release_year"]
        res_com = release["res_com"]
        weather = release["weather"]
        release_number = release["release_number"]
        upgrade_ids = release.get("upgrade_ids", [])

        if release_year == "2021":
            file_key = (
                f"nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
                f"{release_year}/{res_com}_{weather}_release_{release_number}/metadata/metadata.parquet"
            )
            df = process_parquet_file(file_key, columns_to_keep, s3, bucket_name)
            if df is not None:
                feather_filename = f"{data_dir}/{release_name}_baseline.feather"
                df.to_feather(feather_filename, compression="zstd")
                print(f"Successfully saved DataFrame to {feather_filename}")

        elif (
            release_year == "2022"
            or release_year == "2023"
            or (release_year == "2024" and release_name != "com_2024_amy2018_2")
        ):
            for upgrade_id in upgrade_ids:
                upgrade_id = int(upgrade_id)
                if upgrade_id == 0:
                    file_key = (
                        f"nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
                        f"{release_year}/{res_com}_{weather}_release_{release_number}/metadata/baseline.parquet"
                    )
                else:
                    file_key = (
                        f"nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
                        f"{release_year}/{res_com}_{weather}_release_{release_number}/metadata/"
                        f"upgrade{upgrade_id:02d}.parquet"
                    )

                df = process_parquet_file(file_key, columns_to_keep, s3, bucket_name)
                if df is not None:
                    feather_filename = f"{data_dir}/{release_name}_upgrade{upgrade_id:02d}.feather"
                    df.to_feather(feather_filename, compression="zstd")
                    print(f"Successfully saved DataFrame to {feather_filename}")
                    break

        elif release_year == "2024" and release_name == "com_2024_amy2018_2":
            base_file_key = (
                f"nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
                f"{release_year}/{res_com}_{weather}_release_{release_number}/"
                f"metadata_and_annual_results/by_state_and_county/full/parquet/"
            )

            # Find all available parquet files at once
            all_parquet_files = find_all_2024_comstock_amy2018_2_parquet(base_file_key, s3_client, bucket_name)
            print(f"Found parquet files: {list(all_parquet_files.keys())}")

            for upgrade_id in upgrade_ids:
                upgrade_id = int(upgrade_id)
                if upgrade_id == 0:
                    file_key_suffix = "baseline.parquet"
                    available_files = all_parquet_files.get("baseline", [])
                else:
                    file_key_suffix = f"upgrade{upgrade_id:02d}.parquet"
                    available_files = all_parquet_files.get(f"upgrade_{upgrade_id:02d}", [])

                print(f"Found {len(available_files)} {file_key_suffix} files:")

                if available_files:
                    # Combine all parquet files for this upgrade
                    combined_dfs = []
                    for file_key in available_files:
                        df = process_parquet_file(file_key, columns_to_keep, s3, bucket_name)
                        if df is not None:
                            combined_dfs.append(df)

                    if combined_dfs:
                        # Concatenate all DataFrames
                        combined_df = pd.concat(combined_dfs, ignore_index=True)
                        print(f"Combined {len(combined_dfs)} files into {len(combined_df)} total rows")

                        # Save to feather file
                        feather_filename = f"{data_dir}/{release_name}_upgrade{upgrade_id:02d}.feather"
                        combined_df.to_feather(feather_filename, compression="zstd")
                        print(f"Successfully saved combined DataFrame to {feather_filename}")
                    else:
                        print(f"No valid data found for {file_key_suffix}")
                else:
                    print(f"No files found for {file_key_suffix}")
