import json
import os
from pathlib import Path

import pandas as pd
import pyarrow.fs as fs
import pyarrow.parquet as pq


def process_parquet_file(file_key: str, columns_to_keep: list, s3: fs.S3FileSystem, bucket_name: str) -> pd.DataFrame:
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
