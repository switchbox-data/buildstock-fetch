import json
import os
import re
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, TypedDict

import boto3
from botocore import UNSIGNED
from botocore.config import Config


class BuildStockRelease(TypedDict):
    release_year: str
    res_com: str
    weather: str
    release_number: str
    upgrade_ids: list[str]
    available_data: list[str]


class CommonPrefix(TypedDict):
    Prefix: str


def _find_model_directory(s3_client: Any, bucket_name: str, prefix: str) -> str:
    """
    Find the building_energy_model directory path using breadth-first search.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    # Queue contains tuples of (prefix, depth) where depth starts at 0
    # Ensure prefix ends with / for proper directory listing
    directories_to_search = [(prefix.rstrip("/") + "/", 0)]

    while directories_to_search:
        current_prefix, current_depth = directories_to_search.pop(0)  # Get next directory to search

        # Stop if we've gone too deep
        if current_depth >= 3:
            continue

        # List all directories at current level
        pages = paginator.paginate(Bucket=bucket_name, Prefix=current_prefix, Delimiter="/")

        for page in pages:
            # First check directories at this level
            if "CommonPrefixes" in page:
                for prefix_obj in page["CommonPrefixes"]:
                    dir_path = str(prefix_obj["Prefix"])  # Keep the trailing slash for next level search
                    dir_name = dir_path.rstrip("/").split("/")[-1]

                    # If we found the building_energy_model directory, return its path
                    if "building_energy_model" in dir_name.lower():
                        return dir_path.rstrip("/")  # Remove trailing slash from final result

                    # Add this directory to be searched next, with increased depth
                    directories_to_search.append((dir_path, current_depth + 1))

    return ""


def _get_upgrade_ids(s3_client: Any, bucket_name: str, model_path: str) -> list[str]:
    """
    Get the list of upgrade IDs from the building_energy_model directory.
    Extracts the integer from upgrade=## format.
    """
    if not model_path:
        return []

    upgrade_ids = []
    paginator = s3_client.get_paginator("list_objects_v2")
    upgrade_pages = paginator.paginate(Bucket=bucket_name, Prefix=model_path + "/", Delimiter="/")

    for page in upgrade_pages:
        if "CommonPrefixes" in page:
            for prefix_obj in page["CommonPrefixes"]:
                upgrade_path = prefix_obj["Prefix"].rstrip("/")
                upgrade_dir = upgrade_path.split("/")[-1]
                # Extract the integer from upgrade=##
                if upgrade_dir.startswith("upgrade="):
                    upgrade_num = upgrade_dir.split("=")[1]
                    upgrade_ids.append(upgrade_num)

    return sorted(upgrade_ids, key=int)  # Sort numerically


def _process_zip_file(
    s3_client: Any, bucket_name: str, key: str, zip_files_found: int, unique_files: set[str]
) -> tuple[int, set[str]]:
    """Process a single zip file and return updated count of files found and available data types."""
    available_data = set()
    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = os.path.join(temp_dir, os.path.basename(key))
        s3_client.download_file(bucket_name, key, local_path)
        with zipfile.ZipFile(local_path, "r") as zip_ref:
            for file_name in zip_ref.namelist():
                unique_files.add(file_name)
                # Check for XML files
                if file_name.lower().endswith(".xml"):
                    available_data.add("hpxml")
                # Check for schedule CSV files
                if file_name.lower().endswith(".csv") and "schedule" in file_name.lower():
                    available_data.add("schedule")
    return zip_files_found + 1, available_data


def _check_directory_contents(
    s3_client: Any, bucket_name: str, current_prefix: str, zip_files_found: int, unique_files: set[str]
) -> tuple[int, bool, set[str]]:
    """Check directory contents for zip files and process them."""
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=current_prefix)
    if "Contents" not in response:
        return zip_files_found, False, set()

    available_data = set()
    for obj in response["Contents"]:
        key = obj["Key"]
        if key.endswith(".osm.gz"):
            return zip_files_found, True, set()
        if key.endswith(".zip") and zip_files_found < 5:
            zip_files_found, zip_available_data = _process_zip_file(
                s3_client, bucket_name, key, zip_files_found, unique_files
            )
            available_data.update(zip_available_data)
            if zip_files_found == 5:
                return zip_files_found, True, available_data

    return zip_files_found, False, available_data


def _find_and_process_model_file(s3_client: Any, bucket_name: str, prefix: str) -> set[str]:
    """
    Breadth-first search for the first 5 .zip files in building_energy_model directories.
    Collects unique filenames from all zip files found.
    Returns set of available data types found.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    queue = [(prefix, 0)]  # (prefix, depth)
    visited = set()
    zip_files_found = 0
    unique_files: set[str] = set()
    available_data: set[str] = set()

    while queue and zip_files_found < 5:
        current_prefix, depth = queue.pop(0)
        if current_prefix in visited:
            continue

        visited.add(current_prefix)

        # Check current directory contents
        zip_files_found, should_return, dir_available_data = _check_directory_contents(
            s3_client, bucket_name, current_prefix, zip_files_found, unique_files
        )
        available_data.update(dir_available_data)
        if should_return:
            return available_data

        # Check subdirectories
        pages = paginator.paginate(Bucket=bucket_name, Prefix=current_prefix, Delimiter="/")
        for page in pages:
            if "CommonPrefixes" in page:
                for prefix_obj in page["CommonPrefixes"]:
                    dir_path = prefix_obj["Prefix"]
                    dir_name = dir_path.rstrip("/").split("/")[-1]

                    # Only queue directories that might contain building_energy_model
                    if "building_energy_model" in dir_name.lower() or depth == 0:
                        queue.append((dir_path, depth + 1))

    return available_data


def _process_directory_match(
    s3_client: Any, bucket_name: str, current_prefix: str, expected: str, found_dirs: set[str]
) -> None:
    """Process a matched directory and update found_dirs accordingly."""
    if expected == "building_energy_model":
        available_data = _find_and_process_model_file(s3_client, bucket_name, current_prefix)
        found_dirs.update(available_data - {"building_energy_model"})
    elif expected == "timeseries_individual_buildings":
        found_dirs.add("load_curve_15min")
        found_dirs.add("load_curve_hourly")
        found_dirs.add("load_curve_daily")
        found_dirs.add("load_curve_monthly")
        found_dirs.add("load_curve_annual")


def _check_directory_matches(
    s3_client: Any,
    bucket_name: str,
    current_prefix: str,
    prefix: CommonPrefix,
    expected_dirs: list[str],
    found_dirs: set[str],
) -> None:
    """Check if a directory matches any expected directories and process it."""
    dir_name = prefix["Prefix"].rstrip("/").split("/")[-1]
    for expected in expected_dirs:
        if expected in dir_name and expected not in found_dirs:
            found_dirs.add(expected)
            _process_directory_match(s3_client, bucket_name, current_prefix, expected, found_dirs)


def _find_available_data(s3_client: Any, bucket_name: str, prefix: str) -> list[str]:
    """
    Find the available data directories (building_energy_model, metadata, timeseries_individual_buildings).
    Returns a list of directories that exist, searching up to depth 3.
    """
    found_dirs: set[str] = set()
    expected_dirs = ["building_energy_model", "metadata", "timeseries_individual_buildings"]
    paginator = s3_client.get_paginator("list_objects_v2")

    # Queue for BFS: (prefix, depth)
    queue = [(prefix, 0)]
    visited = set()

    while queue and len(found_dirs) < len(expected_dirs):
        current_prefix, depth = queue.pop(0)
        if depth > 2 or current_prefix in visited:
            continue

        visited.add(current_prefix)
        pages = paginator.paginate(Bucket=bucket_name, Prefix=current_prefix, Delimiter="/")

        for page in pages:
            if "CommonPrefixes" in page:
                # First check current level for matches
                for prefix_obj in page["CommonPrefixes"]:
                    _check_directory_matches(
                        s3_client, bucket_name, current_prefix, prefix_obj, expected_dirs, found_dirs
                    )

                # Then add subdirectories to queue for next level
                for prefix_obj in page["CommonPrefixes"]:
                    queue.append((str(prefix_obj["Prefix"]), depth + 1))

    found_dirs.discard("building_energy_model")
    found_dirs.discard("timeseries_individual_buildings")
    return list(found_dirs)


def _find_upgrade_ids(s3_client: Any, bucket_name: str, prefix: str) -> list[str]:
    """
    Find the unique building_energy_model directory and its upgrade IDs.
    For 2021 releases, returns ['0'] as the only upgrade ID.

    Args:
        s3_client: Boto3 S3 client
        bucket_name: Name of the S3 bucket
        prefix: Path prefix to search under

    Returns:
        list[str]: List of upgrade IDs found in that directory
    """
    # For 2021 releases, return ['0'] as the only upgrade ID
    if "/2021/" in prefix:
        return ["0"]

    model_path = _find_model_directory(s3_client, bucket_name, prefix)
    upgrade_ids = _get_upgrade_ids(s3_client, bucket_name, model_path)
    return upgrade_ids


def _process_release(
    s3_client: Any,
    bucket_name: str,
    release_path: str,
    match: re.Match[str],
    seen_combinations: set[tuple[str, str, str, str]],
    releases: dict[str, BuildStockRelease],
    available_data: list[str],
) -> None:
    """
    Process a release directory and extract its metadata.
    """
    # Check if this is the 2024 pattern by looking at the pattern itself
    if "2024/resstock_dataset_2024.1" in match.string:
        release_year, res_com_type, _, release_number = match.groups()
        weather = "tmy3"
    else:
        release_year, res_com_type, weather, release_number = match.groups()

    combination = (release_year, f"{res_com_type}stock", weather, release_number)

    if combination not in seen_combinations:
        seen_combinations.add(combination)

    # Find the building_energy_model directory and its upgrade IDs
    upgrade_ids = _find_upgrade_ids(s3_client, bucket_name, release_path)

    release_data: BuildStockRelease = {
        "release_year": release_year,
        "res_com": f"{res_com_type}stock",
        "weather": weather,
        "release_number": release_number,
        "upgrade_ids": upgrade_ids,
        "available_data": available_data,
    }
    # Create key following the pattern: {res/com}_{release_year}_{weather}_{release_number}
    key = f"{res_com_type}_{release_year}_{weather}_{release_number}"
    releases[key] = release_data


def resolve_bldgid_sets(
    bucket_name: str = "oedi-data-lake",
    prefix: str = "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock",
    output_file: str = "buildstock_releases.json",
) -> dict[str, BuildStockRelease]:
    """
    Get URLs containing 'building_energy_models' from the NREL S3 bucket.
    Parses the URL structure to extract available releases.

    Args:
        bucket_name (str): Name of the S3 bucket
        prefix (str): Prefix path in the bucket to search
        output_file (str): Path to output JSON file

    Returns:
        dict[str, BuildStockRelease]: Dictionary of releases with keys following pattern:
            {res/com}_{release_year}_{weather}_{release_number}
            Example: "res_2022_tmy3_1"
            Each release includes the path to its building_energy_model directory
            and the list of upgrade IDs available for that release.
    """
    # Initialize S3 client with unsigned requests (for public buckets)
    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    # Dictionary to store releases
    releases: dict[str, BuildStockRelease] = {}
    # Set to track unique combinations
    seen_combinations: set[tuple[str, str, str, str]] = set()

    # Regex pattern to extract components from full release path
    pattern = r"(\d{4})/(res|com)stock_(\w+)_release_(\d+(?:\.\d+)?)"
    # 2024 ResStock TMY3 release 1 has a different pattern from the rest
    pattern_2024_resstock_tmy3_1 = r"(\d{4})/(res|com)stock_dataset_(\d{4})\.(\d+(?:\.\d+)?)"

    # First, list all year directories
    paginator = s3_client.get_paginator("list_objects_v2")
    year_pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix + "/", Delimiter="/")

    for year_page in year_pages:
        # Get common prefixes (directories) at this level
        if "CommonPrefixes" not in year_page:
            continue

        for year_prefix in year_page["CommonPrefixes"]:
            year_path = year_prefix["Prefix"]

            # List contents of the year directory to find release directories
            release_pages = paginator.paginate(Bucket=bucket_name, Prefix=year_path, Delimiter="/")

            for release_page in release_pages:
                if "CommonPrefixes" not in release_page:
                    continue

                for release_prefix in release_page["CommonPrefixes"]:
                    release_path = release_prefix["Prefix"].rstrip("/")

                    # Try to match the pattern against the full path
                    relative_path = release_path.replace(prefix, "").lstrip("/")

                    # Check for 2024 ResStock TMY3 release 1 pattern first
                    match = re.match(pattern_2024_resstock_tmy3_1, relative_path)
                    if match:
                        available_data = _find_available_data(s3_client, bucket_name, release_path)
                        _process_release(
                            s3_client, bucket_name, release_path, match, seen_combinations, releases, available_data
                        )
                        continue

                    # Then check for standard pattern
                    match = re.match(pattern, relative_path)

                    if match:
                        available_data = _find_available_data(s3_client, bucket_name, release_path)
                        _process_release(
                            s3_client, bucket_name, release_path, match, seen_combinations, releases, available_data
                        )

    # Save to JSON file with consistent formatting
    output_path = Path(__file__).parent / output_file
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(releases, f, indent=2, sort_keys=False, ensure_ascii=False)
        f.write("\n")  # Ensure newline at end of file

    return releases


if __name__ == "__main__":
    start_time = time.time()
    releases = resolve_bldgid_sets()
    elapsed_time = time.time() - start_time
    print(f"\nFound {len(releases)} unique releases in {elapsed_time:.2f} seconds")
