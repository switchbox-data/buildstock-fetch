import concurrent.futures
import gc
import json
import os
import shutil
import tempfile
import traceback
import zipfile
from dataclasses import dataclass
from datetime import timedelta
from importlib.resources import files
from pathlib import Path
from typing import Any, Optional, Union, cast

import boto3
import polars as pl
import pyarrow.fs as fs
import pyarrow.parquet as pq
import requests
import requests.adapters
from botocore import UNSIGNED
from botocore.config import Config
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from buildstock_fetch.enhancements.SB_upgrades import (
    _get_required_component_files_for_SB_upgrade,
    _process_SB_upgrade_scenario,
)
from buildstock_fetch.types import FileType, ReleaseYear, ResCom, Weather

from .building import BuildingID
from .constants import LOAD_CURVE_COLUMN_AGGREGATION, METADATA_DIR
from .exception import (
    InvalidProductError,
    InvalidReleaseNameError,
    No15minLoadCurveError,
    NoAggregateLoadCurveError,
    NoAnnualLoadCurveError,
    NoBuildingDataError,
    NoWeatherFileError,
    UnknownAggregationFunctionError,
)

# from buildstock_fetch.main_cli import _get_all_available_releases


@dataclass
class RequestedFileTypes:
    hpxml: bool = False
    schedule: bool = False
    metadata: bool = False
    load_curve_15min: bool = False
    load_curve_hourly: bool = False
    load_curve_daily: bool = False
    load_curve_monthly: bool = False
    load_curve_annual: bool = False
    trip_schedules: bool = False
    weather: bool = False


def _validate_release_name(release_name: str) -> bool:
    """Validate the release name.

    Args:
        release_name: The release name to validate.

    Returns:
        True if the release name is valid, False otherwise.
    """
    # Read the valid release names from the JSON file
    releases_file = files("buildstock_fetch").joinpath("data").joinpath("buildstock_releases.json")
    releases_data = json.loads(Path(str(releases_file)).read_text(encoding="utf-8"))

    # Get the top-level keys as valid release names
    valid_release_names = list(releases_data.keys())
    return release_name in valid_release_names


def _derive_load_curve_dir_name(aggregate_time_step: str) -> str:
    """Derive the load curve directory name from the aggregate time step."""
    if aggregate_time_step == "monthly":
        return "load_curve_monthly"
    elif aggregate_time_step == "hourly":
        return "load_curve_hourly"
    elif aggregate_time_step == "daily":
        return "load_curve_daily"
    else:
        msg = f"Unknown aggregate time step: {aggregate_time_step}"
        raise ValueError(msg)
    return aggregate_time_step


def _resolve_unique_metadata_urls(bldg_ids: list[BuildingID]) -> list[str]:
    """Resolve the unique metadata URLs for a list of building IDs."""
    unique_metadata_urls: list[str] = []
    for bldg_id in bldg_ids:
        metadata_url = bldg_id.get_metadata_url()
        if metadata_url is None:
            continue
        if isinstance(metadata_url, list):
            unique_metadata_urls.extend(metadata_url)
        else:
            unique_metadata_urls.append(metadata_url)
    unique_metadata_urls = list(set(unique_metadata_urls))
    return unique_metadata_urls


def fetch_bldg_ids(
    product: ResCom, release_year: ReleaseYear, weather_file: Weather, release_version: str, state: str, upgrade_id: str
) -> list[BuildingID]:
    """Fetch a list of Building ID's

    Provided a state, returns a list of building ID's for that state.

    Args:
        product: The product type (e.g., 'resstock', 'comstock')
        release_year: The release year (e.g., '2021', '2022')
        weather_file: The weather file type (e.g., 'tmy3')
        release_version: The release version number (e.g., '1')
        state: The state to fetch building ID's for.

    Returns:
        A list of building ID's for the given state.
    """

    if product == "resstock":
        product_str = "res"
    elif product == "comstock":
        product_str = "com"
    else:
        raise InvalidProductError(product)

    release_name = f"{product_str}_{release_year}_{weather_file}_{release_version}"
    if not _validate_release_name(release_name):
        raise InvalidReleaseNameError(release_name)

    # Read the specific partition that matches our criteria
    partition_path = (
        METADATA_DIR
        / f"product={product}"
        / f"release_year={release_year}"
        / f"weather_file={weather_file}"
        / f"release_version={release_version}"
        / f"state={state}"
    )

    # Check if the partition exists
    if not partition_path.exists():
        return []

    # Read the parquet files in the specific partition
    df = pl.read_parquet(str(partition_path))

    # No need to filter since we're already reading the specific partition
    filtered_df = df

    # Convert the filtered data to BuildingID objects
    building_ids = []
    for row in filtered_df.iter_rows(named=True):
        building_id = BuildingID(
            bldg_id=int(row["bldg_id"]),
            release_number=release_version,
            release_year=release_year,
            res_com=product,
            weather=weather_file,
            upgrade_id=upgrade_id,
            state=state,
        )
        building_ids.append(building_id)

    return building_ids


def _download_with_progress(url: str, output_file: Path, progress: Progress, task_id: TaskID) -> int:
    """Download a file with progress tracking."""
    # Get file size first
    response = requests.head(url, timeout=30, verify=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    progress.update(task_id, total=total_size)

    # Download with streaming
    response = requests.get(url, stream=True, timeout=30, verify=True)
    response.raise_for_status()

    downloaded_size = 0

    # For streaming downloads, we still need the traditional approach for progress tracking
    with open(str(output_file), "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                downloaded_size += len(chunk)
                if total_size > 0:
                    progress.update(task_id, completed=downloaded_size)

    return downloaded_size


def _convert_url_to_s3_path(url: str) -> str:
    """Convert HTTP URL to S3 path format.

    Args:
        url: URL in HTTP or S3 format

    Returns:
        S3 path in format s3://bucket/key

    Raises:
        ValueError: If URL format is unsupported
    """
    # Convert HTTP URL to S3 path
    # Example: https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/.../metadata.parquet
    # To: s3://oedi-data-lake/nrel-pds-building-stock/.../metadata.parquet
    if url.startswith("https://oedi-data-lake.s3.amazonaws.com/"):
        s3_path = url.replace("https://oedi-data-lake.s3.amazonaws.com/", "s3://oedi-data-lake/")
    elif url.startswith("https://"):
        # Extract bucket and key from other S3 URLs if needed
        msg = f"Unsupported URL format for S3 filtering: {url}"
        raise ValueError(msg)
    else:
        # If it's already an S3 path, use it directly
        if url.startswith("s3://"):
            s3_path = url
        else:
            msg = f"Invalid URL format: {url}"
            raise ValueError(msg)

    return s3_path


def _download_with_progress_metadata(
    urls: list[str],
    bldg_ids: list[BuildingID],
    output_file: Path,
    progress: Progress,
    task_id: TaskID,
    found_bldg_ids: list[int],
) -> int:
    """Download a metadata file with progress tracking, filtering for only requested bldg_ids from S3."""

    # Extract release info for progress messages
    first_bldg_id = bldg_ids[0]
    release_name = first_bldg_id.get_release_name()
    upgrade = first_bldg_id.upgrade_id
    state = first_bldg_id.state
    progress_suffix = f"{release_name} - (upgrade {upgrade}) - {state}"

    # Get the list of bldg_ids we need to filter for, sorted for optimization
    requested_bldg_ids_set = {bldg_id.bldg_id for bldg_id in bldg_ids}
    requested_bldg_ids = sorted(requested_bldg_ids_set)

    # Create S3 filesystem for anonymous access
    # This allows Polars to read directly from S3 without downloading the entire file
    # Polars can use PyArrow's filesystem for efficient S3 access
    s3_filesystem = fs.S3FileSystem(anonymous=True, region="us-west-2")

    for url in urls:
        s3_path = _convert_url_to_s3_path(url)
        progress.update(task_id, description=f"[yellow](metadata) Connecting to S3: {progress_suffix}")

        try:
            # Get total file size from S3
            # Extract bucket and key from s3_path: s3://bucket/key
            s3_path_parts = s3_path.replace("s3://", "").split("/", 1)
            bucket_name = s3_path_parts[0]
            s3_key = s3_path_parts[1] if len(s3_path_parts) > 1 else ""

            # Use boto3 to get file size (HEAD request)
            progress.update(task_id, description=f"[yellow](metadata) Getting file size: {progress_suffix}")
            s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
            try:
                response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
                total_file_size = response.get("ContentLength", 0)
            except Exception:
                # If we can't get file size, fall back to estimation
                total_file_size = 0

            # Read parquet directly from S3 and filter without downloading entire file
            # Use PyArrow filesystem to open file stream, then Polars can read and filter efficiently
            # Extract bucket and key for PyArrow filesystem
            s3_file_path = f"{bucket_name}/{s3_key}"

            try:
                # Open file as stream using PyArrow filesystem
                # Polars will read parquet metadata and use predicate pushdown to filter efficiently
                progress.update(
                    task_id, description=f"[yellow](metadata) Reading and filtering data from S3: {progress_suffix}"
                )
                with s3_filesystem.open_input_file(s3_file_path) as s3_file:
                    # Read only metadata (footer)
                    parquet_file = pq.ParquetFile(s3_file)
                    pq_metadata = parquet_file.metadata
                    total_rows = pq_metadata.num_rows

                    # Reset file pointer for reading actual data
                    s3_file.seek(0)

                    column_names = pq_metadata.schema.names
                    columns_to_keep = []
                    for col in column_names:
                        if any(
                            keyword in col for keyword in ["upgrade", "bldg_id", "metadata_index"]
                        ) or col.startswith("in."):
                            columns_to_keep.append(col)
                    # Polars will use predicate pushdown to only read relevant row groups
                    df = (
                        pl.scan_parquet(s3_file)
                        .select(columns_to_keep)
                        .filter(pl.col("bldg_id").is_in(requested_bldg_ids))
                        .collect()
                    )
                    found_bldg_ids.extend(df["bldg_id"].to_list())
                    selected_rows = len(df)
                    progress.update(
                        task_id,
                        total=total_file_size * (selected_rows / total_rows),
                        description=f"[yellow](metadata) Writing filtered data: {progress_suffix}",
                    )
            except Exception:
                traceback.print_exc()
                raise

            selected_rows = len(df)

            # Check if output file already exists
            if output_file.exists():
                # Read existing file and combine with new data
                existing_file = pl.scan_parquet(output_file)
                new_df = df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)
                new_file_lazy = pl.LazyFrame(new_df)

                combined_file = pl.concat([existing_file, new_file_lazy])
                # Remove duplicate rows based on bldg_id column
                deduplicated_file = combined_file.collect().unique(subset=["bldg_id"], keep="first")
                deduplicated_file.write_parquet(output_file)
            else:
                # Write filtered data directly
                df.write_parquet(output_file)

            # Calculate downloaded size as ratio
            # There's no way to get file size when downloading parts of the file, so we'll
            # estimate it by multiplying the total file size by the ratio of selected rows to total rows
            downloaded_size = int(total_file_size * (selected_rows / total_rows)) if total_rows > 0 else 0
            progress.update(
                task_id,
                completed=downloaded_size,
                total=downloaded_size,
                description=f"[green](metadata) Download complete: {progress_suffix}",
            )
            gc.collect()
        except Exception:
            # Fallback to old method (download entire file using requests) if S3 filtering fails
            progress.update(
                task_id, description=f"[yellow](metadata) Falling back to full file download: {progress_suffix}"
            )
            return _download_with_progress_metadata_fallback(url, output_file, progress, task_id)
        else:
            # Return successfully if no exception occurred
            return downloaded_size

    # If no URLs were processed, return 0 (shouldn't happen in practice)
    return 0


def _download_with_progress_metadata_fallback(url: str, output_file: Path, progress: Progress, task_id: TaskID) -> int:
    """Fallback method to download entire metadata file (old behavior)."""
    # Get file size first
    response = requests.head(url, timeout=30, verify=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    progress.update(task_id, total=total_size)

    # Download with streaming
    response = requests.get(url, stream=True, timeout=30, verify=True)
    response.raise_for_status()

    downloaded_size = 0

    # Check if output file already exists
    if output_file.exists():
        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as temp_file:
            temp_path = Path(temp_file.name)
            with open(temp_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress.update(task_id, completed=downloaded_size)
            _process_single_metadata_file(temp_path)

            existing_file = pl.scan_parquet(output_file)
            new_file = pl.scan_parquet(temp_path)
            combined_file = pl.concat([existing_file, new_file])
            # Remove duplicate rows based on bldg_id column
            deduplicated_file = combined_file.collect().unique(subset=["bldg_id"], keep="first")
            deduplicated_file.write_parquet(output_file)
            gc.collect()
            os.remove(temp_path)

    else:
        # File doesn't exist, download normally
        with open(str(output_file), "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        progress.update(task_id, completed=downloaded_size)

        _process_single_metadata_file(output_file)

    return downloaded_size


def _process_single_metadata_file(metadata_file: Path) -> None:
    """Process a single metadata file to keep only columns containing specified keywords."""
    # First, get column names without loading data into memory
    schema = pl.scan_parquet(metadata_file).collect_schema()

    # Filter columns to only keep those containing "bldg_id", "upgrade", "metadata_index", or "out."
    # and remove columns that start with "in."
    columns_to_keep = []
    for col in schema:
        if any(keyword in col for keyword in ["bldg_id", "upgrade", "metadata_index"]) or col.startswith("in."):
            columns_to_keep.append(col)

    # Use streaming operations to avoid loading entire file into memory
    # Create a temporary file to write the filtered data
    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet", dir=metadata_file.parent) as temp_file:
        temp_file_path = temp_file.name

        try:
            # Stream the data: select columns and write in one operation
            filtered_metadata_file = pl.scan_parquet(metadata_file).select(columns_to_keep).collect()
            filtered_metadata_file.write_parquet(temp_file_path)

            # Replace the original file with the filtered one
            os.replace(temp_file_path, metadata_file)

            # Force garbage collection to free memory immediately
            gc.collect()

        except Exception:
            # Clean up temp file if something goes wrong
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise
    return


def _get_time_step_grouping_key(aggregate_time_step: str) -> tuple[str, str]:
    """Get the grouping key and format string for a given time step.

    Args:
        aggregate_time_step: The time step to aggregate to ("monthly", "hourly", "daily")

    Returns:
        A tuple of (grouping_key_name, format_string)
    """
    time_step_configs = {
        "monthly": ("year_month", "%Y-%m"),
        "hourly": ("year_month_day_hour", "%Y-%m-%d-%H"),
        "daily": ("year_month_day", "%Y-%m-%d"),
    }

    if aggregate_time_step not in time_step_configs:
        msg = f"Unknown aggregate time step: {aggregate_time_step}"
        raise ValueError(msg)

    return time_step_configs[aggregate_time_step]


def _create_aggregation_expressions(load_curve: pl.DataFrame, column_aggregations: dict[str, str]) -> list[pl.Expr]:
    """Create aggregation expressions for each column based on the aggregation rules.

    Args:
        load_curve: The DataFrame to create expressions for
        column_aggregations: Dictionary mapping column names to aggregation functions

    Returns:
        List of Polars expressions for aggregation
    """
    agg_exprs = []

    for col in load_curve.columns:
        if col in ["timestamp", "year_month", "year_month_day_hour", "year_month_day"]:
            continue

        if col in column_aggregations:
            agg_func = column_aggregations[col]
            if agg_func == "sum":
                agg_exprs.append(pl.col(col).sum().alias(col))
            elif agg_func == "mean":
                agg_exprs.append(pl.col(col).mean().alias(col))
            elif agg_func == "first":
                agg_exprs.append(pl.col(col).first().alias(col))
            else:
                raise UnknownAggregationFunctionError()
        else:
            raise UnknownAggregationFunctionError()

    # Add timestamp aggregation (take the first timestamp of each group)
    agg_exprs.append(pl.col("timestamp").first().alias("timestamp"))

    return agg_exprs


def _aggregate_load_curve_aggregate(
    load_curve: pl.DataFrame, aggregate_time_step: str, release_year: str
) -> pl.DataFrame:
    """Aggregate the 15-minute load curve to specified time step based on aggregation rules."""
    # Read the aggregation rules from CSV
    load_curve_map = None
    if release_year == "2024":
        load_curve_map = LOAD_CURVE_COLUMN_AGGREGATION.joinpath("2024_resstock_load_curve_columns.csv")
    elif release_year == "2022":
        load_curve_map = LOAD_CURVE_COLUMN_AGGREGATION.joinpath("2022_resstock_load_curve_columns.csv")
    else:
        msg = f"Missing load_curve_map for release year: {release_year}"
        raise ValueError(msg)
    aggregation_rules = pl.read_csv(load_curve_map)

    # Create a dictionary mapping column names to their aggregation functions
    column_aggregations = dict(zip(aggregation_rules["name"], aggregation_rules["Aggregate_function"]))

    # Ensure timestamp column exists and convert to datetime if needed
    if "timestamp" not in load_curve.columns:
        msg = "DataFrame must contain a 'timestamp' column"
        raise ValueError(msg)

    # Convert timestamp to datetime if it's not already
    load_curve = load_curve.with_columns(pl.col("timestamp").cast(pl.Datetime))

    # We want to subtract 15 minutes because the original load curve provides information
    # for the previous 15 minutes for each timestamp. For example, the first timestamp is 00:00:15,
    # and the columns correspond to consumption from 00:00:00 to 00:00:15. When aggregating,
    # we want the 00:00:00 timestamp to correspond to the consumption from 00:00:00 to whenever the
    # next timestamp is.
    load_curve = load_curve.with_columns((pl.col("timestamp") - timedelta(minutes=15)).alias("timestamp"))

    # Get the grouping key configuration
    grouping_key, format_string = _get_time_step_grouping_key(aggregate_time_step)

    # Create grouping key
    load_curve = load_curve.with_columns(pl.col("timestamp").dt.strftime(format_string).alias(grouping_key))

    # Create aggregation expressions
    agg_exprs = _create_aggregation_expressions(load_curve, column_aggregations)

    # Perform the aggregation
    aggregate_data = load_curve.group_by(grouping_key).agg(agg_exprs)

    # Sort by timestamp and drop the grouping column
    aggregate_data = aggregate_data.sort("timestamp").drop(grouping_key)

    return aggregate_data


def _download_and_process_aggregate(
    url: str, output_file: Path, progress: Progress, task_id: TaskID, aggregate_time_step: str, release_year: str
) -> int:
    """Download aggregate time step load curve to temporary file, process with Polars, and save result."""
    # Get file size first for progress tracking
    response = requests.head(url, timeout=30, verify=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    progress.update(task_id, total=total_size)

    # Download to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as temp_file:
        temp_path = Path(temp_file.name)

        try:
            # Create session with retry logic
            session = requests.Session()
            retry_strategy = requests.adapters.HTTPAdapter(max_retries=15)  # pyright: ignore[reportAttributeAccessIssue]
            session.mount("http://", retry_strategy)
            session.mount("https://", retry_strategy)

            # Download with streaming to temp file
            response = session.get(url, stream=True, timeout=60, verify=True)
            response.raise_for_status()

            downloaded_size = 0
            with open(temp_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress.update(task_id, completed=downloaded_size)

            # Process with Polars
            load_curve_15min = pl.read_parquet(temp_path)
            load_curve_aggregate = _process_aggregate_load_curve(load_curve_15min, aggregate_time_step, release_year)

            # Save processed file to final destination
            load_curve_aggregate.write_parquet(output_file)

            return downloaded_size

        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()


def download_bldg_data(
    bldg_id: BuildingID,
    file_type: RequestedFileTypes,
    output_dir: Path,
    progress: Optional[Progress] = None,
    task_id: Optional[TaskID] = None,
) -> dict[str, Union[Path, None]]:
    """Download and extract building data for a single building. Only HPXML and schedule files are supported.

    Args:
        bldg_id: A BuildingID object to download data for.
        file_type: RequestedFileTypes object to specify which files to download.
        output_dir: Directory to save the downloaded files.
        progress: Optional Rich progress object for tracking download progress.
        task_id: Optional task ID for progress tracking.

    Returns:
        A list of paths to the downloaded files.
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create a unique temporary directory for this building to avoid race conditions
    temp_dir = output_dir / f"temp_{str(bldg_id.bldg_id).zfill(7)}_{bldg_id.upgrade_id}"
    temp_dir.mkdir(exist_ok=True)

    downloaded_paths: dict[str, Optional[Path]] = {
        "hpxml": None,
        "schedule": None,
    }
    if file_type.hpxml or file_type.schedule:
        download_url = bldg_id.get_building_data_url()
        if download_url is None:
            message = f"Building data is not available for {bldg_id.get_release_name()}"
            raise NoBuildingDataError(message)

        output_file = temp_dir / f"{str(bldg_id.bldg_id).zfill(7)}_upgrade{bldg_id.upgrade_id}.zip"

        # Download with progress tracking if progress object is provided
        if progress and task_id is not None:
            _download_with_progress(download_url, output_file, progress, task_id)
        else:
            response = requests.get(download_url, timeout=30, verify=True)
            response.raise_for_status()
            output_file.write_bytes(response.content)

        # Extract specific files based on file_type
        with zipfile.ZipFile(output_file, "r") as zip_ref:
            zip_file_list = zip_ref.namelist()

            if file_type.hpxml:
                # Find and extract the XML file
                xml_files = [f for f in zip_file_list if f.endswith(".xml")]
                if xml_files:
                    xml_file = xml_files[0]  # Take the first (and only) XML file
                    zip_ref.extract(xml_file, temp_dir)
                    # Rename to the specified convention
                    old_path = temp_dir / xml_file
                    new_name = f"bldg{str(bldg_id.bldg_id).zfill(7)}-up{bldg_id.upgrade_id.zfill(2)}.xml"
                    new_path = (
                        output_dir
                        / bldg_id.get_release_name()
                        / "hpxml"
                        / bldg_id.state
                        / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}"
                        / new_name
                    )
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    # Use shutil.move instead of rename to handle cross-device moves
                    shutil.move(str(old_path), str(new_path))
                    downloaded_paths["hpxml"] = new_path

            if file_type.schedule:
                # Find and extract the schedule CSV file
                schedule_files = [f for f in zip_file_list if "schedule" in f.lower() and f.endswith(".csv")]
                if schedule_files:
                    schedule_file = schedule_files[0]  # Take the first (and only) schedule file
                    zip_ref.extract(schedule_file, temp_dir)
                    # Rename to the specified convention
                    old_path = temp_dir / schedule_file
                    new_name = f"bldg{str(bldg_id.bldg_id).zfill(7)}-up{bldg_id.upgrade_id.zfill(2)}_schedule.csv"
                    new_path = (
                        output_dir
                        / bldg_id.get_release_name()
                        / "schedule"
                        / bldg_id.state
                        / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}"
                        / new_name
                    )
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    # Use shutil.move instead of rename to handle cross-device moves
                    shutil.move(str(old_path), str(new_path))
                    downloaded_paths["schedule"] = new_path

        # Remove the zip file and temp directory after extraction
        output_file.unlink()
        temp_dir.rmdir()  # Remove empty temp directory

    return downloaded_paths


def download_15min_load_curve_with_progress(
    bldg_id: BuildingID, output_dir: Path, progress: Optional[Progress] = None, task_id: Optional[TaskID] = None
) -> Path:
    """Download the 15 min load profile timeseries for a given building with progress tracking.

    Args:
        bldg_id: A BuildingID object to download 15 min load profile timeseries for.
        output_dir: Directory to save the downloaded 15 min load profile timeseries.
        progress: Optional Rich progress object for tracking download progress.
        task_id: Optional task ID for progress tracking.

    Returns:
        Path to the downloaded file.
    """

    # Special case for SwitchBox Analysis upgrades
    if bldg_id.is_SB_upgrade():
        # Download each of the component files for the SB upgrade
        bldg_id_component_list = bldg_id.get_SB_upgrade_component_bldg_ids()
        if bldg_id_component_list is None:
            message = f"15 min load profile timeseries is not available for {bldg_id.get_release_name()}, upgrade {bldg_id.upgrade_id}"
            raise No15minLoadCurveError(message)

        for bldg_id_component in bldg_id_component_list:
            download_url = bldg_id_component.get_15min_load_curve_url()
            if download_url is None:
                message = f"15 min load profile timeseries is not available for {bldg_id_component.get_release_name()}, upgrade {bldg_id_component.upgrade_id}"
                raise No15minLoadCurveError(message)

            output_file_component = (
                output_dir
                / bldg_id.get_release_name()
                / "load_curve_15min"
                / f"state={bldg_id.state}"
                / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}"
                / bldg_id_component.get_output_filename("load_curve_15min")
            )
            output_file_component.parent.mkdir(parents=True, exist_ok=True)

            # Download with progress tracking if progress object is provided
            if progress and task_id is not None:
                _download_with_progress(download_url, output_file_component, progress, task_id)
            else:
                response = requests.get(download_url, timeout=30, verify=True)
                response.raise_for_status()
                output_file_component.write_bytes(response.content)

        output_file = (
            output_dir
            / bldg_id.get_release_name()
            / "load_curve_15min"
            / f"state={bldg_id.state}"
            / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}"
            / f"{str(bldg_id.bldg_id)!s}-{int(bldg_id.upgrade_id)!s}.parquet"
        )
        return output_file

    # Regular upgrade case
    download_url = bldg_id.get_15min_load_curve_url()
    if download_url is None:
        message = f"15 min load profile timeseries is not available for {bldg_id.get_release_name()}"
        raise No15minLoadCurveError(message)

    output_file = (
        output_dir
        / bldg_id.get_release_name()
        / "load_curve_15min"
        / f"state={bldg_id.state}"
        / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}"
        / f"{str(bldg_id.bldg_id)!s}-{int(bldg_id.upgrade_id)!s}.parquet"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Download with progress tracking if progress object is provided
    if progress and task_id is not None:
        _download_with_progress(download_url, output_file, progress, task_id)
    else:
        response = requests.get(download_url, timeout=30, verify=True)
        response.raise_for_status()
        output_file.write_bytes(response.content)

    return output_file


def _add_time_aggregation_columns(load_curve_aggregate: pl.DataFrame, aggregate_time_step: str) -> None:
    """Add time-based columns to the dataframe based on aggregation type.

    Args:
        df: Polars DataFrame with a 'timestamp' column
        aggregate_time_step: Type of aggregation ('hourly', 'daily', 'monthly')
    """
    if aggregate_time_step == "hourly":
        # Add year, month, day, and hour columns
        new_df = load_curve_aggregate.with_columns([
            pl.col("timestamp").dt.year().alias("year"),
            pl.col("timestamp").dt.month().alias("month"),
            pl.col("timestamp").dt.day().alias("day"),
            pl.col("timestamp").dt.hour().alias("hour"),
        ])
        load_curve_aggregate.__dict__.update(new_df.__dict__)
    elif aggregate_time_step == "daily":
        # Add year, month, and day columns
        new_df = load_curve_aggregate.with_columns([
            pl.col("timestamp").dt.year().alias("year"),
            pl.col("timestamp").dt.month().alias("month"),
            pl.col("timestamp").dt.day().alias("day"),
        ])
        load_curve_aggregate.__dict__.update(new_df.__dict__)
    elif aggregate_time_step == "monthly":
        # Add year and month columns
        new_df = load_curve_aggregate.with_columns([
            pl.col("timestamp").dt.year().alias("year"),
            pl.col("timestamp").dt.month().alias("month"),
        ])
        load_curve_aggregate.__dict__.update(new_df.__dict__)


def download_aggregate_time_step_load_curve_with_progress(
    bldg_id: BuildingID,
    output_dir: Path,
    progress: Optional[Progress],
    task_id: Optional[TaskID],
    aggregate_time_step: str,
) -> Path:
    """Download the aggregate time step load profile timeseries for a given building with progress tracking."""

    # Derive load_curve_dir and file_type from aggregate_time_step
    load_curve_dir = _derive_load_curve_dir_name(aggregate_time_step)
    file_type: FileType = cast(FileType, load_curve_dir)

    # Special case for SwitchBox Analysis upgrades
    if bldg_id.is_SB_upgrade():
        # Download each of the component files for the SB upgrade
        bldg_id_component_list = bldg_id.get_SB_upgrade_component_bldg_ids()
        if bldg_id_component_list is None:
            message = f"{aggregate_time_step} load profile timeseries is not available for {bldg_id.get_release_name()}, upgrade {bldg_id.upgrade_id}"
            raise NoAggregateLoadCurveError(message)

        for bldg_id_component in bldg_id_component_list:
            download_url = bldg_id_component.get_aggregate_load_curve_url()
            if download_url is None:
                message = f"{aggregate_time_step} load profile timeseries is not available for {bldg_id_component.get_release_name()}, upgrade {bldg_id_component.upgrade_id}"
                raise NoAggregateLoadCurveError(message)

            output_file_component = (
                output_dir
                / bldg_id.get_release_name()
                / load_curve_dir
                / f"state={bldg_id.state}"
                / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}"
                / bldg_id_component.get_output_filename(file_type)
            )
            output_file_component.parent.mkdir(parents=True, exist_ok=True)

            # Download with progress tracking if progress object is provided
            if progress and task_id is not None:
                _download_with_progress(download_url, output_file_component, progress, task_id)
            else:
                response = requests.get(download_url, timeout=30, verify=True)
                response.raise_for_status()
                output_file_component.write_bytes(response.content)

            # Process aggregation for each component
            component_load_curve_aggregate = pl.read_parquet(output_file_component)
            component_load_curve_aggregate = _process_aggregate_load_curve(
                component_load_curve_aggregate, aggregate_time_step, bldg_id.release_year
            )
            component_load_curve_aggregate.write_parquet(output_file_component)

        output_file = (
            output_dir
            / bldg_id.get_release_name()
            / load_curve_dir
            / f"state={bldg_id.state}"
            / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}"
            / bldg_id.get_output_filename(file_type)
        )
        _process_SB_upgrade_scenario(bldg_id, output_dir, output_file, file_type, aggregate_time_step)
        return output_file

    # Regular upgrade case
    download_url = bldg_id.get_aggregate_load_curve_url()
    if download_url is None:
        message = f"Aggregate load profile timeseries is not available for {bldg_id.get_release_name()}"
        raise NoAggregateLoadCurveError(message)

    output_file = (
        output_dir
        / bldg_id.get_release_name()
        / load_curve_dir
        / f"state={bldg_id.state}"
        / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}"
        / f"{str(bldg_id.bldg_id)!s}-{int(bldg_id.upgrade_id)!s}.parquet"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Download with progress tracking if progress object is provided
    if progress and task_id is not None:
        _download_and_process_aggregate(
            download_url, output_file, progress, task_id, aggregate_time_step, bldg_id.release_year
        )
    else:
        # For non-progress downloads, still use temp file approach for consistency
        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as temp_file:
            temp_path = Path(temp_file.name)
            try:
                response = requests.get(download_url, timeout=30, verify=True)
                response.raise_for_status()
                temp_path.write_bytes(response.content)

                # Process with Polars
                load_curve_15min = pl.read_parquet(temp_path)
                load_curve_aggregate = _process_aggregate_load_curve(
                    load_curve_15min, aggregate_time_step, bldg_id.release_year
                )

                # Save processed file to final destination
                load_curve_aggregate.write_parquet(output_file)
            finally:
                if temp_path.exists():
                    temp_path.unlink()

    return output_file


def _process_aggregate_load_curve(
    preaggregate_load_curve: pl.DataFrame,
    aggregate_time_step: str,
    release_year: str,
) -> pl.DataFrame:
    """Process the aggregate load curve."""
    load_curve_aggregate = _aggregate_load_curve_aggregate(preaggregate_load_curve, aggregate_time_step, release_year)
    _add_time_aggregation_columns(load_curve_aggregate, aggregate_time_step)
    return load_curve_aggregate


def _parse_requested_file_type(file_type: tuple[str, ...]) -> RequestedFileTypes:
    """Parse the file type string into a RequestedFileTypes object."""
    file_type_obj = RequestedFileTypes()

    # Map file type strings to their corresponding attributes
    type_mapping = {
        "hpxml": "hpxml",
        "schedule": "schedule",
        "metadata": "metadata",
        "load_curve_15min": "load_curve_15min",
        "load_curve_hourly": "load_curve_hourly",
        "load_curve_daily": "load_curve_daily",
        "load_curve_monthly": "load_curve_monthly",
        "load_curve_annual": "load_curve_annual",
        "trip_schedules": "trip_schedules",
        "weather": "weather",
    }

    # Set attributes based on what's in the file_type tuple
    for type_str, attr_name in type_mapping.items():
        if type_str in file_type:
            setattr(file_type_obj, attr_name, True)

    return file_type_obj


def _filter_metadata_requested_bldg_ids(
    bldg_ids: list[BuildingID], output_dir: Path, downloaded_paths: list[Path]
) -> None:
    """Process the results of a completed metadata download."""
    metadata_to_bldg_id_mapping: dict[Path, list[int]] = {}
    for bldg_id in bldg_ids:
        output_file = (
            output_dir
            / bldg_id.get_release_name()
            / "metadata"
            / f"state={bldg_id.state}"
            / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}"
            / "metadata.parquet"
        )
        if output_file in downloaded_paths:
            if output_file in metadata_to_bldg_id_mapping:
                metadata_to_bldg_id_mapping[output_file].append(bldg_id.bldg_id)
            else:
                metadata_to_bldg_id_mapping[output_file] = [bldg_id.bldg_id]

    for metadata_file, bldg_id_list in metadata_to_bldg_id_mapping.items():
        # Use streaming operations to avoid loading entire file into memory
        # Stream the data: filter rows, select columns, and write in one operation
        filtered_metadata_file = pl.scan_parquet(metadata_file).filter(pl.col("bldg_id").is_in(bldg_id_list)).collect()

        # Replace the original file with the filtered one
        filtered_metadata_file.write_parquet(metadata_file)

        # Force garbage collection to free memory immediately
        gc.collect()

    return


def _process_annual_load_curve_file(file_path: Path) -> None:
    """Process an annual load curve file to keep only columns containing specified keywords.

    Args:
        file_path: Path to the annual load curve parquet file to process.
    """
    # First, get column names without loading data into memory
    schema = pl.scan_parquet(file_path).collect_schema()

    # Filter columns to only keep those containing "bldg_id", "upgrade", "metadata_index", or "out."
    # and remove columns that start with "in."
    columns_to_keep = []
    for col in schema:
        if (
            any(keyword in col for keyword in ["bldg_id", "upgrade", "metadata_index"]) or col.startswith("out.")
        ) and not col.startswith("in."):
            columns_to_keep.append(col)

    # Use streaming operations to avoid loading entire file into memory
    # Create a temporary file to write the filtered data
    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as temp_file:
        temp_file_path = temp_file.name

    # Stream the data: select columns and write in one operation
    filtered_file = pl.scan_parquet(file_path).select(columns_to_keep).collect()
    filtered_file.write_parquet(temp_file_path)

    # Replace the original file with the filtered one
    # Use shutil.move instead of os.replace to handle cross-device moves
    shutil.move(temp_file_path, file_path)

    # Force garbage collection to free memory immediately
    gc.collect()


def _process_annual_load_curve_results(downloaded_paths: list[Path]) -> None:
    """Process all downloaded annual load curve files to filter columns.

    Args:
        downloaded_paths: List of all downloaded file paths.
    """
    # Filter for annual load curve files
    annual_load_curve_files = [
        path for path in downloaded_paths if "load_curve_annual" in str(path) and path.suffix == ".parquet"
    ]

    # Process each annual load curve file
    for file_path in annual_load_curve_files:
        _process_annual_load_curve_file(file_path)


def _process_download_results(
    future: concurrent.futures.Future,
    bldg_id: BuildingID,
    file_type_obj: RequestedFileTypes,
    downloaded_paths: list[Path],
    failed_downloads: list[str],
    console: Console,
) -> None:
    """Process the results of a completed download."""
    try:
        paths_dict = future.result()
        # Convert dict values to list, filtering out None values
        paths = [path for path in paths_dict.values() if path is not None]
        downloaded_paths.extend(paths)

        if file_type_obj.hpxml and paths_dict["hpxml"] is None:
            failed_downloads.append(f"bldg{str(bldg_id.bldg_id).zfill(7)}-up{bldg_id.upgrade_id.zfill(2)}.xml")
        if file_type_obj.schedule and paths_dict["schedule"] is None:
            failed_downloads.append(f"bldg{str(bldg_id.bldg_id).zfill(7)}-up{bldg_id.upgrade_id.zfill(2)}_schedule.csv")

    except NoBuildingDataError:
        raise
    except Exception as e:
        console.print(f"[red]Download failed for bldg_id {bldg_id}: {e}[/red]")


def _add_metadata_url_to_grouping(
    output_file: Path,
    metadata_url: str | list[str] | None,
    output_file_to_download_url: dict[Path, list[str]],
) -> None:
    """Add metadata URL(s) to the download URL grouping dictionary."""
    if metadata_url is None:
        return
    if output_file not in output_file_to_download_url:
        output_file_to_download_url[output_file] = []
    if isinstance(metadata_url, list):
        output_file_to_download_url[output_file].extend(metadata_url)
    elif metadata_url not in output_file_to_download_url[output_file]:
        output_file_to_download_url[output_file].append(metadata_url)


def _group_bldg_ids_by_output_file_metadata(
    bldg_ids: list[BuildingID], output_dir: Path, failed_downloads: list[str]
) -> tuple[dict[Path, list[BuildingID]], dict[Path, list[str]]]:
    """Group bldg_ids by their output file paths and collect download URLs.

    Returns:
        Tuple of (output_file_to_bldg_ids, output_file_to_download_url) dictionaries.
    """
    output_file_to_bldg_ids: dict[Path, list[BuildingID]] = {}
    output_file_to_download_url: dict[Path, list[str]] = {}

    for bldg_id in bldg_ids:
        if bldg_id.is_SB_upgrade():
            component_bldg_ids = bldg_id.get_SB_upgrade_component_bldg_ids()
            if component_bldg_ids is None:
                failed_downloads.append(str(bldg_id.bldg_id))
                continue
            for component_bldg_id in component_bldg_ids:
                output_file = (
                    output_dir
                    / bldg_id.get_release_name()
                    / "metadata"
                    / f"state={bldg_id.state}"
                    / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}"
                    / f"upgrade{str(int(component_bldg_id.upgrade_id)).zfill(2)}.parquet"
                )
                if output_file not in output_file_to_bldg_ids:
                    output_file_to_bldg_ids[output_file] = []
                output_file_to_bldg_ids[output_file].append(bldg_id)

                component_metadata_url = component_bldg_id.get_metadata_url()
                _add_metadata_url_to_grouping(output_file, component_metadata_url, output_file_to_download_url)
        else:
            output_file = (
                output_dir
                / bldg_id.get_release_name()
                / "metadata"
                / f"state={bldg_id.state}"
                / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}"
                / "metadata.parquet"
            )
            download_url = bldg_id.get_metadata_url()
            if download_url is None:
                failed_downloads.append(str(output_file))
                continue
            if output_file not in output_file_to_bldg_ids:
                output_file_to_bldg_ids[output_file] = []
            output_file_to_bldg_ids[output_file].append(bldg_id)
            _add_metadata_url_to_grouping(output_file, download_url, output_file_to_download_url)

    return output_file_to_bldg_ids, output_file_to_download_url


def _create_metadata_progress_tasks(
    output_file_to_bldg_ids: dict[Path, list[BuildingID]],
    output_file_to_download_url: dict[Path, list[str]],
    progress: Progress,
) -> dict[Path, TaskID]:
    """Create progress tasks for metadata downloads.

    Returns:
        Dictionary mapping output files to their progress task IDs.
    """
    output_file_to_metadata_task = {}
    for output_file, bldg_ids in output_file_to_bldg_ids.items():
        bldg_id = bldg_ids[0]
        state = bldg_id.state
        upgrade = bldg_id.upgrade_id
        release_name = bldg_id.get_release_name()
        metadata_task = progress.add_task(
            f"[yellow](metadata) Identifying rows to download: {release_name} - (upgrade {upgrade}) - {state}",
            total=0,  # Will be updated when we get the file size
        )
        output_file_to_metadata_task[output_file] = metadata_task
        output_file.parent.mkdir(parents=True, exist_ok=True)
    return output_file_to_metadata_task


def _check_missing_bldg_ids(bldg_ids: list[BuildingID], found_bldg_ids: list[int], console: Console) -> bool:
    """Check if all bldg_ids were found and print warning if any are missing.

    Args:
        bldg_ids: List of requested BuildingID objects.
        found_bldg_ids: List of bldg_id integers that were found in the metadata file.
        console: Rich console for printing messages.

    Returns:
        True if all bldg_ids were found, False otherwise.
    """
    requested_bldg_ids = {bldg_id.bldg_id for bldg_id in bldg_ids}
    for found_bldg_id in found_bldg_ids:
        if found_bldg_id not in requested_bldg_ids:
            release_name = bldg_ids[0].get_release_name()
            state = bldg_ids[0].state
            upgrade = bldg_ids[0].upgrade_id
            missing_bldg_ids = found_bldg_id
            missing_bldg_ids_str = str(missing_bldg_ids)
            console.print(
                f"[red]Missing bldg_id in metadata file: {missing_bldg_ids_str} for {release_name} - (upgrade {upgrade}) - {state}[/red]"
            )
            return False
    return True


def _download_metadata_with_progress(
    bldg_ids: list[BuildingID],
    output_dir: Path,
    progress: Progress,
    downloaded_paths: list[Path],
    failed_downloads: list[str],
    console: Console,
) -> tuple[list[Path], list[str]]:
    """Download metadata file with progress tracking."""
    # Output file here refers to the final metadata file that will be created.
    # If the metadata file has multiple counties making up a single final metdata file,
    # it will have one output file and multiple download URL's.
    # If the upgrade is SB upgrade that has multiple components making up a single final upgrade,
    # there will be multiple output files, one for each component.
    output_file_to_bldg_ids, output_file_to_download_url = _group_bldg_ids_by_output_file_metadata(
        bldg_ids, output_dir, failed_downloads
    )

    output_file_to_metadata_task = _create_metadata_progress_tasks(
        output_file_to_bldg_ids, output_file_to_download_url, progress
    )

    for output_file, file_grouped_bldg_ids in output_file_to_bldg_ids.items():
        try:
            download_urls = output_file_to_download_url[output_file]
            found_bldg_ids: list[int] = []
            _download_with_progress_metadata(
                download_urls,
                file_grouped_bldg_ids,
                output_file,
                progress,
                output_file_to_metadata_task[output_file],
                found_bldg_ids,
            )
            if not _check_missing_bldg_ids(file_grouped_bldg_ids, found_bldg_ids, console):
                failed_downloads.append(str(output_file))
                continue
            # If the upgrade is an SB upgrade, there will be multiple output files,
            # one for each component. We therefore do not add each of these to
            # "successful downloads" list. This is done later.
            if file_grouped_bldg_ids[0].is_SB_upgrade():
                continue
            downloaded_paths.append(output_file)
        except Exception as e:
            failed_downloads.append(str(output_file))
            for bldg_id in file_grouped_bldg_ids:
                console.print(f"[red]Download failed for metadata {bldg_id.bldg_id}: {e}[/red]")

    for bldg_id in bldg_ids:
        if bldg_id.is_SB_upgrade():
            # If the upgrade is an SB upgrade, there will be multiple output files.
            # Only here do we add the final metadata file to the "successful downloads" list.
            output_file = (
                output_dir
                / bldg_id.get_release_name()
                / "metadata"
                / f"state={bldg_id.state}"
                / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}"
                / bldg_id.get_output_filename("metadata")
            )
            downloaded_paths.append(output_file)
            _process_SB_upgrade_scenario(bldg_id, output_dir, output_file, "metadata", None)
    return downloaded_paths, failed_downloads


def download_weather_file_with_progress(
    bldg_id: BuildingID, output_dir: Path, progress: Progress, task_id: TaskID
) -> Path:
    """Download weather file with progress tracking."""
    download_url = bldg_id.get_weather_file_url()
    if download_url is None:
        raise NoWeatherFileError()
    output_file = (
        output_dir
        / bldg_id.get_release_name()
        / "weather"
        / f"state={bldg_id.state}"
        / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}"
        / f"{bldg_id.get_weather_station_name()}.csv"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    _download_with_progress(download_url, output_file, progress, task_id)
    return output_file


def _download_building_data_parallel(
    bldg_ids: list[BuildingID],
    file_type_obj: RequestedFileTypes,
    output_dir: Path,
    max_workers: int,
    progress: Progress,
    downloaded_paths: list[Path],
    failed_downloads: list[str],
    console: Console,
) -> None:
    """Download building data (HPXML and schedule files) in parallel."""
    if not (file_type_obj.hpxml or file_type_obj.schedule):
        return

    # Create individual download tasks for each building
    download_tasks = {}
    for i, bldg_id in enumerate(bldg_ids):
        task_id = progress.add_task(
            f"[cyan]Building Data for bldg{bldg_id.bldg_id} (upgrade {bldg_id.upgrade_id})",
            total=0,  # Will be updated when we get the file size
        )
        download_tasks[i] = task_id

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and keep track of future -> bldg_id mapping
        future_to_bldg = {
            executor.submit(
                download_bldg_data, bldg_id, file_type_obj, output_dir, progress, download_tasks[i]
            ): bldg_id
            for i, bldg_id in enumerate(bldg_ids)
        }

        # Process completed futures
        for future in concurrent.futures.as_completed(future_to_bldg):
            bldg_id = future_to_bldg[future]  # Get the correct bldg_id for this future
            _process_download_results(future, bldg_id, file_type_obj, downloaded_paths, failed_downloads, console)


def _download_15min_load_curves_parallel(
    bldg_ids: list[BuildingID],
    output_dir: Path,
    max_workers: int,
    progress: Progress,
    downloaded_paths: list[Path],
    failed_downloads: list[str],
    console: Console,
) -> None:
    """Download 15-minute load curves in parallel with progress tracking."""

    # Create progress tasks based on dataset size
    if len(bldg_ids) > 500:
        load_curve_tasks = _create_batch_progress_tasks_15min(bldg_ids, progress, console)
    else:
        load_curve_tasks = _create_individual_progress_tasks_15min(bldg_ids, progress)

    # Create download functions
    def download_15min_with_task_id(bldg_id: BuildingID, output_dir: Path, task_id: TaskID) -> Path:
        return download_15min_load_curve_with_progress(bldg_id, output_dir, progress, task_id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        if len(bldg_ids) > 500:
            # Process in batches for large datasets
            num_batches = 20
            batch_size = ((len(bldg_ids) + num_batches - 1) // num_batches + 99) // 100 * 100
            future_to_bldg = {}

            for batch_idx in range(0, len(bldg_ids), batch_size):
                batch = bldg_ids[batch_idx : batch_idx + batch_size]
                # Skip empty batches
                if not batch:
                    break

                task_id = load_curve_tasks[batch_idx // batch_size]

                for bldg_id in batch:
                    future = executor.submit(
                        _download_15min_with_batch_progress,
                        bldg_id,
                        output_dir,
                        task_id,
                        progress,
                    )
                    future_to_bldg[future] = bldg_id
        else:
            # Original behavior for smaller datasets
            future_to_bldg = {
                executor.submit(download_15min_with_task_id, bldg_id, output_dir, load_curve_tasks[i]): bldg_id
                for i, bldg_id in enumerate(bldg_ids)
            }

        # Process completed futures
        for future in concurrent.futures.as_completed(future_to_bldg):
            bldg_id = future_to_bldg[future]
            _process_download_future_15min(future, bldg_id, output_dir, downloaded_paths, failed_downloads, console)


def _create_batch_progress_tasks(
    bldg_ids: list[BuildingID], aggregate_time_step: str, progress_suffix: str, progress: Progress, console: Console
) -> dict[int, TaskID]:
    """Create progress tasks for batch processing."""
    num_batches = 20
    # Calculate batch size rounded up to nearest 100
    batch_size = ((len(bldg_ids) + num_batches - 1) // num_batches + 99) // 100 * 100
    console.print(
        f"[blue]Using batch processing: {len(bldg_ids)} buildings split into {num_batches} batches of up to {batch_size} buildings each[/blue]"
    )

    load_curve_tasks = {}
    for i in range(num_batches):
        # Calculate how many buildings are in this batch
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(bldg_ids))
        batch_count = end_idx - start_idx

        # Skip empty or negative batches
        if batch_count <= 0:
            break

        console.print(f"[blue]Batch {i + 1}/{num_batches}: {batch_count} buildings[/blue]")

        task_id = progress.add_task(
            f"[magenta]Batch {i + 1}/{num_batches} ({progress_suffix})",
            total=batch_count,  # Set total to the number of buildings in this batch
        )
        load_curve_tasks[i] = task_id

    return load_curve_tasks


def _create_batch_progress_tasks_15min(
    bldg_ids: list[BuildingID], progress: Progress, console: Console
) -> dict[int, TaskID]:
    """Create progress tasks for 15-minute load curve batch processing."""
    num_batches = 20
    # Calculate batch size rounded up to nearest 100
    batch_size = ((len(bldg_ids) + num_batches - 1) // num_batches + 99) // 100 * 100
    console.print(
        f"[blue]Using batch processing: {len(bldg_ids)} buildings split into {num_batches} batches of up to {batch_size} buildings each[/blue]"
    )

    load_curve_tasks = {}
    for i in range(num_batches):
        # Calculate how many buildings are in this batch
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(bldg_ids))
        batch_count = end_idx - start_idx

        # Skip empty or negative batches
        if batch_count <= 0:
            break

        console.print(f"[blue]Batch {i + 1}/{num_batches}: {batch_count} buildings[/blue]")

        task_id = progress.add_task(
            f"[magenta]Batch {i + 1}/{num_batches} (15min)",
            total=batch_count,  # Set total to the number of buildings in this batch
        )
        load_curve_tasks[i] = task_id

    return load_curve_tasks


def _create_individual_progress_tasks(
    bldg_ids: list[BuildingID], progress_suffix: str, progress: Progress
) -> dict[int, TaskID]:
    """Create progress tasks for individual building processing."""
    load_curve_tasks = {}
    for i, bldg_id in enumerate(bldg_ids):
        task_id = progress.add_task(
            f"[magenta]{progress_suffix} - Building {bldg_id.bldg_id} (upgrade {bldg_id.upgrade_id})",
            total=0,  # Will be updated when we get the file size
        )
        load_curve_tasks[i] = task_id
    return load_curve_tasks


def _create_individual_progress_tasks_15min(bldg_ids: list[BuildingID], progress: Progress) -> dict[int, TaskID]:
    """Create progress tasks for individual 15-minute load curve processing."""
    load_curve_tasks = {}
    for i, bldg_id in enumerate(bldg_ids):
        task_id = progress.add_task(
            f"[magenta]Load curve {bldg_id.bldg_id} (upgrade {bldg_id.upgrade_id})",
            total=0,  # Will be updated when we get the file size
        )
        load_curve_tasks[i] = task_id
    return load_curve_tasks


def _download_aggregate_with_batch_progress(
    bldg_id: BuildingID, output_dir: Path, task_id: TaskID, aggregate_time_step: str, progress: Progress
) -> Path:
    """Download with batch progress tracking."""
    # Download the file without individual progress tracking
    result = download_aggregate_time_step_load_curve_with_progress(bldg_id, output_dir, None, None, aggregate_time_step)
    # Update batch progress by 1
    progress.update(task_id, advance=1)
    return result


def _download_15min_with_batch_progress(
    bldg_id: BuildingID, output_dir: Path, task_id: TaskID, progress: Progress
) -> Path:
    """Download 15-minute load curve with batch progress tracking."""
    # Download the file without individual progress tracking
    result = download_15min_load_curve_with_progress(bldg_id, output_dir, None, None)
    # Update batch progress by 1
    progress.update(task_id, advance=1)
    return result


def _process_download_future(
    future: concurrent.futures.Future,
    bldg_id: BuildingID,
    output_dir: Path,
    aggregate_time_step: str,
    downloaded_paths: list[Path],
    failed_downloads: list[str],
    console: Console,
) -> None:
    """Process a completed download future."""
    try:
        output_file = future.result()
        downloaded_paths.append(output_file)
    except NoAggregateLoadCurveError:
        output_file = (
            output_dir
            / bldg_id.get_release_name()
            / f"load_curve_{aggregate_time_step}"
            / f"state={bldg_id.state}"
            / f"{str(bldg_id.bldg_id)!s}-{int(bldg_id.upgrade_id)!s}.parquet"
        )
        failed_downloads.append(str(output_file))
        console.print(f"[red]{aggregate_time_step} load curve not available for {bldg_id.get_release_name()}[/red]")
        raise
    except Exception as e:
        output_file = (
            output_dir
            / bldg_id.get_release_name()
            / f"load_curve_{aggregate_time_step}"
            / f"state={bldg_id.state}"
            / f"{str(bldg_id.bldg_id)!s}-{int(bldg_id.upgrade_id)!s}.parquet"
        )
        failed_downloads.append(str(output_file))
        console.print(f"[red]Download failed for {aggregate_time_step} load curve {bldg_id.bldg_id}: {e}[/red]")


def _process_download_future_15min(
    future: concurrent.futures.Future,
    bldg_id: BuildingID,
    output_dir: Path,
    downloaded_paths: list[Path],
    failed_downloads: list[str],
    console: Console,
) -> None:
    """Process a completed 15-minute download future."""
    try:
        output_file = future.result()
        downloaded_paths.append(output_file)
        # Special case for SwitchBox Analysis upgrades
        if bldg_id.is_SB_upgrade():
            file_type: FileType = "load_curve_15min"
            aggregate_time_step = "15min"
            _process_SB_upgrade_scenario(bldg_id, output_dir, output_file, file_type, aggregate_time_step)
    except No15minLoadCurveError:
        output_file = (
            output_dir
            / bldg_id.get_release_name()
            / "load_curve_15min"
            / f"state={bldg_id.state}"
            / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}"
            / f"{str(bldg_id.bldg_id)!s}-{int(bldg_id.upgrade_id)!s}.parquet"
        )
        failed_downloads.append(str(output_file))
        console.print(f"[red]15 min load curve not available for {bldg_id.get_release_name()}[/red]")
        raise
    except Exception as e:
        output_file = (
            output_dir
            / bldg_id.get_release_name()
            / "load_curve_15min"
            / f"state={bldg_id.state}"
            / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}"
            / f"{str(bldg_id.bldg_id)!s}-{int(bldg_id.upgrade_id)!s}.parquet"
        )
        failed_downloads.append(str(output_file))
        console.print(f"[red]Download failed for 15 min load curve {bldg_id.bldg_id}: {e}[/red]")


def _download_aggregate_load_curves_parallel(
    bldg_ids: list[BuildingID],
    output_dir: Path,
    aggregate_time_step: str,
    max_workers: int,
    progress: Progress,
    downloaded_paths: list[Path],
    failed_downloads: list[str],
    console: Console,
) -> None:
    """Download aggregate load curves in parallel with progress tracking."""

    progress_suffix = f"{bldg_ids[0].get_release_name()} - (load_curve_{aggregate_time_step}) - {bldg_ids[0].state}"

    # Create progress tasks based on dataset size
    if len(bldg_ids) > 500:
        load_curve_tasks = _create_batch_progress_tasks(
            bldg_ids, aggregate_time_step, progress_suffix, progress, console
        )
    else:
        load_curve_tasks = _create_individual_progress_tasks(bldg_ids, progress_suffix, progress)

    # Create download functions
    def download_aggregate_with_task_id(
        bldg_id: BuildingID, output_dir: Path, task_id: TaskID, aggregate_time_step: str
    ) -> Path:
        return download_aggregate_time_step_load_curve_with_progress(
            bldg_id, output_dir, progress, task_id, aggregate_time_step
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        if len(bldg_ids) > 500:
            # Process in batches for large datasets
            num_batches = 20
            batch_size = ((len(bldg_ids) + num_batches - 1) // num_batches + 99) // 100 * 100
            future_to_bldg = {}

            for batch_idx in range(0, len(bldg_ids), batch_size):
                batch = bldg_ids[batch_idx : batch_idx + batch_size]
                # Skip empty batches
                if not batch:
                    break

                task_id = load_curve_tasks[batch_idx // batch_size]

                for bldg_id in batch:
                    future = executor.submit(
                        _download_aggregate_with_batch_progress,
                        bldg_id,
                        output_dir,
                        task_id,
                        aggregate_time_step,
                        progress,
                    )
                    future_to_bldg[future] = bldg_id
        else:
            # Original behavior for smaller datasets
            future_to_bldg = {
                executor.submit(
                    download_aggregate_with_task_id, bldg_id, output_dir, load_curve_tasks[i], aggregate_time_step
                ): bldg_id
                for i, bldg_id in enumerate(bldg_ids)
            }

        # Process completed futures
        for future in concurrent.futures.as_completed(future_to_bldg):
            bldg_id = future_to_bldg[future]
            _process_download_future(
                future, bldg_id, output_dir, aggregate_time_step, downloaded_paths, failed_downloads, console
            )


def _download_metadata(
    bldg_ids: list[BuildingID],
    output_dir: Path,
    progress: Progress,
    downloaded_paths: list[Path],
    failed_downloads: list[str],
    console: Console,
) -> None:
    """Download metadata file."""
    if not bldg_ids:
        return
    _download_metadata_with_progress(bldg_ids, output_dir, progress, downloaded_paths, failed_downloads, console)
    # Only keep the requested bldg_ids in the metadata file
    _filter_metadata_requested_bldg_ids(bldg_ids, output_dir, downloaded_paths)


def _download_with_progress_annual_load_curve(
    urls: list[str],
    bldg_ids: list[BuildingID],
    output_file: Path,
    progress: Progress,
    task_id: TaskID,
    found_bldg_ids: list[int],
) -> int:
    """Download annual load curve files with progress tracking, filtering for only requested bldg_ids from S3."""

    # Extract release info for progress messages
    first_bldg_id = bldg_ids[0]
    release_name = first_bldg_id.get_release_name()
    upgrade = first_bldg_id.upgrade_id
    state = first_bldg_id.state
    progress_suffix = f"{release_name} - (upgrade {upgrade}) - {state}"

    # Get the list of bldg_ids we need to filter for, sorted for optimization
    requested_bldg_ids_set = {bldg_id.bldg_id for bldg_id in bldg_ids}
    requested_bldg_ids = sorted(requested_bldg_ids_set)

    # Create S3 filesystem for anonymous access
    s3_filesystem = fs.S3FileSystem(anonymous=True, region="us-west-2")

    for url in urls:
        s3_path = _convert_url_to_s3_path(url)
        progress.update(task_id, description=f"[yellow](annual) Connecting to S3: {progress_suffix}")

        try:
            # Get total file size from S3
            s3_path_parts = s3_path.replace("s3://", "").split("/", 1)
            bucket_name = s3_path_parts[0]
            s3_key = s3_path_parts[1] if len(s3_path_parts) > 1 else ""

            # Use boto3 to get file size (HEAD request)
            progress.update(task_id, description=f"[yellow](annual) Getting file size: {progress_suffix}")
            s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
            try:
                response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
                total_file_size = response.get("ContentLength", 0)
            except Exception:
                total_file_size = 0

            s3_file_path = f"{bucket_name}/{s3_key}"

            try:
                progress.update(
                    task_id, description=f"[yellow](annual) Reading and filtering data from S3: {progress_suffix}"
                )
                with s3_filesystem.open_input_file(s3_file_path) as s3_file:
                    parquet_file = pq.ParquetFile(s3_file)
                    pq_metadata = parquet_file.metadata
                    total_rows = pq_metadata.num_rows

                    s3_file.seek(0)

                    column_names = pq_metadata.schema.names
                    columns_to_keep = []
                    for col in column_names:
                        if (
                            any(keyword in col for keyword in ["bldg_id", "upgrade", "metadata_index"])
                            or col.startswith("out.")
                        ) and not col.startswith("in."):
                            columns_to_keep.append(col)

                    # Filter for only requested bldg_ids
                    df = (
                        pl.scan_parquet(s3_file)
                        .select(columns_to_keep)
                        .filter(pl.col("bldg_id").is_in(requested_bldg_ids))
                        .collect()
                    )
                    found_bldg_ids.extend(df["bldg_id"].to_list())
                    selected_rows = len(df)
                    progress.update(
                        task_id,
                        total=total_file_size * (selected_rows / total_rows) if total_rows > 0 else total_file_size,
                        description=f"[yellow](annual) Writing filtered data: {progress_suffix}",
                    )
            except Exception:
                traceback.print_exc()
                raise

            selected_rows = len(df)

            # Check if output file already exists
            if output_file.exists():
                # Read existing file and combine with new data
                existing_file = pl.scan_parquet(output_file)
                new_df = df if isinstance(df, pl.DataFrame) else pl.DataFrame(df)
                new_file_lazy = pl.LazyFrame(new_df)

                # Use how="diagonal" to handle schema mismatches
                combined_file = pl.concat([existing_file, new_file_lazy])
                # Remove duplicate rows based on bldg_id column
                deduplicated_file = combined_file.collect().unique(subset=["bldg_id"], keep="first")
                deduplicated_file.write_parquet(output_file)
            else:
                # Write filtered data directly
                df.write_parquet(output_file)

            # Calculate downloaded size as ratio
            downloaded_size = int(total_file_size * (selected_rows / total_rows)) if total_rows > 0 else 0
            progress.update(
                task_id,
                completed=downloaded_size,
                total=downloaded_size,
                description=f"[green](annual) Download complete: {progress_suffix}",
            )
            gc.collect()
        except Exception:
            # Fallback to old method (download entire file using requests) if S3 filtering fails
            progress.update(
                task_id, description=f"[yellow](annual) Falling back to full file download: {progress_suffix}"
            )
            return _download_with_progress_annual_load_curve_fallback(url, output_file, progress, task_id)
        else:
            return downloaded_size

    return 0


def _download_with_progress_annual_load_curve_fallback(
    url: str, output_file: Path, progress: Progress, task_id: TaskID
) -> int:
    """Fallback method to download entire annual load curve file (old behavior)."""
    # Get file size first
    response = requests.head(url, timeout=30, verify=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    progress.update(task_id, total=total_size)

    # Download with streaming
    response = requests.get(url, stream=True, timeout=30, verify=True)
    response.raise_for_status()

    downloaded_size = 0

    # Check if output file already exists
    if output_file.exists():
        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as temp_file:
            temp_path = Path(temp_file.name)
            with open(temp_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress.update(task_id, completed=downloaded_size)

            existing_file = pl.scan_parquet(output_file)
            new_file = pl.scan_parquet(temp_path)
            # Use how="diagonal" to handle schema mismatches
            combined_file = pl.concat([existing_file, new_file], how="diagonal")
            # Remove duplicate rows based on bldg_id column
            deduplicated_file = combined_file.collect().unique(subset=["bldg_id"], keep="first")
            deduplicated_file.write_parquet(output_file)
            gc.collect()
            os.remove(temp_path)

    else:
        # File doesn't exist, download normally
        with open(str(output_file), "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        progress.update(task_id, completed=downloaded_size)

    return downloaded_size


def download_annual_load_curve_with_progress(
    bldg_ids: list[BuildingID],
    output_file: Path,
    download_urls: list[str],
    progress: Optional[Progress] = None,
    task_id: Optional[TaskID] = None,
) -> Path:
    """Download the annual load curve for given buildings with progress tracking and S3 filtering."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if progress and task_id is not None:
        found_bldg_ids: list[int] = []
        _download_with_progress_annual_load_curve(
            download_urls, bldg_ids, output_file, progress, task_id, found_bldg_ids
        )
    else:
        # Fallback to simple download if no progress tracking
        for download_url in download_urls:
            response = requests.get(download_url, timeout=30, verify=True)
            response.raise_for_status()
            if output_file.exists():
                # Combine with existing file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as temp_file:
                    temp_path = Path(temp_file.name)
                    with open(temp_path, "wb") as file:
                        file.write(response.content)
                    existing_file = pl.scan_parquet(output_file)
                    new_file = pl.scan_parquet(temp_path)
                    combined_file = pl.concat([existing_file, new_file], how="diagonal")
                    deduplicated_file = combined_file.collect().unique(subset=["bldg_id"], keep="first")
                    deduplicated_file.write_parquet(output_file)
                    os.remove(temp_path)
            else:
                with open(output_file, "wb") as file:
                    file.write(response.content)

    return output_file


def _add_bldg_id_to_grouping(
    output_file: Path,
    bldg_id: BuildingID,
    output_file_to_bldg_ids: dict[Path, list[BuildingID]],
) -> None:
    """Add bldg_id to the output file grouping dictionary."""
    if output_file not in output_file_to_bldg_ids:
        output_file_to_bldg_ids[output_file] = []
    output_file_to_bldg_ids[output_file].append(bldg_id)


def _group_bldg_ids_by_output_file_annual_load_curve(
    bldg_ids: list[BuildingID], output_dir: Path, failed_downloads: list[str]
) -> tuple[dict[Path, list[BuildingID]], dict[Path, list[str]]]:
    """Group bldg_ids by their output file paths and collect download URLs."""
    output_file_to_bldg_ids: dict[Path, list[BuildingID]] = {}
    output_file_to_download_url: dict[Path, list[str]] = {}

    for bldg_id in bldg_ids:
        if bldg_id.is_SB_upgrade():
            component_bldg_ids = bldg_id.get_SB_upgrade_component_bldg_ids()
            if component_bldg_ids is None:
                failed_downloads.append(str(bldg_id.bldg_id))
                continue
            for component_bldg_id in component_bldg_ids:
                component_output_filename = component_bldg_id.get_annual_load_curve_filename()
                if component_output_filename is None:
                    message = f"Annual load curve is not available for {component_bldg_id.get_release_name()}, upgrade {component_bldg_id.upgrade_id}"
                    raise NoAnnualLoadCurveError(message)
                output_file = (
                    output_dir
                    / bldg_id.get_release_name()
                    / "load_curve_annual"
                    / f"state={bldg_id.state}"
                    / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}"
                    / component_output_filename
                )
                _add_bldg_id_to_grouping(output_file, bldg_id, output_file_to_bldg_ids)

                component_download_url = component_bldg_id.get_annual_load_curve_url()
                if component_download_url is None:
                    failed_downloads.append(str(output_file))
                    continue
                _add_download_url_to_grouping(output_file, component_download_url, output_file_to_download_url)
        else:
            component_output_filename = bldg_id.get_annual_load_curve_filename()
            if component_output_filename is None:
                message = (
                    f"Annual load curve is not available for {bldg_id.get_release_name()}, upgrade {bldg_id.upgrade_id}"
                )
                raise NoAnnualLoadCurveError(message)
            output_file = (
                output_dir
                / bldg_id.get_release_name()
                / "load_curve_annual"
                / f"state={bldg_id.state}"
                / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}"
                / component_output_filename
            )
            download_url = bldg_id.get_annual_load_curve_url()
            if download_url is None:
                failed_downloads.append(str(output_file))
                continue
            _add_bldg_id_to_grouping(output_file, bldg_id, output_file_to_bldg_ids)
            _add_download_url_to_grouping(output_file, download_url, output_file_to_download_url)

    return output_file_to_bldg_ids, output_file_to_download_url


def _add_download_url_to_grouping(
    output_file: Path,
    download_url: str,
    output_file_to_download_url: dict[Path, list[str]],
) -> None:
    """Add download URL to the download URL grouping dictionary."""
    if download_url is None:
        return
    if output_file not in output_file_to_download_url:
        output_file_to_download_url[output_file] = []
    if download_url not in output_file_to_download_url[output_file]:
        output_file_to_download_url[output_file].append(download_url)


def _create_annual_load_curve_tasks(
    output_file_to_bldg_ids: dict[Path, list[BuildingID]],
    progress: Progress,
) -> dict[Path, TaskID]:
    """Create progress tasks for annual load curve downloads."""
    output_file_to_annual_load_curve_tasks = {}
    for output_file, bldg_ids in output_file_to_bldg_ids.items():
        bldg_id = bldg_ids[0]
        state = bldg_id.state
        upgrade = bldg_id.upgrade_id
        release_name = bldg_id.get_release_name()
        annual_load_curve_task = progress.add_task(
            f"[magenta]Annual load curve {release_name} - (upgrade {upgrade}) - {state}",
            total=0,  # Will be updated when we get the file size
        )
        output_file_to_annual_load_curve_tasks[output_file] = annual_load_curve_task
    return output_file_to_annual_load_curve_tasks


def _download_annual_load_curves_parallel(
    bldg_ids: list[BuildingID],
    output_dir: Path,
    max_workers: int,
    progress: Progress,
    downloaded_paths: list[Path],
    failed_downloads: list[str],
    console: Console,
) -> None:
    """Download annual load curves in parallel with progress tracking."""

    output_file_to_bldg_ids, output_file_to_download_url = _group_bldg_ids_by_output_file_annual_load_curve(
        bldg_ids, output_dir, failed_downloads
    )

    # Create progress tasks for annual load curve downloads
    output_file_to_annual_load_curve_tasks = _create_annual_load_curve_tasks(output_file_to_bldg_ids, progress)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a modified version of the download function that uses the specific task IDs
        def download_annual_with_task_id(
            bldg_ids_for_file: list[BuildingID], output_file: Path, download_urls: list[str], task_id: TaskID
        ) -> Path:
            return download_annual_load_curve_with_progress(
                bldg_ids_for_file, output_file, download_urls, progress, task_id
            )

        future_to_bldg = {
            executor.submit(
                download_annual_with_task_id,
                output_file_to_bldg_ids[output_file],
                output_file,
                download_urls,
                output_file_to_annual_load_curve_tasks[output_file],
            ): output_file
            for output_file, download_urls in output_file_to_download_url.items()
        }

        for future in concurrent.futures.as_completed(future_to_bldg):
            output_file = future_to_bldg[future]
            bldg_ids = output_file_to_bldg_ids[output_file]
            bldg_id = bldg_ids[0]
            if bldg_id.is_SB_upgrade():
                # If the upgrade is an SB upgrade, there will be multiple output files.
                # Only here do we add the final metadata file to the "successful downloads" list.
                annual_load_curve_filename = bldg_id.get_annual_load_curve_filename()
                if annual_load_curve_filename is None:
                    msg = f"Annual load curve filename is not available for {bldg_id.get_release_name()}, upgrade {bldg_id.upgrade_id}"
                    raise ValueError(msg)
                output_file = (
                    output_dir
                    / bldg_id.get_release_name()
                    / "load_curve_annual"
                    / f"state={bldg_id.state}"
                    / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}"
                    / annual_load_curve_filename
                )

                # Check if all required component files exist
                required_component_files = _get_required_component_files_for_SB_upgrade(
                    bldg_id, output_dir, "load_curve_annual"
                )
                if not all(comp_file.exists() for comp_file in required_component_files):
                    continue

                downloaded_paths.append(output_file)
                _process_SB_upgrade_scenario(bldg_id, output_dir, output_file, "load_curve_annual", None)
            else:
                output_filename = bldg_id.get_annual_load_curve_filename()
                try:
                    output_file = future.result()
                    downloaded_paths.append(output_file)
                except NoAnnualLoadCurveError:
                    output_file = (
                        output_dir
                        / bldg_id.get_release_name()
                        / "load_curve_annual"
                        / f"state={bldg_id.state}"
                        / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}"
                        / (output_filename or "")  # TODO: Find a better way
                    )
                    failed_downloads.append(str(output_file))
                    console.print(f"[red]Annual load curve not available for {bldg_id.get_release_name()}[/red]")
                    raise
                except Exception as e:
                    output_file = (
                        output_dir
                        / bldg_id.get_release_name()
                        / "load_curve_annual"
                        / f"state={bldg_id.state}"
                        / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}"
                        / (output_filename or "")  # TODO: Find a better way
                    )
                    failed_downloads.append(str(output_file))
                    console.print(f"[red]Download failed for annual load curve {bldg_id.bldg_id}: {e}[/red]")


def _get_parquet_files_for_state(s3_client: Any, bucket: str, s3_prefix: str) -> list[str]:
    """Get list of parquet files for a given S3 prefix."""
    paginator = s3_client.get_paginator("list_objects_v2")
    parquet_files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                parquet_files.append(obj["Key"])
    return parquet_files


def _download_and_read_parquet_files(
    s3_client: Any, bucket: str, parquet_files: list[str], output_dir: Path
) -> list[Any]:
    """Download and read parquet files, returning a list of dataframes."""
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dataframes = []
    for s3_key in parquet_files:
        temp_file = output_dir / f"temp_{s3_key.split('/')[-1]}"
        s3_client.download_file(bucket, s3_key, str(temp_file))
        df = pl.read_parquet(str(temp_file))
        state_dataframes.append(df)
        temp_file.unlink()
    return state_dataframes


def _process_state_data(
    s3_client: Any, bucket: str, prefix: str, release: str, state: str, output_dir: Path
) -> tuple[list[Any], bool]:
    """Process data for a single state, returning (dataframes, has_data)."""
    s3_prefix = f"{prefix}release={release}/state={state}/"
    parquet_files = _get_parquet_files_for_state(s3_client, bucket, s3_prefix)

    if not parquet_files:
        return [], False

    state_dataframes = _download_and_read_parquet_files(s3_client, bucket, parquet_files, output_dir)
    if state_dataframes:
        state_combined_df = pl.concat(state_dataframes)
        return [state_combined_df], True
    return [], False


def _save_filtered_state_data(
    state_df: Any, state: str, bldg_ids: list[BuildingID], release: str, output_dir: Path, downloaded_paths: list[Path]
) -> None:
    """Save filtered data for a specific state."""
    bldg_id_list = [str(bldg.bldg_id) for bldg in bldg_ids if bldg.state == state]
    if not bldg_id_list:
        return

    filtered_df = state_df.filter(pl.col("bldg_id").is_in(bldg_id_list))
    if filtered_df.height == 0:
        return

    output_file = output_dir / release / "trip_schedules" / f"state={state}" / "trip_schedules.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.write_parquet(str(output_file))
    downloaded_paths.append(output_file)


def _download_trip_schedules_data(
    bldg_ids: list[BuildingID],
    output_dir: Path,
    downloaded_paths: list[Path],
    bucket: str = "buildstock-fetch",
    prefix: str = "ev_demand/trip_schedules/",
) -> None:
    """
    Download and filter trip schedules data for specific building IDs.

    Args:
        bldg_ids: List of BuildingID objects to filter for.
        output_dir: Directory to save the downloaded files.
        downloaded_paths: List to append successful download paths to.
        bucket: Name of the S3 bucket.
        prefix: S3 prefix for the trip schedules data.

    Raises:
        NoBuildingDataError: If no buildings from bldg_ids are found in any available state data.
    """
    import warnings

    release = bldg_ids[0].get_release_name()
    states_list = list({bldg.state for bldg in bldg_ids})

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    all_dataframes = []
    available_states = []
    unavailable_states = []

    # Process each state
    for state in states_list:
        state_dataframes, has_data = _process_state_data(s3, bucket, prefix, release, state, output_dir)

        if has_data:
            available_states.append(state)
            all_dataframes.extend(state_dataframes)
        else:
            unavailable_states.append(state)

    # Issue warnings for unavailable states
    if unavailable_states:
        warnings.warn(
            f"No trip schedules data found for {release} in states: {', '.join(unavailable_states)}. "
            f"Continuing with available states: {', '.join(available_states)}.",
            stacklevel=2,
        )

    if not all_dataframes:
        msg = f"No trip schedules data found for {release} in any of the requested states: {', '.join(states_list)}"
        raise NoBuildingDataError(msg)

    # Save filtered data for each available state separately
    for i, state_df in enumerate(all_dataframes):
        state = available_states[i]
        _save_filtered_state_data(state_df, state, bldg_ids, release, output_dir, downloaded_paths)

    if not any(bldg.state in available_states for bldg in bldg_ids):
        msg = f"No trip schedules data found for buildings {[bldg.bldg_id for bldg in bldg_ids]} in {release} for any available state"
        raise NoBuildingDataError(msg)


def _download_weather_files_parallel(
    bldg_ids: list[BuildingID],
    output_dir: Path,
    max_workers: int,
    progress: Progress,
    downloaded_paths: list[Path],
    failed_downloads: list[str],
    console: Console,
    weather_states: Union[list[str], None] = None,
) -> None:
    """Download weather files in parallel with progress tracking."""
    # Initialize weather_states to empty list if None
    if weather_states is None:
        weather_states = []

    # Break if weather_states is empty
    if len(weather_states) == 0:
        for bldg_id in bldg_ids:
            output_file = (
                output_dir
                / bldg_id.get_release_name()
                / "weather"
                / f"state={bldg_id.state}"
                / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}"
                / f"{bldg_id.get_weather_station_name()}.csv"
            )
            failed_downloads.append(str(output_file))
            console.print(f"[red]Weather file not available for {bldg_id.get_release_name()}[/red]")
        return
    # Create progress tasks for weather file downloads
    weather_file_tasks = {}
    for i, bldg_id in enumerate(bldg_ids):
        if bldg_id.state in weather_states:
            task_id = progress.add_task(
                f"[magenta]Weather file {bldg_id.bldg_id} (upgrade {bldg_id.upgrade_id})",
                total=0,  # Will be updated when we get the file size
            )
            weather_file_tasks[i] = task_id
        else:
            output_file = (
                output_dir
                / bldg_id.get_release_name()
                / "weather"
                / f"state={bldg_id.state}"
                / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}"
                / f"{bldg_id.get_weather_station_name()}.csv"
            )
            failed_downloads.append(str(output_file))
            console.print(f"[red]Weather file not available for {bldg_id.get_release_name()}[/red]")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a modified version of the download function that uses the specific task IDs
        def download_weather_file_with_task_id(bldg_id: BuildingID, output_dir: Path, task_id: TaskID) -> Path:
            return download_weather_file_with_progress(bldg_id, output_dir, progress, task_id)

        future_to_bldg = {
            executor.submit(download_weather_file_with_task_id, bldg_id, output_dir, weather_file_tasks[i]): bldg_id
            for i, bldg_id in enumerate(bldg_ids)
        }

        for future in concurrent.futures.as_completed(future_to_bldg):
            bldg_id = future_to_bldg[future]
            try:
                output_file = future.result()
                downloaded_paths.append(output_file)
            except NoWeatherFileError:
                output_file = (
                    output_dir
                    / bldg_id.get_release_name()
                    / "weather"
                    / f"state={bldg_id.state}"
                    / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}"
                    / f"{bldg_id.get_weather_station_name()}.csv"
                )
                failed_downloads.append(str(output_file))
                console.print(f"[red]Weather file not available for {bldg_id.get_release_name()}[/red]")
                raise
            except Exception as e:
                output_file = (
                    output_dir
                    / bldg_id.get_release_name()
                    / "weather"
                    / f"state={bldg_id.state}"
                    / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}"
                    / f"{bldg_id.get_weather_station_name()}.csv"
                )
                failed_downloads.append(str(output_file))
                console.print(f"[red]Download failed for weather file {bldg_id.bldg_id}: {e}[/red]")
                raise


def _print_download_summary(downloaded_paths: list[Path], failed_downloads: list[str], console: Console) -> None:
    """Print a summary of the download results."""
    console.print("\n[bold green]Download complete![/bold green]")
    console.print(f"[green]Successfully downloaded: {len(downloaded_paths)} files[/green]")
    if failed_downloads:
        console.print(f"[red]Failed downloads: {len(failed_downloads)} files[/red]")
        for failed in failed_downloads:
            console.print(f"  [red]• {failed}[/red]")


def fetch_bldg_data(
    bldg_ids: list[BuildingID],
    file_type: tuple[str, ...],
    output_dir: Path,
    max_workers: int = 5,
    weather_states: Union[list[str], None] = None,
) -> tuple[list[Path], list[str]]:
    """Download building data for a given list of building ids

    Downloads the data for the given building ids and returns list of paths to the downloaded files.

    Args:
        bldg_ids: A list of BuildingID objects to download data for.

    Returns:
        A list of paths to the downloaded files.
    """
    file_type_obj = _parse_requested_file_type(file_type)
    console = Console()

    # Initialize weather_states to empty list if None
    if weather_states is None:
        weather_states = []

    downloaded_paths: list[Path] = []
    failed_downloads: list[str] = []

    # Calculate total files to download
    total_files = 0
    if file_type_obj.metadata:
        unique_metadata_urls = _resolve_unique_metadata_urls(bldg_ids)
        total_files += len(unique_metadata_urls)  # Add metadata file
    if file_type_obj.load_curve_15min:
        total_files += len(bldg_ids)  # Add 15-minute load curve files
    if file_type_obj.load_curve_hourly:
        total_files += len(bldg_ids)  # Add hourly load curve files
    if file_type_obj.load_curve_daily:
        total_files += len(bldg_ids)  # Add daily load curve files
    if file_type_obj.load_curve_monthly:
        total_files += len(bldg_ids)  # Add monthly load curve files
    if file_type_obj.load_curve_annual:
        output_file_to_bldg_ids, _ = _group_bldg_ids_by_output_file_annual_load_curve(bldg_ids, Path("data"), [])
        total_files += len(output_file_to_bldg_ids.keys())  # Add annual load curve files
    if file_type_obj.weather:
        available_bldg_ids = [bldg_id for bldg_id in bldg_ids if bldg_id.state in weather_states]
        total_files += len(available_bldg_ids) * len(weather_states)  # Add weather map files

    console.print(f"\n[bold blue]Starting download of {total_files} files...[/bold blue]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        DownloadColumn(),
        TextColumn("•"),
        TransferSpeedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        _execute_downloads(
            file_type_obj,
            bldg_ids,
            output_dir,
            max_workers,
            progress,
            downloaded_paths,
            failed_downloads,
            console,
            weather_states,
        )

        # TODO: add EV related files
        # TODO: Write a function for downloading EV related files from SB's s3 bucket.
        # It should dynamically build the download url based on the release_name + state combo.
        # Make sure to follow the directory structure for downloading the files.
        if file_type_obj.trip_schedules:
            _download_trip_schedules_data(bldg_ids, output_dir, downloaded_paths)

    _print_download_summary(downloaded_paths, failed_downloads, console)

    return downloaded_paths, failed_downloads


def _execute_downloads(
    file_type_obj: RequestedFileTypes,
    bldg_ids: list[BuildingID],
    output_dir: Path,
    max_workers: int,
    progress: Progress,
    downloaded_paths: list[Path],
    failed_downloads: list[str],
    console: Console,
    weather_states: Union[list[str], None] = None,
) -> None:
    """Execute all requested downloads based on file type configuration."""
    # Initialize weather_states to empty list if None
    if weather_states is None:
        weather_states = []

    # Download building data if requested.
    if file_type_obj.hpxml or file_type_obj.schedule:
        _download_building_data_parallel(
            bldg_ids, file_type_obj, output_dir, max_workers, progress, downloaded_paths, failed_downloads, console
        )

    # Get metadata if requested. Only one building is needed to get the metadata.
    if file_type_obj.metadata:
        _download_metadata(bldg_ids, output_dir, progress, downloaded_paths, failed_downloads, console)

    # Get 15 min load profile timeseries if requested.
    if file_type_obj.load_curve_15min:
        _download_15min_load_curves_parallel(
            bldg_ids, output_dir, max_workers, progress, downloaded_paths, failed_downloads, console
        )

    if file_type_obj.load_curve_hourly:
        aggregate_time_step = "hourly"
        _download_aggregate_load_curves_parallel(
            bldg_ids,
            output_dir,
            aggregate_time_step,
            max_workers,
            progress,
            downloaded_paths,
            failed_downloads,
            console,
        )

    if file_type_obj.load_curve_daily:
        aggregate_time_step = "daily"
        _download_aggregate_load_curves_parallel(
            bldg_ids,
            output_dir,
            aggregate_time_step,
            max_workers,
            progress,
            downloaded_paths,
            failed_downloads,
            console,
        )

    if file_type_obj.load_curve_monthly:
        aggregate_time_step = "monthly"
        _download_aggregate_load_curves_parallel(
            bldg_ids,
            output_dir,
            aggregate_time_step,
            max_workers,
            progress,
            downloaded_paths,
            failed_downloads,
            console,
        )

    # Get annual load curve if requested.
    if file_type_obj.load_curve_annual:
        _download_annual_load_curves_parallel(
            bldg_ids, output_dir, max_workers, progress, downloaded_paths, failed_downloads, console
        )
        # Process annual load curve files to filter columns
        _process_annual_load_curve_results(downloaded_paths)

    # Get weather files if requested.
    if file_type_obj.weather:
        _download_weather_files_parallel(
            bldg_ids, output_dir, max_workers, progress, downloaded_paths, failed_downloads, console, weather_states
        )


if __name__ == "__main__":  # pragma: no cover
    bldg_ids = [
        BuildingID(
            bldg_id=80963,
            release_year="2025",
            res_com="comstock",
            weather="amy2018",
            release_number="1",
            upgrade_id="0",
            state="OH",
        ),
        BuildingID(
            bldg_id=82148,
            release_year="2025",
            res_com="comstock",
            weather="amy2018",
            release_number="1",
            upgrade_id="0",
            state="OH",
        ),
    ]
    file_type = ("load_curve_annual",)
    output_dir = Path("data")
    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    print(downloaded_paths)
    print(failed_downloads)
