import concurrent.futures
import json
import os
import zipfile
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Optional, Union

import polars as pl
import requests
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


class InvalidProductError(ValueError):
    """Raised when an invalid product is provided."""

    pass


class InvalidReleaseNameError(ValueError):
    """Raised when an invalid release name is provided."""

    pass


class NoBuildingDataError(ValueError):
    """Raised when no building data is available for a given release."""

    pass


class NoMetadataError(ValueError):
    """Raised when no metadata is available for a given release."""

    pass


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


@dataclass
class BuildingID:
    bldg_id: int
    release_number: str = "1"
    release_year: str = "2022"
    res_com: str = "resstock"
    weather: str = "tmy3"
    upgrade_id: str = "0"
    state: str = "NY"

    @property
    def base_url(self) -> str:
        return (
            f"https://oedi-data-lake.s3.amazonaws.com/"
            "nrel-pds-building-stock/"
            "end-use-load-profiles-for-us-building-stock/"
            f"{self.release_year}/"
            f"{self.res_com}_{self.weather}_release_{self.release_number}/"
        )

    def get_building_data_url(self) -> str:
        """Generate the S3 download URL for this building."""
        if self.release_year == "2021":
            return ""
        elif self.release_year == "2022":
            return (
                f"{self.base_url}"
                f"building_energy_models/upgrade={self.upgrade_id}/"
                f"bldg{str(self.bldg_id).zfill(7)}-up{str(int(self.upgrade_id)).zfill(2)}.zip"
            )
        elif self.release_year == "2023":
            return ""
        elif self.release_year == "2024":
            if self.res_com == "comstock":
                return ""
            elif (self.weather == "amy2018" or self.weather == "tmy3") and self.release_number == "2":
                return (
                    f"{self.base_url}"
                    f"model_and_schedule_files/building_energy_models/upgrade={self.upgrade_id}/"
                    f"bldg{str(self.bldg_id).zfill(7)}-up{str(int(self.upgrade_id)).zfill(2)}.zip"
                )
            else:
                return ""
        elif self.release_year == "2025":
            return (
                f"{self.base_url}"
                f"building_energy_models/upgrade={str(int(self.upgrade_id)).zfill(2)}/"
                f"bldg{str(self.bldg_id).zfill(7)}-up{str(int(self.upgrade_id)).zfill(2)}.zip"
            )
        else:
            return ""

    def get_metadata_url(self) -> str:
        """Generate the S3 download URL for this building."""

        if self.release_year == "2021":
            return f"{self.base_url}metadata/metadata.parquet"
        elif self.release_year == "2022" or self.release_year == "2023":
            if self.upgrade_id == "0":
                return f"{self.base_url}metadata/baseline.parquet"
            else:
                return f"{self.base_url}metadata/upgrade{str(int(self.upgrade_id)).zfill(2)}.parquet"
        elif self.release_year == "2024":
            if self.res_com == "comstock" and self.weather == "amy2018" and self.release_number == "2":
                return ""
                # This release does not have a single national metadata file.
                # Instead, it has a metadata file for each county.
                # We need a way to download them all and combine based on the state
            else:
                if self.upgrade_id == "0":
                    return f"{self.base_url}metadata/baseline.parquet"
                else:
                    return f"{self.base_url}metadata/upgrade{str(int(self.upgrade_id)).zfill(2)}.parquet"
        elif (
            self.release_year == "2025"
            and self.res_com == "comstock"
            and self.weather == "amy2018"
            and self.release_number == "1"
        ):
            return ""
            # This release does not have a single national metadata file.
            # Instead, it has a metadata file for each county.
            # We need a way to download them all and combine based on the state
        else:
            return ""

    def get_15min_timeseries_url(self) -> str:
        """Generate the S3 download URL for this building."""
        if self.release_year == "2021":
            if self.upgrade_id != "0":
                return ""  # This release only has baseline timeseries
            else:
                return (
                    f"{self.base_url}timeseries_individual_buildings/"
                    f"by_state/upgrade={self.upgrade_id}/"
                    f"state={self.state}/"
                    f"bldg{self.bldg_id!s}-up{int(self.upgrade_id)!s}.parquet"
                )

        elif self.release_year == "2022" or self.release_year == "2023":
            return (
                f"{self.base_url}timeseries_individual_buildings/"
                f"by_state/upgrade={self.upgrade_id}/"
                f"state={self.state}/"
                f"bldg{self.bldg_id!s}-up{int(self.upgrade_id)!s}.parquet"
            )
        elif self.release_year == "2024":
            if self.res_com == "resstock" and self.weather == "tmy3" and self.release_number == "1":
                return ""
            else:
                return (
                    f"{self.base_url}timeseries_individual_buildings/"
                    f"by_state/upgrade={self.upgrade_id}/"
                    f"state={self.state}/"
                    f"bldg{self.bldg_id!s}-up{int(self.upgrade_id)!s}.parquet"
                )
        elif self.release_year == "2025":
            return (
                f"{self.base_url}timeseries_individual_buildings/"
                f"by_state/upgrade={self.upgrade_id}/"
                f"state={self.state}/"
                f"bldg{self.bldg_id!s}-up{int(self.upgrade_id)!s}.parquet"
            )
        else:
            return ""

    def get_release_name(self) -> str:
        """Generate the release name for this building."""
        res_com_str = "res" if self.res_com == "resstock" else "com"
        return f"{res_com_str}_{self.release_year}_{self.weather}_{self.release_number}"

    def to_json(self) -> str:
        """Convert the building ID object to a JSON string."""
        return json.dumps(asdict(self))


def _validate_release_name(release_name: str) -> bool:
    """Validate the release name.

    Args:
        release_name: The release name to validate.

    Returns:
        True if the release name is valid, False otherwise.
    """
    # Read the valid release names from the JSON file
    releases_file = files("buildstock_fetch.utils").joinpath("buildstock_releases.json")
    with open(str(releases_file)) as f:
        releases_data = json.load(f)

    # Get the top-level keys as valid release names
    valid_release_names = list(releases_data.keys())
    return release_name in valid_release_names


def fetch_bldg_ids(
    product: str, release_year: str, weather_file: str, release_version: str, state: str, upgrade_id: str
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
    # Construct the absolute path to the parquet directory
    parquet_dir = Path(
        str(files("buildstock_fetch.utils").joinpath("building_data").joinpath("combined_metadata.parquet"))
    )

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
        parquet_dir
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
    response = requests.head(url, timeout=30)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    progress.update(task_id, total=total_size)

    # Download with streaming
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    downloaded_size = 0

    with open(output_file, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                downloaded_size += len(chunk)
                if total_size > 0:
                    progress.update(task_id, completed=downloaded_size)

    return downloaded_size


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
        os.makedirs(output_dir, exist_ok=True)

    # Create a unique temporary directory for this building to avoid race conditions
    temp_dir = output_dir / f"temp_{str(bldg_id.bldg_id).zfill(7)}_{bldg_id.upgrade_id}"
    temp_dir.mkdir(exist_ok=True)

    downloaded_paths: dict[str, Optional[Path]] = {
        "hpxml": None,
        "schedule": None,
    }
    if file_type.hpxml or file_type.schedule:
        download_url = bldg_id.get_building_data_url()
        if download_url == "":
            message = f"Building data is not available for {bldg_id.get_release_name()}"
            raise NoBuildingDataError(message)

        output_file = temp_dir / f"{str(bldg_id.bldg_id).zfill(7)}_upgrade{bldg_id.upgrade_id}.zip"

        # Download with progress tracking if progress object is provided
        if progress and task_id is not None:
            _download_with_progress(download_url, output_file, progress, task_id)
        else:
            response = requests.get(download_url, timeout=30)
            response.raise_for_status()
            with open(output_file, "wb") as file:
                file.write(response.content)

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
                    new_path = output_dir / bldg_id.get_release_name() / "hpxml" / bldg_id.state / new_name
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    old_path.rename(new_path)
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
                    new_path = output_dir / bldg_id.get_release_name() / "schedule" / bldg_id.state / new_name
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    old_path.rename(new_path)
                    downloaded_paths["schedule"] = new_path

        # Remove the zip file and temp directory after extraction
        output_file.unlink()
        temp_dir.rmdir()  # Remove empty temp directory

    return downloaded_paths


def _parse_requested_file_type(file_type: tuple[str, ...]) -> RequestedFileTypes:
    """Parse the file type string into a RequestedFileTypes object."""
    file_type_obj = RequestedFileTypes()
    if "hpxml" in file_type:
        file_type_obj.hpxml = True
    if "schedule" in file_type:
        file_type_obj.schedule = True
    if "metadata" in file_type:
        file_type_obj.metadata = True
    if "load_curve_15min" in file_type:
        file_type_obj.load_curve_15min = True
    if "load_curve_hourly" in file_type:
        file_type_obj.load_curve_hourly = True
    if "load_curve_daily" in file_type:
        file_type_obj.load_curve_daily = True
    if "load_curve_monthly" in file_type:
        file_type_obj.load_curve_monthly = True
    if "load_curve_annual" in file_type:
        file_type_obj.load_curve_annual = True
    return file_type_obj


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


def _download_metadata_with_progress(bldg: BuildingID, output_dir: Path, progress: Progress) -> Path:
    """Download metadata file with progress tracking."""
    download_url = bldg.get_metadata_url()
    if download_url == "":
        message = f"Metadata is not available for {bldg.get_release_name()}"
        raise NoMetadataError(message)

    # Create metadata task with progress tracking
    metadata_task = progress.add_task(
        "[yellow]Downloading metadata",
        total=0,  # Will be updated when we get the file size
    )

    # Get file size first
    response = requests.head(download_url, timeout=30)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    progress.update(metadata_task, total=total_size)

    # Download with progress
    output_file = output_dir / bldg.get_release_name() / "metadata" / bldg.state / "metadata.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    _download_with_progress(download_url, output_file, progress, metadata_task)

    return output_file


def fetch_bldg_data(
    bldg_ids: list[BuildingID], file_type: tuple[str, ...], output_dir: Path, max_workers: int = 5
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

    downloaded_paths: list[Path] = []
    failed_downloads: list[str] = []

    # Calculate total files to download
    total_files = len(bldg_ids)
    if file_type_obj.metadata:
        total_files += 1  # Add metadata file

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
        # Create individual download tasks for each building
        download_tasks = {}
        for i, bldg_id in enumerate(bldg_ids):
            task_id = progress.add_task(
                f"[cyan]Building {bldg_id.bldg_id} (upgrade {bldg_id.upgrade_id})",
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

        # Get metadata if requested. Only one building is needed to get the metadata.
        if file_type_obj.metadata:
            bldg = bldg_ids[0]
            metadata_file = _download_metadata_with_progress(bldg, output_dir, progress)
            downloaded_paths.append(metadata_file)

    # Print summary
    console.print("\n[bold green]Download complete![/bold green]")
    console.print(f"[green]Successfully downloaded: {len(downloaded_paths)} files[/green]")
    if failed_downloads:
        console.print(f"[red]Failed downloads: {len(failed_downloads)} files[/red]")
        for failed in failed_downloads:
            console.print(f"  [red]• {failed}[/red]")

    return downloaded_paths, failed_downloads


if __name__ == "__main__":  # pragma: no cover
    bldg_ids = fetch_bldg_ids(
        product="resstock", release_year="2024", weather_file="amy2018", release_version="2", upgrade_id="0", state="NY"
    )
    print(bldg_ids[:10])
