import concurrent.futures
import json
import os
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import polars as pl
import requests


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


class No15minLoadCurveError(ValueError):
    """Raised when no 15 min load profile timeseries is available for a given release."""

    pass


@dataclass
class RequestedFileTypes:
    hpxml: bool = False
    schedule: bool = False
    metadata: bool = False
    load_curve_15min: bool = False
    load_curve_hourly: bool = False
    load_curve_daily: bool = False
    load_curve_weekly: bool = False
    load_curve_monthly: bool = False


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

    def get_15min_load_curve_url(self) -> str:
        """Generate the S3 download URL for this building."""
        if self.release_year == "2021":
            if self.upgrade_id != "0":
                return ""  # This release only has baseline timeseries
            else:
                return (
                    f"{self.base_url}timeseries_individual_buildings/"
                    f"by_state/upgrade={self.upgrade_id}/"
                    f"state={self.state}/"
                    f"{self.bldg_id!s}-{int(self.upgrade_id)!s}.parquet"
                )

        elif self.release_year == "2022" or self.release_year == "2023":
            return (
                f"{self.base_url}timeseries_individual_buildings/"
                f"by_state/upgrade={self.upgrade_id}/"
                f"state={self.state}/"
                f"{self.bldg_id!s}-{int(self.upgrade_id)!s}.parquet"
            )
        elif self.release_year == "2024":
            if self.res_com == "resstock" and self.weather == "tmy3" and self.release_number == "1":
                return ""
            else:
                return (
                    f"{self.base_url}timeseries_individual_buildings/"
                    f"by_state/upgrade={self.upgrade_id}/"
                    f"state={self.state}/"
                    f"{self.bldg_id!s}-{int(self.upgrade_id)!s}.parquet"
                )
        elif self.release_year == "2025":
            return (
                f"{self.base_url}timeseries_individual_buildings/"
                f"by_state/upgrade={self.upgrade_id}/"
                f"state={self.state}/"
                f"{self.bldg_id!s}-{int(self.upgrade_id)!s}.parquet"
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
    releases_file = Path(__file__).parent.parent / "utils" / "buildstock_releases.json"
    with open(releases_file) as f:
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
    parquet_dir = Path(__file__).parent.parent / "utils" / "building_data" / "combined_metadata.parquet"

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
    df = pl.read_parquet(partition_path)

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


def download_bldg_data(
    bldg_id: BuildingID, file_type: RequestedFileTypes, output_dir: Path
) -> dict[str, Union[Path, None]]:
    """Download and extract building data for a single building. Only HPXML and schedule files are supported.

    Args:
        bldg_id: A BuildingID object to download data for.
        file_type: RequestedFileTypes object to specify which files to download.
        output_dir: Directory to save the downloaded files.

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
        response = requests.get(download_url, timeout=30)
        response.raise_for_status()

        output_file = temp_dir / f"{str(bldg_id.bldg_id).zfill(7)}_upgrade{bldg_id.upgrade_id}.zip"
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


def download_metadata(bldg_id: BuildingID, output_dir: Path) -> Path:
    """Download the metadata for a given building.

    Args:
        bldg_id: A BuildingID object to download metadata for.
        output_dir: Directory to save the downloaded metadata.
    """

    download_url = bldg_id.get_metadata_url()
    if download_url == "":
        message = f"Metadata is not available for {bldg_id.get_release_name()}"
        raise NoMetadataError(message)
    response = requests.get(download_url, timeout=30)
    response.raise_for_status()
    output_file = output_dir / bldg_id.get_release_name() / "metadata" / bldg_id.state / "metadata.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as file:
        file.write(response.content)
    return output_file


def download_15min_load_curve(bldg_id: BuildingID, output_dir: Path) -> Path:
    """Download the 15 min load profile timeseries for a given building.

    Args:
        bldg_id: A BuildingID object to download 15 min load profile timeseries for.
        output_dir: Directory to save the downloaded 15 min load profile timeseries.
    """

    download_url = bldg_id.get_15min_load_curve_url()
    if download_url == "":
        message = f"15 min load profile timeseries is not available for {bldg_id.get_release_name()}"
        raise No15minLoadCurveError(message)
    response = requests.get(download_url, timeout=30)
    response.raise_for_status()
    output_file = (
        output_dir / bldg_id.get_release_name() / "load_curve_15min" / bldg_id.state / "load_curve_15min.parquet"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as file:
        file.write(response.content)
    return output_file


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
    if "load_curve_weekly" in file_type:
        file_type_obj.load_curve_weekly = True
    if "load_curve_monthly" in file_type:
        file_type_obj.load_curve_monthly = True
    return file_type_obj


def _download_building_data_parallel(
    bldg_ids: list[BuildingID], file_type_obj: RequestedFileTypes, output_dir: Path, max_workers: int
) -> tuple[list[Path], list[str]]:
    """Download building data (HPXML and schedule files) in parallel."""
    downloaded_paths = []
    failed_downloads = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_bldg = {
            executor.submit(download_bldg_data, bldg_id, file_type_obj, output_dir): bldg_id for bldg_id in bldg_ids
        }

        for future in concurrent.futures.as_completed(future_to_bldg):
            bldg_id = future_to_bldg[future]
            try:
                paths_dict = future.result()
                paths = [path for path in paths_dict.values() if path is not None]
                downloaded_paths.extend(paths)

                if file_type_obj.hpxml and paths_dict["hpxml"] is None:
                    failed_downloads.append(f"bldg{str(bldg_id.bldg_id).zfill(7)}-up{bldg_id.upgrade_id.zfill(2)}.xml")
                if file_type_obj.schedule and paths_dict["schedule"] is None:
                    failed_downloads.append(
                        f"bldg{str(bldg_id.bldg_id).zfill(7)}-up{bldg_id.upgrade_id.zfill(2)}_schedule.csv"
                    )
            except NoBuildingDataError:
                failed_downloads.append(f"bldg{str(bldg_id.bldg_id).zfill(7)}-up{bldg_id.upgrade_id.zfill(2)}.xml")
                failed_downloads.append(
                    f"bldg{str(bldg_id.bldg_id).zfill(7)}-up{bldg_id.upgrade_id.zfill(2)}_schedule.csv"
                )
                raise
            except Exception as e:
                print(f"Download failed for bldg_id {bldg_id}: {e}")

    return downloaded_paths, failed_downloads


def _download_metadata_single(bldg_ids: list[BuildingID], output_dir: Path) -> tuple[list[Path], list[str]]:
    """Download metadata for a single building (only one needed)."""
    downloaded_paths = []
    failed_downloads = []

    bldg = bldg_ids[0]
    try:
        output_file = download_metadata(bldg, output_dir)
        downloaded_paths.append(output_file)
    except NoMetadataError:
        output_file = output_dir / bldg.get_release_name() / "metadata" / bldg.state / "metadata.parquet"
        failed_downloads.append(str(output_file))
        raise
    except Exception as e:
        print(f"Download failed for metadata for {bldg.get_release_name()}: {e}")

    return downloaded_paths, failed_downloads


def _download_15min_load_curves_parallel(
    bldg_ids: list[BuildingID], output_dir: Path, max_workers: int
) -> tuple[list[Path], list[str]]:
    """Download 15-minute load curves in parallel."""
    downloaded_paths = []
    failed_downloads = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_bldg = {
            executor.submit(download_15min_load_curve, bldg_id, output_dir): bldg_id for bldg_id in bldg_ids
        }

        for future in concurrent.futures.as_completed(future_to_bldg):
            bldg_id = future_to_bldg[future]
            try:
                output_file = future.result()
                downloaded_paths.append(output_file)
            except No15minLoadCurveError:
                output_file = (
                    output_dir
                    / bldg_id.get_release_name()
                    / "load_curve_15min"
                    / bldg_id.state
                    / "load_curve_15min.parquet"
                )
                failed_downloads.append(str(output_file))
                raise
            except Exception as e:
                output_file = (
                    output_dir
                    / bldg_id.get_release_name()
                    / "load_curve_15min"
                    / bldg_id.state
                    / "load_curve_15min.parquet"
                )
                failed_downloads.append(str(output_file))
                print(f"Download failed for 15 min load profile timeseries for {bldg_id.get_release_name()}: {e}")
                raise

    return downloaded_paths, failed_downloads


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

    downloaded_paths = []
    failed_downloads = []

    # Download building data if requested.
    if file_type_obj.hpxml or file_type_obj.schedule:
        paths, failures = _download_building_data_parallel(bldg_ids, file_type_obj, output_dir, max_workers)
        downloaded_paths.extend(paths)
        failed_downloads.extend(failures)

    # Get metadata if requested.
    if file_type_obj.metadata:
        paths, failures = _download_metadata_single(bldg_ids, output_dir)
        downloaded_paths.extend(paths)
        failed_downloads.extend(failures)

    # Get 15 min load profile timeseries if requested.
    if file_type_obj.load_curve_15min:
        paths, failures = _download_15min_load_curves_parallel(bldg_ids, output_dir, max_workers)
        downloaded_paths.extend(paths)
        failed_downloads.extend(failures)

    return downloaded_paths, failed_downloads


if __name__ == "__main__":  # pragma: no cover
    tmp_ids = [
        BuildingID(
            bldg_id=10, release_year="2022", res_com="resstock", weather="amy2018", release_number="1", upgrade_id="1"
        )
    ]
    tmp_data, tmp_failed = fetch_bldg_data(tmp_ids, ("load_curve_15min"), Path(__file__).parent / "data")
    print(f"Downloaded files: {[str(path) for path in tmp_data]}")
    print(f"Failed downloads: {tmp_failed}")
