import concurrent.futures
import json
import os
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import requests


@dataclass
class RequestedFileTypes:
    hpxml: bool = False
    schedule: bool = False
    metadata: bool = False
    time_series_15min: bool = False
    time_series_hourly: bool = False
    time_series_daily: bool = False
    time_series_weekly: bool = False
    time_series_monthly: bool = False


@dataclass
class BuildingID:
    bldg_id: int
    release_number: str = "1"
    release_year: str = "2022"
    res_com: str = "resstock"
    weather: str = "tmy3"
    upgrade_id: str = "0"
    base_url: str = (
        f"https://oedi-data-lake.s3.amazonaws.com/"
        "nrel-pds-building-stock/"
        "end-use-load-profiles-for-us-building-stock/"
        f"{release_year}/"
        f"{res_com}_{weather}_release_{release_number}/"
    )

    def get_building_data_url(self) -> str:
        """Generate the S3 download URL for this building."""
        return (
            f"{self.base_url}"
            f"building_energy_models/upgrade={self.upgrade_id}/"
            f"bldg{self.bldg_id:07}-up0{self.upgrade_id}.zip"
        )

    def get_metadata_url(self) -> str:
        """Generate the S3 download URL for this building."""
        if self.res_com == "resstock" and self.weather == "tmy3" and self.release_year == "2022":
            if self.upgrade_id == "0":
                return f"{self.base_url}metadata/baseline.parquet"
            else:
                return f"{self.base_url}metadata/upgrade{int(self.upgrade_id):02}.parquet"
        else:
            return f"{self.base_url}metadata/metadata.parquet"

    def get_release_name(self) -> str:
        """Generate the release name for this building."""
        res_com_str = "res" if self.res_com == "resstock" else "com"
        return f"{res_com_str}stock_{self.weather}_release_{self.release_number}"

    def to_json(self) -> str:
        """Convert the building ID object to a JSON string."""
        return json.dumps(asdict(self))


def fetch_bldg_ids(state: str) -> list[BuildingID]:
    """Fetch a list of Building ID's

    Provided a state, returns a list of building ID's for that state.

    Args:
        state: The state to fetch building ID's for.

    Returns:
        A list of building ID's for the given state.
    """
    if state == "MA":
        return [
            BuildingID(bldg_id=7),
            BuildingID(bldg_id=8),
            BuildingID(bldg_id=9),
        ]

    else:
        raise NotImplementedError(f"State {state} not supported")


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
    temp_dir = output_dir / f"temp_{bldg_id.bldg_id:07}_{bldg_id.upgrade_id}"
    temp_dir.mkdir(exist_ok=True)

    downloaded_paths: dict[str, Optional[Path]] = {
        "hpxml": None,
        "schedule": None,
    }
    if file_type.hpxml or file_type.schedule:
        base_url = bldg_id.get_building_data_url()
        response = requests.get(base_url, timeout=30)
        response.raise_for_status()

        output_file = temp_dir / f"{bldg_id.bldg_id:07}_upgrade{bldg_id.upgrade_id}.zip"
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
                    new_name = f"bldg{bldg_id.bldg_id:07}-up{bldg_id.upgrade_id:02}.xml"
                    new_path = output_dir / new_name
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
                    new_name = f"bldg{bldg_id.bldg_id:07}-up{bldg_id.upgrade_id:02}_schedule.csv"
                    new_path = output_dir / new_name
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
    if "time_series_15min" in file_type:
        file_type_obj.time_series_15min = True
    if "time_series_hourly" in file_type:
        file_type_obj.time_series_hourly = True
    if "time_series_daily" in file_type:
        file_type_obj.time_series_daily = True
    if "time_series_weekly" in file_type:
        file_type_obj.time_series_weekly = True
    if "time_series_monthly" in file_type:
        file_type_obj.time_series_monthly = True
    return file_type_obj


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
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and keep track of future -> bldg_id mapping
        future_to_bldg = {
            executor.submit(download_bldg_data, bldg_id, file_type_obj, output_dir): bldg_id for bldg_id in bldg_ids
        }

        # Process completed futures
        for future in concurrent.futures.as_completed(future_to_bldg):
            bldg_id = future_to_bldg[future]  # Get the correct bldg_id for this future
            try:
                paths_dict = future.result()
                # Convert dict values to list, filtering out None values
                paths = [path for path in paths_dict.values() if path is not None]
                downloaded_paths.extend(paths)

                if paths_dict["hpxml"] is None:
                    failed_downloads.append(f"bldg{bldg_id.bldg_id:07}-up{bldg_id.upgrade_id:02}.xml")
                if paths_dict["schedule"] is None:
                    failed_downloads.append(f"bldg{bldg_id.bldg_id:07}-up{bldg_id.upgrade_id:02}_schedule.csv")
            except Exception as e:
                print(f"Download failed for bldg_id {bldg_id}: {e}")

    # Get metadata if requested. Only one building is needed to get the metadata.
    if file_type_obj.metadata:
        bldg = bldg_ids[0]
        base_url = bldg.get_metadata_url()
        response = requests.get(base_url, timeout=30)
        response.raise_for_status()

        output_file = output_dir / f"{bldg.get_release_name()}_metadata.parquet"
        with open(output_file, "wb") as file:
            file.write(response.content)
        downloaded_paths.append(output_file)

    return downloaded_paths, failed_downloads


if __name__ == "__main__":  # pragma: no cover
    tmp_ids = [BuildingID(bldg_id=7), BuildingID(bldg_id=8), BuildingID(bldg_id=9)]
    tmp_data, tmp_failed = fetch_bldg_data(tmp_ids, ("hpxml", "schedule", "metadata"), Path(__file__).parent / "data")
    print(f"Downloaded files: {[str(path) for path in tmp_data]}")
    print(f"Failed downloads: {tmp_failed}")
