import json
import os
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import click
import requests


@dataclass
class FileType:
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


def fetch_bldg_data_core(bldg_ids: list[BuildingID], file_type: FileType, output_dir: Path) -> list[Path]:
    """Download building data for a given list of building ids

    Downloads the data for the given building ids and returns list of paths to the downloaded files.

    Args:
        bldg_ids: A list of BuildingID objects to download data for.
        file_type: Tuple of file types to extract (e.g., "hpxml", "schedule")
        output_dir: Directory to save the downloaded files.

    Returns:
        A list of paths to the downloaded files.
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if not output_dir.exists():
        os.makedirs(output_dir, exist_ok=True)

    downloaded_paths = []

    for bldg_id in bldg_ids:
        if file_type.hpxml or file_type.schedule:
            base_url = bldg_id.get_building_data_url()
            response = requests.get(base_url, timeout=30)
            response.raise_for_status()

            output_file = output_dir / f"{bldg_id.bldg_id:07}_upgrade{bldg_id.upgrade_id}.zip"
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
                        zip_ref.extract(xml_file, output_dir)
                        # Rename to the specified convention
                        old_path = output_dir / xml_file
                        new_name = f"bldg{bldg_id.bldg_id:07}-up{bldg_id.upgrade_id:02}.xml"
                        new_path = output_dir / new_name
                        old_path.rename(new_path)
                        downloaded_paths.append(new_path)

                if file_type.schedule:
                    # Find and extract the schedule CSV file
                    schedule_files = [f for f in zip_file_list if "schedule" in f.lower() and f.endswith(".csv")]
                    if schedule_files:
                        schedule_file = schedule_files[0]  # Take the first (and only) schedule file
                        zip_ref.extract(schedule_file, output_dir)
                        # Rename to the specified convention
                        old_path = output_dir / schedule_file
                        new_name = f"bldg{bldg_id.bldg_id:07}-up{bldg_id.upgrade_id:02}_schedule.csv"
                        new_path = output_dir / new_name
                        old_path.rename(new_path)
                        downloaded_paths.append(new_path)

            # Remove the zip file after extraction
            output_file.unlink()

    # Get metadata if requested
    bldg = bldg_ids[0]
    if file_type.metadata:
        base_url = bldg.get_metadata_url()
        response = requests.get(base_url, timeout=30)
        response.raise_for_status()

        output_file = output_dir / f"{bldg.get_release_name()}_metadata.parquet"
        with open(output_file, "wb") as file:
            file.write(response.content)
        downloaded_paths.append(output_file)

    return downloaded_paths


@click.command()
@click.argument("bldg_ids", nargs=-1, required=True)
@click.argument("file_type", nargs=-1, required=True)
@click.argument("output_dir", default=Path(__file__).parent / "data")
def fetch_bldg_data(bldg_ids: list[BuildingID], file_type: tuple[str], output_dir: Path) -> list[Path]:
    """Download building data for a given list of building ids

    Downloads the data for the given building ids and returns list of paths to the downloaded files.

    Args:
        bldg_ids: A list of BuildingID objects to download data for.

    Returns:
        A list of paths to the downloaded files.
    """
    file_type_obj = FileType()
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

    return fetch_bldg_data_core(bldg_ids, file_type_obj, output_dir)


if __name__ == "__main__":  # pragma: no cover
    tmp_ids = fetch_bldg_ids("MA")
    tmp_data = fetch_bldg_data(tmp_ids)
    print(f"Downloaded files: {[str(path) for path in tmp_data]}")
