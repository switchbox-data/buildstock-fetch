import json
import os
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import click
import requests


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


def fetch_bldg_data_core(bldg_ids: list[BuildingID], file_type: list[str], output_dir: Path) -> list[Path]:
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
        if "hpxml" in file_type or "schedule" in file_type:
            base_url = bldg_id.get_building_data_url()
            response = requests.get(base_url, timeout=30)
            response.raise_for_status()

            output_file = output_dir / f"{bldg_id.bldg_id:07}_upgrade{bldg_id.upgrade_id}.zip"
            with open(output_file, "wb") as file:
                file.write(response.content)

            # Extract specific files based on file_type
            with zipfile.ZipFile(output_file, "r") as zip_ref:
                zip_file_list = zip_ref.namelist()

                if "hpxml" in file_type:
                    # Find and extract the XML file
                    xml_files = [f for f in zip_file_list if f.endswith(".xml")]
                    if xml_files:
                        xml_file = xml_files[0]  # Take the first (and only) XML file
                        zip_ref.extract(xml_file, output_dir)
                        extracted_path = output_dir / xml_file
                        downloaded_paths.append(extracted_path)

                if "schedule" in file_type:
                    # Find and extract the schedule CSV file
                    schedule_files = [f for f in zip_file_list if "schedule" in f.lower() and f.endswith(".csv")]
                    if schedule_files:
                        schedule_file = schedule_files[0]  # Take the first (and only) schedule file
                        zip_ref.extract(schedule_file, output_dir)
                        extracted_path = output_dir / schedule_file
                        downloaded_paths.append(extracted_path)

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
    return fetch_bldg_data_core(bldg_ids, list(file_type), output_dir)


if __name__ == "__main__":  # pragma: no cover
    tmp_ids = fetch_bldg_ids("MA")
    tmp_data = fetch_bldg_data(tmp_ids)
    print(f"Downloaded files: {[str(path) for path in tmp_data]}")
