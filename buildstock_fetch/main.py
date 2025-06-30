import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import requests


@dataclass
class BuildingID:
    bldg_id: int
    release_number: str = "1"
    release_year: str = "2022"
    res_com: str = "resstock"
    weather: str = "tmy3"
    upgrade_id: str = "0"

    def get_download_url(self) -> str:
        """Generate the S3 download URL for this building."""
        if self.release_year == "2021":
            return (
                "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/"
                f"end-use-load-profiles-for-us-building-stock/{self.release_year}/"
                f"{self.res_com}_{self.weather}_release_{self.release_number}/"
                f"building_energy_models/"
                f"bldg{self.bldg_id:07}-up0{self.upgrade_id}.osm.gz"
            )
        else:
            return (
                "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/"
                f"end-use-load-profiles-for-us-building-stock/{self.release_year}/"
                f"{self.res_com}_{self.weather}_release_{self.release_number}/"
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


def fetch_bldg_data(bldg_ids: list[BuildingID]) -> list[Path]:
    """Download building data for a given list of building ids

    Downloads the data for the given building ids and returns list of paths to the downloaded files.

    Args:
        bldg_ids: A list of BuildingID objects to download data for.

    Returns:
        A list of paths to the downloaded files.
    """
    data_dir = Path(__file__).parent.parent / "data"
    downloaded_paths = []
    os.makedirs(data_dir, exist_ok=True)

    for bldg_id in bldg_ids:
        response = requests.get(bldg_id.get_download_url(), timeout=30)
        response.raise_for_status()

        output_path = data_dir / f"{bldg_id.bldg_id:07}_upgrade{bldg_id.upgrade_id}.zip"
        with open(output_path, "wb") as file:
            file.write(response.content)

        downloaded_paths.append(output_path)
    return downloaded_paths


if __name__ == "__main__":  # pragma: no cover
    tmp_ids = fetch_bldg_ids("MA")
    tmp_data = fetch_bldg_data(tmp_ids)
    print(f"Downloaded files: {[str(path) for path in tmp_data]}")
