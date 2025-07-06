import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
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
    parquet_dir = Path("/workspaces/buildstock-fetch/utils/building_data/combined_metadata.parquet")

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
    df = pd.read_parquet(partition_path)

    # No need to filter since we're already reading the specific partition
    filtered_df = df

    # Convert the filtered data to BuildingID objects
    building_ids = []
    for _, row in filtered_df.iterrows():
        building_id = BuildingID(
            bldg_id=int(row["bldg_id"]),
            release_number=release_version,
            release_year=release_year,
            res_com=product,
            weather=weather_file,
            upgrade_id=upgrade_id,
        )
        building_ids.append(building_id)

    return building_ids


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
    tmp_ids = fetch_bldg_ids(
        product="resstock", weather_file="tmy3", release_version="1", release_year="2021", state="MA", upgrade_id="0"
    )
    print(tmp_ids[:5])
