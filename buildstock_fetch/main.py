import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union

import pandas as pd
import requests


class InvalidProductError(ValueError):
    """Raised when an invalid product is provided."""

    pass


class InvalidReleaseNameError(ValueError):
    """Raised when an invalid release name is provided."""

    pass


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
        if (
            self.res_com == "resstock"
            and self.release_year == "2021"
            and self.weather == "tmy3"
            and self.release_number == "1"
        ):
            return (
                "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/"
                f"end-use-load-profiles-for-us-building-stock/{self.release_year}/"
                f"{self.res_com}_{self.weather}_release_{self.release_number}/"
                f"building_energy_models/"
                f"bldg{self.bldg_id:07}-up0{self.upgrade_id}.osm.gz"
            )
        elif (
            self.res_com == "comstock"
            and self.release_year == "2023"
            and self.weather == "amy2018"
            and self.release_number == "1"
        ):
            upgrade_id = int(self.upgrade_id)
            return (
                "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/"
                f"end-use-load-profiles-for-us-building-stock/{self.release_year}/"
                f"{self.res_com}_{self.weather}_release_{self.release_number}/"
                f"building_energy_model_files/upgrade={upgrade_id:02}/"
                f"bldg{self.bldg_id}-up{upgrade_id:02}.osm.gz"
            )
        elif (
            self.res_com == "comstock"
            and self.release_year == "2024"
            and self.weather == "amy2018"
            and self.release_number == "1"
        ):
            upgrade_id = int(self.upgrade_id)
            return (
                "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/"
                f"end-use-load-profiles-for-us-building-stock/{self.release_year}/"
                f"{self.res_com}_{self.weather}_release_{self.release_number}/"
                f"building_energy_models/upgrade={upgrade_id:02}/"
                f"bldg{self.bldg_id:07}-up{upgrade_id:02}.osm.gz"
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
    parquet_dir = Path("/workspaces/buildstock-fetch/utils/building_data/combined_metadata.parquet")

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


def fetch_bldg_data(bldg_ids: list[BuildingID], output_directory: Union[Path, None] = None) -> list[Path]:
    """Download building data for a given list of building ids

    Downloads the data for the given building ids and returns list of paths to the downloaded files.

    Args:
        bldg_ids: A list of BuildingID objects to download data for.

    Returns:
        A list of paths to the downloaded files.
    """
    output_directory = Path(__file__).parent.parent / "data" if output_directory is None else Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    downloaded_paths = []

    for bldg_id in bldg_ids:
        response = requests.get(bldg_id.get_download_url(), timeout=30)
        response.raise_for_status()

        output_path = output_directory / f"{bldg_id.bldg_id:07}_upgrade{bldg_id.upgrade_id}.zip"
        with open(output_path, "wb") as file:
            file.write(response.content)

        downloaded_paths.append(output_path)
    return downloaded_paths


if __name__ == "__main__":  # pragma: no cover
    tmp_ids = fetch_bldg_ids(
        product="resstock", weather_file="tmy3", release_version="1", release_year="2021", state="MA", upgrade_id="0"
    )
