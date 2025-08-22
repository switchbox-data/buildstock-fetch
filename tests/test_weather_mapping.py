import random
import shutil
from importlib.resources import files
from pathlib import Path

import polars as pl
import pytest

from buildstock_fetch.main import fetch_bldg_ids
from utils.resolve_weather_station_id import download_and_extract_weather_station, resolve_weather_station_id

WEATHER_STATION_MAP_FILE = Path(
    str(
        files("buildstock_fetch")
        .joinpath("data")
        .joinpath("weather_station_map")
        .joinpath("weather_station_map.parquet")
    )
)


@pytest.fixture(scope="module")
def weather_station_map():
    """Load the weather station map DataFrame once for all tests."""
    return pl.read_parquet(WEATHER_STATION_MAP_FILE)


@pytest.fixture(scope="function")
def cleanup_downloads():
    # Setup - clean up any existing files before test
    data_dir = Path("data")
    test_output_dir = Path("test_output")

    if data_dir.exists():
        shutil.rmtree(data_dir)
    if test_output_dir.exists():
        shutil.rmtree(test_output_dir)

    yield

    # Teardown - clean up downloaded files after test
    if data_dir.exists():
        shutil.rmtree(data_dir)
    if test_output_dir.exists():
        shutil.rmtree(test_output_dir)


def test_weather_station_map_quality(weather_station_map):
    assert weather_station_map.select("bldg_id").n_unique() == weather_station_map.height


# Test the dataframe that was already built
def test_resolve_weather_station_id(weather_station_map):
    # Randomly select 10 buildings to test
    product = "resstock"
    release_year = "2022"
    weather_file = "amy2012"
    release_version = "1"
    state = "NY"
    upgrade_id = "0"

    bldg_ids = fetch_bldg_ids(product, release_year, weather_file, release_version, state, upgrade_id)
    bldg_ids = random.sample(bldg_ids, 10)

    for bldg_id in bldg_ids:
        weather_station_id = download_and_extract_weather_station(bldg_id)
        assert weather_station_id is not None
        assert (
            weather_station_id
            == weather_station_map.filter(pl.col("bldg_id") == bldg_id.bldg_id)[0].select("weather_station_name").item()
        )


def test_weather_station_mapping():
    product = "resstock"
    release_year = "2022"
    weather_file = "amy2012"
    release_version = "1"
    state = "NY"
    upgrade_id = "0"

    bldg_ids = fetch_bldg_ids(product, release_year, weather_file, release_version, state, upgrade_id)
    bldg_ids = random.sample(bldg_ids, 10)

    weather_station_map = resolve_weather_station_id(
        bldg_ids,
        product,
        release_year,
        weather_file,
        release_version,
        state,
        upgrade_id,
        status_update_interval_seconds=15,  # Update every 15 seconds
        status_update_interval_bldgs=25,  # Also update every 25 buildings
        max_workers=20,  # Use 20 parallel threads for downloading
    )

    for bldg_id in bldg_ids:
        weather_station_id = download_and_extract_weather_station(bldg_id)
        assert weather_station_id is not None
        assert (
            weather_station_id
            == weather_station_map.filter(pl.col("bldg_id") == bldg_id.bldg_id)[0].select("weather_station_name").item()
        )
    pass
