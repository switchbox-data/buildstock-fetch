import random
from importlib.resources import files
from pathlib import Path

import polars as pl

from buildstock_fetch.main import fetch_bldg_ids
from utils.resolve_weather_station_id import download_and_extract_weather_station

WEATHER_STATION_MAP_FILE = Path(
    str(
        files("buildstock_fetch")
        .joinpath("data")
        .joinpath("weather_station_map")
        .joinpath("weather_station_map.parquet")
    )
)


# Test the dataframe that was already built
def test_resolve_weather_station_id():
    # Randomly select 10 buildings to test
    product = "resstock"
    release_year = "2022"
    weather_file = "amy2012"
    release_version = "1"
    state = "NY"
    upgrade_id = "0"

    bldg_ids = fetch_bldg_ids(product, release_year, weather_file, release_version, state, upgrade_id)
    bldg_ids = random.sample(bldg_ids, 10)

    weather_station_map = pl.read_parquet(WEATHER_STATION_MAP_FILE)

    for bldg_id in bldg_ids:
        weather_station_id = download_and_extract_weather_station(bldg_id)
        assert weather_station_id is not None
        assert (
            weather_station_id
            == weather_station_map.filter(pl.col("bldg_id") == bldg_id.bldg_id)[0].select("weather_station_name").item()
        )
