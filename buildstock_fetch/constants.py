from importlib.resources import files
from pathlib import Path

__all__ = [
    "LOAD_CURVE_COLUMN_AGGREGATION",
    "METADATA_DIR",
    "RELEASE_JSON_FILE",
    "WEATHER_FILE_DIR",
]


METADATA_DIR = Path(
    str(files("buildstock_fetch").joinpath("data").joinpath("building_data").joinpath("combined_metadata.parquet"))
)
RELEASE_JSON_FILE = Path(str(files("buildstock_fetch").joinpath("data").joinpath("buildstock_releases.json")))
LOAD_CURVE_COLUMN_AGGREGATION = Path(str(files("buildstock_fetch").joinpath("data").joinpath("load_curve_column_map")))
WEATHER_FILE_DIR = Path(str(files("buildstock_fetch").joinpath("data").joinpath("weather_station_map")))
BUILDSTOCK_RELEASES_FILE = str(files("buildstock_fetch").joinpath("data").joinpath("buildstock_releases.json"))
UPGRADES_LOOKUP_FILE = str(files("buildstock_fetch").joinpath("data").joinpath("buildstock_upgrades_lookup.json"))
SB_ANALYSIS_UPGRADES_FILE = str(files("buildstock_fetch").joinpath("data").joinpath("SB_analysis_upgrades.json"))
