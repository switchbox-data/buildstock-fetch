from pathlib import Path
from typing import TypedDict

from buildstock_fetch.types import ReleaseYear, ResCom, Weather


class Inputs(TypedDict):
    product: ResCom
    release_year: ReleaseYear
    weather_file: Weather
    release_version: str
    states: list[str]
    file_type: list[str]
    upgrade_ids: list[str]
    output_directory: Path
