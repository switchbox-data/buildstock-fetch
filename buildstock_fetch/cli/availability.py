import json
from importlib.resources import files
from pathlib import Path
from typing import Union, cast

from rich.panel import Panel

from .questionary import console

BUILDSTOCK_RELEASES_FILE = str(files("buildstock_fetch").joinpath("data").joinpath("buildstock_releases.json"))
# Upgrade scenario description lookup file
UPGRADES_LOOKUP_FILE = str(files("buildstock_fetch").joinpath("data").joinpath("buildstock_upgrades_lookup.json"))


def get_state_options() -> list[str]:
    return [
        "AL",
        "AZ",
        "AR",
        "CA",
        "CO",
        "CT",
        "DE",
        "FL",
        "GA",
        "HI",
        "ID",
        "IL",
        "IN",
        "IA",
        "KS",
        "KY",
        "LA",
        "ME",
        "MD",
        "MA",
        "MI",
        "MN",
        "MS",
        "MO",
        "MT",
        "NE",
        "NV",
        "NH",
        "NJ",
        "NM",
        "NY",
        "NC",
        "ND",
        "OH",
        "OK",
        "OR",
        "PA",
        "RI",
        "SC",
        "SD",
        "TN",
        "TX",
        "UT",
        "VT",
        "VA",
        "WA",
        "WV",
        "WI",
        "WY",
    ]


def get_all_available_releases() -> dict[str, dict]:
    buildstock_releases = json.loads(Path(BUILDSTOCK_RELEASES_FILE).read_text(encoding="utf-8"))
    return cast(dict[str, dict], buildstock_releases)


def get_available_releases_names() -> list[str]:
    # Read the buildstock releases JSON file
    buildstock_releases = json.loads(Path(BUILDSTOCK_RELEASES_FILE).read_text(encoding="utf-8"))

    # Return the top-level keys as release options
    return list(buildstock_releases.keys())


def get_upgrade_ids_options(release_name: str) -> list[str]:
    available_releases = get_all_available_releases()
    available_upgrade_ids = available_releases[release_name]["upgrade_ids"]
    available_upgrade_ids = [int(upgrade_id) for upgrade_id in available_upgrade_ids]
    available_upgrade_ids.sort()
    available_upgrade_ids = [str(upgrade_id) for upgrade_id in available_upgrade_ids]

    if release_name in json.loads(Path(UPGRADES_LOOKUP_FILE).read_text(encoding="utf-8")):
        upgrade_descriptions = json.loads(Path(UPGRADES_LOOKUP_FILE).read_text(encoding="utf-8"))[release_name][
            "upgrade_descriptions"
        ]
        available_upgrade_ids = [
            f"{upgrade_id}: {upgrade_descriptions[upgrade_id]}"
            for upgrade_id in available_upgrade_ids
            if upgrade_id in upgrade_descriptions
        ]

    return cast(list[str], available_upgrade_ids)


def get_product_type_options() -> list[str]:
    return ["resstock", "comstock"]


def get_release_years_options(available_releases: list[str], product_type: str) -> tuple[list[str], list[str]]:
    # Find available release years
    parsed_releases = parse_buildstock_releases(available_releases)
    available_releases = filter_available_releases(list(parsed_releases.keys()), product_type=product_type)
    available_release_years = list({parsed_releases[release]["release_year"] for release in available_releases})
    available_release_years.sort(reverse=True)  # Sort in descending order (latest first)

    return available_releases, available_release_years


def get_weather_options(
    available_releases: list[str], product_type: str, release_year: str
) -> tuple[list[str], list[str]]:
    parsed_releases = parse_buildstock_releases(available_releases)
    available_releases = filter_available_releases(
        list(parsed_releases.keys()), product_type=product_type, release_year=release_year
    )
    available_weather_files = list({parsed_releases[release]["weather_file"] for release in available_releases})

    # Sort weather files in specific order: tmy3, amy2018, amy2012
    weather_order = ["tmy3", "amy2018", "amy2012"]
    available_weather_files.sort(key=lambda x: weather_order.index(x) if x in weather_order else len(weather_order))

    return available_releases, available_weather_files


def get_release_versions_options(
    available_releases: list[str], product_type: str, release_year: str, weather_file: str
) -> tuple[list[str], list[str]]:
    parsed_releases = parse_buildstock_releases(available_releases)
    available_releases = filter_available_releases(
        list(parsed_releases.keys()), product_type=product_type, release_year=release_year, weather_file=weather_file
    )
    available_release_versions = list({parsed_releases[release]["release_version"] for release in available_releases})

    if (
        product_type == "resstock"
        and weather_file == "tmy3"
        and release_year == "2024"
        and "1" in available_release_versions
    ):
        available_release_versions.remove("1")

    # Define the desired order: "2", "1.1", "1"
    display_order = ["2", "1.1", "1"]

    # Filter available release versions to only include those in the desired order
    ordered_release_versions = []
    for version in display_order:
        if version in available_release_versions:
            ordered_release_versions.append(version)

    return available_releases, ordered_release_versions


def filter_available_releases(
    available_releases: list[str],
    product_type: Union[str, None] = None,
    release_year: Union[str, None] = None,
    weather_file: Union[str, None] = None,
    release_version: Union[str, None] = None,
) -> list[str]:
    parsed_releases = parse_buildstock_releases(available_releases)
    filtered_releases = []
    if product_type is not None:
        filtered_releases = [
            release for release in parsed_releases if parsed_releases[release]["product"] == product_type
        ]
    if release_year is not None:
        filtered_releases = [
            release for release in filtered_releases if parsed_releases[release]["release_year"] == release_year
        ]
    if weather_file is not None:
        filtered_releases = [
            release for release in filtered_releases if parsed_releases[release]["weather_file"] == weather_file
        ]
    if release_version is not None:
        filtered_releases = [
            release for release in filtered_releases if parsed_releases[release]["release_version"] == release_version
        ]
    return filtered_releases


def parse_buildstock_releases(buildstock_releases: list[str]) -> dict[str, dict]:
    """Parse buildstock releases and extract components from keys in format: {product}_{release_year}_{weather_file}_{release_version}"""
    parsed_releases = {}

    for release_name in buildstock_releases:
        # Split the release name by underscore
        parts = release_name.split("_")

        product = parts[0]  # e.g., "res" or "com"
        if product == "res":
            product = "resstock"
        elif product == "com":
            product = "comstock"
        else:
            message = f"Invalid product type: {product}"
            console.print(Panel(message, title="Error", border_style="red"))
            raise ValueError(message)

        release_year = parts[1]  # e.g., "2021"
        weather_file = parts[2]  # e.g., "tmy3" or "amy2018"
        release_version = parts[3]  # e.g., "1" or "1.1"

        parsed_releases[release_name] = {
            "product": product,
            "release_year": release_year,
            "weather_file": weather_file,
            "release_version": release_version,
        }
    return parsed_releases
