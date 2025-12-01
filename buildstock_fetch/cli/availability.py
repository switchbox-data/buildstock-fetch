import json
from importlib.resources import files
from pathlib import Path
from typing import cast

BUILDSTOCK_RELEASES_FILE = str(files("buildstock_fetch").joinpath("data").joinpath("buildstock_releases.json"))


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
