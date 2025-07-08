import json
from pathlib import Path

import questionary
import typer
from rich.console import Console
from rich.panel import Panel

# Initialize Rich console
console = Console()

# Create Typer app
app = typer.Typer(
    name="buildstock-fetch",
    help="A CLI tool to fetch data from the BuildStock API",
    rich_markup_mode="rich",
    no_args_is_help=False,
)


# File configuration
BUILDSTOCK_RELEASES_FILE = Path(__file__).parent.parent / "utils" / "buildstock_releases.json"


def _filter_available_releases(
    available_releases: list[str],
    product_type: str | None = None,
    release_year: str | None = None,
    weather_file: str | None = None,
    release_version: str | None = None,
) -> list[str]:
    parsed_releases = _parse_buildstock_releases(available_releases)
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


def _get_release_years(available_releases: list[str], product_type: str) -> tuple[list[str], list[str]]:
    # Find available release years
    parsed_releases = _parse_buildstock_releases(available_releases)
    available_releases = _filter_available_releases(list(parsed_releases.keys()), product_type=product_type)
    available_release_years = list({parsed_releases[release]["release_year"] for release in available_releases})
    available_release_years.sort()

    return available_releases, available_release_years


def _get_state_options() -> list[str]:
    return [
        "AL",
        "AK",
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


def _get_file_type_options() -> list[str]:
    return [
        "metadata",
        "15min_load_curve",
        "hourly_load_curve",
        "monthly_load_curve",
        "daily_load_curve",
        "annual_load_curve",
    ]


def _get_weather_options() -> list[str]:
    return ["tmy3", "amy2018"]


def _get_product_type_options() -> list[str]:
    return ["resstock", "comstock"]


def _get_available_releases() -> list[str]:
    # Read the buildstock releases JSON file
    with open(BUILDSTOCK_RELEASES_FILE) as f:
        buildstock_releases = json.load(f)

    # Return the top-level keys as release options
    return list(buildstock_releases.keys())


def _parse_buildstock_releases(buildstock_releases: list[str]) -> dict[str, dict]:
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


def main_callback(
    release_version: int = typer.Option(None, "--release_version", "-r"),
    state: str = typer.Option(None, "--state", "-s"),
    file_type: str = typer.Option(None, "--file_type", "-f"),
) -> None:
    """
    DBF CLI tool. Run without arguments for interactive mode.
    """

    # Retrieve available releases
    available_releases = _get_available_releases()

    # If no arguments provided, run interactive mode
    if not any([release_version, state, file_type]):
        console.print(Panel("BuildStock Fetch Interactive CLI", title="BuildStock Fetch CLI", border_style="blue"))
        console.print("Welcome to the BuildStock Fetch CLI!")
        console.print("This tool allows you to fetch data from the NREL BuildStock API.")
        console.print("Please select the release information and file type you would like to fetch:")

        # Release version options
        product_type = questionary.select("Select product type:", choices=_get_product_type_options()).ask()
        _, release_years = _get_release_years(available_releases, product_type)
        selected_release_year = questionary.select("Select release year:", choices=release_years).ask()

    # Process the data
    print(f"Result: {product_type}, {selected_release_year}")


app.callback(invoke_without_command=True)(main_callback)


if __name__ == "__main__":
    # Print the available release options
    print(_get_available_releases())

    # Print the parsed release options
    print(_parse_buildstock_releases(_get_available_releases()))

    # Print the release years
    available_releases, available_release_years = _get_release_years(_get_available_releases(), "resstock")
    print(available_releases)
    print(available_release_years)
    available_releases, available_release_years = _get_release_years(_get_available_releases(), "comstock")
    print(available_releases)
    print(available_release_years)

    app()
