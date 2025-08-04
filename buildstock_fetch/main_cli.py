import json
import pprint
import random
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from typing import Union, cast

import questionary
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from buildstock_fetch.main import fetch_bldg_data, fetch_bldg_ids

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
BUILDSTOCK_RELEASES_FILE = str(files("buildstock_fetch.utils").joinpath("buildstock_releases.json"))


class InvalidProductError(Exception):
    """Exception raised when an invalid product is provided."""

    pass


def _filter_available_releases(
    available_releases: list[str],
    product_type: Union[str, None] = None,
    release_year: Union[str, None] = None,
    weather_file: Union[str, None] = None,
    release_version: Union[str, None] = None,
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


def _get_product_type_options() -> list[str]:
    return ["resstock", "comstock"]


def _get_release_years_options(available_releases: list[str], product_type: str) -> tuple[list[str], list[str]]:
    # Find available release years
    parsed_releases = _parse_buildstock_releases(available_releases)
    available_releases = _filter_available_releases(list(parsed_releases.keys()), product_type=product_type)
    available_release_years = list({parsed_releases[release]["release_year"] for release in available_releases})
    available_release_years.sort()

    return available_releases, available_release_years


def _get_weather_options(
    available_releases: list[str], product_type: str, release_year: str
) -> tuple[list[str], list[str]]:
    parsed_releases = _parse_buildstock_releases(available_releases)
    available_releases = _filter_available_releases(
        list(parsed_releases.keys()), product_type=product_type, release_year=release_year
    )
    available_weather_files = list({parsed_releases[release]["weather_file"] for release in available_releases})

    # Sort weather files in specific order: tmy3, amy2018, amy2012
    weather_order = ["tmy3", "amy2018", "amy2012"]
    available_weather_files.sort(key=lambda x: weather_order.index(x) if x in weather_order else len(weather_order))

    return available_releases, available_weather_files


def _get_release_versions_options(
    available_releases: list[str], product_type: str, release_year: str, weather_file: str
) -> tuple[list[str], list[str]]:
    parsed_releases = _parse_buildstock_releases(available_releases)
    available_releases = _filter_available_releases(
        list(parsed_releases.keys()), product_type=product_type, release_year=release_year, weather_file=weather_file
    )
    available_release_versions = list({parsed_releases[release]["release_version"] for release in available_releases})

    # Define the desired order: "2", "1.1", "1"
    display_order = ["2", "1.1", "1"]

    # Filter available release versions to only include those in the desired order
    ordered_release_versions = []
    for version in display_order:
        if version in available_release_versions:
            ordered_release_versions.append(version)

    return available_releases, ordered_release_versions


def _get_upgrade_ids_options(release_name: str) -> list[str]:
    available_releases = _get_all_available_releases()
    available_upgrade_ids = available_releases[release_name]["upgrade_ids"]
    available_upgrade_ids = [int(upgrade_id) for upgrade_id in available_upgrade_ids]
    available_upgrade_ids.sort()
    available_upgrade_ids = [str(upgrade_id) for upgrade_id in available_upgrade_ids]

    return cast(list[str], available_upgrade_ids)


def _get_state_options() -> list[str]:
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


def _get_file_type_options(release_name: str) -> list[str]:
    """Get the file type options for a given release name."""

    # Each release has a different set of available data. For example, some releases don't have building data,
    # some don't have 15 min end use load profiles, etc. This function returns the available data for a given release.
    available_releases = _get_all_available_releases()
    return cast(list[str], available_releases[release_name]["available_data"])


def _get_file_type_options_grouped(release_name: str) -> list[dict]:
    """Get file type options grouped by category for questionary checkbox."""
    file_types = _get_file_type_options(release_name)

    # Define categories
    categories = {
        "Simulation Files": ["hpxml", "schedule"],
        "End Use Load Curves": [
            "load_curve_15min",
            "load_curve_hourly",
            "load_curve_daily",
            "load_curve_monthly",
            "load_curve_annual",
        ],
        "Metadata": ["metadata"],
    }

    choices = []
    for category, types in categories.items():
        # Filter available types but maintain the defined order
        available_in_category = [ft for ft in types if ft in file_types]
        if available_in_category:
            choices.append({"name": f"--- {category} ---", "value": None, "disabled": True})
            for file_type in available_in_category:
                choices.append({"name": f"  {file_type}", "value": file_type, "style": "bold"})

    return choices


def _get_available_releases_names() -> list[str]:
    # Read the buildstock releases JSON file
    with open(BUILDSTOCK_RELEASES_FILE) as f:
        buildstock_releases = json.load(f)

    # Return the top-level keys as release options
    return list(buildstock_releases.keys())


def _get_all_available_releases() -> dict[str, dict]:
    with open(BUILDSTOCK_RELEASES_FILE) as f:
        buildstock_releases = json.load(f)
    return cast(dict[str, dict], buildstock_releases)


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


def _validate_output_directory(output_directory: str) -> Union[bool, str]:
    """Validate that the path format is correct for a directory"""
    try:
        path = Path(output_directory)
        # Check if it's a valid path format
        path.resolve()
    except (OSError, ValueError):
        return "Please enter a valid directory path"
    else:
        return True


def _handle_cancellation(result: Union[str, None], message: str = "Operation cancelled by user.") -> str:
    """Handle user cancellation and exit cleanly"""
    if result is None:
        console.print(f"\n[red]{message}[/red]")
        raise typer.Exit(0) from None
    return result


def _run_interactive_mode() -> dict[str, Union[str, list[str]]]:
    """Run the interactive CLI mode"""
    console.print(Panel("BuildStock Fetch Interactive CLI", title="BuildStock Fetch CLI", border_style="blue"))
    console.print("Welcome to the BuildStock Fetch CLI!")
    console.print("This tool allows you to fetch data from the NREL BuildStock API.")
    console.print("Please select the release information and file type you would like to fetch:")

    # Retrieve available releases
    available_releases = _get_available_releases_names()

    # Retrieve product type and filter available release years by product type
    product_type = _handle_cancellation(
        questionary.select("Select product type:", choices=_get_product_type_options()).ask()
    )
    available_releases, release_years = _get_release_years_options(available_releases, product_type)

    # Retrieve release year and filter available weather files by release year
    selected_release_year = _handle_cancellation(
        questionary.select("Select release year:", choices=release_years).ask()
    )
    available_releases, weather_files = _get_weather_options(available_releases, product_type, selected_release_year)

    # Retrieve weather file and filter available release versions by weather file
    selected_weather_file = _handle_cancellation(
        questionary.select("Select weather file:", choices=weather_files).ask()
    )
    available_releases, release_versions = _get_release_versions_options(
        available_releases, product_type, selected_release_year, selected_weather_file
    )

    # Retrieve release version
    selected_release_version = _handle_cancellation(
        questionary.select("Select release version:", choices=release_versions).ask()
    )

    product_short_name = "res" if product_type == "resstock" else "com"
    selected_release_name = (
        f"{product_short_name}_{selected_release_year}_{selected_weather_file}_{selected_release_version}"
    )

    # Retrieve upgrade ids
    selected_upgrade_ids = _handle_cancellation(
        questionary.checkbox(
            "Select upgrade ids:",
            choices=_get_upgrade_ids_options(selected_release_name),
            instruction="Use spacebar to select/deselect options, enter to confirm",
            validate=lambda answer: "You must select at least one upgrade id" if len(answer) == 0 else True,
        ).ask()
    )

    # Retrieve state
    selected_states = _handle_cancellation(
        questionary.checkbox(
            "Select states:",
            choices=_get_state_options(),
            instruction="Use spacebar to select/deselect options, enter to confirm",
            validate=lambda answer: "You must select at least one state" if len(answer) == 0 else True,
        ).ask()
    )

    # Retrieve requested file type
    requested_file_types = _handle_cancellation(
        questionary.checkbox(
            "Select file type:",
            choices=_get_file_type_options_grouped(selected_release_name),
            instruction="Use spacebar to select/deselect options, enter to confirm",
            validate=lambda answer: "You must select at least one file type" if len(answer) == 0 else True,
        ).ask()
    )

    # Retrieve output directory
    output_directory_str = _handle_cancellation(
        questionary.path(
            "Select output directory:",
            default=str(Path.cwd() / "data"),
            only_directories=True,
            validate=_validate_output_directory,
        ).ask()
    )
    output_directory_path = Path(output_directory_str)
    output_directory_path.mkdir(parents=True, exist_ok=True)
    return {
        "product": product_type,
        "release_year": selected_release_year,
        "weather_file": selected_weather_file,
        "release_version": selected_release_version,
        "upgrade_ids": selected_upgrade_ids,
        "states": selected_states,
        "file_type": requested_file_types,
        "output_directory": str(output_directory_path),
    }


def _verify_interactive_inputs(inputs: dict) -> bool:
    """Display the inputs and ask the user to verify them."""
    console = Console()

    table = Table(title="Please verify your selections:")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    for k, v in inputs.items():
        table.add_row(str(k), pprint.pformat(v))
    console.print(Panel(table, border_style="green"))

    try:
        result = questionary.confirm("Are these selections correct?", default=True).ask()
    except KeyboardInterrupt:
        console.print("\n[red]Operation cancelled by user.[/red]")
        raise typer.Exit(0) from None

    if result is None:
        console.print("\n[red]Operation cancelled by user.[/red]")
        raise typer.Exit(0) from None

    return bool(result)


def _print_data_processing_info(inputs: Mapping[str, Union[str, list[str]]]) -> None:
    """Print the data processing information."""
    print("Downloading data for:")
    print(f"Product: {inputs['product']}")
    print(f"Release year: {inputs['release_year']}")
    print(f"Weather file: {inputs['weather_file']}")
    print(f"Release version: {inputs['release_version']}")
    print(f"States: {inputs['states']}")
    print(f"File type: {inputs['file_type']}")
    print(f"Upgrade ids: {inputs['upgrade_ids']}")
    print(f"Output directory: {inputs['output_directory']}")


def _check_unavailable_file_types(inputs: Mapping[str, Union[str, list[str]]]) -> None:
    """Check and print warning for unavailable file types."""
    unavailable_file_types = ["load_curve_hourly", "load_curve_daily", "load_curve_monthly", "load_curve_annual"]
    selected_file_types = inputs["file_type"].split() if isinstance(inputs["file_type"], str) else inputs["file_type"]
    selected_unavailable = [ft for ft in selected_file_types if ft in unavailable_file_types]
    if selected_unavailable:
        console.print("\n[yellow]The following file types are not available yet:[/yellow]")
        for file_type in selected_unavailable:
            console.print(f"  • {file_type}")
        console.print("")


def _fetch_all_building_ids(inputs: Mapping[str, Union[str, list[str]]]) -> list:
    """Fetch building IDs for all states and upgrade IDs."""
    bldg_ids = []
    for state in inputs["states"]:
        state = state.strip()
        if state == "":
            continue
        for upgrade_id in inputs["upgrade_ids"]:
            bldg_id = fetch_bldg_ids(
                str(inputs["product"]),
                str(inputs["release_year"]),
                str(inputs["weather_file"]),
                str(inputs["release_version"]),
                str(state),
                str(upgrade_id),
            )
            bldg_ids.extend(bldg_id)
    return bldg_ids


def _get_user_download_choice(bldg_ids: list) -> list:
    """Ask user whether to download all files or a sample."""
    if not bldg_ids:
        return []

    # Group building IDs by state-upgrade_id pairs
    state_upgrade_groups: dict[tuple[str, str], list] = {}
    for bldg_id in bldg_ids:
        state_upgrade_key = (bldg_id.state, bldg_id.upgrade_id)
        if state_upgrade_key not in state_upgrade_groups:
            state_upgrade_groups[state_upgrade_key] = []
        state_upgrade_groups[state_upgrade_key].append(bldg_id)

    # Print summary for each state-upgrade_id pair
    console.print(f"\nThere are {len(bldg_ids)} files for this release:")
    for (state, upgrade_id), bldg_list in state_upgrade_groups.items():
        console.print(f"  • State {state}, Upgrade {upgrade_id}: {len(bldg_list)} buildings")

    choice = _handle_cancellation(
        questionary.select(
            "Would you like to download all files or a sample of them?",
            choices=["Download all files", "Download a sample"],
        ).ask()
    )

    if choice == "Download all files":
        return bldg_ids
    else:
        selected_bldg_ids = []

        # For each state-upgrade_id pair, ask user for sample size
        for (state, upgrade_id), bldg_list in state_upgrade_groups.items():
            total_for_state_upgrade = len(bldg_list)

            sample_size_str = _handle_cancellation(
                questionary.text(
                    f"Enter the number of files to download for State {state}, Upgrade {upgrade_id} (0-{total_for_state_upgrade}):",
                    validate=lambda text, max_val=total_for_state_upgrade: (
                        text.isdigit() and 0 <= int(text) <= max_val
                    )
                    or f"Please enter a number between 0 and {max_val}",
                ).ask()
            )

            sample_size = int(sample_size_str)
            if sample_size == 0:
                console.print(f"[yellow]No files will be downloaded for State {state}, Upgrade {upgrade_id}.[/yellow]")
                continue

            # Randomly select the specified number of building IDs for this state-upgrade_id pair
            selected_for_state_upgrade = random.sample(bldg_list, sample_size)
            selected_bldg_ids.extend(selected_for_state_upgrade)
            console.print(f"[green]Selected {sample_size} buildings for State {state}, Upgrade {upgrade_id}.[/green]")

        if not selected_bldg_ids:
            console.print("[yellow]No files selected for download.[/yellow]")

        return selected_bldg_ids


def _validate_required_inputs(inputs: dict[str, Union[str, list[str]]]) -> Union[str, bool]:
    """Validate that all required inputs are provided."""
    if not all([
        inputs["product"],
        inputs["release_year"],
        inputs["weather_file"],
        inputs["release_version"],
        inputs["states"],
        inputs["file_type"],
        inputs["upgrade_ids"],
        inputs["output_directory"],
    ]):
        return "Please provide all required inputs"
    return True


def _validate_release_name(inputs: dict[str, Union[str, list[str]]]) -> Union[str, bool]:
    """Validate the release name."""
    available_releases = _get_all_available_releases()

    if inputs["product"] == "resstock":
        product_short_name = "res"
    elif inputs["product"] == "comstock":
        product_short_name = "com"
    else:
        raise InvalidProductError

    release_name = f"{product_short_name}_{inputs['release_year']}_{inputs['weather_file']}_{inputs['release_version']}"
    if release_name not in available_releases:
        return f"Invalid release name: {release_name}"
    return True


def _validate_upgrade_ids(inputs: dict[str, Union[str, list[str]]], release_name: str) -> Union[str, bool]:
    """Validate upgrade IDs."""
    available_releases = _get_all_available_releases()
    for upgrade_id in inputs["upgrade_ids"]:
        if int(upgrade_id) not in [
            int(upgrade_id_val) for upgrade_id_val in available_releases[release_name]["upgrade_ids"]
        ]:
            return f"Invalid upgrade id: {upgrade_id}"
    return True


def _validate_file_types(inputs: dict[str, Union[str, list[str]]], release_name: str) -> Union[str, bool]:
    """Validate file types."""
    available_releases = _get_all_available_releases()
    for file_type in inputs["file_type"]:
        if file_type not in available_releases[release_name]["available_data"]:
            return f"Invalid file type: {file_type}"
    return True


def _validate_states(inputs: dict[str, Union[str, list[str]]]) -> Union[str, bool]:
    """Validate states."""
    for state in inputs["states"]:
        if state not in _get_state_options():
            return f"Invalid state: {state}"
    return True


def _validate_direct_inputs(inputs: dict[str, Union[str, list[str]]]) -> Union[str, bool]:
    """Validate the direct inputs"""
    # Check required inputs
    required_check = _validate_required_inputs(inputs)
    if required_check is not True:
        return required_check

    # Check release name
    release_check = _validate_release_name(inputs)
    if release_check is not True:
        return release_check

    # Get release name for further validation
    product_short_name = "res" if inputs["product"] == "resstock" else "com"
    release_name = f"{product_short_name}_{inputs['release_year']}_{inputs['weather_file']}_{inputs['release_version']}"

    # Check upgrade IDs
    upgrade_check = _validate_upgrade_ids(inputs, release_name)
    if upgrade_check is not True:
        return upgrade_check

    # Check file types
    file_type_check = _validate_file_types(inputs, release_name)
    if file_type_check is not True:
        return file_type_check

    # Check states
    state_check = _validate_states(inputs)
    if state_check is not True:
        return state_check

    # Check output directory
    output_directory_validation = _validate_output_directory(str(inputs["output_directory"]))
    if output_directory_validation is not True:
        return f"Invalid output directory: {inputs['output_directory']}"

    return True


# Module-level option definitions
PRODUCT_OPTION = typer.Option(None, "--product", "-p", help='"resstock" or "comstock"')
RELEASE_YEAR_OPTION = typer.Option(None, "--release_year", "-y", help="Release year (typically 2021 or later)")
WEATHER_FILE_OPTION = typer.Option(None, "--weather_file", "-w", help='"tmy3", "amy2012", "amy2018"')
RELEASE_VERSION_OPTION = typer.Option(None, "--release_version", "-r", help="1, 1.1, or 2")
STATES_OPTION = typer.Option(
    None, "--states", "-s", help="List of states (multiple can be provided, inside quotes and separated by spaces)"
)
FILE_TYPE_OPTION = typer.Option(
    None,
    "--file_type",
    "-f",
    help="List of file types (multiple can be provided, inside quotes and separated by spaces)",
)
UPGRADE_ID_OPTION = typer.Option(
    None, "--upgrade_id", "-u", help="Upgrade IDs (multiple can be provided, inside quotes and separated by spaces)"
)
OUTPUT_DIRECTORY_OPTION = typer.Option(None, "--output_directory", "-o", help='e.g., "data" or "../output"')


def main_callback(
    product: str = PRODUCT_OPTION,
    release_year: str = RELEASE_YEAR_OPTION,
    weather_file: str = WEATHER_FILE_OPTION,
    release_version: int = RELEASE_VERSION_OPTION,
    states: str = STATES_OPTION,
    file_type: str = FILE_TYPE_OPTION,
    upgrade_id: str = UPGRADE_ID_OPTION,
    output_directory: str = OUTPUT_DIRECTORY_OPTION,
) -> None:
    """
    Buildstock Fetch CLI tool. Run without arguments for interactive mode.
    """

    # If no arguments provided, run interactive mode
    if not any([product, release_year, weather_file, release_version, states, file_type]):
        try:
            while True:
                inputs = _run_interactive_mode()
                if _verify_interactive_inputs(inputs):
                    break
                console.print("[yellow]Let's try again...[/yellow]")
        except KeyboardInterrupt:
            console.print("\n[red]Operation cancelled by user.[/red]")
            raise typer.Exit(0) from None
    else:
        states_list = states.split() if states else []
        upgrade_ids_list = upgrade_id.split() if upgrade_id else ["0"]
        file_type_list = file_type.split() if file_type else []

        direct_inputs: dict[str, Union[str, list[str]]] = {
            "product": product,
            "release_year": release_year,
            "weather_file": weather_file,
            "release_version": str(release_version),
            "states": states_list,
            "file_type": file_type_list,
            "upgrade_ids": upgrade_ids_list,
            "output_directory": output_directory,
        }
        try:
            validation_result = _validate_direct_inputs(direct_inputs)
            if validation_result is not True:
                console.print(f"\n[red]{validation_result}[/red]")
                raise typer.Exit(1) from None
        except InvalidProductError:
            console.print(f"\n[red]Invalid product: {direct_inputs['product']}[/red]")
            raise typer.Exit(1) from None

        inputs = direct_inputs

    # Process the data
    _print_data_processing_info(inputs)
    _check_unavailable_file_types(inputs)

    # Fetch the building ids and download data
    bldg_ids = _fetch_all_building_ids(inputs)

    # Ask user about download choice
    selected_bldg_ids = _get_user_download_choice(bldg_ids)

    if selected_bldg_ids:
        file_type_tuple = (
            tuple(inputs["file_type"].split()) if isinstance(inputs["file_type"], str) else tuple(inputs["file_type"])
        )
        output_dir = inputs["output_directory"]
        if isinstance(output_dir, list):
            output_dir = output_dir[0] if output_dir else "."
        fetch_bldg_data(selected_bldg_ids, file_type_tuple, Path(output_dir))
    else:
        console.print("[yellow]No files selected for download.[/yellow]")


app.command()(main_callback)


if __name__ == "__main__":
    # Print the available release options
    print(_get_available_releases_names())

    # Print the parsed release options
    print(_parse_buildstock_releases(_get_available_releases_names()))

    # Print the release years
    available_releases, available_release_years = _get_release_years_options(
        _get_available_releases_names(), "resstock"
    )
    print(available_releases)
    print(available_release_years)
    available_releases, available_release_years = _get_release_years_options(
        _get_available_releases_names(), "comstock"
    )
    print(available_releases)
    print(available_release_years)

    app()
