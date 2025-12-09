import pprint
from pathlib import Path
from typing import Union, cast

import questionary
import tomli
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from buildstock_fetch.main import fetch_bldg_data, fetch_bldg_ids
from buildstock_fetch.types import ReleaseYear, ResCom, Weather

from .availability import (
    check_weather_file_availability,
    check_weather_map_available_states,
    get_all_available_releases,
    get_available_releases_names,
    get_product_type_options,
    get_release_versions_options,
    get_release_years_options,
    get_state_options,
    get_upgrade_ids_options,
    get_weather_options,
    parse_buildstock_releases,
)
from .inputs import Inputs
from .questionary import checkbox_str, handle_cancellation, select
from .validate import (
    validate_file_types,
    validate_output_directory,
    validate_release_name,
    validate_states,
    validate_upgrade_ids,
)

# Initialize Rich console
console = Console()

# Create Typer app
app = typer.Typer(
    name="buildstock-fetch",
    help="A CLI tool to fetch data from the BuildStock API",
    rich_markup_mode="rich",
    no_args_is_help=False,
)

# Module-level option definitions
PRODUCT_OPTION = typer.Option(None, "--product", "-p", help='"resstock" or "comstock"')
RELEASE_YEAR_OPTION = typer.Option(None, "--release_year", "-y", help="Release year (typically 2021 or later)")
WEATHER_FILE_OPTION = typer.Option(None, "--weather_file", "-w", help='"tmy3", "amy2012", "amy2018"')
RELEASE_VERSION_OPTION = typer.Option(None, "--release_version", "-r", help="1, 1.1, or 2")
STATES_OPTION = typer.Option(
    None, "--states", "-st", help="List of states (multiple can be provided, inside quotes and separated by spaces)"
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
SAMPLE_OPTION = typer.Option(
    None,
    "--sample",
    "-sm",
    help="Number of building IDs to download across all upgrades (only applies to direct inputs)",
)
VERSION_OPTION = typer.Option(False, "--version", "-v", help="Show version information and exit")

# File types that haven't been implemented yet
UNAVAILABLE_FILE_TYPES: list[str] = []


class InvalidProductError(Exception):
    """Exception raised when an invalid product is provided."""

    pass


def main_callback(
    product: ResCom = PRODUCT_OPTION,
    release_year: ReleaseYear = RELEASE_YEAR_OPTION,
    weather_file: Weather = WEATHER_FILE_OPTION,
    release_version: float = RELEASE_VERSION_OPTION,
    states: str = STATES_OPTION,
    file_type: str = FILE_TYPE_OPTION,
    upgrade_id: str = UPGRADE_ID_OPTION,
    output_directory: Path = OUTPUT_DIRECTORY_OPTION,
    sample: str = SAMPLE_OPTION,
    version: bool = VERSION_OPTION,
) -> None:
    """
    Buildstock Fetch CLI tool. Run without arguments for interactive mode.
    """

    # Handle version option first
    if version:
        _show_version()

    # If no arguments provided, run interactive mode
    if not any([product, release_year, weather_file, release_version, states, file_type]):
        inputs = _run_interactive_mode_wrapper()
        sample_value = None
    else:
        inputs, sample_value = _process_direct_inputs(
            product,
            release_year,
            weather_file,
            release_version,
            states,
            file_type,
            upgrade_id,
            output_directory,
            sample,
        )

    # Process the data
    _print_data_processing_info(inputs)
    _process_data_download(inputs, sample_value)


def _show_version() -> None:
    """Display version information and exit."""
    version = _get_version()
    console.print(f"buildstock-fetch version {version}")
    raise typer.Exit(0) from None


def _run_interactive_mode_wrapper() -> Inputs:
    """Run interactive mode with error handling."""
    try:
        while True:
            inputs = _run_interactive_mode()
            if _verify_interactive_inputs(inputs):
                return inputs
            console.print("[yellow]Let's try again...[/yellow]")
    except KeyboardInterrupt:
        console.print("\n[red]Operation cancelled by user.[/red]")
        raise typer.Exit(0) from None


def _process_direct_inputs(
    product: ResCom,
    release_year: ReleaseYear,
    weather_file: Weather,
    release_version: float,
    states: str,
    file_type: str,
    upgrade_id: str,
    output_directory: Path,
    sample: Union[str, None] = None,
) -> tuple[Inputs, Union[str, None]]:
    """Process direct command line inputs."""
    states_list = states.split() if states else []
    upgrade_ids_list = upgrade_id.split() if upgrade_id else ["0"]
    file_type_list = file_type.split() if file_type else []

    direct_inputs: Inputs = {
        "product": product,
        "release_year": release_year,
        "weather_file": weather_file,
        "release_version": _normalize_release_version(release_version),
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

    return direct_inputs, sample


def _run_interactive_mode() -> Inputs:
    """Run the interactive CLI mode"""
    console.print(Panel("BuildStock Fetch Interactive CLI", title="BuildStock Fetch CLI", border_style="blue"))
    console.print("Welcome to the BuildStock Fetch CLI!")
    console.print("This tool allows you to fetch data from the NREL BuildStock API.")
    console.print("Please select the release information and file type you would like to fetch:")

    # Retrieve available releases
    available_releases = get_available_releases_names()

    # Retrieve product type and filter available release years by product type
    product_type: ResCom = handle_cancellation(select("Select product type", choices=get_product_type_options()))
    available_releases, release_years = get_release_years_options(available_releases, product_type)

    # Retrieve release year and filter available weather files by release year
    selected_release_year: ReleaseYear = handle_cancellation(select("Select release year:", choices=release_years))
    available_releases, weather_files = get_weather_options(available_releases, product_type, selected_release_year)

    # Retrieve weather file and filter available release versions by weather file
    selected_weather_file: Weather = handle_cancellation(select("Select weather file:", choices=weather_files))
    available_releases, release_versions = get_release_versions_options(
        available_releases, product_type, selected_release_year, selected_weather_file
    )

    # Retrieve release version
    selected_release_version = handle_cancellation(select("Select release version:", choices=release_versions))

    product_short_name = "res" if product_type == "resstock" else "com"
    selected_release_name = (
        f"{product_short_name}_{selected_release_year}_{selected_weather_file}_{selected_release_version}"
    )

    # Retrieve upgrade ids
    upgrade_options = get_upgrade_ids_options(selected_release_name)
    selected_upgrade_ids_raw = handle_cancellation(
        checkbox_str(
            "Select upgrade ids:",
            choices=upgrade_options,
            instruction="Use spacebar to select/deselect options, 'a' to select all, 'i' to invert selection, enter to confirm",
            validate=lambda answer: "You must select at least one upgrade id" if len(answer) == 0 else True,
        )
    )

    # Extract upgrade ID integers from the selected options
    selected_upgrade_ids = []
    for option in selected_upgrade_ids_raw:
        if ":" in option:
            # Extract the integer before the colon
            upgrade_id = option.split(":")[0].strip()
            selected_upgrade_ids.append(upgrade_id)
        else:
            selected_upgrade_ids.append(option)

    # Retrieve state
    selected_states: list[str] = cast(
        list[str],
        handle_cancellation(
            checkbox_str(
                "Select states:",
                choices=get_state_options(),
                instruction="Use spacebar to select/deselect options, enter to confirm",
                validate=lambda answer: "You must select at least one state" if len(answer) == 0 else True,
            )
        ),
    )

    # Retrieve requested file type
    requested_file_types: list[str] = handle_cancellation(
        questionary.checkbox(
            "Select file type:",
            choices=_get_file_type_options_grouped(selected_release_name, selected_states),
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
            validate=validate_output_directory,
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
        "output_directory": output_directory_path,
    }


def _verify_interactive_inputs(inputs: Inputs) -> bool:
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


def _normalize_release_version(release_version: Union[str, float, int]) -> str:
    """
    Convert release_version to integer if it's a whole number, then to string.

    Args:
        release_version: The release version to normalize

    Returns:
        str: Normalized release version string
    """
    # TODO: they use semantic versioning, we can try using symver
    try:
        release_version_float = float(release_version)
        if release_version_float.is_integer():
            return str(int(release_version_float))
        else:
            return str(release_version)
    except (ValueError, TypeError):
        return str(release_version)


def _get_file_type_options(release_name: str) -> list[str]:
    """Get the file type options for a given release name."""

    # Each release has a different set of available data. For example, some releases don't have building data,
    # some don't have 15 min end use load profiles, etc. This function returns the available data for a given release.
    available_releases = get_all_available_releases()
    return cast(list[str], available_releases[release_name]["available_data"])


def _get_file_type_options_grouped(release_name: str, selected_states: list[str]) -> list[dict]:
    """Get file type options grouped by category for questionary checkbox."""
    file_types = _get_file_type_options(release_name)

    # TODO: If a trip_schedule table was built for any of the states in this release, "trip_schedules" will be included
    # in the file_types list above. However, the state that the user wants to download files for may not have the trip_schedule tables built for it yet.
    # So, we need to check if the release_name + state combo is in the available trip_schedules_states list.
    # If not, we need to remove "trip_schedules" from the file_types list, so it doesn't show up in the questionary checkbox.
    # Remember that users can select multiple states, so as long as one release_name + state combo is in the available trip_schedules_states list,
    # we should include "trip_schedules" in the file_types list. and then later on, we need to handle different states differently.

    trip_schedule_availabile = False
    available_releases = get_all_available_releases()
    for state in selected_states:
        if (
            "trip_schedules" in available_releases[release_name]["available_data"]
            and state in available_releases[release_name]["trip_schedule_states"]
        ):
            trip_schedule_availabile = True

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
        "EV": ["trip_schedules"],
        "Weather": ["weather"],
    }

    choices: list[dict] = []
    for category, types in categories.items():
        if category == "EV":
            _add_ev_category_choices(choices, category, types, trip_schedule_availabile)
        else:
            _add_standard_category_choices(choices, category, types, file_types)

    return choices


def _add_ev_category_choices(
    choices: list[dict], category: str, types: list[str], trip_schedule_available: bool
) -> None:
    """Add EV category choices to the choices list."""
    for file_type in types:
        if file_type == "trip_schedules" and trip_schedule_available:
            choices.append({"name": f"--- {category} ---", "value": None, "disabled": True})
            choices.append({"name": f"  {file_type}", "value": file_type, "style": "bold"})


def _add_standard_category_choices(choices: list[dict], category: str, types: list[str], file_types: list[str]) -> None:
    """Add standard category choices to the choices list."""
    available_in_category = [ft for ft in types if ft in file_types]
    if available_in_category:
        choices.append({"name": f"--- {category} ---", "value": None, "disabled": True})
        for file_type in available_in_category:
            choices.append({"name": f"  {file_type}", "value": file_type, "style": "bold"})


def _handle_cancellation(result: Union[str, None], message: str = "Operation cancelled by user.") -> str:
    """Handle user cancellation and exit cleanly"""
    if result is None:
        console.print(f"\n[red]{message}[/red]")
        raise typer.Exit(0) from None
    return result


def _print_data_processing_info(inputs: Inputs) -> None:
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


def _print_unavailable_file_types_warning(selected_unavailable_file_types: list[str]) -> None:
    """Print warning for unavailable file types."""
    if selected_unavailable_file_types:
        console.print("\n[yellow]The following file types are not available yet:[/yellow]")
        for file_type in selected_unavailable_file_types:
            console.print(f"  • {file_type}")
        console.print("")


def _check_unavailable_file_types(inputs: Inputs) -> tuple[list[str], list[str]]:
    """Check and print warning for unavailable file types."""
    selected_file_types = inputs["file_type"].split() if isinstance(inputs["file_type"], str) else inputs["file_type"]
    # Create a copy to avoid modifying the original list
    available_file_types = selected_file_types.copy()
    selected_unavailable_file_types = []

    for ft in selected_file_types:
        if ft in UNAVAILABLE_FILE_TYPES:
            selected_unavailable_file_types.append(ft)
            # Only remove if it exists in available_file_types
            if ft in available_file_types:
                available_file_types.remove(ft)

    if "weather" in selected_file_types:
        check_weather_file_availability(inputs, available_file_types, selected_unavailable_file_types)

    if "trip_schedules" in selected_file_types:
        product_short_name = "res" if inputs["product"] == "resstock" else "com"
        input_release_name = (
            f"{product_short_name}_{inputs['release_year']}_{inputs['weather_file']}_{inputs['release_version']}"
        )
        available_releases = get_all_available_releases()
        availble_trip_schedule_states = available_releases[input_release_name]["trip_schedule_states"]
        for state in inputs["states"]:
            if state not in availble_trip_schedule_states:
                console.print(f"[yellow]The following state is not available for trip schedules: {state}[/yellow]")
                selected_unavailable_file_types.append(f"trip_schedules_{state}")

    _print_unavailable_file_types_warning(selected_unavailable_file_types)

    return available_file_types, selected_unavailable_file_types


def _fetch_all_building_ids(inputs: Inputs) -> list:
    """Fetch building IDs for all states and upgrade IDs."""
    bldg_ids = []
    for state in inputs["states"]:
        state = state.strip()
        if state == "":
            continue
        for upgrade_id in inputs["upgrade_ids"]:
            bldg_id = fetch_bldg_ids(
                inputs["product"],
                inputs["release_year"],
                inputs["weather_file"],
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

            # Select the first N building IDs for this state-upgrade_id pair
            selected_for_state_upgrade = bldg_list[:sample_size]
            selected_bldg_ids.extend(selected_for_state_upgrade)
            console.print(f"[green]Selected {sample_size} buildings for State {state}, Upgrade {upgrade_id}.[/green]")

        if not selected_bldg_ids:
            console.print("[yellow]No files selected for download.[/yellow]")

        return selected_bldg_ids


def _select_bldg_ids_for_sample(bldg_ids: list, sample: Union[str, None]) -> list:
    """Return building IDs sampled per (state, upgrade_id) pair.

    - Group by (state, upgrade_id)
    - If sample is "all"/"a": return all IDs from each group
    - If sample is an integer string N: return first N from each group
    - Invalid or non-positive sample returns an empty list
    """
    if sample is None:
        return []

    # Group building IDs by (state, upgrade_id)
    state_upgrade_groups: dict[tuple[str, str], list] = {}
    for bldg_id in bldg_ids:
        key = (bldg_id.state, bldg_id.upgrade_id)
        if key not in state_upgrade_groups:
            state_upgrade_groups[key] = []
        state_upgrade_groups[key].append(bldg_id)

    sample_str = str(sample).strip().lower()
    if sample_str in ("all", "a"):
        selected_all: list = []
        for group_ids in state_upgrade_groups.values():
            selected_all.extend(group_ids)
        console.print(f"[green]Selected all {len(selected_all)} building IDs across all state/upgrade pairs.[/green]")
        return selected_all

    try:
        sample_size = int(sample_str)
    except ValueError:
        console.print(f"[red]Invalid value for --sample: {sample}. Use an integer, 'all', or 'a'.[/red]")
        return []

    if sample_size <= 0:
        console.print("[yellow]Sample size must be greater than 0.[/yellow]")
        return []

    # Select first N from each (state, upgrade_id) group
    selected_per_group: list = []
    for (state, upgrade_id), group_ids in state_upgrade_groups.items():
        take = min(sample_size, len(group_ids))
        selected_per_group.extend(group_ids[:take])
        console.print(
            f"[green]Sampling {take} building IDs for State {state}, Upgrade {upgrade_id} (of {len(group_ids)} available).[/green]"
        )

    if not selected_per_group and len(bldg_ids) > 0:
        console.print("[yellow]No building IDs selected after sampling per group.[/yellow]")

    return selected_per_group


def _validate_required_inputs(inputs: Inputs) -> Union[str, bool]:
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


def _validate_direct_inputs(inputs: Inputs) -> bool | str:
    """Validate the direct inputs"""
    # Check required inputs
    required_check = _validate_required_inputs(inputs)
    if required_check is not True:
        return required_check

    # Check release name
    release_check = validate_release_name(inputs)
    if release_check is not True:
        return release_check

    # Get release name for further validation
    product_short_name = "res" if inputs["product"] == "resstock" else "com"
    release_name = f"{product_short_name}_{inputs['release_year']}_{inputs['weather_file']}_{inputs['release_version']}"

    # Check upgrade IDs
    upgrade_check = validate_upgrade_ids(inputs, release_name)
    if upgrade_check is not True:
        return upgrade_check

    # Check file types
    file_type_check = validate_file_types(inputs, release_name)
    if file_type_check is not True:
        return file_type_check

    # Check states
    state_check = validate_states(inputs)
    if state_check is not True:
        return state_check

    # Check output directory
    output_directory_validation = validate_output_directory(inputs["output_directory"])
    if output_directory_validation is not True:
        return f"Invalid output directory: {inputs['output_directory']}"

    return True


def _get_version() -> str:
    """Get the version from pyproject.toml."""
    try:
        # Get the path to pyproject.toml (assuming it's in the project root)
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"

        with open(pyproject_path, "rb") as f:
            data = tomli.load(f)
            version = data["project"]["version"]
            return str(version)
    except (FileNotFoundError, KeyError, Exception):
        return "unknown"


def _process_data_download(inputs: Inputs, sample: Union[str, None] = None) -> None:
    """Process data download based on available file types."""
    available_file_types, unavailable_file_types = _check_unavailable_file_types(inputs)
    if "weather" in inputs["file_type"]:
        available_weather_states, unavailable_weather_states = check_weather_map_available_states(inputs)
    else:
        available_weather_states = None

    if len(available_file_types) > 0:
        # Fetch the building ids and download data
        bldg_ids = _fetch_all_building_ids(inputs)

        # If sample is provided (direct input mode), use it to sample building IDs
        if sample is not None:
            selected_bldg_ids = _select_bldg_ids_for_sample(bldg_ids, sample)
        else:
            # Ask user about download choice (interactive mode)
            selected_bldg_ids = _get_user_download_choice(bldg_ids)

        if selected_bldg_ids:
            file_type_tuple = (
                tuple(inputs["file_type"].split())
                if isinstance(inputs["file_type"], str)
                else tuple(inputs["file_type"])
            )
            output_dir = inputs["output_directory"]
            if isinstance(output_dir, list):
                output_dir = output_dir[0] if output_dir else "."
            fetch_bldg_data(
                selected_bldg_ids, file_type_tuple, Path(output_dir), weather_states=available_weather_states
            )
        else:
            console.print("[yellow]No files selected for download.[/yellow]")
    else:
        console.print("[yellow]None of the selected file types are available for download.[/yellow]")


app.command()(main_callback)


if __name__ == "__main__":
    # Print the available release options
    print(get_available_releases_names())

    # Print the parsed release options
    print(parse_buildstock_releases(get_available_releases_names()))

    # Print the release years
    available_releases, available_release_years = get_release_years_options(get_available_releases_names(), "resstock")
    print(available_releases)
    print(available_release_years)
    available_releases, available_release_years = get_release_years_options(get_available_releases_names(), "comstock")
    print(available_releases)
    print(available_release_years)

    app()
