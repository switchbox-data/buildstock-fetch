import importlib.metadata
import pprint
import re
from pathlib import Path
from typing import Annotated, NamedTuple, cast, get_args

import questionary
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typing_extensions import Never

from buildstock_fetch.building import BuildingID
from buildstock_fetch.main import fetch_bldg_data, fetch_bldg_ids
from buildstock_fetch.types import (
    FileType,
    ReleaseVersion,
    ReleaseYear,
    ResCom,
    Sample,
    UpgradeID,
    USStateCode,
    Weather,
    is_valid_state_code,
)

from .inputs import InputsFinal, InputsMaybe
from .releases import BuildstockReleases

# File types that haven't been implemented yet
UNAVAILABLE_FILE_TYPES: list[str] = []
console = Console()


app = typer.Typer(
    name="buildstock-fetch",
    help="A CLI tool to fetch data from the BuildStock API",
    rich_markup_mode="rich",
    no_args_is_help=False,
)


def _show_app_version(value: bool) -> None:
    if not value:
        return
    try:
        version = importlib.metadata.version("buildstock-fetch")
    except importlib.metadata.PackageNotFoundError:
        console.print("Package not found")
    else:
        console.print(f"buildstock-fetch version {version}")
    raise typer.Exit()


class BuildingsGroup(NamedTuple):
    state: USStateCode
    upgrade_id: UpgradeID
    buildings: list[BuildingID]


@app.command()
def main(  # noqa: C901
    _: Annotated[bool | None, typer.Option("--version", callback=_show_app_version)] = None,
    product: Annotated[ResCom | None, typer.Option("--product", "-p", help='"resstock" or "comstock"')] = None,
    release_year: Annotated[
        ReleaseYear | None, typer.Option("--release_year", "-y", help="Release year (typically 2021 or later)")
    ] = None,
    release_version: Annotated[
        ReleaseVersion | None,
        typer.Option(
            "--release_version",
            "-r",
        ),
    ] = None,
    weather_file: Annotated[Weather | None, typer.Option("--weather_file", "-w")] = None,
    states_raw: Annotated[
        str | None,
        typer.Option(
            "--states",
            "-st",
            help="List of states (multiple can be provided, inside quotes and separated by spaces)",
        ),
    ] = None,
    file_types_raw: Annotated[
        str | None,
        typer.Option(
            "--file_type",
            "-f",
            help="List of file types (multiple can be provided, inside quotes and separated by spaces)",
        ),
    ] = None,
    upgrade_id_raw: Annotated[
        str | None,
        typer.Option(
            "--upgrade_id", "-u", help="Upgrade IDs (multiple can be provided, inside quotes and separated by spaces)"
        ),
    ] = None,
    output_directory: Annotated[
        Path | None, typer.Option("--output_directory", "-o", help='e.g., "data" or "../output"')
    ] = None,
    sample_raw: Annotated[
        int | None,
        typer.Option(
            "--sample",
            "-sm",
            help="Number of building IDs to download across all upgrades. Use 0 for all buildings.",
        ),
    ] = None,
    threads: Annotated[
        int, typer.Option("--threads", "-t", help="Number of files to download at the same time", min=1, max=50)
    ] = 15,
) -> None:
    states = _parse_and_validate_states(states_raw) if states_raw else None
    file_types = _parse_and_validate_file_types(file_types_raw) if file_types_raw else None
    upgrade_ids = _parse_and_validate_upgrade_ids(upgrade_id_raw) if upgrade_id_raw else None
    sample = _validate_sample(sample_raw) if sample_raw is not None else None

    inputs = InputsMaybe(
        product=product,
        release_year=release_year,
        release_version=release_version,
        weather_file=weather_file,
        states=states,
        file_types=file_types,
        upgrade_ids=upgrade_ids,
        output_directory=output_directory,
    )

    inputs_finalized = inputs.is_finalized()

    if not (inputs_finalized or sample):
        console.print(Panel("BuildStock Fetch Interactive CLI", title="BuildStock Fetch CLI", border_style="blue"))
        console.print("Welcome to the BuildStock Fetch CLI!")
        console.print("This tool allows you to fetch data from the NREL BuildStock API.")
        console.print("Please select the release information and file type you would like to fetch:")

    releases = BuildstockReleases.from_json()

    if inputs.product is None:
        inputs.product = select_product(releases, inputs)

    if inputs.release_year is None:
        inputs.release_year = select_release_year(releases, inputs)

    if inputs.weather_file is None:
        inputs.weather_file = select_weather_file(releases, inputs)

    if inputs.release_version is None:
        inputs.release_version = select_release_version(releases, inputs)

    if inputs.upgrade_ids is None:
        inputs.upgrade_ids = select_upgrade_ids(releases, inputs)

    if inputs.states is None:
        inputs.states = select_states()

    if inputs.file_types is None:
        inputs.file_types = select_file_types(releases, inputs)

    if inputs.output_directory is None:
        inputs.output_directory = select_output_directory()

    inputs_final = InputsFinal.from_finalized_maybe(inputs)

    if not inputs_finalized:
        verify_inputs(inputs_final)
        display_cli_args(inputs_final, threads, sample)

    display_download_parameters(inputs_final)
    release = releases.filter_one(inputs)

    if "trip_schedules" in inputs.file_types:
        for state in inputs.states:
            if state not in release.trip_schedule_states:
                console.print(f"[yellow]The following state is not available for trip schedules: {state}[/yellow]")

    building_groups = fetch_building_groups(inputs_final)
    if sample is not None:
        buildings = set()
        for group in building_groups:
            buildings |= set(group.buildings[: -1 if sample == "all" else sample])
    else:
        buildings = get_buildings_sample(building_groups)

    fetch_bldg_data(
        list(buildings),
        tuple(inputs_final.file_types),
        inputs_final.output_directory,
        max_workers=threads,
        weather_states=[state for state in inputs_final.states if state in release.weather_map_available_states],
    )


def select_product(releases: BuildstockReleases, inputs: InputsMaybe) -> ResCom:
    available_releases = releases.filter_at_least_one(inputs)
    products = sorted(available_releases.products)
    products.reverse()
    result = questionary.select("Select product type", choices=products).ask()
    return cast(ResCom, result) or cancel()


def select_release_year(releases: BuildstockReleases, inputs: InputsMaybe) -> ReleaseYear:
    available_releases = releases.filter_at_least_one(inputs)
    years = sorted(available_releases.release_years)
    years.reverse()
    result = questionary.select("Select release year:", choices=years).ask()
    return cast(ReleaseYear, result) or cancel()


def select_weather_file(releases: BuildstockReleases, inputs: InputsMaybe) -> Weather:
    available_releases = releases.filter_at_least_one(inputs)
    desired_order = ["tmy3", "amy2018", "amy2012"]
    weathers = sorted(available_releases.weathers, key=lambda _: desired_order.index(_))
    result = questionary.select("Select weather file", choices=weathers).ask()
    return cast(Weather, result) or cancel()


def select_release_version(releases: BuildstockReleases, inputs: InputsMaybe) -> ReleaseVersion:
    available_releases = releases.filter_at_least_one(inputs)
    versions = set(available_releases.release_versions)
    if (
        inputs.product == "resstock"
        and inputs.weather_file == "tmy3"
        and inputs.release_year == "2024"
        and "1" in versions
    ):
        versions.remove("1")
    desired_order = ["2", "1.1", "1"]
    versions_sorted = sorted(versions, key=lambda _: desired_order.index(_))
    result = questionary.select("Select release version:", choices=versions_sorted).ask()
    return cast(ReleaseVersion, result) or cancel()


def select_upgrade_ids(releases: BuildstockReleases, inputs: InputsMaybe) -> set[UpgradeID]:
    release = releases.filter_one(inputs)
    upgrades_sorted = sorted(release.upgrades, key=lambda _: int(_.id))
    choices = [
        questionary.Choice(
            title=f"{upgrade.id}: {upgrade.description}" if upgrade.description else str(upgrade.id), value=upgrade.id
        )
        for upgrade in upgrades_sorted
    ]
    result = questionary.checkbox(
        "Select upgrade ids:",
        choices=choices,
        instruction="Use spacebar to select/deselect options, 'a' to select all, 'i' to invert selection, enter to confirm",
        validate=lambda answer: "You must select at least one upgrade id" if len(answer) == 0 else True,
    ).ask()
    if not result:
        cancel()
    return set(result)


def select_states() -> set[USStateCode]:
    states = sorted(get_args(USStateCode))
    result = questionary.checkbox(
        "Select states:",
        choices=states,
        instruction="Use spacebar to select/deselect options, enter to confirm",
        validate=lambda answer: "You must select at least one state" if len(answer) == 0 else True,
    ).ask()
    if not result:
        cancel()
    return set(result)


def select_file_types(releases: BuildstockReleases, inputs: InputsMaybe) -> set[FileType]:
    available_releases = releases.filter_at_least_one(inputs)
    choices = []
    if "metadata" in available_releases.file_types:
        choices.append(_category_choice("Metadata"))
        choices.append(_filetype_choice("metadata"))

    end_use_load_curves = (
        "load_curve_15min",
        "load_curve_hourly",
        "load_curve_daily",
        "load_curve_monthly",
        "load_curve_annual",
    )
    if set(end_use_load_curves) & available_releases.file_types:
        choices.append(_category_choice("End Use Load Curves"))
        choices.extend(_filetype_choice(ft) for ft in end_use_load_curves if ft in available_releases.file_types)

    simulation_files = (
        "hpxml",
        "schedule",
    )
    if set(simulation_files) & available_releases.file_types:
        choices.append(_category_choice("Simulation Files"))
        choices.extend(_filetype_choice(ft) for ft in simulation_files if ft in available_releases.file_types)

    if "weather" in available_releases.file_types:
        choices.append(_category_choice("Weather"))
        choices.append(_filetype_choice("weather"))

    result = questionary.checkbox(
        "Select file type:",
        choices=choices,
        instruction="Use spacebar to select/deselect options, enter to confirm",
        validate=lambda answer: "You must select at least one file type" if len(answer) == 0 else True,
    ).ask()
    if not result:
        cancel()
    return set(result)


def select_output_directory() -> Path:
    result = questionary.path(
        "Select output directory:",
        default=str(Path.cwd() / "data"),
        only_directories=True,
        validate=_validate_output_directory,
    ).ask()
    return Path(cast(str, result))


def verify_inputs(inputs: InputsFinal) -> bool:
    console = Console()
    table = Table(title="Please verify your selections:")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("product", pprint.pformat(inputs.product))
    table.add_row("release_year", pprint.pformat(inputs.release_year))
    table.add_row("weather_file", pprint.pformat(inputs.weather_file))
    table.add_row("release_version", pprint.pformat(inputs.release_version))
    table.add_row("upgrade_ids", pprint.pformat(sorted(str(_) for _ in inputs.upgrade_ids)))
    table.add_row("states", pprint.pformat(sorted(inputs.states)))
    table.add_row("file_type", pprint.pformat(sorted(inputs.file_types, key=get_args(FileType).index)))
    table.add_row("output_directory", pprint.pformat(inputs.output_directory))

    console.print(Panel(table, border_style="green"))

    try:
        result = questionary.confirm("Are these selections correct?", default=True).ask()
    except KeyboardInterrupt:
        cancel()

    if result is False:
        cancel(1)

    if result is None:
        cancel()

    return bool(result)


def get_buildings_sample(building_groups: list[BuildingsGroup]) -> set[BuildingID]:
    total_buildings = sum(len(_.buildings) for _ in building_groups)
    console.print(f"\nThere are {total_buildings} files for this release")
    for group in building_groups:
        console.print(f"  • State {group.state}, Upgrade {group.upgrade_id}: {len(group.buildings)} buildings")

    choice = questionary.select(
        "Would you like to download all files or a sample of them?",
        choices=["Download all files", "Download a sample"],
    ).ask()

    if not choice:
        cancel()

    result = set()
    if choice == "Download all files":
        for group in building_groups:
            result |= set(group.buildings)
        return result

    for group in building_groups:
        total_for_state_upgrade = len(group.buildings)
        sample_size_str = questionary.text(
            f"Enter the number of files to download for State {group.state}, Upgrade {group.upgrade_id} (0-{total_for_state_upgrade}):",
            validate=lambda text, max_val=total_for_state_upgrade: (text.isdigit() and 0 <= int(text) <= max_val)
            or f"Please enter a number between 0 and {max_val}",
        ).ask()

        if sample_size_str is None:
            cancel()

        sample_size = int(sample_size_str)
        if sample_size == 0:
            console.print(
                f"[yellow]No files will be downloaded for State {group.state}, Upgrade {group.upgrade_id}.[/yellow]"
            )
            continue
        selected_for_state_upgrade = group.buildings[:sample_size]
        result |= set(selected_for_state_upgrade)
        console.print(
            f"[green]Selected {sample_size} buildings for State {group.state}, Upgrade {group.upgrade_id}.[/green]"
        )
    if not result:
        console.print("[yellow]No files selected for download.[/yellow]")
    return result


def display_cli_args(inputs: InputsFinal, threads: int, sample: Sample | None) -> None:
    console.print("[blue]Paste this in the command line to launch the script again with the same arguments[/]")
    display = (
        "bsf \\\n\t"
        f"--product {inputs.product}\\\n\t"
        f"--release_year {inputs.release_year}\\\n\t"
        f"--weather_file {inputs.weather_file}\\\n\t"
        f"--release_version {inputs.release_version}\\\n\t"
        f'--states "{" ".join(sorted(inputs.states))}"\\\n\t'
        f'--file_type "{" ".join(sorted(inputs.file_types, key=get_args(FileType).index))}"\\\n\t'
        f'--upgrade_id "{" ".join(map(str, sorted(inputs.upgrade_ids)))}"\\\n\t'
        f'--output_directory "{inputs.output_directory.as_posix()}"\\\n\t'
        f"--threads {threads}"
    )
    if sample is not None:
        display += f"\\\n\t--sample {0 if sample == 'all' else sample}"
    console.print(display)
    questionary.press_any_key_to_continue().ask()


def display_download_parameters(inputs: InputsFinal) -> None:
    display = (
        "Downloading data for:\n"
        f"Product: {inputs.product}\n"
        f"Release year: {inputs.release_year}\n"
        f"Weather file: {inputs.weather_file}\n"
        f"Release version: {inputs.release_version}\n"
        f"States: {sorted(inputs.states)}\n"
        f"File type: {sorted(inputs.file_types, key=get_args(FileType).index)}\n"
        f"Upgrade ids: {[str(_) for _ in sorted(inputs.upgrade_ids)]}\n"
        f"Output directory: {inputs.output_directory}\n"
    )
    console.print(display)


def fetch_building_groups(inputs: InputsFinal) -> list[BuildingsGroup]:
    return [
        BuildingsGroup(
            state,  # type: ignore[invalid-argument-type]  # sorted() loses Literal type, but value is USStateCode
            upgrade_id,
            fetch_bldg_ids(
                inputs.product, inputs.release_year, inputs.weather_file, inputs.release_version, state, str(upgrade_id)
            ),
        )
        for state in sorted(inputs.states)
        for upgrade_id in sorted(inputs.upgrade_ids)
    ]


def _category_choice(name: str) -> questionary.Choice:
    return questionary.Choice(title=f"--- {name} ---", value=None, disabled=True)  # type: ignore[arg-type]


def _filetype_choice(file_type: FileType) -> questionary.Choice:
    return questionary.Choice(title=file_type, value=file_type)


def _validate_output_directory(output_directory: str) -> bool | str:
    try:
        path = Path(output_directory)
        path.resolve()
    except (OSError, ValueError):
        return "Please enter a valid directory path"
    else:
        return True


def _parse_and_validate_states(value: str) -> set[USStateCode]:
    states = [v for _ in re.split(" +", value) if (v := _.upper().strip())]
    bad_states = [state for state in states if not is_valid_state_code(state)]
    if bad_states:
        raise typer.BadParameter(message=", ".join(bad_states), param_hint="states")
    return cast(set[USStateCode], states)


def _parse_and_validate_file_types(value: str) -> set[FileType]:
    allowed_values = get_args(FileType)
    file_types_raw = [_.strip() for _ in re.split(" +", value)]
    bad_file_types = [_ for _ in file_types_raw if _ not in allowed_values]
    if bad_file_types:
        raise typer.BadParameter(message=", ".join(bad_file_types), param_hint="file_type")
    return cast(set[FileType], set(file_types_raw))


def _parse_and_validate_upgrade_ids(value: str) -> set[UpgradeID]:
    return {UpgradeID(_.strip()) for _ in re.split(" +", value)}


def _validate_sample(value: int) -> Sample:
    if value < 0:
        raise typer.BadParameter(message=str(value), param_hint="sample")
    if value == 0:
        return "all"
    return value


def cancel(result: int = 0, message: str = "Operation cancelled by user.") -> Never:
    console.print(f"\n[red]{message}[/red]")
    raise typer.Exit(result) from None


if __name__ == "__main__":
    app()
