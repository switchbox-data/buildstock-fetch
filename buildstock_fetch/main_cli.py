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


def _get_release_version_options() -> list[str]:
    return ["2021", "2022", "2023", "2024", "2025"]


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


def _get_available_release_options() -> list[str]:
    return ["2021", "2022", "2023", "2024", "2025"]


def main_callback(
    release_version: int = typer.Option(None, "--release_version", "-r"),
    state: str = typer.Option(None, "--state", "-s"),
    file_type: str = typer.Option(None, "--file_type", "-f"),
) -> None:
    """
    DBF CLI tool. Run without arguments for interactive mode.
    """

    # If no arguments provided, run interactive mode
    if not any([release_version, state, file_type]):
        console.print(Panel("BuildStock Fetch Interactive CLI", title="BuildStock Fetch CLI", border_style="blue"))
        console.print("Welcome to the BuildStock Fetch CLI!")
        console.print("This tool allows you to fetch data from the NREL BuildStock API.")
        console.print("Please select the release information and file type you would like to fetch:")

        # Release version options
        release_version = questionary.select(
            "Select release version:", choices=["2021", "2022", "2023", "2024", "2025"]
        ).ask()

        # State options (all US states)
        state = questionary.select(
            "Select state:",
            choices=[
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
            ],
        ).ask()

        # File type options
        file_type = questionary.select(
            "Select file type:",
            choices=[
                "metadata",
                "15min_load_curve",
                "hourly_load_curve",
                "monthly_load_curve",
                "daily_load_curve",
                "annual_load_curve",
            ],
        ).ask()

    # Process the data
    print(f"Result: {release_version}, {state}, {file_type}")


app.callback(invoke_without_command=True)(main_callback)


if __name__ == "__main__":
    app()
