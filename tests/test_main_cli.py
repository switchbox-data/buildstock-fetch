import re
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from buildstock_fetch.main_cli import app


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


@pytest.fixture(scope="function")
def cleanup_downloads():
    # Setup - clean up any existing files before test
    data_dir = Path("data")
    test_output_dir = Path("test_output")

    if data_dir.exists():
        shutil.rmtree(data_dir)
    if test_output_dir.exists():
        shutil.rmtree(test_output_dir)

    yield

    # Teardown - clean up downloaded files after test
    if data_dir.exists():
        shutil.rmtree(data_dir)
    if test_output_dir.exists():
        shutil.rmtree(test_output_dir)


# Create a test runner
runner = CliRunner()


@patch("questionary.confirm")
@patch("questionary.path")
@patch("questionary.select")
@patch("questionary.checkbox")
@patch("questionary.text")
def test_interactive_mode(mock_text, mock_checkbox, mock_select, mock_path, mock_confirm, cleanup_downloads):
    """Test interactive mode with mocked questionary."""
    # Mock the questionary responses
    mock_select.return_value.ask.side_effect = ["resstock", "2021", "tmy3", "1", "Download a sample"]
    mock_checkbox.return_value.ask.side_effect = [["0"], ["CA", "MA"], ["metadata"]]
    mock_path.return_value.ask.return_value = str(Path.cwd() / "test_output")
    mock_confirm.return_value.ask.return_value = True
    # Mock the text input for sample size (not used in this test case but needed)
    mock_text.return_value.ask.return_value = "5"

    result = runner.invoke(app, [])

    # Debug output
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
    print(f"Exception: {result.exception}")

    assert result.exit_code == 0
    assert "BuildStock Fetch Interactive CLI" in result.stdout
    assert "Welcome to the BuildStock Fetch CLI!" in result.stdout
    assert "Please select the release information and file type you would like to fetch:" in result.stdout
    # Check that the result contains the expected values but with dynamic path and new print format
    assert "Downloading data for:" in result.stdout
    assert "Product: resstock" in result.stdout
    assert "Release year: 2021" in result.stdout
    assert "Weather file: tmy3" in result.stdout
    assert "Release version: 1" in result.stdout
    assert "States: ['CA', 'MA']" in result.stdout
    assert "File type: ['metadata']" in result.stdout
    assert "Upgrade ids: ['0']" in result.stdout
    assert "test_output" in result.stdout

    # Test abort path: confirmation returns False
    mock_select.return_value.ask.side_effect = ["resstock", "2021", "tmy3", "1", "Download a sample"]
    mock_checkbox.return_value.ask.side_effect = [["0"], ["CA", "MA"], ["metadata"]]
    mock_path.return_value.ask.return_value = str(Path.cwd() / "test_output")
    mock_confirm.return_value.ask.return_value = False

    result = runner.invoke(app, [])

    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
    print(f"Exception: {result.exception}")

    assert result.exit_code == 1

    # Mock the questionary responses for another valid run
    mock_select.return_value.ask.side_effect = ["comstock", "2024", "amy2018", "1", "Download a sample"]
    mock_checkbox.return_value.ask.side_effect = [["7"], ["CA", "TX"], ["load_curve_15min", "metadata"]]
    mock_path.return_value.ask.return_value = str(Path.cwd() / "test_output")
    mock_confirm.return_value.ask.return_value = True
    result = runner.invoke(app, [])

    # Debug output
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
    print(f"Exception: {result.exception}")

    assert result.exit_code == 0
    assert "BuildStock Fetch Interactive CLI" in result.stdout
    assert "Welcome to the BuildStock Fetch CLI!" in result.stdout
    assert "Please select the release information and file type you would like to fetch:" in result.stdout
    # Check that the result contains the expected values but with dynamic path and new print format
    assert "Downloading data for:" in result.stdout
    assert "Product: comstock" in result.stdout
    assert "Release year: 2024" in result.stdout
    assert "Weather file: amy2018" in result.stdout
    assert "Release version: 1" in result.stdout
    assert "States: ['CA', 'TX']" in result.stdout
    assert "File type: ['load_curve_15min', 'metadata']" in result.stdout
    assert "Upgrade ids: ['7']" in result.stdout
    assert "test_output" in result.stdout

    # Mock the questionary responses for another valid run
    mock_select.return_value.ask.side_effect = ["resstock", "2024", "amy2018", "2", "Download a sample"]
    mock_checkbox.return_value.ask.side_effect = [["7"], ["CA", "TX"], ["load_curve_monthly", "metadata"]]
    mock_path.return_value.ask.return_value = str(Path.cwd() / "test_output")
    mock_confirm.return_value.ask.return_value = True
    result = runner.invoke(app, [])

    # Debug output
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
    print(f"Exception: {result.exception}")

    assert result.exit_code == 0
    assert "BuildStock Fetch Interactive CLI" in result.stdout
    assert "Welcome to the BuildStock Fetch CLI!" in result.stdout
    assert "Please select the release information and file type you would like to fetch:" in result.stdout
    # Check that the result contains the expected values but with dynamic path and new print format
    assert "Downloading data for:" in result.stdout
    assert "Product: resstock" in result.stdout
    assert "Release year: 2024" in result.stdout
    assert "Weather file: amy2018" in result.stdout
    assert "Release version: 2" in result.stdout
    assert "States: ['CA', 'TX']" in result.stdout
    assert "File type: ['load_curve_monthly', 'metadata']" in result.stdout
    assert "Upgrade ids: ['7']" in result.stdout
    assert "test_output" in result.stdout

    # Test weather file selection
    mock_select.return_value.ask.side_effect = ["resstock", "2022", "amy2012", "1", "Download a sample"]
    mock_checkbox.return_value.ask.side_effect = [["0"], ["NY"], ["weather"]]
    mock_path.return_value.ask.return_value = str(Path.cwd() / "test_output")
    mock_confirm.return_value.ask.return_value = True
    result = runner.invoke(app, [])

    # Debug output
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
    print(f"Exception: {result.exception}")

    assert result.exit_code == 0
    assert "BuildStock Fetch Interactive CLI" in result.stdout
    assert "Welcome to the BuildStock Fetch CLI!" in result.stdout
    assert "Please select the release information and file type you would like to fetch:" in result.stdout
    # Check that the result contains the expected values but with dynamic path and new print format
    assert "Downloading data for:" in result.stdout
    assert "Product: resstock" in result.stdout
    assert "Release year: 2022" in result.stdout
    assert "Weather file: amy2012" in result.stdout
    assert "Release version: 1" in result.stdout
    assert "States: ['NY']" in result.stdout
    assert "File type: ['weather']" in result.stdout
    assert "Upgrade ids: ['0']" in result.stdout
    assert "test_output" in result.stdout


@patch("questionary.confirm")
@patch("questionary.path")
@patch("questionary.select")
@patch("questionary.checkbox")
@patch("questionary.text")
def test_interactive_mode_sample_download(
    mock_text, mock_checkbox, mock_select, mock_path, mock_confirm, cleanup_downloads
):
    """Test interactive mode with sample download option."""
    # Mock the questionary responses for sample download
    mock_select.return_value.ask.side_effect = ["resstock", "2021", "tmy3", "1", "Download a sample"]
    mock_checkbox.return_value.ask.side_effect = [["0"], ["CA"], ["metadata"]]
    mock_path.return_value.ask.return_value = str(Path.cwd() / "test_output")
    mock_confirm.return_value.ask.return_value = True
    # Mock the text input for sample size (for upgrade 0)
    mock_text.return_value.ask.return_value = "3"

    result = runner.invoke(app, [])

    # Debug output
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
    print(f"Exception: {result.exception}")

    assert result.exit_code == 0
    assert "BuildStock Fetch Interactive CLI" in result.stdout
    assert "Welcome to the BuildStock Fetch CLI!" in result.stdout
    assert "Downloading data for:" in result.stdout
    assert "Product: resstock" in result.stdout
    assert "Release year: 2021" in result.stdout
    assert "Weather file: tmy3" in result.stdout
    assert "Release version: 1" in result.stdout
    assert "States: ['CA']" in result.stdout
    assert "File type: ['metadata']" in result.stdout
    assert "Upgrade ids: ['0']" in result.stdout
    assert "test_output" in result.stdout
    # Check for sample download messages
    assert "files for this release" in result.stdout
    assert "Selected 3 buildings for State CA, Upgrade 0" in result.stdout


@patch("questionary.confirm")
@patch("questionary.path")
@patch("questionary.select")
@patch("questionary.checkbox")
@patch("questionary.text")
def test_interactive_mode_zero_sample(
    mock_text, mock_checkbox, mock_select, mock_path, mock_confirm, cleanup_downloads
):
    """Test interactive mode with zero sample download option."""
    # Mock the questionary responses for zero sample download
    mock_select.return_value.ask.side_effect = ["resstock", "2021", "tmy3", "1", "Download a sample"]
    mock_checkbox.return_value.ask.side_effect = [["0"], ["CA"], ["metadata"]]
    mock_path.return_value.ask.return_value = str(Path.cwd() / "test_output")
    mock_confirm.return_value.ask.return_value = True
    # Mock the text input for sample size (0 for upgrade 0)
    mock_text.return_value.ask.return_value = "0"

    result = runner.invoke(app, [])

    # Debug output
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
    print(f"Exception: {result.exception}")

    assert result.exit_code == 0
    assert "BuildStock Fetch Interactive CLI" in result.stdout
    assert "Welcome to the BuildStock Fetch CLI!" in result.stdout
    assert "Downloading data for:" in result.stdout
    assert "Product: resstock" in result.stdout
    assert "Release year: 2021" in result.stdout
    assert "Weather file: tmy3" in result.stdout
    assert "Release version: 1" in result.stdout
    assert "States: ['CA']" in result.stdout
    assert "File type: ['metadata']" in result.stdout
    assert "Upgrade ids: ['0']" in result.stdout
    assert "test_output" in result.stdout
    # Check for zero sample download messages
    assert "files for this release" in result.stdout
    assert "No files will be downloaded for State CA, Upgrade 0" in result.stdout
    assert "No files selected for download" in result.stdout


def test_cli_direct_arguments(cleanup_downloads):
    """Test CLI with direct command line arguments (non-interactive mode)."""
    # Test with direct arguments for metadata download
    result = runner.invoke(
        app,
        [
            "--product",
            "resstock",
            "--release_year",
            "2021",
            "--weather_file",
            "tmy3",
            "--release_version",
            "1",
            "--states",
            "CA",
            "--file_type",
            "metadata",
            "--upgrade_id",
            "0",
            "--output_directory",
            "test_output",
        ],
    )

    # Debug output
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
    print(f"Exception: {result.exception}")

    assert "Downloading data for:" in result.stdout
    assert "Product: resstock" in result.stdout
    assert "Release year: 2021" in result.stdout
    assert "Weather file: tmy3" in result.stdout
    assert "Release version: 1" in result.stdout
    assert "States: ['CA']" in result.stdout
    assert "File type: ['metadata']" in result.stdout
    assert "Upgrade ids: ['0']" in result.stdout
    assert "test_output" in result.stdout


def test_cli_multiple_file_types(cleanup_downloads):
    """Test CLI with multiple file types."""
    result = runner.invoke(
        app,
        [
            "--product",
            "comstock",
            "--release_year",
            "2023",
            "--weather_file",
            "amy2018",
            "--release_version",
            "2",
            "--states",
            "CA TX",
            "--file_type",
            "load_curve_15min metadata",
            "--upgrade_id",
            "7",
            "--output_directory",
            "test_output",
        ],
    )

    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
    print(f"Exception: {result.exception}")

    assert "Downloading data for:" in result.stdout
    assert "Product: comstock" in result.stdout
    assert "Release year: 2023" in result.stdout
    assert "Weather file: amy2018" in result.stdout
    assert "Release version: 2" in result.stdout
    assert "States: ['CA', 'TX']" in result.stdout
    assert "File type: ['load_curve_15min', 'metadata']" in result.stdout
    assert "Upgrade ids: ['7']" in result.stdout


def test_cli_multiple_upgrade_ids(cleanup_downloads):
    """Test CLI with multiple upgrade IDs."""
    result = runner.invoke(
        app,
        [
            "--product",
            "resstock",
            "--release_year",
            "2022",
            "--weather_file",
            "tmy3",
            "--release_version",
            "1",
            "--states",
            "NY",
            "--file_type",
            "hpxml schedule",
            "--upgrade_id",
            "0 1 2",
            "--output_directory",
            "test_output",
        ],
    )

    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
    print(f"Exception: {result.exception}")

    assert "Downloading data for:" in result.stdout
    assert "Product: resstock" in result.stdout
    assert "Release year: 2022" in result.stdout
    assert "Weather file: tmy3" in result.stdout
    assert "Release version: 1" in result.stdout
    assert "States: ['NY']" in result.stdout
    assert "File type: ['hpxml', 'schedule']" in result.stdout
    assert "Upgrade ids: ['0', '1', '2']" in result.stdout


def test_cli_invalid_arguments(cleanup_downloads):
    """Test CLI with invalid arguments."""
    # Test with invalid product
    result = runner.invoke(
        app,
        [
            "--product",
            "invalid_product",
            "--release_year",
            "2021",
            "--weather_file",
            "tmy3",
            "--release_version",
            "1",
            "--states",
            "CA",
            "--file_type",
            "metadata",
            "--output_directory",
            "test_output",
        ],
    )

    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
    print(f"Exception: {result.exception}")

    assert result.exit_code == 1  # Invalid product should exit with code 1
    assert "Invalid product" in result.stdout

    # Test with invalid release year
    result = runner.invoke(
        app,
        [
            "--product",
            "resstock",
            "--release_year",
            "2000",  # Invalid year
            "--weather_file",
            "tmy3",
            "--release_version",
            "1",
            "--states",
            "CA",
            "--file_type",
            "metadata",
            "--output_directory",
            "test_output",
        ],
    )

    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
    print(f"Exception: {result.exception}")

    assert result.exit_code == 1
    assert "Invalid release name" in result.stdout

    # Test with invalid state
    result = runner.invoke(
        app,
        [
            "--product",
            "resstock",
            "--release_year",
            "2021",
            "--weather_file",
            "tmy3",
            "--release_version",
            "1",
            "--states",
            "ABCDEFG",  # Invalid state
            "--file_type",
            "metadata",
            "--output_directory",
            "test_output",
        ],
    )

    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
    print(f"Exception: {result.exception}")

    assert result.exit_code == 1
    assert "Invalid state" in result.stdout


def test_cli_help():
    """Test CLI help functionality."""
    # Test help command
    result = runner.invoke(app, ["--help"])

    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")

    # Strip ANSI escape codes for easier text checking
    clean_output = strip_ansi_codes(result.stdout)

    assert result.exit_code == 0
    assert "Buildstock Fetch CLI tool. Run without arguments for interactive mode." in clean_output
    assert "--product" in clean_output
    assert "--release_year" in clean_output
    assert "--weather_file" in clean_output
    assert "--release_version" in clean_output
    assert "--states" in clean_output
    assert "--file_type" in clean_output
    assert "--upgrade_id" in clean_output
    assert "--output_directory" in clean_output
