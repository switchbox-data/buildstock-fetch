import re
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from buildstock_fetch.cli.main import app


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


@pytest.fixture(scope="function")
def cleanup_downloads():
    data_dir = Path("data")
    test_output_dir = Path("test_output")

    if data_dir.exists():
        shutil.rmtree(data_dir)
    if test_output_dir.exists():
        shutil.rmtree(test_output_dir)

    yield

    if data_dir.exists():
        shutil.rmtree(data_dir)
    if test_output_dir.exists():
        shutil.rmtree(test_output_dir)


runner = CliRunner()


@patch("questionary.press_any_key_to_continue", new=Mock())
@patch("questionary.confirm")
@patch("questionary.path")
@patch("questionary.select")
@patch("questionary.checkbox")
@patch("questionary.text")
def test_interactive_mode_zero_sample(
    mock_text, mock_checkbox, mock_select, mock_path, mock_confirm, cleanup_downloads
):
    """Test interactive mode with zero sample download option."""
    mock_select.return_value.ask.side_effect = ["resstock", "2021", "tmy3", "1", "Download a sample"]
    mock_checkbox.return_value.ask.side_effect = [["0"], ["CA"], ["metadata"]]
    mock_path.return_value.ask.return_value = str(Path.cwd() / "test_output")
    mock_confirm.return_value.ask.return_value = True
    mock_text.return_value.ask.return_value = "0"

    result = runner.invoke(app, [])

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
    assert "files for this release" in result.stdout
    assert "No files will be downloaded for State CA, Upgrade 0" in result.stdout
    assert "No files selected for download" in result.stdout


def test_cli_invalid_arguments(cleanup_downloads):
    """Test CLI with invalid arguments."""
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
    assert result.exit_code == 2

    result = runner.invoke(
        app,
        [
            "--product",
            "resstock",
            "--release_year",
            "2000",
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
    assert result.exit_code == 2

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
            "ABCDEFG",
            "--file_type",
            "metadata",
            "--output_directory",
            "test_output",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for states" in result.stderr


def test_cli_help():
    """Test CLI help functionality."""
    result = runner.invoke(app, ["--help"])
    clean_output = strip_ansi_codes(result.stdout)

    assert result.exit_code == 0
    assert "--product" in clean_output
    assert "--release_year" in clean_output
    assert "--weather_file" in clean_output
    assert "--release_version" in clean_output
    assert "--states" in clean_output
    assert "--file_type" in clean_output
    assert "--upgrade_id" in clean_output
    assert "--output_directory" in clean_output
