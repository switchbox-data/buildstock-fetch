from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from buildstock_fetch.main_cli import app

# Create a test runner
runner = CliRunner()


@patch("questionary.path")
@patch("questionary.select")
@patch("questionary.checkbox")
def test_interactive_mode(mock_checkbox, mock_select, mock_path):
    """Test interactive mode with mocked questionary."""
    # Mock the questionary responses
    mock_select.return_value.ask.side_effect = ["resstock", "2021", "tmy3", "1", "0"]
    mock_checkbox.return_value.ask.side_effect = [["CA", "MA"], "metadata"]
    mock_path.return_value.ask.return_value = str(Path.cwd() / "test_output")

    result = runner.invoke(app, [])

    # Debug output
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
    print(f"Exception: {result.exception}")

    assert result.exit_code == 0
    assert "BuildStock Fetch Interactive CLI" in result.stdout
    assert "Welcome to the BuildStock Fetch CLI!" in result.stdout
    assert "Please select the release information and file type you would like to fetch:" in result.stdout
    # Check that the result contains the expected values but with dynamic path
    assert "Result: resstock, 2021, tmy3, 1, 0, ['CA', 'MA'], metadata," in result.stdout
    assert "test_output" in result.stdout

    # Mock the questionary responses. Check for error message when no states are selected
    mock_select.return_value.ask.side_effect = ["resstock", "2021", "tmy3", "1", "0"]
    mock_checkbox.return_value.ask.return_value = []

    result = runner.invoke(app, [])

    # Debug output
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
    print(f"Exception: {result.exception}")

    assert result.exit_code == 1

    # Mock the questionary responses
    mock_select.return_value.ask.side_effect = ["comstock", "2024", "amy2018", "1", "7"]
    mock_checkbox.return_value.ask.side_effect = [["CA", "TX"], ["15min_load_curve", "metadata"]]
    mock_path.return_value.ask.return_value = str(Path.cwd() / "test_output")
    result = runner.invoke(app, [])

    # Debug output
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
    print(f"Exception: {result.exception}")

    assert result.exit_code == 0
    assert "BuildStock Fetch Interactive CLI" in result.stdout
    assert "Welcome to the BuildStock Fetch CLI!" in result.stdout
    assert "Please select the release information and file type you would like to fetch:" in result.stdout
    # Check that the result contains the expected values but with dynamic path
    assert "Result: comstock, 2024, amy2018, 1, 7, ['CA', 'TX'], ['15min_load_curve', 'metadata']," in result.stdout
    assert "test_output" in result.stdout
