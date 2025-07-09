from unittest.mock import patch

from typer.testing import CliRunner

from buildstock_fetch.main_cli import app

# Create a test runner
runner = CliRunner()


@patch("questionary.select")
@patch("questionary.checkbox")
def test_interactive_mode(mock_checkbox, mock_select):
    """Test interactive mode with mocked questionary."""
    # Mock the questionary responses
    mock_select.return_value.ask.side_effect = ["resstock", "2021", "tmy3", "1"]
    mock_checkbox.return_value.ask.return_value = ["MA"]

    result = runner.invoke(app, [])

    # Debug output
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
    print(f"Exception: {result.exception}")

    assert result.exit_code == 0
    assert "BuildStock Fetch Interactive CLI" in result.stdout
    assert "Welcome to the BuildStock Fetch CLI!" in result.stdout
    assert "Please select the release information and file type you would like to fetch:" in result.stdout
    assert "Result: resstock, 2021, tmy3, 1, ['MA']" in result.stdout

    # Mock the questionary responses
    mock_select.return_value.ask.side_effect = ["comstock", "2021", "tmy3", "1"]
    mock_checkbox.return_value.ask.return_value = ["CA", "TX"]

    result = runner.invoke(app, [])

    # Debug output
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
    print(f"Exception: {result.exception}")

    assert result.exit_code == 0
    assert "BuildStock Fetch Interactive CLI" in result.stdout
    assert "Welcome to the BuildStock Fetch CLI!" in result.stdout
    assert "Please select the release information and file type you would like to fetch:" in result.stdout
    assert "Result: comstock, 2021, tmy3, 1, ['CA', 'TX']" in result.stdout
