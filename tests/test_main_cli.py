from unittest.mock import patch

from typer.testing import CliRunner

from buildstock_fetch.main_cli import app

# Create a test runner
runner = CliRunner()


@patch("questionary.select")
def test_interactive_mode(mock_questionary):
    """Test interactive mode with mocked questionary."""
    # Mock the questionary responses
    mock_questionary.return_value.ask.side_effect = ["resttock", "2022", "tmy3"]

    result = runner.invoke(app, [])

    # Debug output
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
    print(f"Exception: {result.exception}")

    assert result.exit_code == 0
    assert "BuildStock Fetch Interactive CLI" in result.stdout
    assert "Welcome to the BuildStock Fetch CLI!" in result.stdout
    assert "Please select the release information and file type you would like to fetch:" in result.stdout
    assert "Result: resttock, 2022, tmy3" in result.stdout
