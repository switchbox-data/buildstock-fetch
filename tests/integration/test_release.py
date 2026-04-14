"""
End-to-end test that installs the library from PyPI and runs all tests.
This test verifies that the published package works correctly.
"""

import importlib.metadata
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.mark.network
def test_pypi_installation_and_tests():
    """Test that the library can be installed from PyPI and all tests pass."""

    # Create a temporary directory for the test environment
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Install the library from PyPI using uv
        install_result = subprocess.run(
            ["uv", "pip", "install", "buildstock-fetch"], capture_output=True, text=True, cwd=temp_path
        )

        # Check if installation was successful
        assert install_result.returncode == 0, f"Installation failed: {install_result.stderr}"

        # Verify the package is installed
        import_result = subprocess.run(
            [sys.executable, "-c", "import buildstock_fetch; print('Import successful')"],
            capture_output=True,
            text=True,
            cwd=temp_path,
        )

        assert import_result.returncode == 0, f"Import failed: {import_result.stderr}"

        # Test that the CLI command works
        cli_result = subprocess.run(
            [sys.executable, "-m", "buildstock_fetch", "--help"], capture_output=True, text=True, cwd=temp_path
        )

        assert cli_result.returncode == 0, f"CLI help failed: {cli_result.stderr}"


@pytest.mark.network
def test_slow_run_all_tests():
    """Run all tests in the test directory using pytest with the PyPI-installed package."""

    # Get the local library version
    local_library_version = importlib.metadata.version("buildstock-fetch")

    # Get the PyPI version by querying PyPI API
    try:
        with urllib.request.urlopen("https://pypi.org/pypi/buildstock-fetch/json") as response:
            pypi_data = json.loads(response.read())
            pypi_library_version = pypi_data["info"]["version"]
    except Exception:
        # Fallback: Install from PyPI in a temp environment to get the version
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Install the library from PyPI
            install_result = subprocess.run(
                ["uv", "pip", "install", "buildstock-fetch"],
                capture_output=True,
                text=True,
                cwd=temp_path,
            )

            assert install_result.returncode == 0, f"Installation failed: {install_result.stderr}"

            # Get the version from the PyPI-installed package
            version_result = subprocess.run(
                [sys.executable, "-c", "import buildstock_fetch; print(buildstock_fetch.__version__)"],
                capture_output=True,
                text=True,
                cwd=temp_path,
            )

            assert version_result.returncode == 0, f"Version retrieval failed: {version_result.stderr}"

            pypi_library_version = version_result.stdout.strip()

    # Only run these tests if the local library version is different from the PyPI library version
    if local_library_version == pypi_library_version:
        assert True
        return

    # Create a temporary directory for the test environment
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Install the library from PyPI in the temp environment
        install_result = subprocess.run(
            ["uv", "pip", "install", "buildstock-fetch"], capture_output=True, text=True, cwd=temp_path
        )

        # Check if installation was successful
        assert install_result.returncode == 0, f"Installation failed: {install_result.stderr}"

        # Copy test files to temp directory (excluding this file to avoid recursion)
        integration_test_dir = Path(__file__).parent
        root_test_dir = integration_test_dir.parent
        temp_test_dir = temp_path / "tests"
        temp_test_dir.mkdir()

        # Copy conftest.py first (needed for fixtures)
        conftest_file = root_test_dir / "conftest.py"
        if conftest_file.exists():
            shutil.copy2(conftest_file, temp_test_dir)

        # Copy specific test files
        specific_tests = [
            integration_test_dir / "test_main_.py",
            integration_test_dir / "test_main_cli.py",
            root_test_dir / "test_building.py",
            root_test_dir / "test_types.py",
            integration_test_dir / "test_read.py",
            root_test_dir / "test_mixed_upgrade.py",
            root_test_dir / "test_scenarios.py",
        ]
        for test_file in specific_tests:
            if test_file.exists():
                shutil.copy2(test_file, temp_test_dir)

        # Run pytest on the copied test files in the temp environment
        test_result = subprocess.run(
            [sys.executable, "-m", "pytest", str(temp_test_dir), "-v", "--disable-recording"],
            capture_output=True,
            text=True,
            cwd=temp_path,
        )

        # Check if tests passed
        assert test_result.returncode == 0, (
            f"Tests failed with return code {test_result.returncode}. Output: {test_result.stdout}. Errors: {test_result.stderr}"
        )
