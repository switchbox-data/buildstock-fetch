"""
End-to-end test that installs the library from PyPI and runs all tests.
This test verifies that the published package works correctly.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


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


def test_package_functionality():
    """Test basic functionality of the installed package."""

    # Test that we can import the main modules
    try:
        import buildstock_fetch
        from buildstock_fetch.main import fetch_bldg_data, fetch_bldg_ids
        from buildstock_fetch.main_cli import app
    except ImportError as e:
        raise AssertionError(f"Failed to import modules: {e}")

    # Test that the package has the expected attributes
    assert hasattr(buildstock_fetch, "__version__"), "Package should have __version__"

    # Test that the CLI app is available
    assert app is not None, "CLI app should be available"


def test_run_all_tests():
    """Run all tests in the test directory using pytest with the PyPI-installed package."""

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
        test_dir = Path(__file__).parent
        temp_test_dir = temp_path / "tests"
        temp_test_dir.mkdir()

        # Copy specific test files
        specific_tests = ["test_main.py", "test_main_cli.py"]
        for test_name in specific_tests:
            test_file = test_dir / test_name
            if test_file.exists():
                shutil.copy2(test_file, temp_test_dir)

        # Run pytest on the copied test files in the temp environment
        test_result = subprocess.run(
            [sys.executable, "-m", "pytest", str(temp_test_dir), "-v"], capture_output=True, text=True, cwd=temp_path
        )

        # Check if tests passed
        assert test_result.returncode == 0, (
            f"Tests failed with return code {test_result.returncode}. Output: {test_result.stdout}. Errors: {test_result.stderr}"
        )
