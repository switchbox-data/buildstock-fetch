"""Shared pytest fixtures for all test modules."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from buildstock_fetch.main import fetch_bldg_data, fetch_bldg_ids


@pytest.fixture(scope="function")
def cleanup_downloads():
    """Fixture to provide a temporary directory for test downloads.

    This fixture:
    1. Creates a temporary directory for test data
    2. Yields the path to the test
    3. Automatically cleans up the temporary directory after the test completes

    This ensures each test starts with a clean slate and doesn't interfere with
    any work-in-progress data in the working directory.
    """
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="module")
def integration_test_data():
    """Download shared test data for integration tests.

    Downloads metadata and 15min load curves for:
    - 10 buildings (5 from NY, 5 from AL)
    - Release: res_2024_tmy3_2
    - Upgrades: 0, 4, 8

    Uses a temporary directory to avoid interfering with work-in-progress data.
    The temporary directory is cleaned up automatically after tests complete.
    """
    tmpdir = TemporaryDirectory()
    try:
        tmp_path = Path(tmpdir.name)
        test_data_dir = tmp_path / "data"
        test_outputs_dir = tmp_path / "outputs"
        test_data_dir.mkdir(parents=True, exist_ok=True)
        test_outputs_dir.mkdir(parents=True, exist_ok=True)
        # Fetch building IDs for NY and AL
        ny_bldg_ids_upgrade0 = fetch_bldg_ids(
            product="resstock",
            release_year="2024",
            weather_file="tmy3",
            release_version="2",
            state="NY",
            upgrade_id="0",
        )

        al_bldg_ids_upgrade0 = fetch_bldg_ids(
            product="resstock",
            release_year="2024",
            weather_file="tmy3",
            release_version="2",
            state="AL",
            upgrade_id="0",
        )

        # Select first 5 from each state
        ny_bldgs = ny_bldg_ids_upgrade0[:5]
        al_bldgs = al_bldg_ids_upgrade0[:5]

        # Create building IDs for all upgrade combinations
        bldg_ids_to_download = []
        for upgrade in ["0", "4", "8"]:
            for bldg in ny_bldgs:
                bldg_ids_to_download.append(
                    type(bldg)(
                        bldg_id=bldg.bldg_id,
                        release_number="2",
                        release_year="2024",
                        res_com="resstock",
                        weather="tmy3",
                        upgrade_id=upgrade,
                        state="NY",
                    )
                )
            for bldg in al_bldgs:
                bldg_ids_to_download.append(
                    type(bldg)(
                        bldg_id=bldg.bldg_id,
                        release_number="2",
                        release_year="2024",
                        res_com="resstock",
                        weather="tmy3",
                        upgrade_id=upgrade,
                        state="AL",
                    )
                )

        # Download metadata and load curves
        fetch_bldg_data(
            bldg_ids=bldg_ids_to_download,
            file_type=("metadata", "load_curve_15min"),
            output_dir=test_data_dir,
            max_workers=5,
        )

        # Return building info for tests to use
        yield {
            "ny_bldg_ids": [b.bldg_id for b in ny_bldgs],
            "al_bldg_ids": [b.bldg_id for b in al_bldgs],
            "data_path": test_data_dir,
            "outputs_path": test_outputs_dir,
        }
    finally:
        tmpdir.cleanup()
