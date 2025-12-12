"""Shared pytest fixtures for all test modules."""

import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="function")
def cleanup_downloads():
    """Fixture to clean up downloaded data before and after tests.

    This fixture:
    1. Removes any existing 'data' directory before the test runs
    2. Yields control to the test
    3. Removes the 'data' directory after the test completes

    This ensures each test starts with a clean slate and doesn't leave
    downloaded files behind.
    """
    # Setup - clean up any existing files before test
    data_dir = Path("data")
    if data_dir.exists():
        shutil.rmtree(data_dir)

    yield

    # Teardown - clean up downloaded files after test
    if data_dir.exists():
        shutil.rmtree(data_dir)
