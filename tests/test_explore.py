import shutil
from pathlib import Path

import pytest

from buildstock_fetch.building import BuildingID
from buildstock_fetch.explore import DownloadedData, filter_downloads
from buildstock_fetch.main import fetch_bldg_data


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


@pytest.mark.usefixtures("cleanup_downloads")
def test_metadata_upgrade_located():
    bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
    _ = fetch_bldg_data(bldg_ids, ("metadata",), Path("./data"))
    files = DownloadedData(filter_downloads(Path("./data")))
    assert len(files)
    assert files.upgrade_ids() == frozenset({"0"})
    assert files.states() == frozenset({"NY"})
    assert files.file_types() == frozenset({"metadata"})

    filtered = files.filter(state="NY", file_type="metadata")
    assert len(filtered)
    assert filtered.upgrade_ids() == frozenset({"0"})
    assert filtered.states() == frozenset({"NY"})
    assert filtered.file_types() == frozenset({"metadata"})

    assert not filtered.filter(state="AL")
