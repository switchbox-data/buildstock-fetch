from pathlib import Path

import pytest

from buildstock_fetch.building import BuildingID
from buildstock_fetch.explore import DownloadedData, filter_downloads
from buildstock_fetch.main import fetch_bldg_data


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
