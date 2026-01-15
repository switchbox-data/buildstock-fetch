from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from buildstock_fetch.explore import DownloadedData, filter_downloads
from buildstock_fetch.main_new import download_and_process_all, list_buildings
from buildstock_fetch.types import normalize_upgrade_id


@pytest.fixture
def target_folder():
    with TemporaryDirectory() as td:
        yield Path(td)


@pytest.mark.asyncio
async def test_metadata_upgrade_located(target_folder: Path):
    buildings = list_buildings("res_2024_tmy3_2", "NY", normalize_upgrade_id("0"), 5)
    _ = await download_and_process_all(target_folder, buildings, ["metadata"])
    files = DownloadedData(filter_downloads(target_folder))
    assert len(files)
    assert files.upgrades() == frozenset({"0"})
    assert files.states() == frozenset({"NY"})
    assert files.file_types() == frozenset({"metadata"})

    filtered = files.filter(state="NY", file_type="metadata")
    assert len(filtered)
    assert filtered.upgrades() == frozenset({"0"})
    assert filtered.states() == frozenset({"NY"})
    assert filtered.file_types() == frozenset({"metadata"})

    assert not filtered.filter(state="AL")
