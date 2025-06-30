import shutil
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent))

from buildstock_fetch.main import BuildingID, fetch_bldg_data_core, fetch_bldg_ids


@pytest.fixture(scope="function")
def cleanup_downloads():
    # Setup - clean up any existing files before test
    data_dir = Path("data")
    if data_dir.exists():
        shutil.rmtree(data_dir)

    yield

    # Teardown - clean up downloaded files after test
    if data_dir.exists():
        shutil.rmtree(data_dir)


def test_fetch_bldg_ids():
    assert fetch_bldg_ids("MA") == [
        BuildingID(bldg_id=7),
        BuildingID(bldg_id=8),
        BuildingID(bldg_id=9),
    ]


def test_building_id_config():
    res_2024_amy = {"release_number": "1", "release_year": "2022", "res_com": "resstock", "weather": "tmy3"}

    bldg = BuildingID(bldg_id=7, **res_2024_amy)
    assert bldg.bldg_id == 7
    assert bldg.upgrade_id == "0"
    assert bldg.release_number == "1"
    assert bldg.release_year == "2022"
    assert bldg.weather == "tmy3"
    assert bldg.res_com == "resstock"


def test_fetch_bldg_data(cleanup_downloads):
    fetch_bldg_data_core(
        bldg_ids=[BuildingID(bldg_id=7), BuildingID(bldg_id=8)], file_type=["hpxml"], output_dir=Path("data")
    )
    assert Path("data/0000007/bldg0000007.hpxml").exists()
    assert Path("data/0000008/bldg0000008.hpxml").exists()

    fetch_bldg_data_core(
        bldg_ids=[BuildingID(bldg_id=7), BuildingID(bldg_id=8)], file_type=["schedule"], output_dir=Path("data")
    )
    assert Path("data/0000007/bldg0000007.sched.h5").exists()
    assert Path("data/0000008/bldg0000008.sched.h5").exists()
    fetch_bldg_data_core(
        bldg_ids=[BuildingID(bldg_id=7), BuildingID(bldg_id=8)],
        file_type=["hpxml", "schedule"],
        output_dir=Path("data"),
    )
    assert Path("data/0000007/bldg0000007.hpxml").exists()
    assert Path("data/0000008/bldg0000008.hpxml").exists()
    assert Path("data/0000007/bldg0000007.sched.h5").exists()
    assert Path("data/0000008/bldg0000008.sched.h5").exists()
