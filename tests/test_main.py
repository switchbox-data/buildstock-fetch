import shutil
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent))

from buildstock_fetch.main import BuildingID, fetch_bldg_data, fetch_bldg_ids


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
    assert fetch_bldg_ids("resstock", "2021", "tmy3", "1", "MA", "0")[0].bldg_id == 355537
    assert fetch_bldg_ids("resstock", "2021", "tmy3", "1", "MA", "0")[0].res_com == "resstock"
    assert fetch_bldg_ids("resstock", "2021", "tmy3", "1", "MA", "0")[0].release_year == "2021"
    assert fetch_bldg_ids("resstock", "2021", "tmy3", "1", "MA", "0")[0].weather == "tmy3"
    assert fetch_bldg_ids("resstock", "2021", "tmy3", "1", "MA", "0")[0].release_number == "1"
    assert fetch_bldg_ids("resstock", "2021", "tmy3", "1", "MA", "0")[0].upgrade_id == "0"

    assert fetch_bldg_ids("resstock", "2021", "tmy3", "1", "MA", "0")[1].bldg_id == 24415
    assert fetch_bldg_ids("resstock", "2021", "tmy3", "1", "MA", "0")[2].bldg_id == 487404
    assert fetch_bldg_ids("resstock", "2021", "tmy3", "1", "MA", "0")[3].bldg_id == 355634


def test_building_id_config():
    res_2022_tmy3_1 = {"release_number": "1", "release_year": "2022", "res_com": "resstock", "weather": "tmy3"}

    bldg = BuildingID(bldg_id=7, **res_2022_tmy3_1)
    assert bldg.bldg_id == 7
    assert bldg.upgrade_id == "0"
    assert bldg.release_number == "1"
    assert bldg.release_year == "2022"
    assert bldg.weather == "tmy3"
    assert bldg.res_com == "resstock"


def test_fetch_bldg_data(cleanup_downloads):
    fetch_bldg_data([BuildingID(bldg_id=7), BuildingID(bldg_id=8)])
    assert Path("data/0000007_upgrade0.zip").exists()
    assert Path("data/0000008_upgrade0.zip").exists()
