import shutil
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent))

from buildstock_fetch.main import BuildingID, RequestedFileTypes, download_bldg_data, fetch_bldg_ids


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
    # Test fetching HPXML files
    download_bldg_data(
        bldg_ids=[BuildingID(bldg_id=7), BuildingID(bldg_id=8)],
        file_type=RequestedFileTypes(hpxml=True),
        output_dir=Path("data"),
    )
    assert Path("data/bldg0000007-up00.xml").exists()
    assert Path("data/bldg0000008-up00.xml").exists()

    # Test fetching schedule files
    download_bldg_data(
        bldg_ids=[BuildingID(bldg_id=7), BuildingID(bldg_id=8)],
        file_type=RequestedFileTypes(schedule=True),
        output_dir=Path("data"),
    )
    assert Path("data/bldg0000007-up00_schedule.csv").exists()
    assert Path("data/bldg0000008-up00_schedule.csv").exists()

    # Test fetching both HPXML and schedule files
    download_bldg_data(
        bldg_ids=[BuildingID(bldg_id=7), BuildingID(bldg_id=8)],
        file_type=RequestedFileTypes(hpxml=True, schedule=True),
        output_dir=Path("data"),
    )
    assert Path("data/bldg0000007-up00.xml").exists()
    assert Path("data/bldg0000008-up00.xml").exists()
    assert Path("data/bldg0000007-up00_schedule.csv").exists()
    assert Path("data/bldg0000008-up00_schedule.csv").exists()

    # Test fetching metadata
    bldg = BuildingID(bldg_id=7)
    download_bldg_data(
        bldg_ids=[bldg],
        file_type=RequestedFileTypes(metadata=True),
        output_dir=Path("data"),
    )
    print(bldg.get_metadata_url())
    print(bldg.get_release_name())
    assert Path("data/resstock_tmy3_release_1_metadata.parquet").exists()
