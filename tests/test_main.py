import shutil
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent))

from buildstock_fetch.main import (
    BuildingID,
    RequestedFileTypes,
    _parse_requested_file_type,
    download_bldg_data,
    fetch_bldg_data,
    fetch_bldg_ids,
)


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
    res_2024_amy = {"release_number": "1", "release_year": "2022", "res_com": "resstock", "weather": "tmy3"}

    bldg = BuildingID(bldg_id=7, **res_2024_amy)
    assert bldg.bldg_id == 7
    assert bldg.upgrade_id == "0"
    assert bldg.release_number == "1"
    assert bldg.release_year == "2022"
    assert bldg.weather == "tmy3"
    assert bldg.res_com == "resstock"


def test_parse_requested_file_type():
    assert _parse_requested_file_type(("hpxml", "schedule")) == RequestedFileTypes(hpxml=True, schedule=True)
    assert _parse_requested_file_type(("hpxml", "schedule", "metadata")) == RequestedFileTypes(
        hpxml=True, schedule=True, metadata=True
    )
    assert _parse_requested_file_type(("hpxml", "schedule", "metadata", "time_series_15min")) == RequestedFileTypes(
        hpxml=True, schedule=True, metadata=True, time_series_15min=True
    )
    assert _parse_requested_file_type((
        "hpxml",
        "schedule",
        "metadata",
        "time_series_15min",
        "time_series_hourly",
    )) == RequestedFileTypes(hpxml=True, schedule=True, metadata=True, time_series_15min=True, time_series_hourly=True)
    assert _parse_requested_file_type((
        "hpxml",
        "schedule",
        "metadata",
        "time_series_15min",
        "time_series_hourly",
        "time_series_daily",
    )) == RequestedFileTypes(
        hpxml=True,
        schedule=True,
        metadata=True,
        time_series_15min=True,
        time_series_hourly=True,
        time_series_daily=True,
    )
    assert _parse_requested_file_type((
        "hpxml",
        "schedule",
        "metadata",
        "time_series_15min",
        "time_series_hourly",
        "time_series_daily",
        "time_series_weekly",
    )) == RequestedFileTypes(
        hpxml=True,
        schedule=True,
        metadata=True,
        time_series_15min=True,
        time_series_hourly=True,
        time_series_daily=True,
        time_series_weekly=True,
    )


def test_download_bldg_data(cleanup_downloads):
    # Test fetching HPXML files
    bldg_id = BuildingID(bldg_id=7)
    download_bldg_data(
        bldg_id=bldg_id,
        file_type=RequestedFileTypes(hpxml=True),
        output_dir=Path("data"),
    )
    assert Path(f"data/{bldg_id.get_release_name()}/hpxml/{bldg_id.state}/bldg0000007-up00.xml").exists()

    # Test fetching schedule files
    bldg_id = BuildingID(bldg_id=7)
    download_bldg_data(
        bldg_id=bldg_id,
        file_type=RequestedFileTypes(schedule=True),
        output_dir=Path("data"),
    )
    assert Path(f"data/{bldg_id.get_release_name()}/schedule/{bldg_id.state}/bldg0000007-up00_schedule.csv").exists()

    # Test fetching both HPXML and schedule files
    bldg_id = BuildingID(bldg_id=7)
    download_bldg_data(
        bldg_id=bldg_id,
        file_type=RequestedFileTypes(hpxml=True, schedule=True),
        output_dir=Path("data"),
    )
    assert Path(f"data/{bldg_id.get_release_name()}/hpxml/{bldg_id.state}/bldg0000007-up00.xml").exists()
    assert Path(f"data/{bldg_id.get_release_name()}/schedule/{bldg_id.state}/bldg0000007-up00_schedule.csv").exists()


def test_fetch_bldg_data(cleanup_downloads):
    bldg_ids = [BuildingID(bldg_id=7), BuildingID(bldg_id=8), BuildingID(bldg_id=11)]
    file_type = ("hpxml", "schedule", "metadata")
    output_dir = Path("data")
    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    print(downloaded_paths)
    print(failed_downloads)
    assert len(downloaded_paths) == 7
    assert Path(f"data/{bldg_ids[0].get_release_name()}/hpxml/{bldg_ids[0].state}/bldg0000007-up00.xml").exists()
    assert Path(
        f"data/{bldg_ids[0].get_release_name()}/schedule/{bldg_ids[0].state}/bldg0000007-up00_schedule.csv"
    ).exists()
    assert Path(f"data/{bldg_ids[1].get_release_name()}/hpxml/{bldg_ids[1].state}/bldg0000008-up00.xml").exists()
    assert Path(
        f"data/{bldg_ids[1].get_release_name()}/schedule/{bldg_ids[1].state}/bldg0000008-up00_schedule.csv"
    ).exists()
    assert Path(f"data/{bldg_ids[2].get_release_name()}/hpxml/{bldg_ids[2].state}/bldg0000011-up00.xml").exists()
    assert Path(
        f"data/{bldg_ids[2].get_release_name()}/schedule/{bldg_ids[2].state}/bldg0000011-up00_schedule.csv"
    ).exists()
    assert Path(f"data/{bldg_ids[0].get_release_name()}/metadata/{bldg_ids[0].state}/metadata.parquet").exists()
