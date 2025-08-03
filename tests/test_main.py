import shutil
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent))

from buildstock_fetch.main import (
    BuildingID,
    No15minLoadCurveError,
    NoBuildingDataError,
    NoMetadataError,
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
    res_2022_tmy3_1 = {"release_number": "1", "release_year": "2022", "res_com": "resstock", "weather": "tmy3"}

    bldg = BuildingID(bldg_id=7, **res_2022_tmy3_1)
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
    assert _parse_requested_file_type(("hpxml", "schedule", "metadata", "load_curve_15min")) == RequestedFileTypes(
        hpxml=True, schedule=True, metadata=True, load_curve_15min=True
    )
    assert _parse_requested_file_type((
        "hpxml",
        "schedule",
        "metadata",
        "load_curve_15min",
        "load_curve_hourly",
    )) == RequestedFileTypes(hpxml=True, schedule=True, metadata=True, load_curve_15min=True, load_curve_hourly=True)
    assert _parse_requested_file_type((
        "hpxml",
        "schedule",
        "metadata",
        "load_curve_15min",
        "load_curve_hourly",
        "load_curve_daily",
    )) == RequestedFileTypes(
        hpxml=True,
        schedule=True,
        metadata=True,
        load_curve_15min=True,
        load_curve_hourly=True,
        load_curve_daily=True,
    )
    assert _parse_requested_file_type((
        "hpxml",
        "schedule",
        "metadata",
        "load_curve_15min",
        "load_curve_hourly",
        "load_curve_daily",
    )) == RequestedFileTypes(
        hpxml=True,
        schedule=True,
        metadata=True,
        load_curve_15min=True,
        load_curve_hourly=True,
        load_curve_daily=True,
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

    # Test 2021 release - should raise NoBuildingDataError
    bldg_ids = [BuildingID(bldg_id=7, release_year="2021", res_com="resstock", weather="tmy3", upgrade_id="1")]
    file_type = ("hpxml", "schedule")
    output_dir = Path("data")

    with pytest.raises(
        NoBuildingDataError, match=f"Building data is not available for {bldg_ids[0].get_release_name()}"
    ):
        fetch_bldg_data(bldg_ids, file_type, output_dir)

    # Test 2023 release - should raise NoBuildingDataError
    bldg_ids = [BuildingID(bldg_id=7, release_year="2023", res_com="comstock", weather="amy2018", upgrade_id="1")]
    file_type = ("hpxml", "schedule")
    output_dir = Path("data")

    with pytest.raises(
        NoBuildingDataError, match=f"Building data is not available for {bldg_ids[0].get_release_name()}"
    ):
        fetch_bldg_data(bldg_ids, file_type, output_dir)

    # Test 2024 comstock release - should raise NoBuildingDataError
    bldg_ids = [BuildingID(bldg_id=7, release_year="2024", res_com="comstock", weather="amy2018", upgrade_id="1")]
    file_type = ("hpxml", "schedule")
    output_dir = Path("data")

    with pytest.raises(
        NoBuildingDataError, match=f"Building data is not available for {bldg_ids[0].get_release_name()}"
    ):
        fetch_bldg_data(bldg_ids, file_type, output_dir)

    # Test 2024 resstock release - should work fine
    bldg_ids = [
        BuildingID(
            bldg_id=7, release_year="2024", res_com="resstock", weather="tmy3", upgrade_id="1", release_number="2"
        )
    ]
    file_type = ("hpxml", "schedule")
    output_dir = Path("data")
    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    print(downloaded_paths)
    print(failed_downloads)
    assert len(downloaded_paths) == 2
    assert len(failed_downloads) == 0
    assert Path(f"data/{bldg_ids[0].get_release_name()}/hpxml/{bldg_ids[0].state}/bldg0000007-up01.xml").exists()
    assert Path(
        f"data/{bldg_ids[0].get_release_name()}/schedule/{bldg_ids[0].state}/bldg0000007-up01_schedule.csv"
    ).exists()


def test_fetch_metadata(cleanup_downloads):
    bldg_ids = [
        BuildingID(
            bldg_id=7, release_year="2024", res_com="resstock", weather="tmy3", upgrade_id="1", release_number="2"
        )
    ]
    file_type = ("metadata",)
    output_dir = Path("data")
    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    print(downloaded_paths)
    print(failed_downloads)
    assert len(downloaded_paths) == 1
    assert len(failed_downloads) == 0
    assert Path(f"data/{bldg_ids[0].get_release_name()}/metadata/{bldg_ids[0].state}/metadata.parquet").exists()

    # Test 2024 comstock release - should raise NoMetadataError
    bldg_ids = [
        BuildingID(
            bldg_id=7, release_year="2024", res_com="comstock", weather="amy2018", upgrade_id="0", release_number="2"
        )
    ]
    file_type = ("metadata",)
    output_dir = Path("data")

    with pytest.raises(NoMetadataError, match=f"Metadata is not available for {bldg_ids[0].get_release_name()}"):
        fetch_bldg_data(bldg_ids, file_type, output_dir)

    # Test 2025 comstock release - should raise NoMetadataError
    bldg_ids = [
        BuildingID(
            bldg_id=7, release_year="2025", res_com="comstock", weather="amy2018", upgrade_id="0", release_number="1"
        )
    ]
    file_type = ("metadata",)
    output_dir = Path("data")

    with pytest.raises(NoMetadataError, match=f"Metadata is not available for {bldg_ids[0].get_release_name()}"):
        fetch_bldg_data(bldg_ids, file_type, output_dir)


def test_fetch_15min_load_curve(cleanup_downloads):
    bldg_ids = [
        BuildingID(
            bldg_id=100041,
            release_year="2021",
            res_com="comstock",
            weather="tmy3",
            release_number="1",
            upgrade_id="0",
            state="AZ",
        )
    ]
    file_type = ("load_curve_15min",)
    output_dir = Path("data")
    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 1
    assert len(failed_downloads) == 0
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_15min/{bldg_id.state}/bldg{str(bldg_id.bldg_id).zfill(7)}-up{str(int(bldg_id.upgrade_id)).zfill(2)}_load_curve_15min.parquet"
    ).exists()

    bldg_ids = [
        BuildingID(
            bldg_id=10, release_year="2022", res_com="resstock", weather="amy2018", release_number="1", upgrade_id="1"
        )
    ]
    file_type = ("load_curve_15min",)
    output_dir = Path("data")
    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 1
    assert len(failed_downloads) == 0
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_ids[0].get_release_name()}/load_curve_15min/{bldg_ids[0].state}/bldg{str(bldg_id.bldg_id).zfill(7)}-up{str(int(bldg_id.upgrade_id)).zfill(2)}_load_curve_15min.parquet"
    ).exists()

    bldg_ids = [
        BuildingID(
            bldg_id=237237,
            release_year="2023",
            res_com="comstock",
            weather="amy2018",
            release_number="1",
            upgrade_id="1",
            state="OH",
        )
    ]
    file_type = ("load_curve_15min",)
    output_dir = Path("data")
    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 1
    assert len(failed_downloads) == 0
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_15min/{bldg_id.state}/bldg{str(bldg_id.bldg_id).zfill(7)}-up{str(int(bldg_id.upgrade_id)).zfill(2)}_load_curve_15min.parquet"
    ).exists()

    bldg_ids = [
        BuildingID(
            bldg_id=100041,
            release_year="2024",
            res_com="resstock",
            weather="tmy3",
            release_number="1",
            upgrade_id="0",
            state="NY",
        )
    ]
    with pytest.raises(
        No15minLoadCurveError,
        match=f"15 min load profile timeseries is not available for {bldg_ids[0].get_release_name()}",
    ):
        fetch_bldg_data(bldg_ids, file_type, output_dir)

    bldg_ids = [
        BuildingID(
            bldg_id=100000,
            release_year="2024",
            res_com="resstock",
            weather="tmy3",
            release_number="2",
            upgrade_id="0",
            state="OH",
        ),
        BuildingID(
            bldg_id=100058,
            release_year="2024",
            res_com="resstock",
            weather="tmy3",
            release_number="2",
            upgrade_id="2",
            state="NY",
        ),
    ]
    file_type = ("load_curve_15min",)
    output_dir = Path("data")
    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 2
    assert len(failed_downloads) == 0
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_15min/{bldg_id.state}/bldg{str(bldg_id.bldg_id).zfill(7)}-up{str(int(bldg_id.upgrade_id)).zfill(2)}_load_curve_15min.parquet"
    ).exists()
    bldg_id = bldg_ids[1]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_15min/{bldg_id.state}/bldg{str(bldg_id.bldg_id).zfill(7)}-up{str(int(bldg_id.upgrade_id)).zfill(2)}_load_curve_15min.parquet"
    ).exists()

    bldg_ids = [
        BuildingID(
            bldg_id=4849,
            release_year="2025",
            res_com="comstock",
            weather="amy2018",
            release_number="1",
            upgrade_id="0",
            state="AK",
        ),
        BuildingID(
            bldg_id=4850,
            release_year="2025",
            res_com="comstock",
            weather="amy2018",
            release_number="1",
            upgrade_id="0",
            state="AK",
        ),
    ]
    file_type = ("load_curve_15min",)
    output_dir = Path("data")
    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 2
    assert len(failed_downloads) == 0
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_15min/{bldg_id.state}/bldg{str(bldg_id.bldg_id).zfill(7)}-up{str(int(bldg_id.upgrade_id)).zfill(2)}_load_curve_15min.parquet"
    ).exists()
    bldg_id = bldg_ids[1]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_15min/{bldg_id.state}/bldg{str(bldg_id.bldg_id).zfill(7)}-up{str(int(bldg_id.upgrade_id)).zfill(2)}_load_curve_15min.parquet"
    ).exists()
