import json
import shutil
import sys
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from buildstock_fetch.main import (
    LOAD_CURVE_COLUMN_AGGREGATION,
    BuildingID,
    No15minLoadCurveError,
    NoAnnualLoadCurveError,
    NoBuildingDataError,
    RequestedFileTypes,
    _aggregate_load_curve_aggregate,
    _parse_requested_file_type,
    download_bldg_data,
    fetch_bldg_data,
    fetch_bldg_ids,
)
from buildstock_fetch.main_cli import BUILDSTOCK_RELEASES_FILE


@pytest.fixture(scope="function")
def buildstock_releases_json():
    return json.loads(Path(BUILDSTOCK_RELEASES_FILE).read_text(encoding="utf-8"))


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
    # 2021 resstock release
    assert fetch_bldg_ids("resstock", "2021", "tmy3", "1", "MA", "0")[0].bldg_id == 355537
    assert fetch_bldg_ids("resstock", "2021", "tmy3", "1", "MA", "0")[0].res_com == "resstock"
    assert fetch_bldg_ids("resstock", "2021", "tmy3", "1", "MA", "0")[0].release_year == "2021"
    assert fetch_bldg_ids("resstock", "2021", "tmy3", "1", "MA", "0")[0].weather == "tmy3"
    assert fetch_bldg_ids("resstock", "2021", "tmy3", "1", "MA", "0")[0].release_number == "1"
    assert fetch_bldg_ids("resstock", "2021", "tmy3", "1", "MA", "0")[0].upgrade_id == "0"

    assert fetch_bldg_ids("resstock", "2021", "tmy3", "1", "MA", "0")[1].bldg_id == 24415
    assert fetch_bldg_ids("resstock", "2021", "tmy3", "1", "MA", "0")[2].bldg_id == 487404
    assert fetch_bldg_ids("resstock", "2021", "tmy3", "1", "MA", "0")[3].bldg_id == 355634

    # 2021 comstock tmy3 release
    assert fetch_bldg_ids("comstock", "2021", "tmy3", "1", "MA", "0")[0].bldg_id == 271060
    assert fetch_bldg_ids("comstock", "2021", "tmy3", "1", "MA", "0")[0].res_com == "comstock"
    assert fetch_bldg_ids("comstock", "2021", "tmy3", "1", "MA", "0")[0].release_year == "2021"
    assert fetch_bldg_ids("comstock", "2021", "tmy3", "1", "MA", "0")[0].weather == "tmy3"
    assert fetch_bldg_ids("comstock", "2021", "tmy3", "1", "MA", "0")[0].release_number == "1"
    assert fetch_bldg_ids("comstock", "2021", "tmy3", "1", "MA", "0")[0].upgrade_id == "0"

    assert fetch_bldg_ids("comstock", "2021", "tmy3", "1", "MA", "0")[1].bldg_id == 9944
    assert fetch_bldg_ids("comstock", "2021", "tmy3", "1", "MA", "0")[2].bldg_id == 245624
    assert fetch_bldg_ids("comstock", "2021", "tmy3", "1", "MA", "0")[3].bldg_id == 155275

    # 2021 comstock tmy3 release
    assert fetch_bldg_ids("comstock", "2021", "amy2018", "1", "AL", "0")[0].bldg_id == 297733
    assert fetch_bldg_ids("comstock", "2021", "amy2018", "1", "AL", "0")[0].res_com == "comstock"
    assert fetch_bldg_ids("comstock", "2021", "amy2018", "1", "AL", "0")[0].release_year == "2021"
    assert fetch_bldg_ids("comstock", "2021", "amy2018", "1", "AL", "0")[0].weather == "amy2018"
    assert fetch_bldg_ids("comstock", "2021", "amy2018", "1", "AL", "0")[0].release_number == "1"
    assert fetch_bldg_ids("comstock", "2021", "amy2018", "1", "AL", "0")[0].upgrade_id == "0"

    assert fetch_bldg_ids("comstock", "2021", "amy2018", "1", "AL", "0")[1].bldg_id == 190793
    assert fetch_bldg_ids("comstock", "2021", "amy2018", "1", "AL", "0")[2].bldg_id == 288229
    assert fetch_bldg_ids("comstock", "2021", "amy2018", "1", "AL", "0")[3].bldg_id == 290474

    # 2022 resstock releases
    assert fetch_bldg_ids("resstock", "2022", "amy2012", "1", "MI", "0")[1].bldg_id == 460128
    assert fetch_bldg_ids("resstock", "2022", "amy2018", "1.1", "CO", "0")[2].bldg_id == 242983
    assert fetch_bldg_ids("resstock", "2022", "tmy3", "1", "IN", "0")[3].bldg_id == 307400

    # 2023 comstock releases
    assert fetch_bldg_ids("comstock", "2023", "amy2018", "1", "AZ", "0")[1].bldg_id == 8597
    assert fetch_bldg_ids("comstock", "2023", "amy2018", "1", "AZ", "0")[2].bldg_id == 9161
    assert fetch_bldg_ids("comstock", "2023", "amy2018", "2", "MI", "0")[3].bldg_id == 163941

    # 2024 resstock releases
    assert fetch_bldg_ids("resstock", "2024", "amy2018", "2", "IL", "0")[1].bldg_id == 233842
    assert fetch_bldg_ids("resstock", "2024", "amy2018", "2", "IL", "0")[2].bldg_id == 245818
    assert fetch_bldg_ids("resstock", "2024", "tmy3", "2", "ID", "0")[3].bldg_id == 18149

    # 2024 comstock releases
    assert fetch_bldg_ids("comstock", "2024", "amy2018", "1", "GA", "0")[1].bldg_id == 104039
    assert fetch_bldg_ids("comstock", "2024", "amy2018", "1", "GA", "0")[2].bldg_id == 97452
    assert fetch_bldg_ids("comstock", "2024", "amy2018", "2", "DC", "0")[3].bldg_id == 117988


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
    assert Path(
        f"data/{bldg_id.get_release_name()}/hpxml/{bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg0000007-up00.xml"
    ).exists()

    # Test fetching schedule files
    bldg_id = BuildingID(bldg_id=7)
    download_bldg_data(
        bldg_id=bldg_id,
        file_type=RequestedFileTypes(schedule=True),
        output_dir=Path("data"),
    )
    assert Path(
        f"data/{bldg_id.get_release_name()}/schedule/{bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg0000007-up00_schedule.csv"
    ).exists()

    # Test fetching both HPXML and schedule files
    bldg_id = BuildingID(bldg_id=7)
    download_bldg_data(
        bldg_id=bldg_id,
        file_type=RequestedFileTypes(hpxml=True, schedule=True),
        output_dir=Path("data"),
    )
    assert Path(
        f"data/{bldg_id.get_release_name()}/hpxml/{bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg0000007-up00.xml"
    ).exists()
    assert Path(
        f"data/{bldg_id.get_release_name()}/schedule/{bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg0000007-up00_schedule.csv"
    ).exists()


def test_fetch_bldg_data(cleanup_downloads):
    bldg_ids = [BuildingID(bldg_id=7), BuildingID(bldg_id=8), BuildingID(bldg_id=11)]
    file_type = ("hpxml", "schedule", "metadata")
    output_dir = Path("data")
    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    print(downloaded_paths)
    print(failed_downloads)
    assert len(downloaded_paths) == 7
    assert Path(
        f"data/{bldg_ids[0].get_release_name()}/hpxml/{bldg_ids[0].state}/upgrade={str(int(bldg_ids[0].upgrade_id)).zfill(2)}/bldg0000007-up00.xml"
    ).exists()
    assert Path(
        f"data/{bldg_ids[0].get_release_name()}/schedule/{bldg_ids[0].state}/upgrade={str(int(bldg_ids[0].upgrade_id)).zfill(2)}/bldg0000007-up00_schedule.csv"
    ).exists()
    assert Path(
        f"data/{bldg_ids[1].get_release_name()}/hpxml/{bldg_ids[1].state}/upgrade={str(int(bldg_ids[1].upgrade_id)).zfill(2)}/bldg0000008-up00.xml"
    ).exists()
    assert Path(
        f"data/{bldg_ids[1].get_release_name()}/schedule/{bldg_ids[1].state}/upgrade={str(int(bldg_ids[1].upgrade_id)).zfill(2)}/bldg0000008-up00_schedule.csv"
    ).exists()
    assert Path(
        f"data/{bldg_ids[2].get_release_name()}/hpxml/{bldg_ids[2].state}/upgrade={str(int(bldg_ids[2].upgrade_id)).zfill(2)}/bldg0000011-up00.xml"
    ).exists()
    assert Path(
        f"data/{bldg_ids[2].get_release_name()}/schedule/{bldg_ids[2].state}/upgrade={str(int(bldg_ids[2].upgrade_id)).zfill(2)}/bldg0000011-up00_schedule.csv"
    ).exists()
    assert Path(
        f"data/{bldg_ids[0].get_release_name()}/metadata/state={bldg_ids[0].state}/upgrade={str(int(bldg_ids[0].upgrade_id)).zfill(2)}/metadata.parquet"
    ).exists()

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
    assert Path(
        f"data/{bldg_ids[0].get_release_name()}/hpxml/{bldg_ids[0].state}/upgrade={str(int(bldg_ids[0].upgrade_id)).zfill(2)}/bldg0000007-up01.xml"
    ).exists()
    assert Path(
        f"data/{bldg_ids[0].get_release_name()}/schedule/{bldg_ids[0].state}/upgrade={str(int(bldg_ids[0].upgrade_id)).zfill(2)}/bldg0000007-up01_schedule.csv"
    ).exists()


def test_fetch_metadata(cleanup_downloads):
    METADATA_COLUMNS = ["bldg_id", "upgrade", "in."]
    NOT_METADATA_COLUMNS = ["out."]
    bldg_ids = [
        BuildingID(
            bldg_id=303470, release_year="2024", res_com="resstock", weather="tmy3", upgrade_id="1", release_number="2"
        )
    ]
    file_type = ("metadata",)
    output_dir = Path("data")
    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    print(downloaded_paths)
    print(failed_downloads)
    assert len(downloaded_paths) == 1
    assert len(failed_downloads) == 0
    metadata_file = pl.read_parquet(downloaded_paths[0])
    # Check that each required column pattern is a prefix of at least one actual column name
    for required_col in METADATA_COLUMNS:
        found = any(actual_col.startswith(required_col) for actual_col in metadata_file.columns)
        assert found
    for not_required_col in NOT_METADATA_COLUMNS:
        found = any(actual_col.startswith(not_required_col) for actual_col in metadata_file.columns)
        assert not found
    assert Path(
        f"data/{bldg_ids[0].get_release_name()}/metadata/state={bldg_ids[0].state}/upgrade={str(int(bldg_ids[0].upgrade_id)).zfill(2)}/metadata.parquet"
    ).exists()

    # Test 2024 comstock release
    bldg_ids = [
        BuildingID(
            bldg_id=19713,
            release_year="2024",
            res_com="comstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="2",
        ),
        BuildingID(
            bldg_id=658,
            release_year="2024",
            res_com="comstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="2",
        ),
        BuildingID(
            bldg_id=659,
            release_year="2024",
            res_com="comstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="2",
        ),
    ]
    file_type = ("metadata",)
    output_dir = Path("data")

    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 1
    assert len(failed_downloads) == 0
    assert Path(
        f"data/{bldg_ids[0].get_release_name()}/metadata/state={bldg_ids[0].state}/upgrade={str(int(bldg_ids[0].upgrade_id)).zfill(2)}/metadata.parquet"
    ).exists()

    # Test 2024 comstock release
    bldg_ids = [
        BuildingID(
            bldg_id=21023,
            release_year="2024",
            res_com="comstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="2",
            state="NY",
        ),
        BuildingID(
            bldg_id=18403,
            release_year="2024",
            res_com="comstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="2",
            state="NY",
        ),
        BuildingID(
            bldg_id=70769,
            release_year="2024",
            res_com="comstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="2",
            state="NV",
        ),
        BuildingID(
            bldg_id=68227,
            release_year="2024",
            res_com="comstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="2",
            state="NV",
        ),
    ]
    file_type = ("metadata",)
    output_dir = Path("data")

    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 2
    assert len(failed_downloads) == 0
    assert Path(
        f"data/{bldg_ids[0].get_release_name()}/metadata/state={bldg_ids[0].state}/upgrade={str(int(bldg_ids[0].upgrade_id)).zfill(2)}/metadata.parquet"
    ).exists()
    assert Path(
        f"data/{bldg_ids[-1].get_release_name()}/metadata/state={bldg_ids[-1].state}/upgrade={str(int(bldg_ids[-1].upgrade_id)).zfill(2)}/metadata.parquet"
    ).exists()

    # Test 2025 comstock release - should fail
    bldg_ids = [
        BuildingID(
            bldg_id=150914,
            release_year="2025",
            res_com="comstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="1",
            state="MA",
        ),
        BuildingID(
            bldg_id=149336,
            release_year="2025",
            res_com="comstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="1",
            state="MA",
        ),
        BuildingID(
            bldg_id=87123,
            release_year="2025",
            res_com="comstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="1",
            state="MO",
        ),
        BuildingID(
            bldg_id=87232,
            release_year="2025",
            res_com="comstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="1",
            state="MO",
        ),
    ]
    file_type = ("metadata",)
    output_dir = Path("data")

    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 2
    assert len(failed_downloads) == 0
    assert Path(
        f"data/{bldg_ids[0].get_release_name()}/metadata/state={bldg_ids[0].state}/upgrade={str(int(bldg_ids[0].upgrade_id)).zfill(2)}/metadata.parquet"
    ).exists()
    assert Path(
        f"data/{bldg_ids[2].get_release_name()}/metadata/state={bldg_ids[2].state}/upgrade={str(int(bldg_ids[2].upgrade_id)).zfill(2)}/metadata.parquet"
    ).exists()


def test_fetch_metadata_relevant_bldg_id(cleanup_downloads):
    METADATA_COLUMNS = ["bldg_id", "upgrade", "in."]
    NOT_METADATA_COLUMNS = ["out."]
    bldg_ids = [
        BuildingID(
            bldg_id=320214,
            release_year="2024",
            res_com="resstock",
            weather="tmy3",
            upgrade_id="1",
            release_number="2",
            state="NY",
        ),
        BuildingID(
            bldg_id=95261,
            release_year="2024",
            res_com="resstock",
            weather="tmy3",
            upgrade_id="1",
            release_number="2",
            state="NY",
        ),
        BuildingID(
            bldg_id=95272,
            release_year="2024",
            res_com="resstock",
            weather="tmy3",
            upgrade_id="1",
            release_number="2",
            state="NY",
        ),
    ]
    file_type = ("metadata",)
    output_dir = Path("data")
    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    print(downloaded_paths)
    print(failed_downloads)
    assert len(downloaded_paths) == 1
    assert len(failed_downloads) == 0
    assert Path(
        f"data/{bldg_ids[0].get_release_name()}/metadata/state={bldg_ids[0].state}/upgrade={str(int(bldg_ids[0].upgrade_id)).zfill(2)}/metadata.parquet"
    ).exists()
    metadata_file_path = downloaded_paths[0]
    metadata_file = pl.read_parquet(metadata_file_path)
    # Check that each required column pattern is a prefix of at least one actual column name
    for required_col in METADATA_COLUMNS:
        found = any(actual_col.startswith(required_col) for actual_col in metadata_file.columns)
        assert found
    for not_required_col in NOT_METADATA_COLUMNS:
        found = any(actual_col.startswith(not_required_col) for actual_col in metadata_file.columns)
        assert not found
    assert metadata_file.height == 3
    assert metadata_file.filter(pl.col("bldg_id") == 320214).height == 1
    assert metadata_file.filter(pl.col("bldg_id") == 95261).height == 1
    assert metadata_file.filter(pl.col("bldg_id") == 95272).height == 1


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
        f"data/{bldg_id.get_release_name()}/load_curve_15min/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{str(bldg_id.bldg_id).zfill(7)}_load_curve_15min.parquet"
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
        f"data/{bldg_ids[0].get_release_name()}/load_curve_15min/state={bldg_ids[0].state}/upgrade={str(int(bldg_ids[0].upgrade_id)).zfill(2)}/bldg{str(bldg_id.bldg_id).zfill(7)}_load_curve_15min.parquet"
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
        f"data/{bldg_id.get_release_name()}/load_curve_15min/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{str(bldg_id.bldg_id).zfill(7)}_load_curve_15min.parquet"
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
        f"data/{bldg_id.get_release_name()}/load_curve_15min/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{str(bldg_id.bldg_id).zfill(7)}_load_curve_15min.parquet"
    ).exists()
    bldg_id = bldg_ids[1]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_15min/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{str(bldg_id.bldg_id).zfill(7)}_load_curve_15min.parquet"
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
        f"data/{bldg_id.get_release_name()}/load_curve_15min/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{str(bldg_id.bldg_id).zfill(7)}_load_curve_15min.parquet"
    ).exists()
    bldg_id = bldg_ids[1]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_15min/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{str(bldg_id.bldg_id).zfill(7)}_load_curve_15min.parquet"
    ).exists()


def _assert_annual_load_curve_columns(
    file_path: Path, required_columns: list[str], not_required_columns: list[str]
) -> None:
    """Helper function to assert column patterns in annual load curve files.

    Args:
        file_path: Path to the parquet file to check
        required_columns: List of column patterns that must be present (checked as prefix)
        not_required_columns: List of column patterns that must not be present (checked as prefix)
    """
    annual_load_curve_file = pl.read_parquet(file_path)
    # Check that each required column pattern is a prefix of at least one actual column name
    for required_col in required_columns:
        found = any(actual_col.startswith(required_col) for actual_col in annual_load_curve_file.columns)
        assert found
    for not_required_col in not_required_columns:
        found = any(actual_col.startswith(not_required_col) for actual_col in annual_load_curve_file.columns)
        assert not found


def test_fetch_annual_load_curve(cleanup_downloads):
    ANNUAL_LOAD_CURVE_COLUMNS = ["bldg_id", "upgrade", "out."]
    NOT_ANNUAL_LOAD_CURVE_COLUMNS = ["in."]
    # 2021 release - should raise NoAnnualLoadCurveError
    bldg_ids = [BuildingID(bldg_id=7, release_year="2021", res_com="resstock", weather="tmy3", upgrade_id="1")]
    file_type = ("load_curve_annual",)
    output_dir = Path("data")

    with pytest.raises(
        NoAnnualLoadCurveError, match=f"Annual load curve is not available for {bldg_ids[0].get_release_name()}"
    ):
        fetch_bldg_data(bldg_ids, file_type, output_dir)

    bldg_ids = [BuildingID(bldg_id=8, release_year="2021", res_com="comstock", weather="amy2018", upgrade_id="1")]
    file_type = ("load_curve_annual",)
    output_dir = Path("data")

    with pytest.raises(
        NoAnnualLoadCurveError, match=f"Annual load curve is not available for {bldg_ids[0].get_release_name()}"
    ):
        fetch_bldg_data(bldg_ids, file_type, output_dir)

    # 2022 release - should work fine
    bldg_ids = [
        BuildingID(
            bldg_id=9,
            release_year="2022",
            res_com="resstock",
            weather="tmy3",
            upgrade_id="1",
            release_number="1",
            state="AL",
        ),
        BuildingID(
            bldg_id=10, release_year="2022", res_com="resstock", weather="amy2018", upgrade_id="0", release_number="1.1"
        ),
    ]
    file_type = ("load_curve_annual",)
    output_dir = Path("data")
    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 2
    _assert_annual_load_curve_columns(downloaded_paths[0], ANNUAL_LOAD_CURVE_COLUMNS, NOT_ANNUAL_LOAD_CURVE_COLUMNS)
    _assert_annual_load_curve_columns(downloaded_paths[1], ANNUAL_LOAD_CURVE_COLUMNS, NOT_ANNUAL_LOAD_CURVE_COLUMNS)
    assert len(failed_downloads) == 0
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_annual/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/{bldg_id.get_annual_load_curve_filename()}"
    ).exists()
    bldg_id = bldg_ids[1]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_annual/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/{bldg_id.get_annual_load_curve_filename()}"
    ).exists()

    # 2023 release - should work fine
    bldg_ids = [
        BuildingID(
            bldg_id=9, release_year="2023", res_com="comstock", weather="amy2018", upgrade_id="1", release_number="1"
        ),
        BuildingID(
            bldg_id=10, release_year="2023", res_com="comstock", weather="amy2018", upgrade_id="0", release_number="2"
        ),
    ]
    file_type = ("load_curve_annual",)
    output_dir = Path("data")
    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 2
    assert len(failed_downloads) == 0
    _assert_annual_load_curve_columns(downloaded_paths[0], ANNUAL_LOAD_CURVE_COLUMNS, NOT_ANNUAL_LOAD_CURVE_COLUMNS)
    _assert_annual_load_curve_columns(downloaded_paths[1], ANNUAL_LOAD_CURVE_COLUMNS, NOT_ANNUAL_LOAD_CURVE_COLUMNS)
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_annual/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/{bldg_id.get_annual_load_curve_filename()}"
    ).exists()
    bldg_id = bldg_ids[1]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_annual/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/{bldg_id.get_annual_load_curve_filename()}"
    ).exists()

    # 2024 comstock amy2018 v2 release - should work fine
    bldg_ids = [
        BuildingID(
            bldg_id=5634,
            release_year="2024",
            res_com="comstock",
            weather="amy2018",
            upgrade_id="1",
            release_number="2",
            state="AL",
        ),
        BuildingID(
            bldg_id=78270,
            release_year="2024",
            res_com="comstock",
            weather="amy2018",
            upgrade_id="1",
            release_number="2",
            state="DE",
        ),
    ]
    file_type = ("load_curve_annual",)
    output_dir = Path("data")
    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 2
    assert len(failed_downloads) == 0
    _assert_annual_load_curve_columns(downloaded_paths[0], ANNUAL_LOAD_CURVE_COLUMNS, NOT_ANNUAL_LOAD_CURVE_COLUMNS)
    _assert_annual_load_curve_columns(downloaded_paths[1], ANNUAL_LOAD_CURVE_COLUMNS, NOT_ANNUAL_LOAD_CURVE_COLUMNS)
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_annual/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/{bldg_id.get_annual_load_curve_filename()}"
    ).exists()
    bldg_id = bldg_ids[1]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_annual/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/{bldg_id.get_annual_load_curve_filename()}"
    ).exists()

    # 2024 resstock tmy3 v1 release - should crash
    bldg_ids = [
        BuildingID(
            bldg_id=100000,
            release_year="2024",
            res_com="resstock",
            weather="tmy3",
            upgrade_id="0",
            release_number="1",
            state="NY",
        ),
        BuildingID(
            bldg_id=100058,
            release_year="2024",
            res_com="resstock",
            weather="tmy3",
            upgrade_id="2",
            release_number="1",
            state="NY",
        ),
    ]
    file_type = ("load_curve_annual",)
    output_dir = Path("data")

    with pytest.raises(
        NoAnnualLoadCurveError, match=f"Annual load curve is not available for {bldg_ids[0].get_release_name()}"
    ):
        fetch_bldg_data(bldg_ids, file_type, output_dir)

    # 2025 comstock amy2018 v1 release - should work fine
    bldg_ids = [
        BuildingID(
            bldg_id=61336,
            release_year="2025",
            res_com="comstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="1",
            state="FL",
        ),
        BuildingID(
            bldg_id=52360,
            release_year="2025",
            res_com="comstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="1",
            state="CT",
        ),
    ]
    file_type = ("load_curve_annual",)
    output_dir = Path("data")

    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 2
    assert len(failed_downloads) == 0
    _assert_annual_load_curve_columns(downloaded_paths[0], ANNUAL_LOAD_CURVE_COLUMNS, NOT_ANNUAL_LOAD_CURVE_COLUMNS)
    _assert_annual_load_curve_columns(downloaded_paths[1], ANNUAL_LOAD_CURVE_COLUMNS, NOT_ANNUAL_LOAD_CURVE_COLUMNS)
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_annual/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/{bldg_id.get_annual_load_curve_filename()}"
    ).exists()
    bldg_id = bldg_ids[1]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_annual/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/{bldg_id.get_annual_load_curve_filename()}"
    ).exists()


def test_fetch_monthly_load_curve(cleanup_downloads):
    # 2024 release - should work fine
    bldg_ids = [
        BuildingID(
            bldg_id=173038,
            release_year="2024",
            res_com="resstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="2",
            state="RI",
        ),
        BuildingID(
            bldg_id=119411,
            release_year="2024",
            res_com="resstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="2",
            state="RI",
        ),
    ]
    file_type = ("load_curve_monthly",)
    output_dir = Path("data")

    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 2
    assert len(failed_downloads) == 0
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_monthly/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_monthly.parquet"
    ).exists()
    bldg_id = bldg_ids[1]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_monthly/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_monthly.parquet"
    ).exists()

    monthly_load_curve = pl.read_parquet(downloaded_paths[0])
    assert "year" in monthly_load_curve.columns
    assert "month" in monthly_load_curve.columns
    assert all(monthly_load_curve["year"] == monthly_load_curve["timestamp"].dt.year())
    assert all(monthly_load_curve["month"] == monthly_load_curve["timestamp"].dt.month())


def test_fetch_hourly_load_curve(cleanup_downloads):
    # 2022 release
    bldg_ids = [
        BuildingID(
            bldg_id=386459,
            release_year="2022",
            res_com="resstock",
            weather="amy2012",
            upgrade_id="0",
            release_number="1",
            state="ME",
        ),
        BuildingID(
            bldg_id=247072,
            release_year="2022",
            res_com="resstock",
            weather="amy2012",
            upgrade_id="0",
            release_number="1",
            state="ME",
        ),
    ]
    file_type = ("load_curve_hourly",)
    output_dir = Path("data")

    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 2
    assert len(failed_downloads) == 0
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_hourly/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_hourly.parquet"
    ).exists()
    bldg_id = bldg_ids[1]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_hourly/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_hourly.parquet"
    ).exists()

    hourly_load_curve = pl.read_parquet(downloaded_paths[0])
    timestamps = hourly_load_curve["timestamp"].to_list()
    assert len(timestamps) == 8784
    assert timestamps[1] - timestamps[0] == timedelta(hours=1)
    assert "year" in hourly_load_curve.columns
    assert "month" in hourly_load_curve.columns
    assert "day" in hourly_load_curve.columns
    assert "hour" in hourly_load_curve.columns
    assert all(hourly_load_curve["year"] == hourly_load_curve["timestamp"].dt.year())
    assert all(hourly_load_curve["month"] == hourly_load_curve["timestamp"].dt.month())
    assert all(hourly_load_curve["day"] == hourly_load_curve["timestamp"].dt.day())
    assert all(hourly_load_curve["hour"] == hourly_load_curve["timestamp"].dt.hour())

    bldg_ids = [
        BuildingID(
            bldg_id=43469,
            release_year="2022",
            res_com="resstock",
            weather="amy2012",
            upgrade_id="0",
            release_number="1.1",
            state="FL",
        ),
        BuildingID(
            bldg_id=25395,
            release_year="2022",
            res_com="resstock",
            weather="amy2012",
            upgrade_id="0",
            release_number="1.1",
            state="FL",
        ),
    ]
    file_type = ("load_curve_hourly",)
    output_dir = Path("data")

    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 2
    assert len(failed_downloads) == 0
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_hourly/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_hourly.parquet"
    ).exists()
    bldg_id = bldg_ids[1]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_hourly/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_hourly.parquet"
    ).exists()

    hourly_load_curve = pl.read_parquet(downloaded_paths[0])
    timestamps = hourly_load_curve["timestamp"].to_list()
    assert len(timestamps) == 8784
    assert timestamps[1] - timestamps[0] == timedelta(hours=1)
    assert "year" in hourly_load_curve.columns
    assert "month" in hourly_load_curve.columns
    assert "day" in hourly_load_curve.columns
    assert "hour" in hourly_load_curve.columns
    assert all(hourly_load_curve["year"] == hourly_load_curve["timestamp"].dt.year())
    assert all(hourly_load_curve["month"] == hourly_load_curve["timestamp"].dt.month())
    assert all(hourly_load_curve["day"] == hourly_load_curve["timestamp"].dt.day())
    assert all(hourly_load_curve["hour"] == hourly_load_curve["timestamp"].dt.hour())

    bldg_ids = [
        BuildingID(
            bldg_id=349561,
            release_year="2022",
            res_com="resstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="1.1",
            state="CO",
        ),
        BuildingID(
            bldg_id=547230,
            release_year="2022",
            res_com="resstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="1.1",
            state="CO",
        ),
    ]
    file_type = ("load_curve_hourly",)
    output_dir = Path("data")

    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 2
    assert len(failed_downloads) == 0
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_hourly/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_hourly.parquet"
    ).exists()
    bldg_id = bldg_ids[1]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_hourly/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_hourly.parquet"
    ).exists()

    hourly_load_curve = pl.read_parquet(downloaded_paths[0])
    timestamps = hourly_load_curve["timestamp"].to_list()
    assert len(timestamps) == 8760
    assert timestamps[1] - timestamps[0] == timedelta(hours=1)
    assert "year" in hourly_load_curve.columns
    assert "month" in hourly_load_curve.columns
    assert "day" in hourly_load_curve.columns
    assert "hour" in hourly_load_curve.columns
    assert all(hourly_load_curve["year"] == hourly_load_curve["timestamp"].dt.year())
    assert all(hourly_load_curve["month"] == hourly_load_curve["timestamp"].dt.month())
    assert all(hourly_load_curve["day"] == hourly_load_curve["timestamp"].dt.day())
    assert all(hourly_load_curve["hour"] == hourly_load_curve["timestamp"].dt.hour())

    bldg_ids = [
        BuildingID(
            bldg_id=37283,
            release_year="2022",
            res_com="resstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="1",
            state="GA",
        ),
        BuildingID(
            bldg_id=436794,
            release_year="2022",
            res_com="resstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="1",
            state="GA",
        ),
    ]
    file_type = ("load_curve_hourly",)
    output_dir = Path("data")

    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 2
    assert len(failed_downloads) == 0
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_hourly/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_hourly.parquet"
    ).exists()
    bldg_id = bldg_ids[1]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_hourly/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_hourly.parquet"
    ).exists()

    hourly_load_curve = pl.read_parquet(downloaded_paths[0])
    timestamps = hourly_load_curve["timestamp"].to_list()
    assert len(timestamps) == 8760
    assert timestamps[1] - timestamps[0] == timedelta(hours=1)
    assert "year" in hourly_load_curve.columns
    assert "month" in hourly_load_curve.columns
    assert "day" in hourly_load_curve.columns
    assert "hour" in hourly_load_curve.columns
    assert all(hourly_load_curve["year"] == hourly_load_curve["timestamp"].dt.year())
    assert all(hourly_load_curve["month"] == hourly_load_curve["timestamp"].dt.month())
    assert all(hourly_load_curve["day"] == hourly_load_curve["timestamp"].dt.day())
    assert all(hourly_load_curve["hour"] == hourly_load_curve["timestamp"].dt.hour())

    bldg_ids = [
        BuildingID(
            bldg_id=276508,
            release_year="2022",
            res_com="resstock",
            weather="tmy3",
            upgrade_id="0",
            release_number="1",
            state="DE",
        ),
        BuildingID(
            bldg_id=330393,
            release_year="2022",
            res_com="resstock",
            weather="tmy3",
            upgrade_id="0",
            release_number="1",
            state="DE",
        ),
    ]
    file_type = ("load_curve_hourly",)
    output_dir = Path("data")

    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 2
    assert len(failed_downloads) == 0
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_hourly/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_hourly.parquet"
    ).exists()
    bldg_id = bldg_ids[1]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_hourly/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_hourly.parquet"
    ).exists()

    bldg_ids = [
        BuildingID(
            bldg_id=394846,
            release_year="2022",
            res_com="resstock",
            weather="tmy3",
            upgrade_id="0",
            release_number="1.1",
            state="IN",
        ),
        BuildingID(
            bldg_id=159923,
            release_year="2022",
            res_com="resstock",
            weather="tmy3",
            upgrade_id="0",
            release_number="1.1",
            state="IN",
        ),
    ]
    file_type = ("load_curve_hourly",)
    output_dir = Path("data")

    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 2
    assert len(failed_downloads) == 0
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_hourly/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_hourly.parquet"
    ).exists()
    bldg_id = bldg_ids[1]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_hourly/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_hourly.parquet"
    ).exists()

    hourly_load_curve = pl.read_parquet(downloaded_paths[0])
    timestamps = hourly_load_curve["timestamp"].to_list()
    assert len(timestamps) == 8760
    assert timestamps[1] - timestamps[0] == timedelta(hours=1)
    assert "year" in hourly_load_curve.columns
    assert "month" in hourly_load_curve.columns
    assert "day" in hourly_load_curve.columns
    assert "hour" in hourly_load_curve.columns
    assert all(hourly_load_curve["year"] == hourly_load_curve["timestamp"].dt.year())
    assert all(hourly_load_curve["month"] == hourly_load_curve["timestamp"].dt.month())
    assert all(hourly_load_curve["day"] == hourly_load_curve["timestamp"].dt.day())
    assert all(hourly_load_curve["hour"] == hourly_load_curve["timestamp"].dt.hour())

    bldg_ids = [
        BuildingID(
            bldg_id=37283,
            release_year="2022",
            res_com="resstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="1",
            state="GA",
        ),
        BuildingID(
            bldg_id=436794,
            release_year="2022",
            res_com="resstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="1",
            state="GA",
        ),
    ]
    file_type = ("load_curve_hourly",)
    output_dir = Path("data")

    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 2
    assert len(failed_downloads) == 0
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_hourly/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_hourly.parquet"
    ).exists()
    bldg_id = bldg_ids[1]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_hourly/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_hourly.parquet"
    ).exists()

    hourly_load_curve = pl.read_parquet(downloaded_paths[0])
    timestamps = hourly_load_curve["timestamp"].to_list()
    assert len(timestamps) == 8760
    assert timestamps[1] - timestamps[0] == timedelta(hours=1)
    assert "year" in hourly_load_curve.columns
    assert "month" in hourly_load_curve.columns
    assert "day" in hourly_load_curve.columns
    assert "hour" in hourly_load_curve.columns
    assert all(hourly_load_curve["year"] == hourly_load_curve["timestamp"].dt.year())
    assert all(hourly_load_curve["month"] == hourly_load_curve["timestamp"].dt.month())
    assert all(hourly_load_curve["day"] == hourly_load_curve["timestamp"].dt.day())
    assert all(hourly_load_curve["hour"] == hourly_load_curve["timestamp"].dt.hour())

    # 2024 release - should work fine
    bldg_ids = [
        BuildingID(
            bldg_id=173038,
            release_year="2024",
            res_com="resstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="2",
            state="RI",
        ),
        BuildingID(
            bldg_id=119411,
            release_year="2024",
            res_com="resstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="2",
            state="RI",
        ),
    ]
    file_type = ("load_curve_hourly",)
    output_dir = Path("data")

    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 2
    assert len(failed_downloads) == 0
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_hourly/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_hourly.parquet"
    ).exists()
    bldg_id = bldg_ids[1]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_hourly/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_hourly.parquet"
    ).exists()

    hourly_load_curve = pl.read_parquet(downloaded_paths[0])
    timestamps = hourly_load_curve["timestamp"].to_list()
    assert len(timestamps) == 8760
    assert timestamps[1] - timestamps[0] == timedelta(hours=1)
    assert "year" in hourly_load_curve.columns
    assert "month" in hourly_load_curve.columns
    assert "day" in hourly_load_curve.columns
    assert "hour" in hourly_load_curve.columns
    assert all(hourly_load_curve["year"] == hourly_load_curve["timestamp"].dt.year())
    assert all(hourly_load_curve["month"] == hourly_load_curve["timestamp"].dt.month())
    assert all(hourly_load_curve["day"] == hourly_load_curve["timestamp"].dt.day())
    assert all(hourly_load_curve["hour"] == hourly_load_curve["timestamp"].dt.hour())


def test_fetch_daily_load_curve(cleanup_downloads):
    # 2022 release
    bldg_ids = [
        BuildingID(
            bldg_id=386459,
            release_year="2022",
            res_com="resstock",
            weather="amy2012",
            upgrade_id="0",
            release_number="1",
            state="ME",
        ),
        BuildingID(
            bldg_id=247072,
            release_year="2022",
            res_com="resstock",
            weather="amy2012",
            upgrade_id="0",
            release_number="1",
            state="ME",
        ),
    ]
    file_type = ("load_curve_daily",)
    output_dir = Path("data")

    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 2
    assert len(failed_downloads) == 0
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_daily/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_daily.parquet"
    ).exists()
    bldg_id = bldg_ids[1]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_daily/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_daily.parquet"
    ).exists()

    daily_load_curve = pl.read_parquet(downloaded_paths[0])
    timestamps = daily_load_curve["timestamp"].to_list()
    assert len(timestamps) == 366
    assert timestamps[1] - timestamps[0] == timedelta(days=1)
    assert "year" in daily_load_curve.columns
    assert "month" in daily_load_curve.columns
    assert "day" in daily_load_curve.columns
    assert all(daily_load_curve["year"] == daily_load_curve["timestamp"].dt.year())
    assert all(daily_load_curve["month"] == daily_load_curve["timestamp"].dt.month())
    assert all(daily_load_curve["day"] == daily_load_curve["timestamp"].dt.day())

    bldg_ids = [
        BuildingID(
            bldg_id=43469,
            release_year="2022",
            res_com="resstock",
            weather="amy2012",
            upgrade_id="0",
            release_number="1.1",
            state="FL",
        ),
        BuildingID(
            bldg_id=25395,
            release_year="2022",
            res_com="resstock",
            weather="amy2012",
            upgrade_id="0",
            release_number="1.1",
            state="FL",
        ),
    ]
    file_type = ("load_curve_daily",)
    output_dir = Path("data")

    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 2
    assert len(failed_downloads) == 0
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_daily/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_daily.parquet"
    ).exists()
    bldg_id = bldg_ids[1]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_daily/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_daily.parquet"
    ).exists()

    daily_load_curve = pl.read_parquet(downloaded_paths[0])
    timestamps = daily_load_curve["timestamp"].to_list()
    assert len(timestamps) == 366
    assert timestamps[1] - timestamps[0] == timedelta(days=1)
    assert "year" in daily_load_curve.columns
    assert "month" in daily_load_curve.columns
    assert "day" in daily_load_curve.columns
    assert all(daily_load_curve["year"] == daily_load_curve["timestamp"].dt.year())
    assert all(daily_load_curve["month"] == daily_load_curve["timestamp"].dt.month())
    assert all(daily_load_curve["day"] == daily_load_curve["timestamp"].dt.day())

    bldg_ids = [
        BuildingID(
            bldg_id=349561,
            release_year="2022",
            res_com="resstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="1.1",
            state="CO",
        ),
        BuildingID(
            bldg_id=547230,
            release_year="2022",
            res_com="resstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="1.1",
            state="CO",
        ),
    ]
    file_type = ("load_curve_daily",)
    output_dir = Path("data")

    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 2
    assert len(failed_downloads) == 0
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_daily/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_daily.parquet"
    ).exists()
    bldg_id = bldg_ids[1]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_daily/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_daily.parquet"
    ).exists()

    daily_load_curve = pl.read_parquet(downloaded_paths[0])
    timestamps = daily_load_curve["timestamp"].to_list()
    assert len(timestamps) == 365
    assert timestamps[1] - timestamps[0] == timedelta(days=1)
    assert "year" in daily_load_curve.columns
    assert "month" in daily_load_curve.columns
    assert "day" in daily_load_curve.columns
    assert all(daily_load_curve["year"] == daily_load_curve["timestamp"].dt.year())
    assert all(daily_load_curve["month"] == daily_load_curve["timestamp"].dt.month())
    assert all(daily_load_curve["day"] == daily_load_curve["timestamp"].dt.day())

    bldg_ids = [
        BuildingID(
            bldg_id=37283,
            release_year="2022",
            res_com="resstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="1",
            state="GA",
        ),
        BuildingID(
            bldg_id=436794,
            release_year="2022",
            res_com="resstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="1",
            state="GA",
        ),
    ]
    file_type = ("load_curve_daily",)
    output_dir = Path("data")

    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 2
    assert len(failed_downloads) == 0
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_daily/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_daily.parquet"
    ).exists()
    bldg_id = bldg_ids[1]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_daily/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_daily.parquet"
    ).exists()

    daily_load_curve = pl.read_parquet(downloaded_paths[0])
    timestamps = daily_load_curve["timestamp"].to_list()
    assert len(timestamps) == 365
    assert timestamps[1] - timestamps[0] == timedelta(days=1)
    assert "year" in daily_load_curve.columns
    assert "month" in daily_load_curve.columns
    assert "day" in daily_load_curve.columns
    assert all(daily_load_curve["year"] == daily_load_curve["timestamp"].dt.year())
    assert all(daily_load_curve["month"] == daily_load_curve["timestamp"].dt.month())
    assert all(daily_load_curve["day"] == daily_load_curve["timestamp"].dt.day())

    bldg_ids = [
        BuildingID(
            bldg_id=276508,
            release_year="2022",
            res_com="resstock",
            weather="tmy3",
            upgrade_id="0",
            release_number="1",
            state="DE",
        ),
        BuildingID(
            bldg_id=330393,
            release_year="2022",
            res_com="resstock",
            weather="tmy3",
            upgrade_id="0",
            release_number="1",
            state="DE",
        ),
    ]
    file_type = ("load_curve_daily",)
    output_dir = Path("data")

    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 2
    assert len(failed_downloads) == 0
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_daily/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_daily.parquet"
    ).exists()
    bldg_id = bldg_ids[1]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_daily/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_daily.parquet"
    ).exists()

    bldg_ids = [
        BuildingID(
            bldg_id=394846,
            release_year="2022",
            res_com="resstock",
            weather="tmy3",
            upgrade_id="0",
            release_number="1.1",
            state="IN",
        ),
        BuildingID(
            bldg_id=159923,
            release_year="2022",
            res_com="resstock",
            weather="tmy3",
            upgrade_id="0",
            release_number="1.1",
            state="IN",
        ),
    ]
    file_type = ("load_curve_daily",)
    output_dir = Path("data")

    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 2
    assert len(failed_downloads) == 0
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_daily/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_daily.parquet"
    ).exists()
    bldg_id = bldg_ids[1]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_daily/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_daily.parquet"
    ).exists()

    daily_load_curve = pl.read_parquet(downloaded_paths[0])
    timestamps = daily_load_curve["timestamp"].to_list()
    assert len(timestamps) == 365
    assert timestamps[1] - timestamps[0] == timedelta(days=1)
    assert "year" in daily_load_curve.columns
    assert "month" in daily_load_curve.columns
    assert "day" in daily_load_curve.columns
    assert all(daily_load_curve["year"] == daily_load_curve["timestamp"].dt.year())
    assert all(daily_load_curve["month"] == daily_load_curve["timestamp"].dt.month())
    assert all(daily_load_curve["day"] == daily_load_curve["timestamp"].dt.day())

    bldg_ids = [
        BuildingID(
            bldg_id=37283,
            release_year="2022",
            res_com="resstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="1",
            state="GA",
        ),
        BuildingID(
            bldg_id=436794,
            release_year="2022",
            res_com="resstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="1",
            state="GA",
        ),
    ]
    file_type = ("load_curve_daily",)
    output_dir = Path("data")

    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 2
    assert len(failed_downloads) == 0
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_daily/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_daily.parquet"
    ).exists()
    bldg_id = bldg_ids[1]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_daily/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_daily.parquet"
    ).exists()

    daily_load_curve = pl.read_parquet(downloaded_paths[0])
    timestamps = daily_load_curve["timestamp"].to_list()
    assert len(timestamps) == 365
    assert timestamps[1] - timestamps[0] == timedelta(days=1)
    assert "year" in daily_load_curve.columns
    assert "month" in daily_load_curve.columns
    assert "day" in daily_load_curve.columns
    assert all(daily_load_curve["year"] == daily_load_curve["timestamp"].dt.year())
    assert all(daily_load_curve["month"] == daily_load_curve["timestamp"].dt.month())
    assert all(daily_load_curve["day"] == daily_load_curve["timestamp"].dt.day())

    # 2024 release - should work fine
    bldg_ids = [
        BuildingID(
            bldg_id=173038,
            release_year="2024",
            res_com="resstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="2",
            state="RI",
        ),
        BuildingID(
            bldg_id=119411,
            release_year="2024",
            res_com="resstock",
            weather="amy2018",
            upgrade_id="0",
            release_number="2",
            state="RI",
        ),
    ]
    file_type = ("load_curve_daily",)
    output_dir = Path("data")

    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 2
    assert len(failed_downloads) == 0
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_daily/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_daily.parquet"
    ).exists()
    bldg_id = bldg_ids[1]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_daily/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/bldg{bldg_id.bldg_id:07d}_load_curve_daily.parquet"
    ).exists()

    daily_load_curve = pl.read_parquet(downloaded_paths[0])
    timestamps = daily_load_curve["timestamp"].to_list()
    assert len(timestamps) == 365
    assert timestamps[1] - timestamps[0] == timedelta(days=1)
    assert "year" in daily_load_curve.columns
    assert "month" in daily_load_curve.columns
    assert "day" in daily_load_curve.columns
    assert all(daily_load_curve["year"] == daily_load_curve["timestamp"].dt.year())
    assert all(daily_load_curve["month"] == daily_load_curve["timestamp"].dt.month())
    assert all(daily_load_curve["day"] == daily_load_curve["timestamp"].dt.day())


def test_fetch_weather_station_name(cleanup_downloads):
    bldg_ids = [
        BuildingID(
            bldg_id=67, release_year="2022", res_com="resstock", weather="amy2012", upgrade_id="0", release_number="1"
        ),
        BuildingID(
            bldg_id=69, release_year="2022", res_com="resstock", weather="amy2012", upgrade_id="0", release_number="1"
        ),
        BuildingID(
            bldg_id=132, release_year="2022", res_com="resstock", weather="amy2012", upgrade_id="0", release_number="1"
        ),
        BuildingID(
            bldg_id=161, release_year="2022", res_com="resstock", weather="amy2012", upgrade_id="0", release_number="1"
        ),
    ]

    expected_weather_station_names = [
        "G3600130",
        "G3600810",
        "G3600810",
        "G3600710",
    ]

    for i, bldg_id in enumerate(bldg_ids):
        weather_station_name = bldg_id.get_weather_station_name()
        assert weather_station_name == expected_weather_station_names[i]


def test_fetch_weather_file(cleanup_downloads, buildstock_releases_json):
    bldg_ids = [
        BuildingID(
            bldg_id=376421,
            release_year="2022",
            res_com="resstock",
            weather="amy2012",
            upgrade_id="0",
            release_number="1",
            state="NY",
        ),
        BuildingID(
            bldg_id=347694,
            release_year="2022",
            res_com="resstock",
            weather="amy2012",
            upgrade_id="0",
            release_number="1",
            state="NY",
        ),
        BuildingID(
            bldg_id=47568,
            release_year="2022",
            res_com="resstock",
            weather="amy2012",
            upgrade_id="0",
            release_number="1",
            state="NY",
        ),
        BuildingID(
            bldg_id=200309,
            release_year="2022",
            res_com="resstock",
            weather="amy2012",
            upgrade_id="0",
            release_number="1",
            state="NY",
        ),
    ]

    file_type = ("weather",)
    output_dir = Path("data")

    release_name = "res_2022_amy2012_1"
    weather_states = buildstock_releases_json[release_name]["weather_map_available_states"]

    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir, weather_states=weather_states)
    assert len(downloaded_paths) == len(bldg_ids)
    assert len(failed_downloads) == 0
    for bldg_id in bldg_ids:
        assert Path(
            f"data/{bldg_id.get_release_name()}/weather/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/{bldg_id.get_weather_station_name()}.csv"
        ).exists()

    # Invalid weather file test
    bldg_ids = [
        BuildingID(
            bldg_id=67, release_year="2024", res_com="comstock", weather="amy2018", upgrade_id="0", release_number="1"
        ),
    ]
    file_type = ("weather",)
    output_dir = Path("data")

    release_name = "com_2024_amy2018_1"
    if "weather_map_available_states" in buildstock_releases_json[release_name]:
        weather_states = buildstock_releases_json[release_name]["weather_map_available_states"]
    else:
        weather_states = []
    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir, weather_states=weather_states)
    assert len(downloaded_paths) == 0
    assert len(failed_downloads) == 1


def _verify_aggregation_values(matching_15min, row, column_aggregations):
    """Verify that aggregated values match the expected aggregation of 15-minute data."""
    for name, rule in column_aggregations.items():
        if name == "timestamp":
            continue

        if name not in matching_15min.columns:
            continue

        if rule == "sum":
            # Use a small tolerance for floating point precision issues
            assert abs(matching_15min[name].sum() - row[name]) < 0.1
        elif rule == "mean":
            assert abs(matching_15min[name].mean() - row[name]) < 0.1
        elif rule == "first":
            assert abs(matching_15min[name].first() - row[name]) < 0.1


def _analyze_aggregation_data(load_curve_15min, load_curve_aggregate, column_aggregations, aggregate_timestep):
    """Analyze aggregation data by finding matching 15-minute data for each timestep."""
    for row in load_curve_aggregate.iter_rows(named=True):
        agg_timestamp = row["timestamp"]

        if aggregate_timestep == "hourly":
            # Use the same grouping logic as the aggregation function
            grouping_key = agg_timestamp.strftime("%Y-%m-%d-%H")

            # Find matching 15-minute data using the same grouping logic
            matching_15min = load_curve_15min.filter(pl.col("timestamp").dt.strftime("%Y-%m-%d-%H") == grouping_key)

        elif aggregate_timestep == "daily":
            # Use the same grouping logic as the aggregation function
            grouping_key = agg_timestamp.strftime("%Y-%m-%d")

            # Find all 15-minute data for this day using the same grouping logic
            matching_15min = load_curve_15min.filter(pl.col("timestamp").dt.strftime("%Y-%m-%d") == grouping_key)

        elif aggregate_timestep == "monthly":
            # Use the same grouping logic as the aggregation function
            grouping_key = agg_timestamp.strftime("%Y-%m")

            # Find all 15-minute data for this month using the same grouping logic
            matching_15min = load_curve_15min.filter(pl.col("timestamp").dt.strftime("%Y-%m") == grouping_key)

        # Verify aggregation values
        _verify_aggregation_values(matching_15min, row, column_aggregations)


def test_aggregation_functions(cleanup_downloads):
    """Test aggregation functions for different time steps."""
    aggregate_timesteps = ["hourly", "monthly", "daily"]

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
    ]
    file_type = ("load_curve_15min",)
    output_dir = Path("data")
    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 1
    assert len(failed_downloads) == 0
    load_curve_15min = pl.read_parquet(downloaded_paths[0])

    load_curve_map = LOAD_CURVE_COLUMN_AGGREGATION.joinpath("2024_resstock_load_curve_columns.csv")
    aggregation_rules = pl.read_csv(load_curve_map)
    column_aggregations = dict(zip(aggregation_rules["name"], aggregation_rules["Aggregate_function"]))

    for aggregate_timestep in aggregate_timesteps:
        # Create a copy of the 15min data and apply the same timestamp adjustment as the aggregation function
        load_curve_15min_processed = load_curve_15min.with_columns(
            (pl.col("timestamp") - timedelta(minutes=15)).alias("timestamp")
        )

        load_curve_aggregate = _aggregate_load_curve_aggregate(load_curve_15min, aggregate_timestep, "2024")
        if aggregate_timestep == "monthly":
            assert load_curve_aggregate.shape[0] == 12
        elif aggregate_timestep == "hourly":
            assert load_curve_aggregate.shape[0] == 8760
        elif aggregate_timestep == "daily":
            assert load_curve_aggregate.shape[0] == 365

        # Analyze aggregation data for this timestep using the processed 15min data
        _analyze_aggregation_data(
            load_curve_15min_processed, load_curve_aggregate, column_aggregations, aggregate_timestep
        )

    bldg_ids = [
        BuildingID(
            bldg_id=386459,
            release_year="2022",
            res_com="resstock",
            weather="amy2012",
            upgrade_id="0",
            release_number="1",
            state="ME",
        )
    ]
    file_type = ("load_curve_15min",)
    output_dir = Path("data")
    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 1
    assert len(failed_downloads) == 0
    load_curve_15min = pl.read_parquet(downloaded_paths[0])

    load_curve_map = LOAD_CURVE_COLUMN_AGGREGATION.joinpath("2022_resstock_load_curve_columns.csv")
    aggregation_rules = pl.read_csv(load_curve_map)
    column_aggregations = dict(zip(aggregation_rules["name"], aggregation_rules["Aggregate_function"]))

    for aggregate_timestep in aggregate_timesteps:
        # Create a copy of the 15min data and apply the same timestamp adjustment as the aggregation function
        load_curve_15min_processed = load_curve_15min.with_columns(
            (pl.col("timestamp") - timedelta(minutes=15)).alias("timestamp")
        )

        load_curve_aggregate = _aggregate_load_curve_aggregate(load_curve_15min, aggregate_timestep, "2022")
        if aggregate_timestep == "monthly":
            assert load_curve_aggregate.shape[0] == 12
        elif aggregate_timestep == "hourly":
            assert load_curve_aggregate.shape[0] == 8784
        elif aggregate_timestep == "daily":
            assert load_curve_aggregate.shape[0] == 366

        # Analyze aggregation data for this timestep using the processed 15min data
        _analyze_aggregation_data(
            load_curve_15min_processed, load_curve_aggregate, column_aggregations, aggregate_timestep
        )


def test_fetch_trip_schedules(cleanup_downloads):
    # 2025 comstock amy2018 v1 release - should work fine
    bldg_ids = [
        BuildingID(
            bldg_id=320214,
            release_year="2024",
            res_com="resstock",
            weather="tmy3",
            upgrade_id="0",
            release_number="2",
            state="NY",
        )
    ]
    file_type = ("trip_schedules",)
    output_dir = Path("data")
    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 1
    assert len(failed_downloads) == 0
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/trip_schedules/state={bldg_id.state}/trip_schedules.parquet"
    ).exists()

    # 2025 comstock amy2018 v1 release - should work fine
    bldg_ids = [
        BuildingID(
            bldg_id=320214,
            release_year="2024",
            res_com="resstock",
            weather="tmy3",
            upgrade_id="0",
            release_number="2",
            state="NY",
        ),
        BuildingID(
            bldg_id=216071,
            release_year="2024",
            res_com="resstock",
            weather="tmy3",
            upgrade_id="0",
            release_number="2",
            state="NY",
        ),
    ]
    file_type = ("trip_schedules",)
    output_dir = Path("data")
    downloaded_paths, failed_downloads = fetch_bldg_data(bldg_ids, file_type, output_dir)
    assert len(downloaded_paths) == 1
    assert len(failed_downloads) == 0
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/trip_schedules/state={bldg_id.state}/trip_schedules.parquet"
    ).exists()
