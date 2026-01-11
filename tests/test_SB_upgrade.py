import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from buildstock_fetch.building import BuildingID
from buildstock_fetch.main import fetch_bldg_data


def test_SB_upgrade_load_curve_15min():
    bldg_ids = [
        BuildingID(
            bldg_id=361520,
            release_number="2",
            release_year="2024",
            res_com="resstock",
            weather="tmy3",
            state="IN",
            upgrade_id="17",
        )
    ]
    file_type = ("load_curve_15min",)
    output_dir = Path("data")

    _, _ = fetch_bldg_data(bldg_ids, file_type, output_dir)
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_15min/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/{str(bldg_id.bldg_id)!s}-{int(bldg_id.upgrade_id)!s}.parquet"
    ).exists()


def test_SB_upgrade_load_curve_hourly():
    bldg_ids = [
        BuildingID(
            bldg_id=361520,
            release_number="2",
            release_year="2024",
            res_com="resstock",
            weather="tmy3",
            state="IN",
            upgrade_id="17",
        )
    ]
    file_type = ("load_curve_hourly",)
    output_dir = Path("data")

    _, _ = fetch_bldg_data(bldg_ids, file_type, output_dir)
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_hourly/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/{str(bldg_id.bldg_id)!s}-{int(bldg_id.upgrade_id)!s}.parquet"
    ).exists()


def test_SB_upgrade_load_curve_daily():
    bldg_ids = [
        BuildingID(
            bldg_id=361520,
            release_number="2",
            release_year="2024",
            res_com="resstock",
            weather="tmy3",
            state="IN",
            upgrade_id="17",
        )
    ]
    file_type = ("load_curve_daily",)
    output_dir = Path("data")

    _, _ = fetch_bldg_data(bldg_ids, file_type, output_dir)
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_daily/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/{str(bldg_id.bldg_id)!s}-{int(bldg_id.upgrade_id)!s}.parquet"
    ).exists()


def test_SB_upgrade_load_curve_monthly():
    bldg_ids = [
        BuildingID(
            bldg_id=361520,
            release_number="2",
            release_year="2024",
            res_com="resstock",
            weather="tmy3",
            state="IN",
            upgrade_id="17",
        )
    ]
    file_type = ("load_curve_monthly",)
    output_dir = Path("data")

    _, _ = fetch_bldg_data(bldg_ids, file_type, output_dir)
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_monthly/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/{str(bldg_id.bldg_id)!s}-{int(bldg_id.upgrade_id)!s}.parquet"
    ).exists()


def test_SB_upgrade_annual_load_curve():
    bldg_ids = [
        BuildingID(
            bldg_id=361520,
            release_number="2",
            release_year="2024",
            res_com="resstock",
            weather="tmy3",
            state="IN",
            upgrade_id="17",
        )
    ]
    file_type = ("load_curve_annual",)
    output_dir = Path("data")

    _, _ = fetch_bldg_data(bldg_ids, file_type, output_dir)
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_annual/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/{str(bldg_id.bldg_id)!s}-{int(bldg_id.upgrade_id)!s}.parquet"
    ).exists()


def test_SB_upgrade_metadata():
    bldg_ids = [
        BuildingID(
            bldg_id=361520,
            release_number="2",
            release_year="2024",
            res_com="resstock",
            weather="tmy3",
            state="IN",
            upgrade_id="17",
        )
    ]
    file_type = ("metadata",)
    output_dir = Path("data")

    _, _ = fetch_bldg_data(bldg_ids, file_type, output_dir)
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/metadata/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/metadata.parquet"
    ).exists()
