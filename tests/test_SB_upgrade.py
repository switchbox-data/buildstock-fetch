import sys
import pytest
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from buildstock_fetch.building import BuildingID

def test_SB_upgrade_load_curve_15min():
    bldg_id = BuildingID(bldg_id=7, upgrade_id="17")
    output_dir = Path("data")
    output_file = output_dir / bldg_id.get_release_name() / "load_curve_15min" / f"state={bldg_id.state}" / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}" / f"bldg{str(bldg_id.bldg_id).zfill(7)}-up{str(int(bldg_id.upgrade_id)).zfill(2)}.parquet"
    assert output_file.exists()

def test_SB_upgrade_load_curve_hourly():
    bldg_id = BuildingID(bldg_id=7, upgrade_id="17")
    output_dir = Path("data")
    output_file = output_dir / bldg_id.get_release_name() / "load_curve_hourly" / f"state={bldg_id.state}" / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}" / f"bldg{str(bldg_id.bldg_id).zfill(7)}-up{str(int(bldg_id.upgrade_id)).zfill(2)}.parquet"
    assert output_file.exists()

def test_SB_upgrade_load_curve_daily():
    bldg_id = BuildingID(bldg_id=7, upgrade_id="17")
    output_dir = Path("data")
    output_file = output_dir / bldg_id.get_release_name() / "load_curve_daily" / f"state={bldg_id.state}" / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}" / f"bldg{str(bldg_id.bldg_id).zfill(7)}-up{str(int(bldg_id.upgrade_id)).zfill(2)}.parquet"
    assert output_file.exists()

def test_SB_upgrade_load_curve_monthly():
    bldg_id = BuildingID(bldg_id=7, upgrade_id="17")
    output_dir = Path("data")
    output_file = output_dir / bldg_id.get_release_name() / "load_curve_monthly" / f"state={bldg_id.state}" / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}" / f"bldg{str(bldg_id.bldg_id).zfill(7)}-up{str(int(bldg_id.upgrade_id)).zfill(2)}.parquet"
    assert output_file.exists()

def test_SB_upgrade_annual_load_curve():
    bldg_id = BuildingID(bldg_id=7, upgrade_id="17")
    output_dir = Path("data")
    output_file = output_dir / bldg_id.get_release_name() / "load_curve_annual" / f"state={bldg_id.state}" / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}" / f"bldg{str(bldg_id.bldg_id).zfill(7)}-up{str(int(bldg_id.upgrade_id)).zfill(2)}.parquet"
    assert output_file.exists()

def test_SB_upgrade_metadata():
    bldg_id = BuildingID(bldg_id=7, upgrade_id="17")
    output_dir = Path("data")
    output_file = output_dir / bldg_id.get_release_name() / "metadata" / f"state={bldg_id.state}" / f"upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}" / "metadata.parquet"
    assert output_file.exists()
