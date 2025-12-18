import shutil
from pathlib import Path

import polars as pl
import pytest

from buildstock_fetch.building import BuildingID
from buildstock_fetch.main import fetch_bldg_data
from buildstock_fetch.read import BuildStockRead, FileTypeNotAvailableError, NoUpgradesFoundError


@pytest.fixture(scope="function")
def cleanup_downloads():
    """Fixture to clean up downloaded data before and after tests.

    This fixture:
    1. Removes any existing 'data' directory before the test runs
    2. Yields control to the test
    3. Removes the 'data' directory after the test completes

    This ensures each test starts with a clean slate and doesn't leave
    downloaded files behind.
    """
    # Setup - clean up any existing files before test
    data_dir = Path("data")

    if data_dir.exists():
        shutil.rmtree(data_dir)

    yield

    # Teardown - clean up downloaded files after test
    if data_dir.exists():
        shutil.rmtree(data_dir)


@pytest.mark.usefixtures("cleanup_downloads")
def test_read_metadata_single_upgrade():
    bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
    _ = fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))
    bsr = BuildStockRead(
        data_path="./data",
        release="res_2022_tmy3_1",
        states="NY",
    )
    metadata = bsr.read_metadata(upgrades="0")
    assert isinstance(metadata, pl.LazyFrame)
    df = metadata.collect()
    assert "bldg_id" in df.columns
    assert df.height > 0


@pytest.mark.usefixtures("cleanup_downloads")
def test_read_metadata_multiple_upgrades():
    """Test reading metadata for multiple upgrades."""
    bldg_ids = [
        BuildingID(bldg_id=7, upgrade_id="0"),
        BuildingID(bldg_id=7, upgrade_id="1"),
    ]
    fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

    bsr = BuildStockRead(
        data_path="data",
        release="res_2022_tmy3_1",
        states="NY",
    )

    metadata = bsr.read_metadata(upgrades=["0", "1"])

    df = metadata.collect()

    # Should have records from both upgrades
    assert 0 in df["upgrade"].to_list()
    assert 1 in df["upgrade"].to_list()


@pytest.mark.usefixtures("cleanup_downloads")
def test_read_metadata_auto_detect_upgrades():
    """Test reading metadata with auto-detected upgrades."""

    bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
    fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

    bsr = BuildStockRead(
        data_path="data",
        release="res_2022_tmy3_1",
        states="NY",
    )

    # Call without specifying upgrades - should auto-detect
    metadata = bsr.read_metadata()
    df = metadata.collect()
    assert df.height > 0


@pytest.mark.usefixtures("cleanup_downloads")
def test_read_load_curve_15min():
    """Test reading 15-minute load curve data."""
    bldg_ids = [
        BuildingID(
            bldg_id=122289, upgrade_id="0", release_year="2024", weather="tmy3", release_number="2", res_com="resstock"
        )
    ]
    fetch_bldg_data(bldg_ids, ("load_curve_15min",), Path("data"))

    bsr = BuildStockRead(
        data_path="data",
        release="res_2024_tmy3_2",
        states="NY",
    )

    load_curve = bsr.read_load_curve_15min(upgrades="0")
    assert isinstance(load_curve, pl.LazyFrame)

    df = load_curve.collect()
    assert df.height > 0
    assert "timestamp" in df.columns
    assert "bldg_id" in df.columns


@pytest.mark.usefixtures("cleanup_downloads")
def test_read_load_curve_hourly():
    """Test reading hourly load curve data."""
    bldg_ids = [
        BuildingID(
            bldg_id=122289,
            upgrade_id="0",
            release_year="2024",
            weather="tmy3",
            release_number="2",
            res_com="resstock",
        )
    ]
    fetch_bldg_data(bldg_ids, ("load_curve_hourly",), Path("data"))

    bsr = BuildStockRead(
        data_path="data",
        release="res_2024_tmy3_2",
        states="NY",
    )

    load_curve = bsr.read_load_curve_hourly(upgrades="0")
    assert isinstance(load_curve, pl.LazyFrame)

    df = load_curve.collect()
    assert df.height > 0
    assert "timestamp" in df.columns


@pytest.mark.usefixtures("cleanup_downloads")
def test_read_load_curve_daily():
    """Test reading daily load curve data."""
    bldg_ids = [
        BuildingID(
            bldg_id=122289,
            upgrade_id="0",
            release_year="2024",
            weather="tmy3",
            release_number="2",
            res_com="resstock",
        )
    ]
    fetch_bldg_data(bldg_ids, ("load_curve_daily",), Path("data"))
    bsr = BuildStockRead(
        data_path="data",
        release="res_2024_tmy3_2",
        states="NY",
    )

    load_curve = bsr.read_load_curve_daily(upgrades="0")
    assert isinstance(load_curve, pl.LazyFrame)

    df = load_curve.collect()

    assert df.height > 0

    assert "timestamp" in df.columns


@pytest.mark.usefixtures("cleanup_downloads")
def test_read_load_curve_annual():
    """Test reading annual load curve data."""
    bldg_ids = [
        BuildingID(
            bldg_id=122289,
            upgrade_id="0",
            release_year="2024",
            weather="tmy3",
            release_number="2",
            res_com="resstock",
        )
    ]
    fetch_bldg_data(bldg_ids, ("load_curve_annual",), Path("data"))
    bsr = BuildStockRead(
        data_path="data",
        release="res_2024_tmy3_2",
        states="NY",
    )

    load_curve = bsr.read_load_curve_annual(upgrades="0")
    assert isinstance(load_curve, pl.LazyFrame)

    df = load_curve.collect()
    assert df.height > 0
    assert "bldg_id" in df.columns


@pytest.mark.usefixtures("cleanup_downloads")
def test_read_load_curve_not_available_for_release():
    """Test that LoadCurveNotFoundError is raised for unavailable data type in release."""
    bldg_ids = [
        BuildingID(
            bldg_id=122289,
            upgrade_id="0",
            release_year="2024",
            weather="tmy3",
            release_number="2",
            res_com="resstock",
        )
    ]
    fetch_bldg_data(bldg_ids, ("load_curve_annual",), Path("data"))
    bsr = BuildStockRead(
        data_path="data",
        release="res_2024_tmy3_2",
        states="NY",
    )
    # RES_2024_TMY3_2 should have load_curve_15min available, but not on disk
    with pytest.raises((FileTypeNotAvailableError, NoUpgradesFoundError)):
        bsr.read_load_curve_15min(upgrades="0")


@pytest.mark.usefixtures("cleanup_downloads")
def test_sampling_with_seed():
    """Test that sampling with seed is reproducible."""
    bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]

    fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))
    bldg_ids = [
        BuildingID(
            bldg_id=122289,
            upgrade_id="0",
            release_year="2024",
            weather="tmy3",
            release_number="2",
            res_com="resstock",
        )
    ]
    fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))
    bsr1 = BuildStockRead(
        data_path="data",
        release="res_2024_tmy3_2",
        states="NY",
        sample_n=100,
        random=42,
    )
    bsr2 = BuildStockRead(
        data_path="data",
        release="res_2024_tmy3_2",
        states="NY",
        sample_n=100,
        random=42,
    )

    # Should have same sampled buildings
    assert bsr1.sampled_buildings == bsr2.sampled_buildings
