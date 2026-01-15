from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pytest

from buildstock_fetch.main_new import download_and_process_all, list_buildings
from buildstock_fetch.read import BuildStockRead, FileTypeNotAvailableError, NoUpgradesFoundError
from buildstock_fetch.types import normalize_upgrade_id


@pytest.fixture
def target_folder():
    with TemporaryDirectory() as td:
        yield Path(td)


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.network
async def test_read_metadata_single_upgrade(target_folder: Path):
    buildings = list_buildings("res_2022_tmy3_1", "NY", normalize_upgrade_id("0"), 1)
    _ = await download_and_process_all(target_folder, buildings, ["metadata"])
    bsr = BuildStockRead(
        data_path=target_folder,
        release="res_2022_tmy3_1",
        states="NY",
    )
    metadata = bsr.read_metadata(upgrades="0")
    assert isinstance(metadata, pl.LazyFrame)
    df = metadata.collect()
    assert "bldg_id" in df.columns
    assert df.height > 0


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.network
async def test_read_metadata_multiple_upgrades(target_folder: Path):
    """Test reading metadata for multiple upgrades."""
    buildings = [
        *list_buildings("res_2022_tmy3_1", "NY", normalize_upgrade_id("0"), 1),
        *list_buildings("res_2022_tmy3_1", "NY", normalize_upgrade_id("1"), 1),
    ]
    _ = await download_and_process_all(target_folder, buildings, ["metadata"])

    bsr = BuildStockRead(
        data_path=target_folder,
        release="res_2022_tmy3_1",
        states="NY",
    )

    metadata = bsr.read_metadata(upgrades=["0", "1"])

    df = metadata.collect()

    # Should have records from both upgrades
    assert 0 in df["upgrade"].to_list()
    assert 1 in df["upgrade"].to_list()


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.network
async def test_read_metadata_auto_detect_upgrades(target_folder: Path):
    """Test reading metadata with auto-detected upgrades."""

    buildings = list_buildings("res_2022_tmy3_1", "NY", normalize_upgrade_id("0"), 1)
    _ = await download_and_process_all(target_folder, buildings, ["metadata"])

    bsr = BuildStockRead(
        data_path=target_folder,
        release="res_2022_tmy3_1",
        states="NY",
    )

    # Call without specifying upgrades - should auto-detect
    metadata = bsr.read_metadata()
    df = metadata.collect()
    assert df.height > 0


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.network
async def test_read_load_curve_15min(target_folder: Path):
    """Test reading 15-minute load curve data."""
    buildings = list_buildings("res_2024_tmy3_2", "NY", normalize_upgrade_id("0"), 1)
    _ = await download_and_process_all(target_folder, buildings, ["load_curve_15min"])

    bsr = BuildStockRead(
        data_path=target_folder,
        release="res_2024_tmy3_2",
        states="NY",
    )

    load_curve = bsr.read_load_curve_15min(upgrades="0")
    assert isinstance(load_curve, pl.LazyFrame)

    df = load_curve.collect()
    assert df.height > 0
    assert "timestamp" in df.columns
    assert "bldg_id" in df.columns


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.network
async def test_read_load_curve_hourly(target_folder: Path):
    """Test reading hourly load curve data."""
    buildings = list_buildings("res_2024_tmy3_2", "NY", normalize_upgrade_id("0"), 1)
    _ = await download_and_process_all(target_folder, buildings, ["load_curve_hourly"])

    bsr = BuildStockRead(
        data_path=target_folder,
        release="res_2024_tmy3_2",
        states="NY",
    )

    load_curve = bsr.read_load_curve_hourly(upgrades="0")
    assert isinstance(load_curve, pl.LazyFrame)

    df = load_curve.collect()
    assert df.height > 0
    assert "timestamp" in df.columns


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.network
async def test_read_load_curve_daily(target_folder: Path):
    """Test reading daily load curve data."""
    buildings = list_buildings("res_2024_tmy3_2", "NY", normalize_upgrade_id("0"), 1)
    _ = await download_and_process_all(target_folder, buildings, ["load_curve_daily"])
    bsr = BuildStockRead(
        data_path=target_folder,
        release="res_2024_tmy3_2",
        states="NY",
    )

    load_curve = bsr.read_load_curve_daily(upgrades="0")
    assert isinstance(load_curve, pl.LazyFrame)

    df = load_curve.collect()

    assert df.height > 0

    assert "timestamp" in df.columns


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.network
async def test_read_load_curve_annual(target_folder: Path):
    """Test reading annual load curve data."""
    buildings = list_buildings("res_2024_tmy3_2", "NY", normalize_upgrade_id("0"), 1)
    _ = await download_and_process_all(target_folder, buildings, ["load_curve_annual"])
    bsr = BuildStockRead(
        data_path=target_folder,
        release="res_2024_tmy3_2",
        states="NY",
    )

    load_curve = bsr.read_load_curve_annual(upgrades="0")
    assert isinstance(load_curve, pl.LazyFrame)

    df = load_curve.collect()
    assert df.height > 0
    assert "bldg_id" in df.columns


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.network
async def test_read_load_curve_not_available_for_release(target_folder: Path):
    """Test that LoadCurveNotFoundError is raised for unavailable data type in release."""
    buildings = list_buildings("res_2024_tmy3_2", "NY", normalize_upgrade_id("0"), 1)
    _ = await download_and_process_all(target_folder, buildings, ["load_curve_annual"])
    bsr = BuildStockRead(
        data_path=target_folder,
        release="res_2024_tmy3_2",
        states="NY",
    )
    # RES_2024_TMY3_2 should have load_curve_15min available, but not on disk
    with pytest.raises((FileTypeNotAvailableError, NoUpgradesFoundError)):
        _ = bsr.read_load_curve_15min(upgrades="0")


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.network
async def test_sampling_with_seed(target_folder: Path):
    """Test that sampling with seed is reproducible."""
    buildings = list_buildings("res_2024_tmy3_2", "NY", normalize_upgrade_id("0"), 200)
    _ = await download_and_process_all(target_folder, buildings, ["metadata"])

    bsr1 = BuildStockRead(
        data_path=target_folder,
        release="res_2024_tmy3_2",
        states="NY",
        sample_n=100,
        random=42,
    )
    bsr2 = BuildStockRead(
        data_path=target_folder,
        release="res_2024_tmy3_2",
        states="NY",
        sample_n=100,
        random=42,
    )

    # Should have same sampled buildings
    assert bsr1.sampled_buildings == bsr2.sampled_buildings
