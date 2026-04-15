from pathlib import Path

import polars as pl
import pytest

from buildstock_fetch.read import BuildStockRead, NoUpgradesFoundError


def _write_parquet_partition(
    base_path: Path,
    release: str,
    file_type: str,
    state: str,
    upgrade: str,
    filename: str,
    df: pl.DataFrame,
) -> None:
    target_dir = base_path / release / file_type / f"state={state}" / f"upgrade={int(upgrade):02d}"
    target_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(target_dir / filename)


@pytest.fixture
def local_buildstock_data(tmp_path: Path) -> Path:
    release_metadata = "res_2022_tmy3_1"
    release_load_curves = "res_2024_tmy3_2"

    _write_parquet_partition(
        tmp_path,
        release_metadata,
        "metadata",
        "NY",
        "0",
        "metadata.parquet",
        pl.DataFrame(
            {
                "bldg_id": [101, 102],
                "in.geometry_floor_area": [1200, 1400],
                "weight": [1.0, 1.5],
            }
        ),
    )
    _write_parquet_partition(
        tmp_path,
        release_metadata,
        "metadata",
        "NY",
        "1",
        "metadata.parquet",
        pl.DataFrame(
            {
                "bldg_id": [201, 202],
                "in.geometry_floor_area": [1500, 1600],
                "upgrade.cost_usd": [5000, 7000],
                "weight": [0.9, 1.1],
            }
        ),
    )

    for file_type, df in {
        "load_curve_15min": pl.DataFrame(
            {
                "timestamp": ["2024-01-01T00:00:00", "2024-01-01T00:15:00"],
                "bldg_id": [301, 301],
                "out.electricity.total.energy_consumption": [1.2, 1.3],
            }
        ),
        "load_curve_hourly": pl.DataFrame(
            {
                "timestamp": ["2024-01-01T00:00:00", "2024-01-01T01:00:00"],
                "bldg_id": [301, 301],
                "out.electricity.total.energy_consumption": [4.8, 5.2],
            }
        ),
        "load_curve_daily": pl.DataFrame(
            {
                "timestamp": ["2024-01-01", "2024-01-02"],
                "bldg_id": [301, 301],
                "out.electricity.total.energy_consumption": [110.0, 111.0],
            }
        ),
        "load_curve_annual": pl.DataFrame(
            {
                "bldg_id": [301],
                "out.electricity.total.energy_consumption": [40123.0],
                "weight": [1.0],
            }
        ),
    }.items():
        _write_parquet_partition(
            tmp_path,
            release_load_curves,
            file_type,
            "NY",
            "0",
            f"{file_type}.parquet",
            df,
        )

    return tmp_path


def test_read_metadata_single_upgrade(local_buildstock_data: Path):
    """Reads baseline metadata from a single on-disk upgrade partition."""
    bsr = BuildStockRead(data_path=local_buildstock_data, release="res_2022_tmy3_1", states="NY")

    df = bsr.read_metadata(upgrades="0").collect()

    assert set(df["bldg_id"]) == {101, 102}
    assert set(df["upgrade"]) == {0}
    assert "bldg_id" in df.columns


def test_read_metadata_multiple_upgrades(local_buildstock_data: Path):
    """Reads metadata across upgrades and preserves diagonal-concat columns."""
    bsr = BuildStockRead(data_path=local_buildstock_data, release="res_2022_tmy3_1", states="NY")

    df = bsr.read_metadata(upgrades=["0", "1"]).collect()

    assert set(df["upgrade"]) == {0, 1}
    assert set(df["bldg_id"]) == {101, 102, 201, 202}
    assert "upgrade.cost_usd" in df.columns
    assert df.filter(pl.col("upgrade") == 0)["upgrade.cost_usd"].null_count() == 2


def test_read_metadata_auto_detect_upgrades(local_buildstock_data: Path):
    """Auto-detects available metadata upgrades when none are specified."""
    bsr = BuildStockRead(data_path=local_buildstock_data, release="res_2022_tmy3_1", states="NY")

    df = bsr.read_metadata().collect()

    assert df.height == 4
    assert set(df["upgrade"]) == {0, 1}


@pytest.mark.parametrize(
    ("method_name", "expected_columns"),
    [
        ("read_load_curve_15min", {"timestamp", "bldg_id"}),
        ("read_load_curve_hourly", {"timestamp", "bldg_id"}),
        ("read_load_curve_daily", {"timestamp", "bldg_id"}),
        ("read_load_curve_annual", {"bldg_id"}),
    ],
)
def test_read_load_curves(local_buildstock_data: Path, method_name: str, expected_columns: set[str]):
    """Reads a single load-curve parquet partition for each supported cadence under test."""
    bsr = BuildStockRead(data_path=local_buildstock_data, release="res_2024_tmy3_2", states="NY")

    df = getattr(bsr, method_name)(upgrades="0").collect()

    assert df.height > 0
    assert expected_columns.issubset(df.columns)
    assert set(df["upgrade"]) == {0}


def test_read_load_curve_missing_on_disk_raises(local_buildstock_data: Path):
    """Raises when the release supports a file type but no matching on-disk upgrades exist."""
    bsr = BuildStockRead(data_path=local_buildstock_data, release="com_2023_amy2018_1", states="NY")

    with pytest.raises(NoUpgradesFoundError):
        bsr.read_load_curve_annual(upgrades="0").collect()


def test_sampling_with_seed_is_reproducible(local_buildstock_data: Path):
    """Uses metadata sampling deterministically when given the same random seed."""
    bsr1 = BuildStockRead(
        data_path=local_buildstock_data,
        release="res_2022_tmy3_1",
        states="NY",
        sample_n=2,
        random=42,
    )
    bsr2 = BuildStockRead(
        data_path=local_buildstock_data,
        release="res_2022_tmy3_1",
        states="NY",
        sample_n=2,
        random=42,
    )

    assert bsr1.sampled_buildings == bsr2.sampled_buildings


@pytest.fixture
def chunked_load_curve_data(tmp_path: Path) -> Path:
    """Creates a chunked on-disk layout with multiple parquet files per state+upgrade."""
    release = "res_2024_tmy3_2"

    for upgrade, rows in {
        "0": [(1001, 1200.0), (1002, 1300.0), (1003, 1400.0), (1004, 1500.0)],
        "1": [(2001, 2200.0), (2002, 2300.0)],
    }.items():
        _write_parquet_partition(
            tmp_path,
            release,
            "metadata",
            "NY",
            upgrade,
            "metadata.parquet",
            pl.DataFrame(
                {
                    "bldg_id": [bldg_id for bldg_id, area in rows],
                    "in.geometry_floor_area": [area for _, area in rows],
                    "weight": [1.0] * len(rows),
                }
            ),
        )

    for filename, df in {
        "bucket_0000_up00.parquet": pl.DataFrame(
            {
                "timestamp": ["2024-01-01T00:00:00", "2024-01-01T00:15:00"],
                "bldg_id": [1001, 1002],
                "out.electricity.total.energy_consumption": [1.1, 1.2],
            }
        ),
        "bucket_0001_up00.parquet": pl.DataFrame(
            {
                "timestamp": ["2024-01-01T00:00:00", "2024-01-01T00:15:00"],
                "bldg_id": [1003, 1004],
                "out.electricity.total.energy_consumption": [1.3, 1.4],
            }
        ),
    }.items():
        _write_parquet_partition(tmp_path, release, "load_curve_15min", "NY", "0", filename, df)

    for filename, df in {
        "bucket_0000_up00.parquet": pl.DataFrame(
            {
                "timestamp": ["2024-01-01T00:00:00", "2024-01-01T01:00:00"],
                "bldg_id": [1001, 1002],
                "out.electricity.total.energy_consumption": [4.1, 4.2],
            }
        ),
        "bucket_0001_up00.parquet": pl.DataFrame(
            {
                "timestamp": ["2024-01-01T00:00:00", "2024-01-01T01:00:00"],
                "bldg_id": [1003, 1004],
                "out.electricity.total.energy_consumption": [5.1, 5.2],
            }
        ),
        "bucket_0002_up01.parquet": pl.DataFrame(
            {
                "timestamp": ["2024-01-01T00:00:00", "2024-01-01T01:00:00"],
                "bldg_id": [2001, 2002],
                "out.electricity.total.energy_consumption": [6.1, 6.2],
            }
        ),
    }.items():
        upgrade = "1" if "up01" in filename else "0"
        _write_parquet_partition(tmp_path, release, "load_curve_hourly", "NY", upgrade, filename, df)

    for filename, df in {
        "bucket_0000_up00.parquet": pl.DataFrame(
            {
                "timestamp": ["2024-01-01", "2024-02-01"],
                "bldg_id": [1001, 1002],
                "out.electricity.total.energy_consumption": [100.0, 101.0],
            }
        ),
        "bucket_0001_up00.parquet": pl.DataFrame(
            {
                "timestamp": ["2024-01-01", "2024-02-01"],
                "bldg_id": [1003, 1004],
                "out.electricity.total.energy_consumption": [102.0, 103.0],
            }
        ),
    }.items():
        _write_parquet_partition(tmp_path, release, "load_curve_monthly", "NY", "0", filename, df)

    _write_parquet_partition(
        tmp_path,
        release,
        "load_curve_hourly",
        "AL",
        "0",
        "bucket_0000_up00.parquet",
        pl.DataFrame(
            {
                "timestamp": ["2024-01-01T00:00:00"],
                "bldg_id": [9001],
                "out.electricity.total.energy_consumption": [8.8],
            }
        ),
    )

    return tmp_path


def test_read_load_curve_hourly_multiple_chunk_files(chunked_load_curve_data: Path):
    """Reads all hourly rows when one state+upgrade is split across multiple chunk files."""
    bsr = BuildStockRead(data_path=chunked_load_curve_data, release="res_2024_tmy3_2", states="NY")

    df = bsr.read_load_curve_hourly(upgrades="0").collect()

    assert set(df["bldg_id"]) == {1001, 1002, 1003, 1004}
    assert set(df["upgrade"]) == {0}


def test_read_load_curve_monthly_multiple_chunk_files(chunked_load_curve_data: Path):
    """Reads monthly data across multiple chunk files instead of assuming one file per building."""
    bsr = BuildStockRead(data_path=chunked_load_curve_data, release="res_2024_tmy3_2", states="NY")

    df = bsr.read_load_curve_monthly(upgrades="0").collect()

    assert set(df["bldg_id"]) == {1001, 1002, 1003, 1004}
    assert "timestamp" in df.columns


def test_read_load_curve_filters_building_ids_across_chunks(chunked_load_curve_data: Path):
    """Applies bldg_id filters correctly when the matching rows live in different chunk files."""
    bsr = BuildStockRead(data_path=chunked_load_curve_data, release="res_2024_tmy3_2", states="NY")

    df = bsr.read_load_curve_hourly(upgrades="0", building_ids=[1002, 1004]).collect()

    assert set(df["bldg_id"]) == {1002, 1004}
    assert df.height == 2


def test_read_load_curve_15min_filters_subset_within_shared_bucket(chunked_load_curve_data: Path):
    """Returns only requested buildings even when a 15-minute chunk file contains extra buildings."""
    bsr = BuildStockRead(data_path=chunked_load_curve_data, release="res_2024_tmy3_2", states="NY")

    df = bsr.read_load_curve_15min(upgrades="0", building_ids=[1001]).collect()

    assert set(df["bldg_id"]) == {1001}
    assert df.height == 1


def test_read_load_curve_sampling_applies_across_chunks(chunked_load_curve_data: Path):
    """Filters load-curve rows by sampled buildings even when those buildings span chunk files."""
    bsr = BuildStockRead(
        data_path=chunked_load_curve_data,
        release="res_2024_tmy3_2",
        states="NY",
        sample_n=2,
        random=42,
    )

    sampled = bsr.sampled_buildings
    assert sampled is not None

    df = bsr.read_load_curve_hourly(upgrades="0").collect()

    assert set(df["bldg_id"]) == sampled


def test_read_load_curve_multiple_upgrades_across_chunks(chunked_load_curve_data: Path):
    """Reads chunked hourly data from more than one upgrade partition in a single query."""
    bsr = BuildStockRead(data_path=chunked_load_curve_data, release="res_2024_tmy3_2", states="NY")

    df = bsr.read_load_curve_hourly(upgrades=["0", "1"]).collect()

    assert set(df["upgrade"]) == {0, 1}
    assert set(df["bldg_id"]) == {1001, 1002, 1003, 1004, 2001, 2002}


def test_read_load_curve_multiple_states_across_chunks(chunked_load_curve_data: Path):
    """Keeps state filtering correct when multiple states each have chunked load-curve files."""
    bsr = BuildStockRead(data_path=chunked_load_curve_data, release="res_2024_tmy3_2", states="NY")

    df = bsr.read_load_curve_hourly(upgrades="0").collect()

    assert set(df["state"]) == {"NY"}
    assert 9001 not in set(df["bldg_id"])
