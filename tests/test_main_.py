from collections.abc import Collection
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pytest

from buildstock_fetch.building_ import Building
from buildstock_fetch.main_new import download_and_process_all, list_buildings
from buildstock_fetch.releases import RELEASES
from buildstock_fetch.types import FileType, ReleaseKey, normalize_upgrade_id

TEST_METADATA_MERGING_SKIP_RELEASES: set[ReleaseKey] = {"com_2025_amy2018_3"}
TEST_METADATA_MERGING_RELEASES = RELEASES.keys - TEST_METADATA_MERGING_SKIP_RELEASES

TEST_ANNUAL_LOAD_CURVE_MERGING_SKIP_RELEASES: set[ReleaseKey] = {"com_2025_amy2018_3"}
TEST_ANNUAL_LOAD_CURVE_MERGING_RELEASES = (
    RELEASES.filter(file_types=["load_curve_annual"]).keys - TEST_ANNUAL_LOAD_CURVE_MERGING_SKIP_RELEASES
)


@pytest.fixture
def target_folder():
    with TemporaryDirectory() as td:
        yield Path(td)


def first_building_id_str(buildings: Collection[Building]) -> str:
    building = next(iter(buildings))
    return f"{building.release}-{building.upgrade}-{building.state}"


@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.network
@pytest.mark.parametrize(
    ["buildings"],
    [
        (buildings,)
        for release in sorted(RELEASES.keys)
        for state in ("NY",)
        for upgrade in (
            [normalize_upgrade_id("0")]
            + ([normalize_upgrade_id("5")] if normalize_upgrade_id("5") in RELEASES[release].upgrades else [])
        )
        if (buildings := list_buildings(release, state, upgrade, 5))
    ],
    ids=first_building_id_str,
)
async def test_metadata_has_required_fields_and_exists_in_paths(target_folder: Path, buildings: Collection[Building]):
    METADATA_COLUMNS = ["bldg_id", "upgrade", "in.", "upgrade."]
    NOT_METADATA_COLUMNS = ["out."]
    await download_and_process_all(target_folder, buildings, ["metadata"])
    filenames: set[Path] = set()
    for building in buildings:
        path = (
            target_folder
            / building.release
            / "metadata"
            / f"state={building.state}"
            / f"upgrade={building.upgrade.zfill(2)}"
        )
        files = list(path.iterdir())
        assert len(files) == 1
        assert files[0].suffix == ".parquet"
        filename = files[0]
        lf = pl.scan_parquet(filename)
        assert not lf.filter(pl.col("bldg_id") == building.id).collect().is_empty()
        filenames.add(filename)
    for filename in filenames:
        lf = pl.scan_parquet(filename)
        columns = lf.collect_schema().keys()
        for required_col in METADATA_COLUMNS:
            if required_col == "upgrade." and (
                "upgrade=00" in str(filename)
                or "res_2024_tmy3_1" in str(filename)
                or ("2022" not in str(filename) and "res_2024" not in str(filename) and "res_2025" not in str(filename))
            ):
                continue
            assert any(c.startswith(required_col) for c in columns)
        for alien_col in NOT_METADATA_COLUMNS:
            assert not any(c.startswith(alien_col) for c in columns)

    assert set(pl.read_parquet(list(filenames))["bldg_id"]) == {_.id for _ in buildings}


# @pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.network
@pytest.mark.parametrize(
    "buildings_partitioned",
    [
        [buildings[:5], buildings[-5:]]
        for release in sorted(TEST_METADATA_MERGING_RELEASES)
        for state in ("NY",)
        for upgrade in (normalize_upgrade_id("0"),)
        if (buildings := list_buildings(release, state, upgrade, 50))
    ],
    ids=lambda _: first_building_id_str(_[0]),
)
async def test_metadata_merging(target_folder: Path, buildings_partitioned: list[Collection[Building]]):
    METADATA_COLUMNS = ["bldg_id", "upgrade", "in.", "upgrade."]
    NOT_METADATA_COLUMNS = ["out."]

    for building_chunk in buildings_partitioned:
        await download_and_process_all(target_folder, building_chunk, ["metadata"])

    buildings = [building for partition in buildings_partitioned for building in partition]
    filenames: set[Path] = set()

    for building in buildings:
        path = (
            target_folder
            / building.release
            / "metadata"
            / f"state={building.state}"
            / f"upgrade={building.upgrade.zfill(2)}"
        )
        files = list(path.iterdir())
        assert len(files) == 1
        assert files[0].suffix == ".parquet"
        filename = files[0]
        lf = pl.scan_parquet(filename)
        assert not lf.filter(pl.col("bldg_id") == building.id).collect().is_empty()
        filenames.add(filename)
    for filename in filenames:
        lf = pl.scan_parquet(filename)
        columns = lf.collect_schema().keys()
        for required_col in METADATA_COLUMNS:
            if required_col == "upgrade." and (
                "upgrade=00" in str(filename)
                or "res_2024_tmy3_1" in str(filename)
                or ("2022" not in str(filename) and "res_2024" not in str(filename) and "res_2025" not in str(filename))
            ):
                continue
            assert any(c.startswith(required_col) for c in columns)
        for alien_col in NOT_METADATA_COLUMNS:
            assert not any(c.startswith(alien_col) for c in columns)

    assert set(pl.read_parquet(list(filenames))["bldg_id"]) == {_.id for _ in buildings}


RELEASES_WITH_LOAD_CURVES: list[ReleaseKey] = [
    "com_2024_amy2018_1",
    "com_2024_amy2018_2",
    "res_2022_amy2012_1.1",
    "res_2022_amy2012_1",
    "res_2022_amy2018_1.1",
    "res_2022_amy2018_1",
    "res_2022_tmy3_1",
    "res_2022_tmy3_1.1",
    "res_2024_amy2018_2",
    "res_2024_tmy3_2",
]


@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.network
@pytest.mark.parametrize(
    ["buildings"],
    [
        (buildings,)
        for release in sorted(RELEASES.filter(file_types={"load_curve_annual"}).keys)
        for state in ("NY",)
        for upgrade in (normalize_upgrade_id("0"),)
        if (buildings := list_buildings(release, state, upgrade, 5))
    ],
    ids=first_building_id_str,
)
async def test_load_curve_annual(target_folder: Path, buildings: Collection[Building]):
    ANNUAL_LOAD_CURVE_COLUMNS = ["bldg_id", "upgrade", "out."]
    NOT_ANNUAL_LOAD_CURVE_COLUMNS = ["in."]
    await download_and_process_all(target_folder, buildings, ["load_curve_annual"])
    filenames: set[Path] = set()
    for building in buildings:
        path = (
            target_folder
            / building.release
            / "load_curve_annual"
            / f"state={building.state}"
            / f"upgrade={building.upgrade.zfill(2)}"
        )
        files = list(path.iterdir())
        assert len(files) == 1
        assert files[0].suffix == ".parquet"
        filename = files[0]
        lf = pl.scan_parquet(filename)
        assert not lf.filter(pl.col("bldg_id") == building.id).collect().is_empty()
        filenames.add(filename)
    for filename in filenames:
        lf = pl.scan_parquet(filename)
        columns = lf.collect_schema().keys()
        for required_col in ANNUAL_LOAD_CURVE_COLUMNS:
            assert any(c.startswith(required_col) for c in columns)
        for alien_col in NOT_ANNUAL_LOAD_CURVE_COLUMNS:
            assert not any(c.startswith(alien_col) for c in columns)

    assert set(pl.read_parquet(list(filenames))["bldg_id"]) == {_.id for _ in buildings}


@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.network
@pytest.mark.parametrize(
    "buildings_partitioned",
    [
        [buildings[:5], buildings[-5:]]
        for release in sorted(TEST_ANNUAL_LOAD_CURVE_MERGING_RELEASES)
        for state in ("NY", "AL")
        for upgrade in (normalize_upgrade_id("0"),)
        if (buildings := list_buildings(release, state, upgrade, 50))
    ],
    ids=lambda _: first_building_id_str(_[0]),
)
async def test_annual_load_curves_merging(target_folder: Path, buildings_partitioned: list[Collection[Building]]):
    ANNUAL_LOAD_CURVE_COLUMNS = ["bldg_id", "upgrade", "out."]
    NOT_ANNUAL_LOAD_CURVE_COLUMNS = ["in."]

    for building_chunk in buildings_partitioned:
        await download_and_process_all(target_folder, building_chunk, ["load_curve_annual"])

    buildings = [building for partition in buildings_partitioned for building in partition]
    filenames: set[Path] = set()

    for building in buildings:
        path = (
            target_folder
            / building.release
            / "load_curve_annual"
            / f"state={building.state}"
            / f"upgrade={building.upgrade.zfill(2)}"
        )
        files = list(path.iterdir())
        assert len(files) == 1
        assert files[0].suffix == ".parquet"
        filename = files[0]
        lf = pl.scan_parquet(filename)
        assert not lf.filter(pl.col("bldg_id") == building.id).collect().is_empty()
        filenames.add(filename)
    for filename in filenames:
        lf = pl.scan_parquet(filename)
        columns = lf.collect_schema().keys()
        for required_col in ANNUAL_LOAD_CURVE_COLUMNS:
            assert any(c.startswith(required_col) for c in columns)
        for alien_col in NOT_ANNUAL_LOAD_CURVE_COLUMNS:
            assert not any(c.startswith(alien_col) for c in columns)

    assert set(pl.read_parquet(list(filenames))["bldg_id"]) == {_.id for _ in buildings}


@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.network
@pytest.mark.parametrize(
    ["buildings"],
    [
        (buildings,)
        for release in RELEASES_WITH_LOAD_CURVES
        for state in ("NY",)
        for upgrade in sorted(RELEASES[release].upgrades)[:2]
        if (buildings := list_buildings(release, state, upgrade, 5))
    ],
    ids=first_building_id_str,
)
async def test_load_curves(target_folder: Path, buildings: list[Building]):
    file_types: list[FileType] = ["load_curve_15min", "load_curve_hourly", "load_curve_daily", "load_curve_monthly"]
    await download_and_process_all(target_folder, buildings, file_types)
    for building in buildings:
        for file_type in file_types:
            path = (
                target_folder
                / building.release
                / file_type
                / f"state={building.state}"
                / f"upgrade={building.upgrade.zfill(2)}"
            )
            filenames = [n for n in path.iterdir() if n.suffix == ".parquet" and str(building.id) in n.name]
            assert len(filenames) == 1
