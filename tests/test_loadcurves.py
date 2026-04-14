from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from urllib.parse import urlparse

import httpx
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from buildstock_fetch.building_ import Building
from buildstock_fetch.loadcurves import download_and_process_load_curves_batch, group_buildings
from buildstock_fetch.types import FileType, UpgradeID


def _make_building(*, bldg_id: int) -> Building:
    return Building(
        id=bldg_id,
        release="res_2024_tmy3_2",
        upgrade=UpgradeID("0"),
        state="NY",
        cached_county=None,
    )


def _write_source_parquet(path: Path, *, start_hour: int, load_values: list[float], labels: list[str]) -> pl.DataFrame:
    df = pl.DataFrame({
        "timestamp": [
            datetime(2024, 1, 1, start_hour, 0, 15),
            datetime(2024, 1, 1, start_hour, 15, 15),
            datetime(2024, 1, 1, start_hour, 30, 15),
        ],
        "example_load_kw": load_values,
        "example_label": labels,
    })
    df.write_parquet(path)
    return df


def _write_aggregation_source_parquet(
    path: Path,
    *,
    timestamps: list[datetime],
    bldg_id: int,
    load_values: list[float],
    temp_values: list[float],
) -> pl.DataFrame:
    df = pl.DataFrame({
        "timestamp": timestamps,
        "bldg_id": [bldg_id] * len(timestamps),
        "example_load_kw": load_values,
        "example_temp_c": temp_values,
    })
    df.write_parquet(path)
    return df


def _install_local_download_stubs(
    monkeypatch: pytest.MonkeyPatch,
    source_by_building_id: dict[int, Path],
) -> None:
    async def fake_estimate_download_size(client: httpx.AsyncClient, urls: object) -> float:
        return float(len(source_by_building_id))

    @asynccontextmanager
    async def fake_download(client: httpx.AsyncClient, url: str, progress: object):
        building_id = int(Path(urlparse(url).path).name.split("-")[0])
        yield SimpleNamespace(name=str(source_by_building_id[building_id]))

    monkeypatch.setattr("buildstock_fetch.loadcurves.estimate_download_size", fake_estimate_download_size)
    monkeypatch.setattr("buildstock_fetch.loadcurves.download", fake_download)


def _install_partial_failure_download_stubs(
    monkeypatch: pytest.MonkeyPatch,
    source_by_building_id: dict[int, Path],
    *,
    failed_building_ids: set[int],
) -> None:
    async def fake_estimate_download_size(client: httpx.AsyncClient, urls: object) -> float:
        return float(len(source_by_building_id))

    @asynccontextmanager
    async def fake_download(client: httpx.AsyncClient, url: str, progress: object):
        building_id = int(Path(urlparse(url).path).name.split("-")[0])
        if building_id in failed_building_ids:
            msg = f"download failed for building {building_id}"
            raise RuntimeError(msg)
        yield SimpleNamespace(name=str(source_by_building_id[building_id]))

    monkeypatch.setattr("buildstock_fetch.loadcurves.estimate_download_size", fake_estimate_download_size)
    monkeypatch.setattr("buildstock_fetch.loadcurves.download", fake_download)


def _install_aggregation_rules_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load_aggregation_rules(release: str) -> list[pl.Expr]:
        return [
            pl.col("example_load_kw").sum(),
            pl.col("example_temp_c").mean(),
            pl.col("bldg_id").first(),
        ]

    monkeypatch.setattr("buildstock_fetch.loadcurves._load_aggregation_rules", fake_load_aggregation_rules)


async def _run_batch(
    tmp_path: Path,
    building: Building,
    aggregate: Literal["load_curve_15min", "load_curve_hourly", "load_curve_daily", "load_curve_monthly"],
) -> list[Path]:
    async with httpx.AsyncClient() as client:
        return await download_and_process_load_curves_batch(
            tmp_path,
            client,
            [aggregate],
            [building],
            semaphore=None,
            processing_semaphore=None,
        )


@pytest.mark.asyncio
async def test_download_and_process_load_curve_15min_same_building_twice_is_idempotent_and_preserves_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    building = _make_building(bldg_id=1234567)
    source_path = tmp_path / "source-1234567.parquet"
    expected_df = _write_source_parquet(
        source_path,
        start_hour=0,
        load_values=[1.25, 2.5, 3.75],
        labels=["a", "b", "c"],
    )
    _install_local_download_stubs(monkeypatch, {building.id: source_path})

    first_paths = await _run_batch(tmp_path, building, "load_curve_15min")
    second_paths = await _run_batch(tmp_path, building, "load_curve_15min")

    expected_target = tmp_path / building.file_path("load_curve_15min")
    assert first_paths == [expected_target]
    assert second_paths == [expected_target]
    assert expected_target.exists()

    written_files = list(expected_target.parent.glob("*.parquet"))
    assert written_files == [expected_target]

    actual_df = pl.read_parquet(expected_target)
    assert_frame_equal(actual_df, expected_df)


@pytest.mark.asyncio
async def test_download_and_process_load_curve_15min_writes_each_building_to_expected_file_and_preserves_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    building_a = _make_building(bldg_id=1234567)
    building_b = _make_building(bldg_id=2345678)
    source_a = tmp_path / "source-1234567.parquet"
    source_b = tmp_path / "source-2345678.parquet"
    expected_a = _write_source_parquet(source_a, start_hour=0, load_values=[1.0, 2.0, 3.0], labels=["a", "b", "c"])
    expected_b = _write_source_parquet(source_b, start_hour=1, load_values=[4.0, 5.0, 6.0], labels=["x", "y", "z"])
    _install_local_download_stubs(monkeypatch, {building_a.id: source_a, building_b.id: source_b})

    async with httpx.AsyncClient() as client:
        written_paths = await download_and_process_load_curves_batch(
            tmp_path,
            client,
            ["load_curve_15min"],
            [building_a, building_b],
            semaphore=None,
            processing_semaphore=None,
        )

    expected_target_a = tmp_path / building_a.file_path("load_curve_15min")
    expected_target_b = tmp_path / building_b.file_path("load_curve_15min")
    assert sorted(written_paths) == sorted([expected_target_a, expected_target_b])
    assert expected_target_a.exists()
    assert expected_target_b.exists()

    actual_a = pl.read_parquet(expected_target_a)
    actual_b = pl.read_parquet(expected_target_b)
    assert_frame_equal(actual_a, expected_a)
    assert_frame_equal(actual_b, expected_b)


@pytest.mark.asyncio
async def test_download_and_process_load_curves_batch_isolates_building_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failed_building = _make_building(bldg_id=1234567)
    successful_building = _make_building(bldg_id=2345678)
    source_path = tmp_path / "source-2345678.parquet"
    expected_df = _write_source_parquet(
        source_path,
        start_hour=1,
        load_values=[4.0, 5.0, 6.0],
        labels=["x", "y", "z"],
    )
    _install_partial_failure_download_stubs(
        monkeypatch,
        {successful_building.id: source_path},
        failed_building_ids={failed_building.id},
    )

    async with httpx.AsyncClient() as client:
        written_paths = await download_and_process_load_curves_batch(
            tmp_path,
            client,
            ["load_curve_15min"],
            [failed_building, successful_building],
            semaphore=None,
            processing_semaphore=None,
        )

    expected_successful_target = tmp_path / successful_building.file_path("load_curve_15min")
    expected_failed_target = tmp_path / failed_building.file_path("load_curve_15min")
    assert written_paths == [expected_successful_target]
    assert expected_successful_target.exists()
    assert not expected_failed_target.exists()

    actual_df = pl.read_parquet(expected_successful_target)
    assert_frame_equal(actual_df, expected_df)


@pytest.mark.asyncio
async def test_download_and_process_load_curves_batch_deduplicates_duplicate_curve_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    building = _make_building(bldg_id=1234567)
    source_path = tmp_path / "source-1234567.parquet"
    expected_df = _write_source_parquet(
        source_path,
        start_hour=0,
        load_values=[1.25, 2.5, 3.75],
        labels=["a", "b", "c"],
    )
    _install_local_download_stubs(monkeypatch, {building.id: source_path})

    async with httpx.AsyncClient() as client:
        written_paths = await download_and_process_load_curves_batch(
            tmp_path,
            client,
            ["load_curve_15min", "load_curve_15min"],
            [building],
            semaphore=None,
            processing_semaphore=None,
        )

    expected_target = tmp_path / building.file_path("load_curve_15min")
    assert written_paths == [expected_target]
    assert expected_target.exists()
    assert list(expected_target.parent.glob("*.parquet")) == [expected_target]

    actual_df = pl.read_parquet(expected_target)
    assert_frame_equal(actual_df, expected_df)


@pytest.mark.asyncio
async def test_download_and_process_load_curves_batch_returns_empty_without_estimating_or_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fail_estimate_download_size(client: httpx.AsyncClient, urls: object) -> float:
        raise AssertionError("estimate_download_size should not be called for an empty buildings list")

    monkeypatch.setattr("buildstock_fetch.loadcurves.estimate_download_size", fail_estimate_download_size)

    async with httpx.AsyncClient() as client:
        written_paths = await download_and_process_load_curves_batch(
            tmp_path,
            client,
            ["load_curve_15min"],
            [],
            semaphore=None,
            processing_semaphore=None,
        )

    assert written_paths == []


@pytest.mark.asyncio
async def test_download_and_process_load_curves_batch_merges_contents_when_two_buildings_share_an_output_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    building_a = _make_building(bldg_id=1234567)
    building_b = _make_building(bldg_id=2345678)
    source_a = tmp_path / "source-1234567.parquet"
    source_b = tmp_path / "source-2345678.parquet"
    expected_a = _write_source_parquet(source_a, start_hour=0, load_values=[1.0, 2.0, 3.0], labels=["a", "b", "c"])
    expected_b = _write_source_parquet(source_b, start_hour=1, load_values=[4.0, 5.0, 6.0], labels=["x", "y", "z"])
    _install_local_download_stubs(monkeypatch, {building_a.id: source_a, building_b.id: source_b})

    original_file_path = Building.file_path

    def fake_file_path(self: Building, file_type: FileType) -> Path:
        if self.id in {building_a.id, building_b.id} and file_type == "load_curve_15min":
            return Path(self.release) / "load_curve_15min" / "state=NY" / "upgrade=00" / "shared-output.parquet"
        return original_file_path(self, file_type)

    monkeypatch.setattr(Building, "file_path", fake_file_path)

    async with httpx.AsyncClient() as client:
        written_paths = await download_and_process_load_curves_batch(
            tmp_path,
            client,
            ["load_curve_15min"],
            [building_a, building_b],
            semaphore=None,
            processing_semaphore=None,
        )

    shared_target = tmp_path / fake_file_path(building_a, "load_curve_15min")
    expected_df = (
        pl
        .concat([expected_a, expected_b], how="vertical")
        .sort(["timestamp", "example_label"])
        .with_columns(pl.col("timestamp").cast(pl.Datetime))
    )

    assert written_paths == [shared_target, shared_target]
    assert shared_target.exists()

    actual_df = pl.read_parquet(shared_target).sort(["timestamp", "example_label"])
    expected_df = expected_df.cast(actual_df.schema)
    assert_frame_equal(actual_df, expected_df)


@pytest.mark.asyncio
async def test_download_and_process_load_curve_hourly_aggregates_values_shifts_timestamps_and_adds_schema_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    building = _make_building(bldg_id=1234567)
    source_path = tmp_path / "source-hourly.parquet"
    _write_aggregation_source_parquet(
        source_path,
        timestamps=[
            datetime(2024, 1, 1, 0, 15, 0),
            datetime(2024, 1, 1, 0, 30, 0),
            datetime(2024, 1, 1, 0, 45, 0),
            datetime(2024, 1, 1, 1, 0, 0),
            datetime(2024, 1, 1, 1, 15, 0),
            datetime(2024, 1, 1, 1, 30, 0),
            datetime(2024, 1, 1, 1, 45, 0),
            datetime(2024, 1, 1, 2, 0, 0),
        ],
        bldg_id=building.id,
        load_values=[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        temp_values=[2.0, 4.0, 6.0, 8.0, 1.0, 3.0, 5.0, 7.0],
    )
    _install_local_download_stubs(monkeypatch, {building.id: source_path})
    _install_aggregation_rules_stub(monkeypatch)

    written_paths = await _run_batch(tmp_path, building, "load_curve_hourly")

    expected_target = tmp_path / building.file_path("load_curve_hourly")
    assert written_paths == [expected_target]

    actual_df = pl.read_parquet(expected_target)
    expected_df = pl.DataFrame({
        "timestamp": [datetime(2024, 1, 1, 0, 0, 0), datetime(2024, 1, 1, 1, 0, 0)],
        "example_load_kw": [10.0, 100.0],
        "example_temp_c": [5.0, 4.0],
        "bldg_id": [building.id, building.id],
        "year": [2024, 2024],
        "month": [1, 1],
        "day": [1, 1],
        "hour": [0, 1],
    })
    expected_df = expected_df.cast(actual_df.schema)
    assert_frame_equal(actual_df, expected_df)


@pytest.mark.asyncio
async def test_download_and_process_load_curve_daily_aggregates_values_and_adds_schema_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    building = _make_building(bldg_id=1234567)
    source_path = tmp_path / "source-daily.parquet"
    _write_aggregation_source_parquet(
        source_path,
        timestamps=[
            datetime(2024, 1, 1, 23, 15, 0),
            datetime(2024, 1, 1, 23, 30, 0),
            datetime(2024, 1, 1, 23, 45, 0),
            datetime(2024, 1, 2, 0, 0, 0),
            datetime(2024, 1, 2, 0, 15, 0),
            datetime(2024, 1, 2, 0, 30, 0),
            datetime(2024, 1, 2, 0, 45, 0),
            datetime(2024, 1, 2, 1, 0, 0),
        ],
        bldg_id=building.id,
        load_values=[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        temp_values=[2.0, 4.0, 6.0, 8.0, 1.0, 3.0, 5.0, 7.0],
    )
    _install_local_download_stubs(monkeypatch, {building.id: source_path})
    _install_aggregation_rules_stub(monkeypatch)

    written_paths = await _run_batch(tmp_path, building, "load_curve_daily")

    expected_target = tmp_path / building.file_path("load_curve_daily")
    assert written_paths == [expected_target]

    actual_df = pl.read_parquet(expected_target)
    expected_df = pl.DataFrame({
        "timestamp": [datetime(2024, 1, 1, 0, 0, 0), datetime(2024, 1, 2, 0, 0, 0)],
        "example_load_kw": [10.0, 100.0],
        "example_temp_c": [5.0, 4.0],
        "bldg_id": [building.id, building.id],
        "year": [2024, 2024],
        "month": [1, 1],
        "day": [1, 2],
    })
    expected_df = expected_df.cast(actual_df.schema)
    assert_frame_equal(actual_df, expected_df)


@pytest.mark.asyncio
async def test_download_and_process_load_curve_monthly_aggregates_values_and_adds_schema_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    building = _make_building(bldg_id=1234567)
    source_path = tmp_path / "source-monthly.parquet"
    _write_aggregation_source_parquet(
        source_path,
        timestamps=[
            datetime(2024, 1, 31, 23, 15, 0),
            datetime(2024, 1, 31, 23, 30, 0),
            datetime(2024, 1, 31, 23, 45, 0),
            datetime(2024, 2, 1, 0, 0, 0),
            datetime(2024, 2, 1, 0, 15, 0),
            datetime(2024, 2, 1, 0, 30, 0),
            datetime(2024, 2, 1, 0, 45, 0),
            datetime(2024, 2, 1, 1, 0, 0),
        ],
        bldg_id=building.id,
        load_values=[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        temp_values=[2.0, 4.0, 6.0, 8.0, 1.0, 3.0, 5.0, 7.0],
    )
    _install_local_download_stubs(monkeypatch, {building.id: source_path})
    _install_aggregation_rules_stub(monkeypatch)

    written_paths = await _run_batch(tmp_path, building, "load_curve_monthly")

    expected_target = tmp_path / building.file_path("load_curve_monthly")
    assert written_paths == [expected_target]

    actual_df = pl.read_parquet(expected_target)
    expected_df = pl.DataFrame({
        "timestamp": [datetime(2024, 1, 1, 0, 0, 0), datetime(2024, 2, 1, 0, 0, 0)],
        "example_load_kw": [10.0, 100.0],
        "example_temp_c": [5.0, 4.0],
        "bldg_id": [building.id, building.id],
        "year": [2024, 2024],
        "month": [1, 2],
    })
    expected_df = expected_df.cast(actual_df.schema)
    assert_frame_equal(actual_df, expected_df)


def test_group_buildings():
    def b(id_: int, load_curve_15min: str, load_curve_hourly: str):
        @dataclass(frozen=True)
        class FakeBuilding:
            id: int = id_

            def file_path(self, file_type):
                match file_type:
                    case "load_curve_15min":
                        return load_curve_15min
                    case "load_curve_hourly":
                        return load_curve_hourly
                    case _:
                        raise RuntimeError("Unexpected file type")

        return FakeBuilding()

    result = group_buildings(
        [
            b(0, "0", "0"),
            b(1, "0", "1"),
            b(2, "2", "0"),
            b(3, "1", "1"),
            b(4, "4", "4"),
            b(5, "5", "5"),
            b(6, "2", "2"),
            b(7, "4", "4"),
        ],
        ["load_curve_15min", "load_curve_hourly"],
    )
    indices = [[bldg.id for bldg in groups] for groups in result]
    expected = [
        [0, 1, 2, 3, 6],
        [4, 7],
        [5],
    ]
    assert indices == expected


def test_group_buildings_merges_transitively_connected_groups():
    def b(id_: int, load_curve_15min: str, load_curve_hourly: str):
        @dataclass(frozen=True)
        class FakeBuilding:
            id: int = id_

            def file_path(self, file_type):
                match file_type:
                    case "load_curve_15min":
                        return load_curve_15min
                    case "load_curve_hourly":
                        return load_curve_hourly
                    case _:
                        raise RuntimeError("Unexpected file type")

        return FakeBuilding()

    merged_result = group_buildings(
        [
            b(10, "a", "x"),
            b(11, "b", "y"),
            b(12, "a", "y"),
        ],
        ["load_curve_15min", "load_curve_hourly"],
    )
    merged_indices = [[bldg.id for bldg in groups] for groups in merged_result]
    merged_expected = [
        [10, 11, 12],
    ]
    assert merged_indices == merged_expected
