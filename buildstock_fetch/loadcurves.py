import asyncio
import logging
import random
import shutil
import tempfile
from collections.abc import Collection, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urljoin

import httpx
import polars as pl
import tenacity
from httpx import AsyncClient

from .building_ import Building
from .constants import LOAD_CURVE_COLUMN_AGGREGATION, OEDI_WEB_URL
from .shared import DownloadAndProcessProgress, download, estimate_download_size
from .types import ReleaseKey

AGGREGATION_RULES_CACHE: dict[ReleaseKey, list[pl.Expr]] = {}
_FINALIZE_MERGE_CHUNK_SIZE = 128

LoadCurveAggregate = Literal[
    "load_curve_hourly",
    "load_curve_daily",
    "load_curve_monthly",
]
LoadCurve = Literal[LoadCurveAggregate, "load_curve_15min"]

_TIME_BUCKET = {
    "load_curve_hourly": "1h",
    "load_curve_daily": "1d",
    "load_curve_monthly": "1mo",
}


async def download_and_process_load_curves_batch(
    target_folder: Path,
    client: AsyncClient,
    curves: Collection[LoadCurve],
    buildings: Sequence[Building],
    semaphore: asyncio.Semaphore | None,
    processing_semaphore: asyncio.Semaphore | None,
) -> list[Path]:
    semaphore = semaphore or asyncio.Semaphore(200)
    processing_semaphore = processing_semaphore or asyncio.Semaphore(1)
    if not buildings:
        return []

    sample_size = min(len(buildings), 100)
    sample_download_size = await estimate_download_size(
        client,
        [
            urljoin(OEDI_WEB_URL, building.load_curve_15min_path)
            for building in random.Random(0).sample(buildings, sample_size)
        ],
    )
    estimated_download_size = (sample_download_size / sample_size) * len(buildings)
    progress = DownloadAndProcessProgress(
        estimated_download_size, len(buildings), "Downloading and processing load curves"
    )

    with progress.live():
        tasks = [
            _download_and_process_load_curves_group(
                target_folder, client, curves, group, progress, semaphore, processing_semaphore
            )
            for group in group_buildings(buildings, curves)
        ]
        result = [path for nested in await asyncio.gather(*tasks) for path in nested]

    return result


sem = asyncio.Semaphore(50)


async def _download_and_process_load_curves_for_building_logged(
    target_folder: Path,
    client: AsyncClient,
    curves: Collection[LoadCurve],
    building: Building,
    progress: DownloadAndProcessProgress,
    semaphore: asyncio.Semaphore,
    processing_semaphore: asyncio.Semaphore,
) -> list[Path]:
    try:
        result = await _download_and_process_load_curves_for_building(
            target_folder, client, curves, building, progress, semaphore, processing_semaphore
        )
    except Exception as e:
        logging.getLogger(__name__).error("Error while processing building %s: %s", building, e)  # noqa: TRY400
        return []
    else:
        return result


async def _download_and_process_load_curves_group(
    target_folder: Path,
    client: AsyncClient,
    curves: Collection[LoadCurve],
    buildings: Sequence[Building],
    progress: DownloadAndProcessProgress,
    semaphore: asyncio.Semaphore,
    processing_semaphore: asyncio.Semaphore,
) -> list[Path]:
    if len(buildings) <= 1:
        return await _download_and_process_load_curves_for_building_logged(
            target_folder, client, curves, buildings[0], progress, semaphore, processing_semaphore
        )

    target_folder.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=".loadcurves-", dir=target_folder))
    tasks: list[asyncio.Task[list[Path]]] = []
    try:
        tasks = [
            asyncio.create_task(
                _download_and_process_load_curves_for_building_logged(
                    temp_root / str(building.id), client, curves, building, progress, semaphore, processing_semaphore
                )
            )
            for building in buildings
        ]
        nested_results = await asyncio.gather(*tasks)
        temp_paths_by_target, result_paths = _group_temp_paths_by_target_from_results(
            target_folder, temp_root, buildings, nested_results
        )
        await _finalize_temp_paths_by_target(temp_paths_by_target, processing_semaphore)
        return result_paths
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            _ = await asyncio.gather(*tasks, return_exceptions=True)

        temp_paths_by_target, _ = _group_temp_paths_by_target_from_disk(target_folder, temp_root)
        if temp_paths_by_target:
            await asyncio.shield(_finalize_temp_paths_by_target(temp_paths_by_target, processing_semaphore))
        raise
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


@tenacity.retry(
    retry=tenacity.retry_if_exception_type(httpx.HTTPError),
    wait=tenacity.wait_exponential(2),
    stop=tenacity.stop_after_attempt(9),
    after=lambda e: logging.getLogger(__name__).info("Retrying %s", e),
)
async def _download_and_process_load_curves_for_building(
    target_folder: Path,
    client: AsyncClient,
    curves: Collection[LoadCurve],
    building: Building,
    progress: DownloadAndProcessProgress,
    semaphore: asyncio.Semaphore,
    processing_semaphore: asyncio.Semaphore,
) -> list[Path]:
    url = urljoin(OEDI_WEB_URL, building.load_curve_15min_path)
    async with semaphore, download(client, url, progress) as f:
        file_path = Path(cast(str, f.name))

        tasks = [
            asyncio.create_task(
                _async_process_load_curve_aggregate(target_folder, file_path, aggregate, building, processing_semaphore)
            )
            for aggregate in set(curves)
        ]
        for task in tasks:
            progress.on_processing_started()
            task.add_done_callback(lambda _: progress.on_processing_finished())
        result = await asyncio.gather(*tasks, return_exceptions=False)
        progress.on_building_finished()
        return result


async def _async_process_load_curve_aggregate(
    target_folder: Path,
    file_path: Path,
    aggregate: LoadCurve,
    building: Building,
    semaphore: asyncio.Semaphore,
) -> Path:
    async with semaphore:
        return await asyncio.to_thread(_process_load_curve_aggregate, target_folder, file_path, aggregate, building)


def _process_load_curve_aggregate(
    target_folder: Path, file_path: Path, aggregate: LoadCurve, building: Building
) -> Path:
    target_path = target_folder / building.file_path(aggregate)
    target_path.parent.mkdir(exist_ok=True, parents=True)
    if aggregate == "load_curve_15min":
        _ = shutil.copy2(file_path, target_path)
        return target_path
    aggregation_rules = _load_aggregation_rules(building.release)

    bucket = _TIME_BUCKET[aggregate]

    # We want to subtract 15 minutes because the original load curve provides information
    # for the previous 15 minutes for each timestamp. For example, the first timestamp is 00:00:15,
    # and the columns correspond to consumption from 00:00:00 to 00:00:15. When aggregating,
    # we want the 00:00:00 timestamp to correspond to the consumption from 00:00:00 to whenever the
    # next timestamp is.
    lf = pl.scan_parquet(file_path).with_columns(
        (pl.col("timestamp").cast(pl.Datetime) - timedelta(minutes=15)).alias("timestamp")
    )

    lf = lf.with_columns(pl.col("timestamp").dt.truncate(bucket).alias("_bucket_ts"))

    lf = lf.group_by("_bucket_ts").agg(aggregation_rules).rename({"_bucket_ts": "timestamp"}).sort("timestamp")

    match aggregate:
        case "load_curve_hourly":
            lf = lf.with_columns([
                pl.col("timestamp").dt.year().alias("year"),
                pl.col("timestamp").dt.month().alias("month"),
                pl.col("timestamp").dt.day().alias("day"),
                pl.col("timestamp").dt.hour().alias("hour"),
            ])
        case "load_curve_daily":
            lf = lf.with_columns([
                pl.col("timestamp").dt.year().alias("year"),
                pl.col("timestamp").dt.month().alias("month"),
                pl.col("timestamp").dt.day().alias("day"),
            ])
        case "load_curve_monthly":
            lf = lf.with_columns([
                pl.col("timestamp").dt.year().alias("year"),
                pl.col("timestamp").dt.month().alias("month"),
            ])

    lf.sink_parquet(target_path)
    return target_path


async def _async_finalize_load_curve_output(
    target_path: Path,
    source_paths: Sequence[Path],
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        await asyncio.to_thread(_finalize_load_curve_output, target_path, source_paths)


async def _finalize_temp_paths_by_target(
    temp_paths_by_target: dict[Path, list[Path]],
    processing_semaphore: asyncio.Semaphore,
) -> None:
    finalize_tasks = [
        _async_finalize_load_curve_output(target_path, source_paths, processing_semaphore)
        for target_path, source_paths in temp_paths_by_target.items()
    ]
    _ = await asyncio.gather(*finalize_tasks)


def _group_temp_paths_by_target_from_results(
    target_folder: Path,
    temp_root: Path,
    buildings: Sequence[Building],
    nested_results: Sequence[Sequence[Path]],
) -> tuple[dict[Path, list[Path]], list[Path]]:
    temp_paths_by_target: dict[Path, list[Path]] = {}
    result_paths: list[Path] = []
    for building, temp_paths in zip(buildings, nested_results, strict=True):
        building_temp_root = temp_root / str(building.id)
        for temp_path in temp_paths:
            final_target = target_folder / temp_path.relative_to(building_temp_root)
            temp_paths_by_target.setdefault(final_target, []).append(temp_path)
            result_paths.append(final_target)
    return temp_paths_by_target, result_paths


def _group_temp_paths_by_target_from_disk(
    target_folder: Path,
    temp_root: Path,
) -> tuple[dict[Path, list[Path]], list[Path]]:
    temp_paths_by_target: dict[Path, list[Path]] = {}
    result_paths: list[Path] = []
    for temp_path in sorted(temp_root.rglob("*.parquet")):
        relative_path = temp_path.relative_to(temp_root)
        if len(relative_path.parts) < 2:
            continue
        final_target = target_folder.joinpath(*relative_path.parts[1:])
        temp_paths_by_target.setdefault(final_target, []).append(temp_path)
        result_paths.append(final_target)
    return temp_paths_by_target, result_paths


def _finalize_load_curve_output(target_path: Path, source_paths: Sequence[Path]) -> None:
    target_path.parent.mkdir(exist_ok=True, parents=True)
    merge_inputs = list(source_paths)
    if target_path.exists():
        merge_inputs.insert(0, target_path)

    if len(merge_inputs) == 1:
        only_path = merge_inputs[0]
        if only_path != target_path:
            _ = shutil.move(only_path, target_path)
        return

    temp_dir = Path(tempfile.mkdtemp(prefix=f".merge-{target_path.stem}-", dir=target_path.parent))
    try:
        merged_path = _merge_parquet_files_chunked(merge_inputs, temp_dir)
        tmp_target = target_path.with_suffix(".tmp.parquet")
        if tmp_target.exists():
            tmp_target.unlink()
        _ = shutil.move(merged_path, tmp_target)
        _ = tmp_target.replace(target_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _merge_parquet_files_chunked(paths: Sequence[Path], temp_dir: Path) -> Path:
    if not paths:
        msg = "Cannot merge an empty parquet path list"
        raise ValueError(msg)

    current_paths = list(paths)
    round_idx = 0

    # Reduce large fan-in merges into smaller staged merges to keep finalization
    # bounded when many buildings map into the same bucket file.
    while len(current_paths) > 1:
        next_paths: list[Path] = []
        for chunk_idx, start in enumerate(range(0, len(current_paths), _FINALIZE_MERGE_CHUNK_SIZE)):
            chunk_paths = current_paths[start : start + _FINALIZE_MERGE_CHUNK_SIZE]
            chunk_output = temp_dir / f"merge-r{round_idx:02d}-{chunk_idx:04d}.parquet"
            pl.scan_parquet([str(path) for path in chunk_paths]).sink_parquet(chunk_output)
            next_paths.append(chunk_output)
        current_paths = next_paths
        round_idx += 1

    return current_paths[0]


def _load_aggregation_rules(release: ReleaseKey) -> list[pl.Expr]:
    if release in AGGREGATION_RULES_CACHE:
        return AGGREGATION_RULES_CACHE[release]
    filename = LOAD_CURVE_COLUMN_AGGREGATION.joinpath(f"{release}.csv")
    if not filename.exists():
        msg = f"Missing load_curve_map for release: {release}"
        raise ValueError(msg)
    aggregation_rules = pl.read_csv(filename)
    rules_dict = cast(
        dict[str, str], dict(zip(aggregation_rules["name"], aggregation_rules["Aggregate_function"], strict=True))
    )
    result: list[pl.Expr] = []
    for column, rule in rules_dict.items():
        if column == "timestamp":
            continue
        match rule:
            case "sum":
                result.append(pl.col(column).sum())
            case "mean":
                result.append(pl.col(column).mean())
            case "first":
                result.append(pl.col(column).first())
            case rule:
                msg = f"Unknown aggregation function: {rule}"
                raise ValueError(msg)
    AGGREGATION_RULES_CACHE[release] = result
    return result


def group_buildings(buildings: Collection[Building], load_curves: Collection[LoadCurve]) -> list[list[Building]]:  # noqa: C901
    building_list = list(buildings)
    if not building_list:
        return []

    curves = tuple(load_curves)
    parent = list(range(len(building_list)))
    rank = [0] * len(building_list)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            parent[left_root] = right_root
        elif rank[left_root] > rank[right_root]:
            parent[right_root] = left_root
        else:
            parent[right_root] = left_root
            rank[left_root] += 1

    first_seen_by_path: dict[Path, int] = {}
    for index, building in enumerate(building_list):
        for curve in curves:
            path = building.file_path(curve)
            if path in first_seen_by_path:
                union(index, first_seen_by_path[path])
            else:
                first_seen_by_path[path] = index

    groups_by_root: dict[int, list[Building]] = {}
    for index, building in enumerate(building_list):
        root = find(index)
        groups_by_root.setdefault(root, []).append(building)

    return list(groups_by_root.values())
