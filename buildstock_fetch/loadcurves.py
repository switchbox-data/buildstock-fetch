import asyncio
import logging
import random
import shutil
from collections.abc import Collection, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urljoin

import polars as pl
from httpx import AsyncClient

from .building_ import Building
from .constants import LOAD_CURVE_COLUMN_AGGREGATION, OEDI_WEB_URL
from .shared import DownloadAndProcessProgress, download, estimate_download_size
from .types import ReleaseKey

AGGREGATION_RULES_CACHE: dict[ReleaseKey, list[pl.Expr]] = {}

LoadCurveAggregate = Literal[
    "load_curve_hourly",
    "load_curve_daily",
    "load_curve_monthly",
]
LoadCurve = Literal[LoadCurveAggregate, "load_curve_15min"]


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
    tasks = [
        _download_and_process_load_curves_for_building(
            target_folder, client, curves, building, progress, semaphore, processing_semaphore
        )
        for building in buildings
    ]
    with progress.live():
        nested = await asyncio.gather(*tasks, return_exceptions=True)
    exceptions = (_ for _ in nested if isinstance(_, BaseException))
    for e in exceptions:
        logging.getLogger(__name__).exception("Error: %s", e)
    return [_ for n in nested if not isinstance(n, BaseException) for _ in n]


sem = asyncio.Semaphore(50)


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
    lf = pl.scan_parquet(file_path)
    lf = lf.with_columns(pl.col("timestamp").cast(pl.Datetime))

    # We want to subtract 15 minutes because the original load curve provides information
    # for the previous 15 minutes for each timestamp. For example, the first timestamp is 00:00:15,
    # and the columns correspond to consumption from 00:00:00 to 00:00:15. When aggregating,
    # we want the 00:00:00 timestamp to correspond to the consumption from 00:00:00 to whenever the
    # next timestamp is.
    lf = lf.with_columns((pl.col("timestamp") - timedelta(minutes=15)).alias("timestamp"))

    grouping_key, format_string = _get_time_step_grouping_key(aggregate)
    lf = lf.with_columns(pl.col("timestamp").dt.strftime(format_string).alias(grouping_key))

    lf = lf.group_by(grouping_key).agg(aggregation_rules)
    lf = lf.sort("timestamp").drop(grouping_key)
    logging.getLogger(__name__).info("Sinking to %s", target_path)

    # Add time aggregation columns

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
        match rule:
            case "sum":
                result.append(pl.col(column).sum().alias(column))
            case "mean":
                result.append(pl.col(column).mean().alias(column))
            case "first":
                result.append(pl.col(column).first().alias(column))
            case rule:
                msg = f"Unknown aggregation function: {rule}"
                raise ValueError(msg)
    AGGREGATION_RULES_CACHE[release] = result
    return result


def _get_time_step_grouping_key(aggregate: LoadCurveAggregate) -> tuple[str, str]:
    """Get the grouping key and format string for a given time step."""
    match aggregate:
        case "load_curve_hourly":
            return ("year_month_day_hour", "%Y-%m-%d-%H")
        case "load_curve_daily":
            return ("year_month_day", "%Y-%m-%d")
        case "load_curve_monthly":
            return ("year_month", "%Y-%m")
