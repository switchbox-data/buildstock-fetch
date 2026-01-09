import asyncio
from asyncio import create_task
from collections.abc import Collection
from contextlib import AsyncExitStack
from pathlib import Path
from typing import cast

import aiofiles
import polars as pl
from aiofiles.threadpool.binary import AsyncBufferedIOBase
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from types_aiobotocore_s3.client import S3Client

from .building_ import Building
from .shared import DownloadAndProcessProgress, groupby_sorted
from .types import ReleaseKey, USStateCode


async def download_and_process_trip_schedules_batch(
    target_folder: Path, client: S3Client, buildings: Collection[Building]
) -> list[Path]:
    grouped = cast(
        dict[ReleaseKey, dict[USStateCode, list[Building]]],
        {
            release: {
                state: list(state_group) for state, state_group in groupby_sorted(release_group, lambda _: _.state)
            }
            for release, release_group in groupby_sorted(buildings, lambda _: _.release)
        },
    )
    locating_parquet_files_progress = Progress(
        SpinnerColumn(), TextColumn("Locating parquet files"), BarColumn(), TimeRemainingColumn(), transient=True
    )
    locating_parquet_files_progress_task = locating_parquet_files_progress.add_task(
        "Locating parquet files", total=sum(len(_) for _ in grouped.values())
    )
    locating_parquet_files_tasks = [
        create_task(_locate_parquet_files(client, release, state))
        for release, release_group in grouped.items()
        for state in release_group
    ]
    for task in locating_parquet_files_tasks:
        task.add_done_callback(lambda _: locating_parquet_files_progress.advance(locating_parquet_files_progress_task))
    with locating_parquet_files_progress:
        parquet_files = await asyncio.gather(*locating_parquet_files_tasks)
    total_download_size = sum(size for _, _, files in parquet_files for _, size in files)
    total_processes = sum(1 for _, _, files in parquet_files if files)
    progress = DownloadAndProcessProgress(
        total_download_size,
        total_processes,
        "Downloading and processing trip schedules",
    )

    tasks = [
        _download_and_process(target_folder, client, [k for k, _ in s3_keys], grouped[release][state], progress)
        for release, state, s3_keys in parquet_files
        if s3_keys
    ]
    with progress.live():
        result = await asyncio.gather(*tasks)
    return result


async def _download_and_process(
    target_folder: Path,
    client: S3Client,
    s3_keys: list[str],
    buildings: Collection[Building],
    progress: DownloadAndProcessProgress,
) -> Path:
    async with AsyncExitStack() as stack:
        tempfiles: list[AsyncBufferedIOBase] = []
        for _ in s3_keys:
            f = await stack.enter_async_context(aiofiles.tempfile.NamedTemporaryFile("wb", delete_on_close=False))
            tempfiles.append(f)

        download_tasks = [
            _download(client, s3_key, buffer, progress) for s3_key, buffer in zip(s3_keys, tempfiles, strict=True)
        ]
        _ = await asyncio.gather(*download_tasks)

        progress.on_processing_started()

        paths = [Path(cast(str, f.name)) for f in tempfiles]
        processing_result = await _async_process(target_folder, paths, buildings)
        progress.on_processing_finished()
    return processing_result


async def _async_process(
    target_folder: Path, parquet_filenames: Collection[Path], buildings: Collection[Building]
) -> Path:
    return await asyncio.to_thread(_process, target_folder, parquet_filenames, buildings)


def _process(
    target_folder: Path,
    parquet_filenames: Collection[Path],
    buildings: Collection[Building],
) -> Path:
    print(buildings)
    lf = pl.concat([pl.scan_parquet(fn) for fn in parquet_filenames])
    lf = lf.filter(pl.col("bldg_id").is_in({str(_.id) for _ in buildings}))

    # derive output filename
    expected_output_filenames = {_.file_path("trip_schedules") for _ in buildings}
    if len(expected_output_filenames) != 1:
        raise RuntimeError()
    output_filename = target_folder / next(iter(expected_output_filenames))

    if output_filename.exists():
        old_lf = pl.scan_parquet(output_filename)
        combined = pl.concat([old_lf, lf]).unique(subset=["bldg_id"])
        tmp_f = Path(str(output_filename) + ".tmp")
        combined.sink_parquet(tmp_f)
        _ = tmp_f.rename(output_filename)
    else:
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        lf.sink_parquet(output_filename)
    return output_filename


async def _download(
    client: S3Client,
    s3_key: str,
    buffer: AsyncBufferedIOBase,
    progress: DownloadAndProcessProgress,
) -> None:
    resp = await client.get_object(Bucket="buildstock-fetch", Key=s3_key)
    body = resp["Body"]
    async for chunk in body.iter_chunks():
        if not chunk:
            continue
        written = await buffer.write(chunk)
        progress.on_chunk_downloaded(written)
    await buffer.close()


async def _locate_parquet_files(
    client: S3Client,
    release: ReleaseKey,
    state: USStateCode,
) -> tuple[ReleaseKey, USStateCode, list[tuple[str, int]]]:
    result: list[tuple[str, int]] = []
    paginator = client.get_paginator("list_objects_v2")
    async for page in paginator.paginate(
        Bucket="buildstock-fetch", Prefix=f"ev_demand/trip_schedules/release={release}/state={state}"
    ):
        for obj in page.get("Contents", []):
            if (key := obj.get("Key", "")).endswith(".parquet"):
                result.append((key, obj.get("Size", 0)))
    return (release, state, result)
