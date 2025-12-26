import asyncio
from collections.abc import Collection
from pathlib import Path
from statistics import mean
from typing import Callable, NamedTuple, cast

import aiofiles
import polars as pl
from httpx import AsyncClient
from rich.progress import (
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from buildstock_fetch.building_ import Building
from buildstock_fetch.constants import METADATA_DIR
from buildstock_fetch.releases import RELEASES, BuildstockRelease
from buildstock_fetch.types import (
    ReleaseKey,
    UpgradeID,
    USStateCode,
)


class FileToDownload(NamedTuple):
    url: str
    "url to download"

    path: Path
    "download path"


def list_buildings(
    release: ReleaseKey,
    state: USStateCode,
    upgrade: UpgradeID,
    limit: int | None = None,
) -> list[Building]:
    """Helper function to get a list of Building objects

    Similar to `list_building_ids` but returns `Building` objects instead of list of ints.
    """
    release_obj = RELEASES[release]
    partition_path = (
        METADATA_DIR
        / f"product={release_obj.product}"
        / f"release_year={release_obj.year}"
        / f"weather_file={release_obj.weather}"
        / f"release_version={release_obj.version}"
        / f"state={state}"
    )

    if not partition_path.exists():
        return []

    df = pl.scan_parquet(partition_path)
    if limit:
        df = df.limit(limit)

    schema = df.collect_schema()

    if "county" in schema:
        lst = cast(list[tuple[int, str]], df.select(["bldg_id", "county"]).collect().rows())
    else:
        lst = [(cast(int, _[0]), None) for _ in df.select("bldg_id").collect().rows()]

    return [Building(id_, release, upgrade, state, county) for id_, county in lst]


def list_building_ids(release: BuildstockRelease, state: USStateCode, limit: int | None = None) -> list[int]:
    """Get a list of building ids which exist for the specified release in the specified state"""

    partition_path = (
        METADATA_DIR
        / f"product={release.product}"
        / f"release_year={release.year}"
        / f"weather_file={release.weather}"
        / f"release_version={release.version}"
        / f"state={state}"
    )

    if not partition_path.exists():
        return []

    df = pl.scan_parquet(partition_path)
    if limit:
        df = df.limit(limit)
    return df.collect()["bldg_id"].to_list()


async def download_multiple_files_with_progress_bar(
    client: AsyncClient,
    urls: Collection[FileToDownload],
    description: str = "downloading",
):

    download_column = DownloadColumn()
    progress_bar = Progress(
        SpinnerColumn(),
        TextColumn(description),
        download_column,
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        # FileSizeColumn(),
        # TotalFileSizeColumn(),
    )
    progress_task = progress_bar.add_task(description, total=len(urls))

    totals = []

    def on_file_size(download: FileToDownload, size: int | None):
        totals.append(size or 0)
        mean_ = mean(totals)
        total = len(urls) * mean_
        progress_bar.update(progress_task, total=total)

    def on_chunk(download: FileToDownload, size: int):
        progress_bar.advance(progress_task, size)

    tasks = [download_file(client, download, on_file_size=on_file_size, on_chunk=on_chunk) for download in urls]
    with progress_bar:
        for future in asyncio.as_completed(tasks):
            try:
                r = await future
            except:
                pass
            # progress_bar.advance(progress_task)


async def download(
    client: AsyncClient,
    url: str,
    path: Path,
    chunk_size: int | None = 8192,
    on_chunk: Callable[[str, int], None] | None = None,
) -> str:
    """Download `url` to `path`. Returns the url on successful download"""
    response = await client.head(url)
    _ = response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(path, "wb") as f, client.stream("GET", url) as stream:
        async for data in stream.aiter_bytes(chunk_size):
            _ = await f.write(data)
            if on_chunk:
                on_chunk(url, len(data))
    return url


async def download_file(
    client: AsyncClient,
    download: FileToDownload,
    chunk_size: int | None = 8192,
    on_file_size: Callable[[FileToDownload, int | None], None] | None = None,
    on_chunk: Callable[[FileToDownload, int], None] | None = None,
) -> FileToDownload:

    response = await client.head(download.url)
    _ = response.raise_for_status()
    if on_file_size:
        total_size = int(response.headers.get("content-length"))  # pyright: ignore[reportAny]
        on_file_size(download, total_size)

    download.path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(download.path, "wb") as f, client.stream("GET", download.url) as stream:
        async for data in stream.aiter_bytes(chunk_size):
            _ = await f.write(data)
            if on_chunk:
                on_chunk(download, len(data))
    return download
