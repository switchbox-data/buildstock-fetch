import asyncio
import logging
from collections.abc import AsyncGenerator, Callable, Collection, Iterable, Iterator
from contextlib import asynccontextmanager
from itertools import groupby
from typing import TypeVar, cast, final

import aiofiles
from aiofiles.threadpool.binary import AsyncBufferedReader
from httpx import AsyncClient
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from useful_types import SupportsRichComparisonT


@final
class DownloadAndProcessProgress:
    def __init__(self, estimated_download_size: float, num_files: int, task_description: str) -> None:
        self._total_progress = Progress(
            SpinnerColumn(),
            TextColumn(task_description),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            MofNCompleteColumn(),
        )
        self._total_progress_task = self._total_progress.add_task("Downloading and processing load curves")
        self._total_progress.update(self._total_progress_task, total=num_files)
        self._download_progress = Progress(
            TextColumn("Downloading files"),
            BarColumn(),
            DownloadColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TransferSpeedColumn(),
        )
        self._download_progress_task = self._download_progress.add_task("Downloading files")
        self._download_progress.update(self._download_progress_task, total=estimated_download_size)
        self._processing_progress = Progress(
            TextColumn("Processing files"),
            BarColumn(),
            MofNCompleteColumn(),
            transient=True,
            refresh_per_second=50,
        )
        self._processing_progress_task = self._processing_progress.add_task("Processing")
        self._processing_progress.update(self._processing_progress_task, total=0)
        self._processing_total = 0
        self._processed = 0

    def on_chunk_downloaded(self, chunk_size: int) -> None:
        self._download_progress.advance(self._download_progress_task, chunk_size)

    def on_processing_started(self) -> None:
        self._processing_total += 1
        self._processing_progress.update(self._processing_progress_task, total=self._processing_total)

    def on_processing_finished(self) -> None:
        self._processed += 1
        if self._processed == self._processing_total:
            self._processing_progress.update(
                self._download_progress_task, completed=self._processed, total=self._processing_total
            )
            self._processing_total = 0
            self._processed = 0
        else:
            self._processing_progress.update(self._download_progress_task, completed=self._processed)

    def on_building_finished(self) -> None:
        self._total_progress.advance(self._total_progress_task)

    def live(self) -> Live:
        group = Group(
            self._total_progress,
            Panel(Group(self._download_progress, self._processing_progress)),
        )
        return Live(group)


async def estimate_download_size(client: AsyncClient, urls: Collection[str]) -> float:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("Estimating download size"),
        BarColumn(),
        TimeRemainingColumn(),
        transient=True,
    )
    task = progress.add_task("Estimating download size", total=len(urls))
    tasks = [_estimate_average_download_size_single_file(client, url) for url in sorted(urls)]
    results: list[int] = []
    with progress:
        async for future in asyncio.as_completed(tasks):
            try:
                value = future.result()
                results.append(value)
            except Exception as _:
                logging.getLogger(__name__).exception("An error occured while estimating download size")
            progress.advance(task)
    return sum(results)


async def _estimate_average_download_size_single_file(client: AsyncClient, url: str) -> int:
    response = await client.head(url)
    _ = response.raise_for_status()
    return int(cast(str, response.headers.get("content-length")))


@asynccontextmanager
async def download(
    client: AsyncClient, url: str, progress: DownloadAndProcessProgress
) -> AsyncGenerator[AsyncBufferedReader]:
    async with aiofiles.tempfile.NamedTemporaryFile(delete_on_close=False) as f:
        async with client.stream("GET", url) as stream:
            _ = stream.raise_for_status()
            async for data in stream.aiter_bytes(8192):
                written = await f.write(data)
                progress.on_chunk_downloaded(written)
        await f.close()
        yield f


T = TypeVar("T", covariant=True)


def groupby_sorted(
    iterable: Iterable[T], key: Callable[[T], SupportsRichComparisonT]
) -> Iterator[tuple[SupportsRichComparisonT, Iterator[T]]]:
    return groupby(sorted(iterable, key=key), key=key)
