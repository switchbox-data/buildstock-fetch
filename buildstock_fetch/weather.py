import asyncio
import logging
import random
import shutil
from collections.abc import Collection
from pathlib import Path
from typing import cast
from urllib.parse import urljoin

import httpx
import tenacity
from httpx import AsyncClient

from buildstock_fetch.releases import RELEASES
from buildstock_fetch.shared import DownloadAndProcessProgress, download, estimate_download_size, groupby_sorted

from .building_ import Building
from .constants import OEDI_WEB_URL


async def download_and_process_weather_batch(
    target_folder: Path,
    client: AsyncClient,
    buildings: Collection[Building],
    semaphore: asyncio.Semaphore | None = None,
) -> list[Path]:
    semaphore = semaphore or asyncio.Semaphore(200)
    buildings_in_weather_states: list[Building] = []
    for building in buildings:
        if building.state not in RELEASES[building.release].weather_map_available_states:
            logging.getLogger(__name__).error("Weather file not available for %r", building)
        else:
            buildings_in_weather_states.append(building)

    grouped = [
        (urljoin(OEDI_WEB_URL, p), {b.file_path("weather") for b in buildings})
        for p, buildings in groupby_sorted(buildings_in_weather_states, lambda _: _.weather_path or "")
    ]
    if not grouped:
        return []
    urls = [url for url, _ in grouped]
    sample_size = min(len(urls), 100)
    sample = random.sample(urls, sample_size)
    sample_download_size = await estimate_download_size(client, sample)
    estimated_download_size = (sample_download_size / len(sample)) * len(urls)

    progress = DownloadAndProcessProgress(estimated_download_size, len(urls), "Downloading weather files")

    tasks = [
        _download_and_process_logged(target_folder, client, url, paths, progress, semaphore) for url, paths in grouped
    ]

    with progress.live():
        result = await asyncio.gather(*tasks)
    return [_ for n in result for _ in n]


@tenacity.retry(
    retry=tenacity.retry_if_exception_type(httpx.HTTPError),
    wait=tenacity.wait_exponential(2),
    stop=tenacity.stop_after_attempt(9),
    after=lambda e: logging.getLogger(__name__).warning("Retrying %s", e),
)
async def _download_and_process_logged(
    target_folder: Path,
    client: AsyncClient,
    url: str,
    copy_to_paths: Collection[Path],
    progress: DownloadAndProcessProgress,
    semaphore: asyncio.Semaphore,
) -> list[Path]:
    try:
        return await _download_and_process(target_folder, client, url, copy_to_paths, progress, semaphore)
    except Exception as e:
        logging.getLogger(__name__).exception("Error while processing url %s", url, exc_info=e.with_traceback(None))
        return []


async def _download_and_process(
    target_folder: Path,
    client: AsyncClient,
    url: str,
    copy_to_paths: Collection[Path],
    progress: DownloadAndProcessProgress,
    semaphore: asyncio.Semaphore,
) -> list[Path]:
    result: list[Path] = []
    async with semaphore, download(client, url, progress) as f:
        file_path = Path(cast(str, f.name))
        progress.on_processing_started()
        for path in copy_to_paths:
            path = target_folder / path
            path.parent.mkdir(exist_ok=True, parents=True)
            result.append(Path(shutil.copy2(file_path, path)))
        progress.on_processing_finished()
    progress.on_building_finished()
    return result
