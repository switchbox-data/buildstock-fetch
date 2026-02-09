import asyncio
import logging
from collections.abc import Collection
from pathlib import Path
from typing import cast
from urllib.parse import urljoin

import httpx
import polars as pl
import tenacity
from httpx import AsyncClient

from buildstock_fetch.constants import OEDI_WEB_URL

from .building_ import Building
from .shared import DownloadAndProcessProgress, download, estimate_download_size, groupby_sorted


async def download_and_process_metadata_batch(
    target_folder: Path,
    client: AsyncClient,
    buildings: Collection[Building],
    semaphore: asyncio.Semaphore | None = None,
) -> list[Path]:
    semaphore = semaphore or asyncio.Semaphore(200)
    grouped = {
        urljoin(OEDI_WEB_URL, oedi_metadata_path): {
            local_metadata_path: list(buildings_with_local_metadata_path)
            for local_metadata_path, buildings_with_local_metadata_path in groupby_sorted(
                buildings_with_metadata_path, lambda _: _.file_path("metadata")
            )
        }
        for oedi_metadata_path, buildings_with_metadata_path in groupby_sorted(buildings, lambda _: _.metadata_path)
    }

    download_size = await estimate_download_size(client, grouped.keys())
    progress = DownloadAndProcessProgress(download_size, len(grouped), "Downloading and processing metadata")

    with progress.live():
        tasks = (
            _download_and_process_metadata_logged(
                target_folder, client, oedi_metadata_url, partitions, progress, semaphore
            )
            for oedi_metadata_url, partitions in grouped.items()
        )
        nested = await asyncio.gather(*tasks)
        return [_ for n in nested for _ in n]


async def _download_and_process_metadata_logged(
    target_folder: Path,
    client: AsyncClient,
    oedi_matadata_url: str,
    partitions: dict[Path, list[Building]],
    progress: DownloadAndProcessProgress,
    semaphore: asyncio.Semaphore,
) -> list[Path]:
    with progress.live():
        try:
            return await _download_and_process_metadata(
                target_folder,
                client,
                oedi_matadata_url,
                partitions,
                progress,
                semaphore,
            )
        except Exception as e:
            logging.getLogger(__name__).exception(
                "Error while processing url %s", oedi_matadata_url, exc_info=e.with_traceback(None)
            )
            return []


@tenacity.retry(
    retry=tenacity.retry_if_exception_type(httpx.HTTPError),
    wait=tenacity.wait_exponential(2),
    stop=tenacity.stop_after_attempt(9),
    after=lambda e: logging.getLogger(__name__).info("Retrying %s", e),
)
async def _download_and_process_metadata(
    target_folder: Path,
    client: AsyncClient,
    oedi_matadata_url: str,
    partitions: dict[Path, list[Building]],
    progress: DownloadAndProcessProgress,
    semaphore: asyncio.Semaphore,
) -> list[Path]:
    result: list[Path] = []
    async with semaphore, download(client, oedi_matadata_url, progress) as f:
        progress.on_processing_started()

        # f.name is FileDescriptorOrPath but in our case it's going to be str
        f_path = Path(cast(str, f.name))

        lf = pl.scan_parquet(f_path)
        lf = compact_metadata(lf)

        for rel_path, buildings_here in partitions.items():
            progress.on_processing_started()
            buildings_set = {_.id for _ in buildings_here}
            out_path = target_folder / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = out_path.with_suffix(".tmp.parquet")
            lf = lf.filter(pl.col(name="bldg_id").is_in(buildings_set))
            final_lf = pl.concat([pl.scan_parquet(out_path), lf]).unique() if out_path.exists() else lf
            final_lf.sink_parquet(tmp_path)
            _ = tmp_path.rename(out_path)
            result.append(out_path)
            progress.on_processing_finished()

    progress.on_building_finished()
    return result


def compact_metadata(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema = lf.collect_schema()
    columns_to_keep = [
        col
        for col in schema
        if any(keyword in col for keyword in ["bldg_id", "upgrade", "metadata_index", "weight"])
        or col.startswith("in.")
        or col.startswith("upgrade.")
    ]
    return lf.select(columns_to_keep)
