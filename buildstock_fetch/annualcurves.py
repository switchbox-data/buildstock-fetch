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

from .building_ import Building
from .constants import OEDI_WEB_URL
from .shared import DownloadAndProcessProgress, download, estimate_download_size, groupby_sorted


async def download_and_process_annual_results(
    target_folder: Path,
    client: AsyncClient,
    buildings: Collection[Building],
    semaphore: asyncio.Semaphore | None = None,
) -> list[Path]:
    semaphore = semaphore or asyncio.Semaphore(200)
    grouped = {
        urljoin(OEDI_WEB_URL, oedi_annualcurve_path): {
            local_annualcurve_path: list(buildings_with_local_annualcurve_path)
            for local_annualcurve_path, buildings_with_local_annualcurve_path in groupby_sorted(
                buildings_with_annualcurve_path, lambda _: _.file_path("load_curve_annual")
            )
        }
        for oedi_annualcurve_path, buildings_with_annualcurve_path in groupby_sorted(
            buildings,
            lambda _: _.load_curve_annual_path,
        )
    }
    download_size = await estimate_download_size(client, grouped.keys())
    progress = DownloadAndProcessProgress(download_size, len(grouped), "Downloading and processing annual results")
    tasks = [
        _download_and_process_annual_results_logged(target_folder, client, oedi_url, partitions, progress, semaphore)
        for oedi_url, partitions in grouped.items()
    ]
    with progress.live():
        nested = await asyncio.gather(*tasks)
    return [_ for n in nested for _ in n]


async def _download_and_process_annual_results_logged(
    target_folder: Path,
    client: AsyncClient,
    oedi_url: str,
    partitions: dict[Path, list[Building]],
    progress: DownloadAndProcessProgress,
    semaphore: asyncio.Semaphore,
) -> list[Path]:
    try:
        result = await _download_and_process_annual_results(
            target_folder,
            client,
            oedi_url,
            partitions,
            progress,
            semaphore,
        )
    except Exception as e:
        logging.getLogger(__name__).exception(
            "Error while processing url %s", oedi_url, exc_info=e.with_traceback(None)
        )
        return []
    else:
        return result


@tenacity.retry(
    retry=tenacity.retry_if_exception_type(httpx.HTTPError),
    wait=tenacity.wait_exponential(2),
    stop=tenacity.stop_after_attempt(9),
    after=lambda e: logging.getLogger(__name__).info("Retrying %s", e),
)
async def _download_and_process_annual_results(
    target_folder: Path,
    client: AsyncClient,
    oedi_url: str,
    partitions: dict[Path, list[Building]],
    progress: DownloadAndProcessProgress,
    semaphore: asyncio.Semaphore,
) -> list[Path]:
    result: list[Path] = []
    async with semaphore, download(client, oedi_url, progress) as f:
        progress.on_processing_started()

        # f.name is FileDescriptorOrPath but in our case it's going to be str
        f_path = Path(cast(str, f.name))

        lf = pl.scan_parquet(f_path)
        lf = _compact_metadata(lf)

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


def _compact_metadata(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema = lf.collect_schema()
    columns_to_keep = [
        col
        for col in schema
        if (any(keyword in col for keyword in ["bldg_id", "upgrade", "metadata_index"]) or col.startswith("out."))
        and not col.startswith("in.")
    ]
    return lf.select(columns_to_keep)
