import asyncio
import tempfile
from collections.abc import Collection
from itertools import groupby
from pathlib import Path
from typing import cast
from urllib.parse import urljoin

import polars as pl
from httpx import AsyncClient

from buildstock_fetch.progress import DownloadAndProcessProgress, download, estimate_download_size

from .building_ import Building
from .constants import OEDI_WEB_URL


async def download_and_process_annual_results(
    target_folder: Path,
    client: AsyncClient,
    buildings: Collection[Building],
):
    grouped = {
        urljoin(OEDI_WEB_URL, oedi_annualcurve_path): {
            local_annualcurve_path: list(buildings_with_local_annualcurve_path)
            for local_annualcurve_path, buildings_with_local_annualcurve_path in groupby(
                buildings_with_annualcurve_path, lambda _: _.file_path("load_curve_annual")
            )
        }
        for oedi_annualcurve_path, buildings_with_annualcurve_path in groupby(
            buildings, lambda _: _.load_curve_annual_path
        )
    }
    download_size = await estimate_download_size(client, grouped.keys())
    progress = DownloadAndProcessProgress(download_size, len(grouped), "Downloading and processing annual results")
    with progress.live():
        tasks = [
            _download_and_process_annual_results(target_folder, client, oedi_url, partitions, progress)
            for oedi_url, partitions in grouped.items()
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)


async def _download_and_process_annual_results(
    target_folder: Path,
    client: AsyncClient,
    oedi_url: str,
    partitions: dict[Path, list[Building]],
    progress: DownloadAndProcessProgress,
):
    async with download(client, oedi_url, progress) as f:
        progress.on_processing_started()

        # f.name is FileDescriptorOrPath but in our case it's going to be str
        f_path = Path(cast(str, f.name))

        lf = pl.scan_parquet(f_path)
        lf = _compact_metadata(lf)

        # create a routing dataframe
        route_rows: list[tuple[int, str]] = [
            (building.id, str(path)) for path, buildings in partitions.items() for building in buildings
        ]
        route = pl.DataFrame(route_rows, schema={"bldg_id": pl.Int64, "out_path": pl.Utf8}, orient="row")

        with tempfile.TemporaryDirectory() as tmpdir:
            (
                lf.join(route.lazy(), on="bldg_id", how="inner").sink_parquet(
                    pl.PartitionByKey(
                        tmpdir,
                        by="out_path",
                        file_path=lambda _: Path(cast(str, _.keys[0].raw_value)) / f"{_.part_idx}__.parquet",
                        include_key=False,
                    ),
                    mkdir=True,
                )
            )

            tmp_path = Path(tmpdir)
            tasks = [
                asyncio.to_thread(_merge_partitions, tmp_path, target_folder, part.relative_to(tmp_path))
                for part in tmp_path.rglob("*/*.parquet")
                if part.is_dir()
            ]
            result = await asyncio.gather(*tasks, return_exceptions=False)
            progress.on_processing_finished()
    progress.on_building_finished()
    return result


def _merge_partitions(from_dir: Path, to_dir: Path, path: Path) -> Path:
    out_path = to_dir / path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    in_path = from_dir / path / "*.parquet"

    new_lf = pl.scan_parquet(in_path)

    if out_path.exists():
        old_lf = pl.scan_parquet(out_path)
        combined = pl.concat([old_lf, new_lf]).unique(subset=["bldg_id"])
        tmp_f = Path(str(out_path) + ".tmp")
        combined.sink_parquet(tmp_f, compression="zstd")
        _ = tmp_f.rename(out_path)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        new_lf.sink_parquet(out_path, compression="zstd")

    return out_path


def _compact_metadata(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema = lf.collect_schema()
    columns_to_keep = [
        col
        for col in schema
        if (any(keyword in col for keyword in ["bldg_id", "upgrade", "metadata_index"]) or col.startswith("out."))
        and not col.startswith("in.")
    ]
    return lf.select(columns_to_keep)
