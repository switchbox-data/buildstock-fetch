import asyncio
from collections.abc import Collection
import logging
from pathlib import Path
import shutil
from tempfile import tempdir
import tempfile
from typing import Literal, cast
from urllib.parse import urljoin
import zipfile

from httpx import AsyncClient

from buildstock_fetch.building_ import Building
from buildstock_fetch.constants import OEDI_WEB_URL
from buildstock_fetch.progress import DownloadAndProcessProgress, download, estimate_download_size


_ENERGY_MODEL_FILE_TYPE = Literal["hpxml", "schedule"]


async def download_and_process_energy_models_batch(
    target_folder: Path,
    client: AsyncClient,
    file_types: Collection[_ENERGY_MODEL_FILE_TYPE],
    buildings: Collection[Building],
):
    if not buildings:
        return cast(list[BaseException | Building], [])
    sample_size = min(len(buildings), 100)
    sample_download_size = await estimate_download_size(
        client, [urljoin(OEDI_WEB_URL, building.energy_models_path) for building in buildings]
    )
    estimated_download_size = (sample_download_size / sample_size) * len(buildings)
    progress = DownloadAndProcessProgress(
        estimated_download_size,
        len(buildings),
        "Downloading and processing energy models",
    )
    tasks = [
        _download_and_process_energy_models_for_building(
            target_folder,
            client,
            file_types,
            building,
            progress,
        )
        for building in buildings
    ]
    with progress.live():
        return await asyncio.gather(*tasks, return_exceptions=True)


async def _download_and_process_energy_models_for_building(
    target_folder: Path,
    client: AsyncClient,
    file_types: Collection[_ENERGY_MODEL_FILE_TYPE],
    building: Building,
    progress: DownloadAndProcessProgress,
) -> Building:
    url = urljoin(OEDI_WEB_URL, building.energy_models_path)
    async with download(client, url, progress) as f:
        progress.on_processing_started()
        file_path = Path(cast(str, f.name))
        result_future = await asyncio.to_thread(
            _process_energy_models,
            file_path,
            target_folder,
            file_types,
            building,
        )
        result = await result_future
        progress.on_processing_finished()
        progress.on_building_finished()
        return result


async def _process_energy_models(
    zip_file_path: Path,
    target_folder: Path,
    file_types: Collection[_ENERGY_MODEL_FILE_TYPE],
    building: Building,
):
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref, tempfile.TemporaryDirectory() as td_str:
        td = Path(td_str)
        zip_file_contents = zip_ref.namelist()
        if "hpxml" in file_types:
            xml_file = next((_ for _ in zip_file_contents if _.endswith(".xml")), None)
            if not xml_file:
                logging.getLogger(__name__).error("HPXML not found for building %r", building)
            else:
                _ = zip_ref.extract(xml_file, td)
                from_path = td / xml_file
                target_file = target_folder / building.file_path("hpxml")
                target_file.parent.mkdir(exist_ok=True, parents=True)
                _ = shutil.move(from_path, target_file)
        if "schedule" in file_types:
            csv_file = next((_ for _ in zip_file_contents if _.endswith(".csv")), None)
            if not csv_file:
                logging.getLogger(__name__).error("Schedule not found for building %r", building)
            else:
                from_path = td / csv_file
                target_file = target_folder / building.file_path("schedule")
                target_file.parent.mkdir(exist_ok=True, parents=True)
                _ = zip_ref.extract(csv_file, td)
                _ = shutil.move(from_path, target_file)
    return building
