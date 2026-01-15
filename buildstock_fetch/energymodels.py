import asyncio
import logging
import random
import shutil
import tempfile
import zipfile
from collections.abc import Collection
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urljoin

from httpx import AsyncClient

from .building_ import Building
from .constants import OEDI_WEB_URL
from .shared import DownloadAndProcessProgress, download, estimate_download_size

ENERGY_MODEL_FILE_TYPE = Literal["hpxml", "schedule"]


async def download_and_process_energy_models_batch(
    target_folder: Path,
    client: AsyncClient,
    file_types: Collection[ENERGY_MODEL_FILE_TYPE],
    buildings: Collection[Building],
    semaphore: asyncio.Semaphore | None = None,
    processing_semaphore: asyncio.Semaphore | None = None,
) -> list[Path]:
    semaphore = semaphore or asyncio.Semaphore(200)
    processing_semaphore = processing_semaphore or asyncio.Semaphore(1)
    if not buildings:
        return []
    sample_size = min(len(buildings), 100)
    sample_download_size = await estimate_download_size(
        client,
        [
            urljoin(OEDI_WEB_URL, building.energy_models_path)
            for building in random.sample(list(buildings), sample_size)
        ],
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
            semaphore,
            processing_semaphore,
        )
        for building in buildings
    ]
    with progress.live():
        nested = await asyncio.gather(*tasks, return_exceptions=True)
    for e in (_ for _ in nested if isinstance(_, BaseException)):
        logging.getLogger(__name__).exception("Error: %s", e)
    return [_ for n in nested if not isinstance(n, BaseException) for _ in n]


async def _download_and_process_energy_models_for_building(
    target_folder: Path,
    client: AsyncClient,
    file_types: Collection[ENERGY_MODEL_FILE_TYPE],
    building: Building,
    progress: DownloadAndProcessProgress,
    semaphore: asyncio.Semaphore,
    processing_semaphore: asyncio.Semaphore,
) -> list[Path]:
    url = urljoin(OEDI_WEB_URL, building.energy_models_path)
    async with semaphore, download(client, url, progress) as f:
        progress.on_processing_started()
        file_path = Path(cast(str, f.name))
        result = await _async_process_energy_models(
            file_path,
            target_folder,
            file_types,
            building,
            processing_semaphore,
        )
        progress.on_processing_finished()
        progress.on_building_finished()
        return result


async def _async_process_energy_models(
    zip_file_path: Path,
    target_folder: Path,
    file_types: Collection[ENERGY_MODEL_FILE_TYPE],
    building: Building,
    semaphore: asyncio.Semaphore,
) -> list[Path]:
    async with semaphore:
        return await asyncio.to_thread(_process_energy_models, zip_file_path, target_folder, file_types, building)


def _process_energy_models(
    zip_file_path: Path,
    target_folder: Path,
    file_types: Collection[ENERGY_MODEL_FILE_TYPE],
    building: Building,
) -> list[Path]:
    result: list[Path] = []
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
                result.append(target_file)

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
                result.append(target_file)
    return result
