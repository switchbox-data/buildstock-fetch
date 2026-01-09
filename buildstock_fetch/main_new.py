from collections.abc import Collection
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import aioboto3
import httpx
import polars as pl
from aiobotocore.config import AioConfig
from botocore import UNSIGNED  # pyright: ignore[reportAny]
from httpx import AsyncClient

from buildstock_fetch.annualcurves import download_and_process_annual_results
from buildstock_fetch.building_ import Building
from buildstock_fetch.constants import METADATA_DIR
from buildstock_fetch.energymodels import ENERGY_MODEL_FILE_TYPE, download_and_process_energy_models_batch
from buildstock_fetch.metadata import download_and_process_metadata_batch
from buildstock_fetch.releases import RELEASES, BuildstockRelease
from buildstock_fetch.tripschedules import download_and_process_trip_schedules_batch
from buildstock_fetch.types import (
    FileType,
    ReleaseKey,
    UpgradeID,
    USStateCode,
)
from buildstock_fetch.weather import download_and_process_weather_batch

from .loadcurves import LoadCurve, download_and_process_load_curves_batch


@dataclass(frozen=True)
class AppConfig:
    client: AsyncClient
    executor: ProcessPoolExecutor
    download_path: Path


async def download_and_process_all(
    target_folder: Path,
    buildings: Collection[Building],
    file_types: Collection[FileType],
    max_concurrent_downloads: int = 20,
) -> None:
    file_types_ = set(file_types)
    limits = httpx.Limits(max_connections=max_concurrent_downloads, max_keepalive_connections=10)
    timeout = httpx.Timeout(60)
    s3_session = aioboto3.Session()
    s3_client_ = s3_session.client("s3", config=AioConfig(signature_version=UNSIGNED))  # pyright: ignore[reportUnknownMemberType]
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client, s3_client_ as s3_client:
        if "metadata" in file_types_:
            _ = await download_and_process_metadata_batch(target_folder, client, buildings)
        curves: set[LoadCurve] = {"load_curve_15min", "load_curve_hourly", "load_curve_daily", "load_curve_monthly"}
        if curves & file_types_:
            _ = await download_and_process_load_curves_batch(
                target_folder,
                client,
                curves & file_types_,
                list(buildings),
            )
        if "load_curve_annual" in file_types_:
            _ = await download_and_process_annual_results(
                target_folder,
                client,
                buildings,
            )
        energy_models: set[ENERGY_MODEL_FILE_TYPE] = {"hpxml", "schedule"}
        if energy_models & file_types_:
            _ = await download_and_process_energy_models_batch(
                target_folder,
                client,
                energy_models & file_types_,
                buildings,
            )
        if "trip_schedules" in file_types_:
            _ = await download_and_process_trip_schedules_batch(
                target_folder,
                s3_client,
                buildings,
            )
        if "weather" in file_types:
            _ = await download_and_process_weather_batch(
                target_folder,
                client,
                buildings,
            )


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

    lst: list[tuple[int, str]] | list[tuple[int, None]]

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
