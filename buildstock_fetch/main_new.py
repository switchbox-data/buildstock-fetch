from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import polars as pl
from httpx import AsyncClient

from buildstock_fetch.building_ import Building
from buildstock_fetch.constants import METADATA_DIR
from buildstock_fetch.releases import RELEASES, BuildstockRelease
from buildstock_fetch.types import (
    ReleaseKey,
    UpgradeID,
    USStateCode,
)


@dataclass(frozen=True)
class AppConfig:
    client: AsyncClient
    executor: ProcessPoolExecutor
    download_path: Path


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
