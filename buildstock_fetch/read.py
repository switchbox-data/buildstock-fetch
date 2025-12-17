import logging
from collections.abc import Collection
from functools import cached_property
from os import PathLike
from pathlib import Path
from random import Random
from typing import cast, final, override

import polars as pl

from buildstock_fetch.explore import DownloadedData, filter_downloads
from buildstock_fetch.releases import BuildstockRelease, BuildstockReleases
from buildstock_fetch.types import (
    FileType,
    ReleaseKey,
    UpgradeID,
    USStateCode,
    is_valid_state_code,
    normalize_upgrade_id,
)


class DataNotFoundError(Exception): ...


@final
class InvalidUpgradeForRelease(ValueError):
    def __init__(self, release: BuildstockRelease, *upgrade_ids: UpgradeID) -> None:
        self.release = release
        self.upgrade_ids = upgrade_ids
        super().__init__()

    @override
    def __str__(self) -> str:
        if len(self.upgrade_ids) == 1:
            return (
                f"Upgrade {self.upgrade_ids[0]} is not valid for release {self.release.key}. "
                f"Valid upgrades: {sorted(self.release.upgrade_ids, key=int)}"
            )
        return (
            f"Upgrades {self.upgrade_ids} are not valid for release {self.release.key}. "
            f"Valid upgrades: {sorted(self.release.upgrade_ids, key=int)}"
        )


@final
class MetadataNotFoundError(DataNotFoundError):
    def __init__(self, release: BuildstockRelease, states: Collection[USStateCode] | None) -> None:
        self.release = release
        self.states = states
        super().__init__()

    @override
    def __str__(self) -> str:
        if self.states:
            return (
                f"No metadata found for release {self.release.key} and states {sorted(self.states)}. "
                "Please download metadata using bsf first."
            )
        return f"No metadata found for release {self.release.key}. Please download metadata using bsf first."


@final
class NoUpgradesFoundError(DataNotFoundError):
    def __init__(self, release: BuildstockRelease) -> None:
        self.release = release
        super().__init__()

    @override
    def __str__(self) -> str:
        return f"No upgrades found on disk for release {self.release.key}"


@final
class UpgradeNotFoundError(DataNotFoundError):
    def __init__(
        self, release: BuildstockRelease, available_upgrades: Collection[UpgradeID], *missing_upgrades: UpgradeID
    ) -> None:
        self.release = release
        self.available_upgrades = available_upgrades
        self.missing_upgrades = missing_upgrades
        super().__init__()

    @override
    def __str__(self) -> str:
        if len(self.missing_upgrades) == 1:
            return (
                f"Upgrade {self.missing_upgrades[0]} is not found on disk for release {self.release.key}. "
                f"Available upgrades on disk: {sorted(self.available_upgrades, key=int)}. "
                f"Please download the data using bsf first."
            )
        return (
            f"Upgrades {sorted(self.missing_upgrades, key=int)} are not found on disk for release {self.release.key}. "
            f"Available upgrades on disk: {sorted(self.available_upgrades, key=int)}. "
            f"Please download the data using bsf first."
        )


@final
class FileTypeNotAvailableError(ValueError):
    def __init__(self, release: BuildstockRelease, file_type: FileType) -> None:
        self.release = release
        self.file_type = file_type
        super().__init__()

    @override
    def __str__(self) -> str:
        return (
            f"File type {self.file_type} is not available for release {self.release.key}. "
            f"Available file types are: {sorted(self.release.file_types)}"
        )


@final
class BuildStockRead:
    """Reader class for BuildStock data downloaded with bsf.

    This class provides methods to read metadata and load curve data
    from locally downloaded BuildStock files.

    Args:
        data_path: Path to the data directory (local path or S3 path).
        release: A BuildStockRelease enum member specifying the release.
        states: Optional State or list of States to filter data.
            If None, auto-detects states present on disk.
        sample_n: Optional number of buildings to sample.
        seed: Optional random seed for reproducible sampling.

    Example:
        >>> from buildstock_fetch.read import BuildStockRead
        >>> bsr = BuildStockRead(
        ...     data_path="./data",
        ...     release="res_2024_tmy_2",
        ...     states="NY",
        ... )
        >>> metadata = bsr.read_metadata(upgrades=["0", "1"])

    """

    def __init__(
        self,
        data_path: Path | str,
        release: ReleaseKey | BuildstockRelease,
        states: USStateCode | Collection[USStateCode] | None = None,
        sample_n: int | None = None,
        random: Random | int | None = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.release = release if isinstance(release, BuildstockRelease) else BuildstockReleases.load()[release]

        self.states: list[USStateCode] | None
        if states is None:
            self.states = None
        elif is_valid_state_code(states):
            self.states = [states]
        else:
            self.states = list(states)

        if random is None:
            self.random = Random()
        elif isinstance(random, Random):
            self.random = random
        else:
            self.random = Random(random)

        self.sample_n = sample_n

    @cached_property
    def downloaded_data(self) -> DownloadedData:
        return DownloadedData(
            filter_downloads(
                self.data_path,
                release_key=(self.release.key,),
                state=self.states,
            )
        )

    @cached_property
    def sampled_buildings(self) -> frozenset[int] | None:
        if self.sample_n is None:
            return None
        metadata_files = self.downloaded_data.filter(file_type="metadata", suffix=".parquet")
        if not metadata_files:
            raise MetadataNotFoundError(self.release, self.states)
        first_metadata_file = min(metadata_files, key=lambda _: int(_.upgrade_id))
        df = pl.scan_parquet(first_metadata_file.file_path).select("bldg_id").collect()
        all_building_ids = cast(list[int], df["bldg_id"].unique().to_list())

        if self.sample_n > len(all_building_ids):
            logging.getLogger(__name__).info(
                f"sample_n ({self.sample_n}) exceeds available buildings ({len(all_building_ids)}). "
                + "Returning all buildings without sampling."
            )
            return frozenset(all_building_ids)

        return frozenset(self.random.sample(all_building_ids, self.sample_n))

    def read_metadata(self, upgrade_ids: str | Collection[str] | None = None) -> pl.LazyFrame:
        return self.read_parquets("metadata", upgrade_ids)

    def read_load_curve_15min(self, upgrade_ids: str | Collection[str] | None = None) -> pl.LazyFrame:
        return self.read_parquets("load_curve_15min", upgrade_ids)

    def read_load_curve_hourly(self, upgrade_ids: str | Collection[str] | None = None) -> pl.LazyFrame:
        return self.read_parquets("load_curve_hourly", upgrade_ids)

    def read_load_curve_daily(self, upgrade_ids: str | Collection[str] | None = None) -> pl.LazyFrame:
        return self.read_parquets("load_curve_daily", upgrade_ids)

    def read_load_curve_monthly(self, upgrade_ids: str | Collection[str] | None = None) -> pl.LazyFrame:
        return self.read_parquets("load_curve_monthly", upgrade_ids)

    def read_load_curve_annual(self, upgrade_ids: str | Collection[str] | None = None) -> pl.LazyFrame:
        return self.read_parquets("load_curve_annual", upgrade_ids)

    def read_parquets(self, file_type: FileType, upgrade_ids: str | Collection[str] | None = None) -> pl.LazyFrame:
        if "metadata" not in self.release.file_types:
            raise FileTypeNotAvailableError(self.release, file_type)

        upgrade_ids = self._validate_upgrades(file_type, upgrade_ids)

        files = self.downloaded_data.filter(file_type=file_type, suffix=".parquet", upgrade_id=upgrade_ids)
        lf = pl.scan_parquet([_.file_path for _ in files])
        lf = self._apply_sampling_filter(lf)
        return lf

    def _validate_upgrades(
        self, file_type: FileType, upgrade_ids: str | Collection[str] | None = None
    ) -> frozenset[UpgradeID]:
        if upgrade_ids is None:
            upgrade_ids = None
        elif isinstance(upgrade_ids, str):
            upgrade_ids = [normalize_upgrade_id(upgrade_ids)]
        else:
            upgrade_ids = [normalize_upgrade_id(_) for _ in upgrade_ids]

        # We shouldn't raise an error here - an empty list may be passed intentionally
        if upgrade_ids is not None and not upgrade_ids:
            logging.getLogger(__name__).info("Empty upgrades list got passed into validate_upgrades")
            return frozenset()

        if upgrade_ids and (invalid_upgrades := [_ for _ in upgrade_ids if _ not in self.release.upgrade_ids]):
            raise InvalidUpgradeForRelease(self.release, *invalid_upgrades)  # type: ignore[arg-type]

        available_upgrades = self.downloaded_data.filter(state=self.states, file_type=file_type).upgrade_ids()
        if not available_upgrades:
            raise NoUpgradesFoundError(self.release)

        if upgrade_ids and (missing_upgrades := [_ for _ in upgrade_ids if _ not in available_upgrades]):
            raise UpgradeNotFoundError(self.release, available_upgrades, *missing_upgrades)  # type: ignore[arg-type]

        return frozenset(upgrade_ids or available_upgrades)  # type: ignore [arg-type]

    def _apply_sampling_filter(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Apply sampling filter if sampled_bldgs is set."""
        if self.sampled_buildings is not None:
            return lf.filter(pl.col("bldg_id").is_in(self.sampled_buildings))
        return lf
