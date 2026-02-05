import logging
from collections.abc import Collection
from functools import cached_property
from pathlib import Path
from random import Random
from typing import cast

import polars as pl
from cloudpathlib import S3Path
from typing_extensions import final, override

from buildstock_fetch.explore import DownloadedData, filter_downloads
from buildstock_fetch.releases import BuildstockRelease, BuildstockReleases
from buildstock_fetch.types import (
    FileType,
    ReleaseKey,
    UpgradeID,
    USStateCode,
    is_s3_path,
    is_valid_state_code,
    normalize_upgrade_id,
)


class DataNotFoundError(Exception): ...


@final
class InvalidUpgradeForRelease(ValueError):
    def __init__(self, release: BuildstockRelease, *upgrades: UpgradeID) -> None:
        self.release = release
        self.upgrades = upgrades
        super().__init__()

    @override
    def __str__(self) -> str:
        if len(self.upgrades) == 1:
            return (
                f"Upgrade {self.upgrades[0]} is not valid for release {self.release.key}. "
                f"Valid upgrades: {sorted(self.release.upgrades, key=int)}"
            )
        return (
            f"Upgrades {self.upgrades} are not valid for release {self.release.key}. "
            f"Valid upgrades: {sorted(self.release.upgrades, key=int)}"
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
        random: Optional random state for reproducible sampling (Random instance or int seed).

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
        data_path: Path | S3Path | str,
        release: ReleaseKey | BuildstockRelease,
        states: USStateCode | Collection[USStateCode] | None = None,
        sample_n: int | None = None,
        random: Random | int | None = None,
    ) -> None:
        self.data_path = S3Path(cast(str, data_path)) if is_s3_path(data_path) else Path(cast(str, data_path))
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
    def downloaded_metadata(self) -> DownloadedData:
        return DownloadedData(
            filter_downloads(
                self.data_path,
                release_key=(self.release.key,),
                state=self.states,
                file_type="metadata",
            )
        )

    @cached_property
    def sampled_buildings(self) -> frozenset[int] | None:
        if self.sample_n is None:
            return None
        metadata_files = self.downloaded_metadata.filter(file_type="metadata", suffix=".parquet")
        if not metadata_files:
            raise MetadataNotFoundError(self.release, self.states)

        # Get unique states and find minimum upgrade per state
        # This ensures sampling from all states when multiple states are requested
        states = {f.state for f in metadata_files}
        files_to_read = []
        for state in states:
            state_files = [f for f in metadata_files if f.state == state]
            min_upgrade = min(state_files, key=lambda f: int(f.upgrade)).upgrade
            files_to_read.extend([f for f in state_files if f.upgrade == min_upgrade])

        df = pl.scan_parquet([str(f.file_path) for f in files_to_read]).select("bldg_id").collect()
        all_building_ids = cast(list[int], df["bldg_id"].unique().to_list())

        if self.sample_n > len(all_building_ids):
            logging.getLogger(__name__).info(
                f"sample_n ({self.sample_n}) exceeds available buildings ({len(all_building_ids)}). "
                + "Returning all buildings without sampling."
            )
            return frozenset(all_building_ids)

        return frozenset(self.random.sample(all_building_ids, self.sample_n))

    def read_metadata(
        self, upgrades: str | Collection[str] | None = None, building_ids: Collection[int] | None = None
    ) -> pl.LazyFrame:
        return self.read_parquets("metadata", upgrades, building_ids)

    def read_load_curve_15min(
        self, upgrades: str | Collection[str] | None = None, building_ids: Collection[int] | None = None
    ) -> pl.LazyFrame:
        return self.read_parquets("load_curve_15min", upgrades, building_ids)

    def read_load_curve_hourly(
        self, upgrades: str | Collection[str] | None = None, building_ids: Collection[int] | None = None
    ) -> pl.LazyFrame:
        return self.read_parquets("load_curve_hourly", upgrades, building_ids)

    def read_load_curve_daily(
        self, upgrades: str | Collection[str] | None = None, building_ids: Collection[int] | None = None
    ) -> pl.LazyFrame:
        return self.read_parquets("load_curve_daily", upgrades, building_ids)

    def read_load_curve_monthly(
        self, upgrades: str | Collection[str] | None = None, building_ids: Collection[int] | None = None
    ) -> pl.LazyFrame:
        return self.read_parquets("load_curve_monthly", upgrades, building_ids)

    def read_load_curve_annual(
        self, upgrades: str | Collection[str] | None = None, building_ids: Collection[int] | None = None
    ) -> pl.LazyFrame:
        return self.read_parquets("load_curve_annual", upgrades, building_ids)

    def read_parquets(
        self,
        file_type: FileType,
        upgrades: str | Collection[str] | None = None,
        building_ids: Collection[int] | None = None,
    ) -> pl.LazyFrame:
        if "metadata" not in self.release.file_types:
            raise FileTypeNotAvailableError(self.release, file_type)

        upgrades = self._validate_upgrades(file_type, upgrades)

        # We use different reading strategies based on file type:
        # - Metadata files: Must use diagonal concat because upgrade 0 metadata files have missing
        #   upgrade columns compared to other upgrades, creating schema mismatches which would cause
        #   scan_parquet with globbed files paths to fail. Diagonal concat handles this by filling
        #   missing columns with nulls.
        # - Load curve files: Can use a single glob pattern with hive partitioning since schemas
        #   are consistent across upgrades.

        if file_type == "metadata":
            lf = self._read_metadata_with_diagonal_concat(file_type)
        else:
            lf = pl.scan_parquet(str(self.data_path / self.release.key / file_type) + "/")

        # Apply all filters using Polars (states, upgrades, building_ids, sampled_buildings)
        # Polars automatically infers state/upgrade columns from hive partitioning
        lf = self._apply_filters(lf, upgrades, building_ids)

        return lf

    def _read_metadata_with_diagonal_concat(self, file_type: FileType) -> pl.LazyFrame:
        """Read all metadata files and concat diagonally, then filters will be applied."""
        # Get all metadata files (not filtered by state/upgrade - we'll filter with Polars)
        files = self.downloaded_metadata.filter(file_type=file_type, suffix=".parquet")
        file_paths = [str(file.file_path) for file in files]

        if not file_paths:
            return pl.LazyFrame()

        # Scan each file with hive partitioning and concat diagonally
        # This handles schema mismatches where different upgrades have different columns
        # (e.g., upgrade 0 metadata files have missing upgrade columns)
        lazy_frames = [pl.scan_parquet(file_path, hive_partitioning=True) for file_path in file_paths]
        # Type checker doesn't know concat returns LazyFrame when given LazyFrames
        result = pl.concat(lazy_frames, how="diagonal")
        return cast(pl.LazyFrame, result)

    def _apply_filters(
        self,
        lf: pl.LazyFrame,
        upgrades: frozenset[UpgradeID] | None = None,
        building_ids: Collection[int] | None = None,
    ) -> pl.LazyFrame:
        """Apply all filters (states, upgrades, building_ids, sampled_buildings) using Polars.

        Filter order: Partition filters (state/upgrade) are applied first as they can prune
        entire files/partitions via hive partitioning. Row-level filters (bldg_id) are applied
        after and use predicate pushdown with row group statistics. Polars query optimizer
        collects all predicates automatically, but ordering partition filters first helps
        ensure they're recognized for file pruning.
        """

        # Apply state and upgradepartition filters first - these can prune entire files/partitions
        # See: https://pola.rs/posts/predicate-pushdown-query-optimizer/

        # Apply state filter if states are specified
        if self.states is not None:
            lf = lf.filter(pl.col("state").is_in(self.states))

        # Apply upgrade filter if upgrades are specified
        # Hive partitions treats upgrades as int64
        if upgrades:
            upgrade_values = [int(upgrade) for upgrade in upgrades]
            lf = lf.filter(pl.col("upgrade").is_in(upgrade_values))

        # Apply row-level filters - combine multiple bldg_id filters for efficiency
        building_id_sets = []
        if building_ids is not None:
            building_id_sets.append(set(building_ids))
        if self.sampled_buildings is not None:
            building_id_sets.append(set(self.sampled_buildings))

        if building_id_sets:
            # Intersect all building ID sets and apply single filter
            combined_ids = set.intersection(*building_id_sets) if len(building_id_sets) > 1 else building_id_sets[0]
            if combined_ids:
                lf = lf.filter(pl.col("bldg_id").is_in(list(combined_ids)))

        return lf

    def _available_upgrades(self, file_type: FileType) -> frozenset[UpgradeID]:
        if file_type == "metadata":
            return self.downloaded_metadata.filter(state=self.states, file_type=file_type).upgrades()

        state_upgrade_ids = []
        try:
            for state_path in (self.data_path / self.release.key / file_type).iterdir():
                # Note: This is a little weird if multiple states. Currently returns intersection,
                # so only upgrades available in all states are returned.
                if self.states is None or state_path.name.removeprefix("state=") in self.states:
                    state_upgrade_ids.append({u.name.removeprefix("upgrade=") for u in state_path.iterdir()})
        except Exception as e:
            raise NoUpgradesFoundError(self.release) from e
        return frozenset(normalize_upgrade_id(_) for _ in set.intersection(*state_upgrade_ids))

    def _validate_upgrades(
        self, file_type: FileType, upgrades: str | Collection[str] | None = None
    ) -> frozenset[UpgradeID]:
        if upgrades is None:
            upgrades = None
        elif isinstance(upgrades, str):
            upgrades = [normalize_upgrade_id(upgrades)]
        else:
            upgrades = [normalize_upgrade_id(_) for _ in upgrades]

        # We shouldn't raise an error here - an empty list may be passed intentionally
        if upgrades is not None and not upgrades:
            logging.getLogger(__name__).info("Empty upgrades list got passed into validate_upgrades")
            return frozenset()

        if upgrades and (invalid_upgrades := [_ for _ in upgrades if _ not in self.release.upgrades]):
            raise InvalidUpgradeForRelease(self.release, *cast(tuple[UpgradeID, ...], tuple(invalid_upgrades)))

        available_upgrades = self._available_upgrades(file_type)
        if not available_upgrades:
            raise NoUpgradesFoundError(self.release)

        if upgrades and (missing_upgrades := [_ for _ in upgrades if _ not in available_upgrades]):
            raise UpgradeNotFoundError(
                self.release, available_upgrades, *cast(tuple[UpgradeID, ...], tuple(missing_upgrades))
            )

        if upgrades:
            return frozenset(cast(Collection[UpgradeID], upgrades))
        return frozenset(available_upgrades)
