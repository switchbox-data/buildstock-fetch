"""Mixed upgrade scenario orchestration for multi-year adoption trajectories.

This module provides the MixedUpgradeScenario class for defining and reading
heterogeneous upgrade mixes where buildings progressively adopt different
upgrades over time.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Collection, Iterable
from functools import cached_property
from pathlib import Path
from random import Random

import polars as pl
from typing_extensions import final, override

from buildstock_fetch.explore import DownloadedData, filter_downloads
from buildstock_fetch.read import BuildStockRead, DataNotFoundError
from buildstock_fetch.releases import BuildstockRelease, BuildstockReleases
from buildstock_fetch.scenarios import validate_scenario
from buildstock_fetch.types import (
    FileType,
    ReleaseKey,
    UpgradeID,
    USStateCode,
    is_valid_state_code,
    normalize_upgrade_id,
)


@final
class ScenarioDataNotFoundError(DataNotFoundError):
    """Raised when data for scenario upgrades is not found on disk."""

    def __init__(self, file_type: FileType, missing_upgrades: Collection[UpgradeID]) -> None:
        self.file_type = file_type
        self.missing_upgrades = missing_upgrades
        super().__init__()

    @override
    def __str__(self) -> str:
        return (
            f"Missing {self.file_type} data for upgrades: {sorted(self.missing_upgrades, key=int)}. "
            f"Please download the required upgrades using bsf."
        )


@final
class ScenarioParameterRequiredError(ValueError):
    """Raised when scenario parameter is not provided to MixedUpgradeScenario."""

    @override
    def __str__(self) -> str:
        return "scenario parameter is required for MixedUpgradeScenario"


@final
class BaselineMetadataNotFoundError(DataNotFoundError):
    """Raised when baseline metadata is not found on disk."""

    @override
    def __str__(self) -> str:
        return "No baseline (upgrade 0) metadata found. Please download baseline data using bsf."


@final
class NoBuildingsAvailableError(ValueError):
    """Raised when no buildings are available for sampling."""

    @override
    def __str__(self) -> str:
        return "No buildings available for sampling. Please check that metadata exists."


@final
class YearOutOfRangeError(ValueError):
    """Raised when a year index is out of range."""

    def __init__(self, year_idx: int, max_year: int) -> None:
        self.year_idx = year_idx
        self.max_year = max_year
        super().__init__()

    @override
    def __str__(self) -> str:
        return f"Year index {self.year_idx} is out of range. Valid range: [0, {self.max_year}]"


@final
class MixedUpgradeScenario:
    """Class for orchestrating multi-year adoption trajectories across multiple upgrade scenarios.

    This class enables defining and reading heterogeneous upgrade mixes where buildings
    progressively adopt different upgrades over time. Buildings are sampled once from a
    baseline upgrade, then allocated to different upgrades according to adoption fractions
    per year. Monotonic adoption is enforced: buildings can only move from baseline to
    an upgrade, never backwards.

    Args:
        data_path: Path to the data directory (local path or S3 path).
        pathway_scenario_name: Name of the pathway scenario. will be used to create subdirectories when writing out the metadata and load curves.
        release: A BuildstockRelease or release key string specifying the release.
        states: Optional state code or list of state codes to filter data.
            If None, auto-detects states present on disk.
        sample_n: Optional number of buildings to sample from baseline.
        random: Optional Random instance or seed for reproducible sampling and allocation.
        scenario: Dict mapping upgrade IDs to adoption fractions per year.
            Example: {4: [0.06, 0.18, 0.30], 8: [0.04, 0.12, 0.20]}
            represents 3 years where upgrade 4 grows from 6% to 30% adoption
            and upgrade 8 grows from 4% to 20% adoption.

    Example:
        >>> from buildstock_fetch.mixed_upgrade import MixedUpgradeScenario
        >>> from buildstock_fetch.scenarios import uniform_adoption
        >>> scenario = uniform_adoption(
        ...     upgrade_ids=[4, 8],
        ...     weights={4: 0.6, 8: 0.4},
        ...     adoption_trajectory=[0.1, 0.3, 0.5],
        ... )
        >>> mus = MixedUpgradeScenario(
        ...     data_path="./data",
        ...     pathway_scenario_name="rapid_adoption",
        ...     release="res_2024_tmy3_2",
        ...     states="NY",
        ...     sample_n=1000,
        ...     random=42,
        ...     scenario=scenario,
        ... )
        >>> metadata = mus.read_metadata().collect()
        >>> mus.export_scenario_to_cairo("./scenario.csv")
    """

    def __init__(
        self,
        data_path: str | Path,
        pathway_scenario_name: str,
        release: ReleaseKey | BuildstockRelease,
        states: USStateCode | Collection[USStateCode] | None = None,
        sample_n: int | None = None,
        random: Random | int | None = None,
        scenario: dict[int, list[float]] | None = None,
    ) -> None:
        if scenario is None:
            raise ScenarioParameterRequiredError()

        validate_scenario(scenario)
        self.scenario = scenario
        self.pathway_scenario_name = pathway_scenario_name
        self._upgrade_ids = tuple(scenario)
        self.num_years = len(next(iter(scenario.values())))

        self.data_path = Path(data_path)
        self.release = release if isinstance(release, BuildstockRelease) else BuildstockReleases.load()[release]

        if states is None:
            self.states: list[USStateCode] | None = None
        elif is_valid_state_code(states):
            self.states = [states]
        else:
            self.states = list(states)

        self.random = random if isinstance(random, Random) else Random(random)

        self.sample_n = sample_n
        self._available_bldg_ids_cache: dict[str, dict[int, frozenset[int]]] = {}

        # Baseline reader handles state detection and optional sampling.
        self.baseline_reader = BuildStockRead(
            data_path=data_path,
            release=self.release,
            states=self.states,
            sample_n=sample_n,
            random=self.random,
        )

        sampled = self.baseline_reader.sampled_buildings
        if sampled is not None:
            self.sampled_bldgs: frozenset[int] = sampled
        else:
            # No sampling: fall back to full baseline bldg_id list.
            metadata_files = self.downloaded_data.filter(
                file_type="metadata", suffix=".parquet", upgrade=normalize_upgrade_id("0")
            )
            if not metadata_files:
                raise BaselineMetadataNotFoundError()
            first_file = min(metadata_files, key=lambda x: x.file_path)
            df = pl.scan_parquet(first_file.file_path).select("bldg_id").collect()
            all_bldg_ids = df["bldg_id"].unique().to_list()
            self.sampled_bldgs = frozenset(all_bldg_ids)

        if not self.sampled_bldgs:
            raise NoBuildingsAvailableError()

        num_upgrades = len(scenario)
        print(f"Sampled {len(self.sampled_bldgs)} buildings from baseline (upgrade 0)")
        print(f"Materialized {self.num_years} years of adoption across {num_upgrades} upgrades")

    @cached_property
    def downloaded_data(self) -> DownloadedData:
        """Cached property for downloaded data discovery."""
        return DownloadedData(
            filter_downloads(
                self.data_path,
                release_key=(self.release.key,),
                state=self.states,
            )
        )

    @cached_property
    def _allocation_plan(self) -> tuple[list[int], dict[int, list[int]]]:
        bldgs = list(self.sampled_bldgs)
        self.random.shuffle(bldgs)
        allocations: dict[int, list[int]] = {uid: [] for uid in self._upgrade_ids}
        next_idx = 0
        total = len(bldgs)

        for year_idx in range(self.num_years):
            for uid in self._upgrade_ids:
                target = int(self.scenario[uid][year_idx] * total)
                new = target - len(allocations[uid])
                if new > 0:
                    # Assign contiguous slices once; yearly targets take prefixes.
                    end = next_idx + new
                    allocations[uid].extend(bldgs[next_idx:end])
                    next_idx = end

        return bldgs, allocations

    @cached_property
    def materialized_scenario(self) -> dict[int, dict[int, int]]:
        """Materialize building allocations for all years in the scenario."""
        bldgs, allocations = self._allocation_plan
        total = len(bldgs)
        materialized: dict[int, dict[int, int]] = {}

        for year_idx in range(self.num_years):
            # Start from baseline, then overlay upgrades for this year.
            year_map = dict.fromkeys(bldgs, 0)
            for uid in self._upgrade_ids:
                target = int(self.scenario[uid][year_idx] * total)
                if target:
                    for bldg_id in allocations[uid][:target]:
                        year_map[bldg_id] = uid
            materialized[year_idx] = year_map

        return materialized

    def _validate_years(self, years: list[int] | None) -> list[int]:
        """Validate and resolve years parameter.

        Args:
            years: List of year indices, or None for all years.

        Returns:
            List of valid year indices.

        Raises:
            ValueError: If any year index is out of range.
        """
        if years is None:
            return list(range(self.num_years))

        for year_idx in years:
            if not 0 <= year_idx < self.num_years:
                raise YearOutOfRangeError(year_idx, self.num_years - 1)

        return years

    def _validate_data_availability(self, file_type: FileType) -> None:
        """Validate that all required upgrade data exists on disk.

        Args:
            file_type: Type of data to check (e.g., 'metadata', 'load_curve_15min').

        Raises:
            ScenarioDataNotFoundError: If data for any scenario upgrade is missing.
        """
        required_upgrades = {normalize_upgrade_id(str(uid)) for uid in self._upgrade_ids} | {normalize_upgrade_id("0")}
        missing_upgrades = required_upgrades - self.downloaded_data.filter(file_type=file_type).upgrades()
        if missing_upgrades:
            raise ScenarioDataNotFoundError(file_type, missing_upgrades)

    def _available_bldg_ids_by_upgrade(self, file_type: FileType) -> dict[int, frozenset[int]]:
        per_building_types = {
            "load_curve_15min",
            "load_curve_hourly",
            "load_curve_daily",
            "load_curve_monthly",
            "load_curve_annual",
        }
        if file_type not in per_building_types:
            return {}
        cached = self._available_bldg_ids_cache.get(file_type)
        if cached is not None:
            return cached

        needed_upgrades = {0, *self._upgrade_ids}
        available: dict[int, set[int]] = {uid: set() for uid in needed_upgrades}
        for info in self.downloaded_data.filter(file_type=file_type):
            try:
                upgrade_id = int(info.upgrade)
                # Extract leading building ID (any digit length) from filename
                match = re.match(r"^(\d+)-", info.filename)
                if not match:
                    continue
                bldg_id = int(match.group(1))
            except (TypeError, ValueError):
                continue
            if upgrade_id in available:
                available[upgrade_id].add(bldg_id)

        cached = {uid: frozenset(ids) for uid, ids in available.items()}
        self._available_bldg_ids_cache[file_type] = cached
        return cached

    def _warn_on_missing_ids(
        self,
        file_type: FileType,
        upgrade_id: int,
        year_idx: int,
        requested_ids: list[int],
        available_ids: frozenset[int],
        sample_size: int = 10,
    ) -> None:
        if not requested_ids or not available_ids:
            return
        missing = [bldg_id for bldg_id in requested_ids if bldg_id not in available_ids]
        if not missing:
            return
        logger = logging.getLogger(__name__)
        logger.warning(
            "Missing %s files for upgrade %s in year_idx %s: %s of %s requested IDs not found%s",
            file_type,
            upgrade_id,
            year_idx,
            len(missing),
            len(requested_ids),
            f" (example IDs: {missing[:sample_size]})" if sample_size else "",
        )

    def _read_for_upgrade(
        self,
        file_type: FileType,
        upgrade_id: int,
        bldg_ids: Iterable[int],
        year_idx: int,
        available_ids_by_upgrade: dict[int, frozenset[int]],
    ) -> pl.LazyFrame | None:
        """Read data for a specific upgrade and set of building IDs."""
        bldg_ids_list = list(bldg_ids)
        if not bldg_ids_list:
            return None
        if available_ids_by_upgrade:
            available_ids = available_ids_by_upgrade.get(upgrade_id, frozenset())
            self._warn_on_missing_ids(file_type, upgrade_id, year_idx, bldg_ids_list, available_ids)
        lf = self.baseline_reader.read_parquets(file_type, upgrades=str(upgrade_id), building_ids=bldg_ids_list)
        if file_type == "metadata":
            lf = lf.rename({"upgrade": "upgrade_id"})
        else:
            lf = lf.with_columns(pl.lit(upgrade_id).alias("upgrade_id"))
        return lf.with_columns(pl.lit(year_idx).alias("year"))

    def _process_year(
        self,
        file_type: FileType,
        year_idx: int,
        bldgs: list[int],
        allocations: dict[int, list[int]],
        total: int,
        available_ids_by_upgrade: dict[int, frozenset[int]],
    ) -> pl.LazyFrame | None:
        """Process data for a single year, reading all upgrades and baseline."""
        year_dfs: list[pl.LazyFrame] = []
        total_adopted = 0

        for uid in self._upgrade_ids:
            target = int(self.scenario[uid][year_idx] * total)
            total_adopted += target
            lf = self._read_for_upgrade(file_type, uid, allocations[uid][:target], year_idx, available_ids_by_upgrade)
            if lf is not None:
                year_dfs.append(lf)

        # Baseline = remaining buildings after all upgrade allocations.
        lf = self._read_for_upgrade(file_type, 0, bldgs[total_adopted:], year_idx, available_ids_by_upgrade)
        if lf is not None:
            year_dfs.append(lf)

        if year_dfs:
            return pl.concat(year_dfs, how="vertical_relaxed")
        return None

    def _read_data_for_scenario(self, file_type: FileType, years: list[int] | None = None) -> pl.LazyFrame:
        """Read data across upgrades and years without full materialization."""
        years = sorted(self._validate_years(years))
        self._validate_data_availability(file_type)
        available_ids_by_upgrade = self._available_bldg_ids_by_upgrade(file_type)
        # Build order + per-upgrade adopter lists once; slice per year.
        bldgs, allocations = self._allocation_plan
        total = len(bldgs)

        all_year_dfs: list[pl.LazyFrame] = []
        for year_idx in years:
            year_df = self._process_year(file_type, year_idx, bldgs, allocations, total, available_ids_by_upgrade)
            if year_df is not None:
                all_year_dfs.append(year_df)

        if not all_year_dfs:
            return pl.LazyFrame({"bldg_id": [], "upgrade_id": [], "year": []})

        return pl.concat(all_year_dfs, how="vertical_relaxed")

    def read_metadata(self, years: list[int] | None = None) -> pl.LazyFrame:
        """Read metadata for specified years in the scenario.

        Returns a LazyFrame containing metadata for all buildings and years in the
        scenario. Each row represents one building in one year.

        Args:
            years: List of year indices to include (0-indexed), or None for all years.
                Example: [0, 1, 2] or None

        Returns:
            A Polars LazyFrame with columns:
                - bldg_id: Building ID (from sampled baseline)
                - upgrade_id: Upgrade ID for this building in this year (0 or scenario upgrade)
                - year: Year index (0-indexed)
                - ...: Original metadata columns (e.g., in.state, in.vintage, etc.)

        Raises:
            ValueError: If any year index is out of range.
            ScenarioDataNotFoundError: If metadata for scenario upgrades is not on disk.

        Example:
            >>> # Read metadata for all years
            >>> metadata = mus.read_metadata()
            >>> df = metadata.collect()
            >>>
            >>> # Read metadata for specific years
            >>> metadata_early = mus.read_metadata(years=[0, 1])
        """
        return self._read_data_for_scenario("metadata", years)

    def read_load_curve_15min(self, years: list[int] | None = None) -> pl.LazyFrame:
        """Read 15-minute load curve data for specified years in the scenario.

        Args:
            years: List of year indices to include, or None for all years.

        Returns:
            A Polars LazyFrame with columns:
                - bldg_id: Building ID
                - upgrade_id: Upgrade ID
                - year: Year index
                - timestamp: Timestamp of the load data
                - ...: Energy columns (e.g., out.electricity.total.energy_consumption)

        Raises:
            ValueError: If any year index is out of range.
            ScenarioDataNotFoundError: If load curve data for scenario upgrades is not on disk.
        """
        return self._read_data_for_scenario("load_curve_15min", years)

    def read_load_curve_hourly(self, years: list[int] | None = None) -> pl.LazyFrame:
        """Read hourly load curve data for specified years in the scenario.

        Args:
            years: List of year indices to include, or None for all years.

        Returns:
            A Polars LazyFrame with columns:
                - bldg_id: Building ID
                - upgrade_id: Upgrade ID
                - year: Year index
                - timestamp: Timestamp of the load data
                - ...: Energy columns

        Raises:
            ValueError: If any year index is out of range.
            ScenarioDataNotFoundError: If load curve data for scenario upgrades is not on disk.
        """
        return self._read_data_for_scenario("load_curve_hourly", years)

    def read_load_curve_daily(self, years: list[int] | None = None) -> pl.LazyFrame:
        """Read daily load curve data for specified years in the scenario.

        Args:
            years: List of year indices to include, or None for all years.

        Returns:
            A Polars LazyFrame with columns:
                - bldg_id: Building ID
                - upgrade_id: Upgrade ID
                - year: Year index
                - timestamp: Timestamp of the load data
                - ...: Energy columns

        Raises:
            ValueError: If any year index is out of range.
            ScenarioDataNotFoundError: If load curve data for scenario upgrades is not on disk.
        """
        return self._read_data_for_scenario("load_curve_daily", years)

    def read_load_curve_annual(self, years: list[int] | None = None) -> pl.LazyFrame:
        """Read annual load curve data for specified years in the scenario.

        Args:
            years: List of year indices to include, or None for all years.

        Returns:
            A Polars LazyFrame with columns:
                - bldg_id: Building ID
                - upgrade_id: Upgrade ID
                - year: Year index
                - ...: Annual energy totals

        Raises:
            ValueError: If any year index is out of range.
            ScenarioDataNotFoundError: If load curve data for scenario upgrades is not on disk.
        """
        return self._read_data_for_scenario("load_curve_annual", years)

    def export_scenario_to_cairo(self, output_path: str | Path) -> None:
        """Export scenario to CAIRO-compatible CSV format.

        Creates a CSV file with one row per building and one column per year.
        Each cell contains the upgrade ID for that building in that year.

        Args:
            output_path: Path where the CSV file should be written.

        Output format:
            bldg_id,year_0,year_1,year_2
            405821,0,0,0
            612547,0,4,4
            789234,0,0,8

        Raises:
            ScenarioDataNotFoundError: If data for scenario upgrades is not on disk.

        Example:
            >>> mus.export_scenario_to_cairo("./scenario.csv")
            Exported scenario for 1000 buildings across 3 years to ./scenario.csv
        """
        self._validate_data_availability("metadata")
        bldgs, allocations = self._allocation_plan
        sorted_bldg_ids = sorted(self.sampled_bldgs)
        # Index once for stable, low-overhead column fills.
        idx = {bid: i for i, bid in enumerate(sorted_bldg_ids)}
        total = len(bldgs)

        year_cols = {f"year_{year_idx}": [0] * len(sorted_bldg_ids) for year_idx in range(self.num_years)}
        for year_idx in range(self.num_years):
            col = year_cols[f"year_{year_idx}"]
            # Fill each year column from the precomputed adopter slices.
            for uid in self._upgrade_ids:
                target = int(self.scenario[uid][year_idx] * total)
                if target:
                    for bldg_id in allocations[uid][:target]:
                        col[idx[bldg_id]] = uid

        df = pl.DataFrame({"bldg_id": sorted_bldg_ids, **year_cols})
        output_path = Path(output_path)
        df.write_csv(output_path)

        print(
            f"Exported scenario for {len(self.sampled_bldgs)} buildings across {self.num_years} years to {output_path}"
        )

    def _resolve_scenario_root(self, path: Path | None) -> Path:
        """Resolve scenario root directory based on optional base path."""
        base_path = self.data_path if path is None else Path(path)
        return base_path / self.pathway_scenario_name

    def save_metadata_parquet(self, path: Path | None = None) -> None:
        """Save mixed upgrade metadata to a partitioned parquet dataset.

        Output structure:
            <path>/<scenario_name>/year=<year_int>/metadata.parquet

        Args:
            path: Optional base path to write to. Defaults to the release path.
        """
        scenario_root = self._resolve_scenario_root(path)
        for year_idx in range(self.num_years):
            year_dir = scenario_root / f"year={year_idx}"
            year_dir.mkdir(parents=True, exist_ok=True)
            output_file = year_dir / "metadata.parquet"
            df = self.read_metadata(years=[year_idx]).collect()
            df.write_parquet(output_file)

    def save_hourly_load_parquet(self, path: Path | None = None) -> None:
        """Save mixed upgrade hourly load curves to partitioned parquet datasets.

        Output structure:
            <path>/<scenario_name>/year=<year_int>/<bldg_id>-<upgrade_id>.parquet

        Args:
            path: Optional base path to write to. Defaults to the release path.
        """
        scenario_root = self._resolve_scenario_root(path)
        for year_idx in range(self.num_years):
            year_dir = scenario_root / f"year={year_idx}"
            year_dir.mkdir(parents=True, exist_ok=True)
            df = self.read_load_curve_hourly(years=[year_idx]).collect()
            if df.is_empty():
                continue
            for key, group in df.partition_by(["bldg_id", "upgrade_id"], as_dict=True).items():
                bldg_id, upgrade_id = key
                output_file = year_dir / f"{int(bldg_id)}-{int(upgrade_id)}.parquet"
                group.write_parquet(output_file)
