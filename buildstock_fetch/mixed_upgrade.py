"""Mixed upgrade scenario orchestration for multi-year adoption trajectories.

This module provides the MixedUpgradeScenario class for defining and reading
heterogeneous upgrade mixes where buildings progressively adopt different
upgrades over time.
"""

from __future__ import annotations

from collections.abc import Collection
from functools import cached_property
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    pass


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
class MixedUpgradeScenario:
    """Class for orchestrating multi-year adoption trajectories across multiple upgrade scenarios.

    This class enables defining and reading heterogeneous upgrade mixes where buildings
    progressively adopt different upgrades over time. Buildings are sampled once from a
    baseline upgrade, then allocated to different upgrades according to adoption fractions
    per year. Monotonic adoption is enforced: buildings can only move from baseline to
    an upgrade, never backwards.

    Args:
        data_path: Path to the data directory (local path or S3 path).
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
        release: ReleaseKey | BuildstockRelease,
        states: USStateCode | Collection[USStateCode] | None = None,
        sample_n: int | None = None,
        random: Random | int | None = None,
        scenario: dict[int, list[float]] | None = None,
    ) -> None:
        # Validate and store scenario
        if scenario is None:
            raise ValueError("scenario parameter is required for MixedUpgradeScenario")

        validate_scenario(scenario)
        self.scenario = scenario
        self.num_years = len(next(iter(scenario.values())))

        # Store parameters
        self.data_path = Path(data_path)
        self.release = release if isinstance(release, BuildstockRelease) else BuildstockReleases.load()[release]

        # Normalize states
        if states is None:
            self.states: list[USStateCode] | None = None
        elif is_valid_state_code(states):
            self.states = [states]
        else:
            self.states = list(states)

        # Set up random generator
        if random is None:
            self.random = Random()
        elif isinstance(random, Random):
            self.random = random
        else:
            self.random = Random(random)

        self.sample_n = sample_n

        # Create internal BuildStockRead for baseline (upgrade 0)
        # This handles all the state detection, validation, and sampling
        self.baseline_reader = BuildStockRead(
            data_path=data_path,
            release=self.release,
            states=self.states,
            sample_n=sample_n,
            random=self.random,
        )

        # Get the actual sampled building IDs from baseline reader
        if self.baseline_reader.sampled_buildings is not None:
            self.sampled_bldgs: frozenset[int] = self.baseline_reader.sampled_buildings
        else:
            # No sampling - get all buildings from metadata
            metadata_files = self.downloaded_data.filter(file_type="metadata", suffix=".parquet", upgrade="0")
            if not metadata_files:
                raise DataNotFoundError(
                    "No baseline (upgrade 0) metadata found. Please download baseline data using bsf."
                )
            import polars as pl
            first_file = min(metadata_files, key=lambda x: x.file_path)
            df = pl.scan_parquet(first_file.file_path).select("bldg_id").collect()
            all_bldg_ids = df["bldg_id"].unique().to_list()
            self.sampled_bldgs = frozenset(all_bldg_ids)

        if not self.sampled_bldgs:
            raise ValueError("No buildings available for sampling. Please check that metadata exists.")

        # Log what we did
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
    def materialized_scenario(self) -> dict[int, dict[int, int]]:
        """Materialize building allocations for all years in the scenario.

        This is cached to avoid recomputing the same allocation multiple times.

        Returns:
            Dict mapping year index to dict of {bldg_id: upgrade_id}.
            Example: {0: {100: 0, 101: 0}, 1: {100: 4, 101: 0}}
        """
        # Shuffle building IDs deterministically using the Random instance
        shuffled_bldgs = list(self.sampled_bldgs)
        self.random.shuffle(shuffled_bldgs)

        # Track which buildings have been allocated to upgrades
        allocated_bldgs: dict[int, int] = {}  # {bldg_id: upgrade_id}

        # Materialize each year
        materialized: dict[int, dict[int, int]] = {}

        for year_idx in range(self.num_years):
            year_allocation: dict[int, int] = {}

            # Start with all allocated buildings from previous years (monotonicity)
            year_allocation.update(allocated_bldgs)

            # For each upgrade, allocate new adopters
            for upgrade_id, fractions in self.scenario.items():
                target_fraction = fractions[year_idx]
                target_count = int(target_fraction * len(shuffled_bldgs))

                # Count how many buildings are already in this upgrade
                current_count = sum(1 for uid in allocated_bldgs.values() if uid == upgrade_id)

                # Allocate additional buildings if needed
                new_adopters_needed = target_count - current_count

                if new_adopters_needed > 0:
                    # Find unallocated buildings
                    unallocated = [bid for bid in shuffled_bldgs if bid not in allocated_bldgs]

                    # Allocate next N unallocated buildings
                    for bldg_id in unallocated[:new_adopters_needed]:
                        allocated_bldgs[bldg_id] = upgrade_id
                        year_allocation[bldg_id] = upgrade_id

            # Buildings not allocated to any upgrade remain in baseline (upgrade 0)
            for bldg_id in shuffled_bldgs:
                if bldg_id not in year_allocation:
                    year_allocation[bldg_id] = 0

            materialized[year_idx] = year_allocation

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
                raise ValueError(f"Year index {year_idx} is out of range. Valid range: [0, {self.num_years - 1}]")

        return years

    def _validate_data_availability(self, file_type: FileType) -> None:
        """Validate that all required upgrade data exists on disk.

        Args:
            file_type: Type of data to check (e.g., 'metadata', 'load_curve_15min').

        Raises:
            ScenarioDataNotFoundError: If data for any scenario upgrade is missing.
        """
        # Get all upgrade IDs in the scenario (plus baseline 0)
        required_upgrades = {normalize_upgrade_id(str(uid)) for uid in self.scenario.keys()}
        required_upgrades.add("0")  # Always need baseline

        # Check which upgrades are available on disk using DownloadedData
        available_upgrades = self.downloaded_data.filter(file_type=file_type).upgrades()

        missing_upgrades = required_upgrades - available_upgrades

        if missing_upgrades:
            raise ScenarioDataNotFoundError(file_type, missing_upgrades)

    def _read_data_for_scenario(self, file_type: FileType, years: list[int] | None = None):
        """Internal method to read data across all upgrades and years in the scenario.

        Implements incremental reading optimization (max 2N reads):
        - Year 0: Read all buildings from upgrade 0
        - Year 1+: Only read NEW buildings that transitioned to non-zero upgrades
        - Reuse baseline (upgrade 0) data for buildings that haven't transitioned

        Args:
            file_type: Type of data to read (e.g., 'metadata', 'load_curve_15min').
            years: List of year indices to include, or None for all years.

        Returns:
            LazyFrame with columns: bldg_id, upgrade_id, year, ...(original columns)
        """
        import polars as pl

        # Validate years and data availability
        validated_years = self._validate_years(years)
        self._validate_data_availability(file_type)

        # Track which buildings have been read from each upgrade
        read_cache: dict[int, pl.LazyFrame] = {}  # {upgrade_id: data}
        buildings_read_per_upgrade: dict[int, set[int]] = {}  # {upgrade_id: {bldg_ids}}

        # Collect results for each year
        all_year_dfs: list[pl.LazyFrame] = []

        for year_idx in validated_years:
            year_allocation = self.materialized_scenario[year_idx]

            # Group buildings by upgrade for this year
            upgrade_to_buildings: dict[int, list[int]] = {}
            for bldg_id, upgrade_id in year_allocation.items():
                if upgrade_id not in upgrade_to_buildings:
                    upgrade_to_buildings[upgrade_id] = []
                upgrade_to_buildings[upgrade_id].append(bldg_id)

            # For each upgrade, only read NEW buildings
            year_dfs: list[pl.LazyFrame] = []
            for upgrade_id, building_ids in upgrade_to_buildings.items():
                # Initialize tracking for this upgrade if needed
                if upgrade_id not in buildings_read_per_upgrade:
                    buildings_read_per_upgrade[upgrade_id] = set()

                # Determine which buildings are NEW (not yet read for this upgrade)
                new_buildings = [b for b in building_ids if b not in buildings_read_per_upgrade[upgrade_id]]

                # Read new buildings if any
                if new_buildings:
                    new_data = self.baseline_reader.read_parquets(file_type, upgrades=str(upgrade_id))
                    new_data = new_data.filter(pl.col("bldg_id").is_in(new_buildings))

                    # Add upgrade_id column
                    if file_type == "metadata":
                        new_data = new_data.rename({"upgrade": "upgrade_id"})
                    else:
                        new_data = new_data.with_columns(pl.lit(upgrade_id).alias("upgrade_id"))

                    # Add to cache
                    if upgrade_id in read_cache:
                        read_cache[upgrade_id] = pl.concat([read_cache[upgrade_id], new_data], how="vertical_relaxed")
                    else:
                        read_cache[upgrade_id] = new_data

                    # Mark as read
                    buildings_read_per_upgrade[upgrade_id].update(new_buildings)

                # Get data for all buildings in this upgrade for this year (from cache)
                year_upgrade_data = read_cache[upgrade_id].filter(pl.col("bldg_id").is_in(building_ids))
                year_upgrade_data = year_upgrade_data.with_columns(pl.lit(year_idx).alias("year"))
                year_dfs.append(year_upgrade_data)

            # Concatenate all upgrades for this year
            if year_dfs:
                year_df = pl.concat(year_dfs, how="vertical_relaxed")
                all_year_dfs.append(year_df)

        # Concatenate all years
        if not all_year_dfs:
            return pl.LazyFrame({"bldg_id": [], "upgrade_id": [], "year": []})

        result = pl.concat(all_year_dfs, how="vertical_relaxed")
        return result

    def read_metadata(self, years: list[int] | None = None):
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

    def read_load_curve_15min(self, years: list[int] | None = None):
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

    def read_load_curve_hourly(self, years: list[int] | None = None):
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

    def read_load_curve_daily(self, years: list[int] | None = None):
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

    def read_load_curve_annual(self, years: list[int] | None = None):
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

        This method uses vectorized operations for efficient export.

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
        import polars as pl

        # Validate that all scenario data exists
        self._validate_data_availability("metadata")

        # Build DataFrame using vectorized column construction (optimization)
        # This is more efficient than building row-by-row
        sorted_bldg_ids = sorted(self.sampled_bldgs)

        # Create the DataFrame with bldg_id column and all year columns at once
        df = pl.DataFrame(
            {
                "bldg_id": sorted_bldg_ids,
                **{
                    f"year_{year_idx}": [self.materialized_scenario[year_idx][bid] for bid in sorted_bldg_ids]
                    for year_idx in range(self.num_years)
                },
            }
        )

        # Write to CSV
        output_path = Path(output_path)
        df.write_csv(output_path)

        print(f"Exported scenario for {len(self.sampled_bldgs)} buildings across {self.num_years} years to {output_path}")
