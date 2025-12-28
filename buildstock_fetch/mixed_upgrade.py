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
