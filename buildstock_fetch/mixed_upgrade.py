"""
Mixed upgrade scenario orchestration for multi-year adoption trajectories.

This module provides the MixedUpgradeScenario class for defining and reading
heterogeneous upgrade mixes where buildings progressively adopt different
upgrades over time.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from buildstock_fetch.io import BuildStockRead, BuildStockRelease, DataNotFoundError, State
from buildstock_fetch.scenarios import validate_scenario

if TYPE_CHECKING:
    pass


class ScenarioDataNotFoundError(DataNotFoundError):
    """Raised when data for scenario upgrades is not found on disk."""

    pass


class MixedUpgradeScenario:
    """Class for orchestrating multi-year adoption trajectories across multiple upgrade scenarios.

    This class enables defining and reading heterogeneous upgrade mixes where buildings
    progressively adopt different upgrades over time. Buildings are sampled once from a
    baseline upgrade, then allocated to different upgrades according to adoption fractions
    per year. Monotonic adoption is enforced: buildings can only move from baseline to
    an upgrade, never backwards.

    Args:
        data_path: Path to the data directory (local path or S3 path).
        release: A BuildStockRelease enum member specifying the release.
        states: Optional State or list of States to filter data.
            If None, auto-detects states present on disk.
        sample_n: Optional number of buildings to sample from baseline.
        seed: Optional random seed for reproducible sampling and allocation.
        scenario: Dict mapping upgrade IDs to adoption fractions per year.
            Example: {4: [0.06, 0.18, 0.30], 8: [0.04, 0.12, 0.20]}
            represents 3 years where upgrade 4 grows from 6% to 30% adoption
            and upgrade 8 grows from 4% to 20% adoption.

    Example:
        >>> from buildstock_fetch.mixed_upgrade import MixedUpgradeScenario
        >>> from buildstock_fetch.io import BuildStockRelease, State
        >>> from buildstock_fetch.scenarios import uniform_adoption
        >>> scenario = uniform_adoption(
        ...     upgrade_ids=[4, 8],
        ...     weights={4: 0.6, 8: 0.4},
        ...     adoption_trajectory=[0.1, 0.3, 0.5],
        ... )
        >>> mus = MixedUpgradeScenario(
        ...     data_path="./data",
        ...     release=BuildStockRelease.RES_2024_TMY3_2,
        ...     states=State.NY,
        ...     sample_n=1000,
        ...     seed=42,
        ...     scenario=scenario,
        ... )
        >>> metadata = mus.read_metadata().collect()
        >>> mus.export_scenario_to_cairo("./scenario.csv")
    """

    def __init__(
        self,
        data_path: str | Path,
        release: BuildStockRelease,
        states: State | list[State] | None = None,
        sample_n: int | None = None,
        seed: int | None = None,
        scenario: dict[int, list[float]] | None = None,
    ) -> None:
        # Validate and store scenario
        if scenario is None:
            raise ValueError("scenario parameter is required for MixedUpgradeScenario")

        validate_scenario(scenario)
        self.scenario = scenario
        self.num_years = len(next(iter(scenario.values())))
        self.seed = seed

        # Create internal BuildStockRead for baseline (upgrade 0)
        # This handles all the state detection, validation, and sampling
        self.baseline_reader = BuildStockRead(
            data_path=data_path,
            release=release,
            states=states,
            sample_n=sample_n,
            seed=seed,
        )

        # Store key attributes from baseline reader
        self.path = self.baseline_reader.path
        self.release = self.baseline_reader.release
        self.states = self.baseline_reader.states
        self.sample_n = sample_n
        self.sampled_bldgs = self.baseline_reader.sampled_bldgs

        # Get the actual sampled building IDs
        if self.sampled_bldgs is None:
            # No sampling - get all buildings from baseline
            all_bldg_ids = self.baseline_reader._get_all_bldg_ids_from_metadata()
            self.sampled_bldgs = all_bldg_ids
            self.sample_n = len(all_bldg_ids)

        # Ensure sampled_bldgs is not None for type checking
        if not self.sampled_bldgs:
            raise ValueError("No buildings available for sampling. Please check that metadata exists.")

        # Materialize the scenario: assign buildings to upgrades per year
        self.materialized_scenario = self._materialize_scenario()

        # Log what we did
        num_upgrades = len(scenario)
        print(f"Sampled {len(self.sampled_bldgs)} buildings from baseline (upgrade 0) with seed {seed}")
        print(f"Materialized {self.num_years} years of adoption across {num_upgrades} upgrades")

    def _materialize_scenario(self) -> dict[int, dict[int, int]]:
        """Materialize building allocations for all years in the scenario.

        Returns:
            Dict mapping year index to dict of {bldg_id: upgrade_id}.
            Example: {0: {100: 0, 101: 0}, 1: {100: 4, 101: 0}}
        """
        # Shuffle building IDs deterministically
        shuffled_bldgs = list(self.sampled_bldgs)
        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(shuffled_bldgs)

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
                raise ValueError(
                    f"Year index {year_idx} is out of range. " f"Valid range: [0, {self.num_years - 1}]"
                )

        return years

    def _validate_data_availability(self, data_type: str) -> None:
        """Validate that all required upgrade data exists on disk.

        Args:
            data_type: Type of data to check (e.g., 'metadata', 'load_curve_15min').

        Raises:
            ScenarioDataNotFoundError: If data for any scenario upgrade is missing.
        """
        # Get all upgrade IDs in the scenario (plus baseline 0)
        required_upgrades = set(self.scenario.keys())
        required_upgrades.add(0)  # Always need baseline

        # Check which upgrades are available on disk
        available_upgrades: set[str] = set()
        for state in self.states:
            upgrades_on_disk = self.baseline_reader._get_upgrades_on_disk(data_type, state)
            available_upgrades.update(upgrades_on_disk)

        # Convert required upgrade IDs to strings for comparison
        required_upgrades_str = {str(uid) for uid in required_upgrades}
        missing_upgrades = required_upgrades_str - available_upgrades

        if missing_upgrades:
            raise ScenarioDataNotFoundError(
                f"Missing {data_type} data for upgrades: {sorted(missing_upgrades, key=int)}. "
                f"Available upgrades on disk: {sorted(available_upgrades, key=int)}. "
                f"Please download the required upgrades using bsf."
            )

    def _read_data_for_scenario(self, data_type: str, years: list[int] | None = None) -> pl.LazyFrame:
        """Internal method to read data across all upgrades and years in the scenario.

        Args:
            data_type: Type of data to read (e.g., 'metadata', 'load_curve_15min').
            years: List of year indices to include, or None for all years.

        Returns:
            LazyFrame with columns: bldg_id, upgrade_id, year, ...(original columns)
        """
        # Validate years and data availability
        validated_years = self._validate_years(years)
        self._validate_data_availability(data_type)

        # Determine which upgrades we need to read
        required_upgrades = set(self.scenario.keys())
        required_upgrades.add(0)  # Always need baseline for year 0

        # Create a mapping table: {bldg_id: [(year, upgrade_id), ...]}
        bldg_year_upgrade_mapping: list[tuple[int, int, int]] = []
        for year_idx in validated_years:
            year_allocation = self.materialized_scenario[year_idx]
            for bldg_id, upgrade_id in year_allocation.items():
                bldg_year_upgrade_mapping.append((bldg_id, year_idx, upgrade_id))

        # Read data for each unique upgrade
        upgrade_dfs: list[pl.LazyFrame] = []

        for upgrade_id in required_upgrades:
            # Create a BuildStockRead for this specific upgrade
            upgrade_reader = BuildStockRead(
                data_path=self.path,
                release=self.release,
                states=self.states,
                sample_n=None,  # Don't sample - we'll filter by building IDs
                seed=None,
            )

            # Read the data
            if data_type == "metadata":
                lf = upgrade_reader.read_metadata(upgrades=str(upgrade_id))
            else:
                lf = upgrade_reader._read_load_curve(data_type, upgrades=str(upgrade_id))

            # Filter to only buildings in our sample
            lf = lf.filter(pl.col("bldg_id").is_in(self.sampled_bldgs))

            # Add upgrade_id column if not present (it should be from the data)
            # The data already has 'upgrade' column, we'll rename it to upgrade_id
            if "upgrade" in lf.columns:
                lf = lf.rename({"upgrade": "upgrade_id"})
            else:
                lf = lf.with_columns(pl.lit(upgrade_id).alias("upgrade_id"))

            upgrade_dfs.append(lf)

        # Concatenate all upgrade data
        all_data = pl.concat(upgrade_dfs, how="vertical_relaxed")

        # Create mapping DataFrame for year assignments
        mapping_df = pl.DataFrame(
            {
                "bldg_id": [x[0] for x in bldg_year_upgrade_mapping],
                "year": [x[1] for x in bldg_year_upgrade_mapping],
                "upgrade_id": [x[2] for x in bldg_year_upgrade_mapping],
            }
        )

        # Join data with year assignments
        # For each (bldg_id, upgrade_id) in the data, find all years it appears in the scenario
        result = all_data.join(
            mapping_df.lazy(),
            on=["bldg_id", "upgrade_id"],
            how="inner",
        )

        return result

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
        # Validate that all scenario data exists
        self._validate_data_availability("metadata")

        # Create DataFrame from materialized scenario
        rows: list[dict[str, int]] = []

        for bldg_id in sorted(self.sampled_bldgs):
            row = {"bldg_id": bldg_id}
            for year_idx in range(self.num_years):
                upgrade_id = self.materialized_scenario[year_idx][bldg_id]
                row[f"year_{year_idx}"] = upgrade_id
            rows.append(row)

        # Create DataFrame and write to CSV
        df = pl.DataFrame(rows)
        output_path = Path(output_path)
        df.write_csv(output_path)

        print(f"Exported scenario for {len(self.sampled_bldgs)} buildings " f"across {self.num_years} years to {output_path}")
