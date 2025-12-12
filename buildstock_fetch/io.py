"""
Module for reading BuildStock data downloaded with bsf.

Provides a user-friendly interface to read metadata and load curve data
from locally downloaded BuildStock files.

Usage:
    from buildstock_fetch.io import BuildStockRead, BuildStockRelease, State

    bsr = BuildStockRead(
        data_path="./data",
        release=BuildStockRelease.RES_2024_TMY3_2,
        states=State.NY
    )
    metadata_df = bsr.read_metadata(upgrades=["0", "1", "2"])
"""

from __future__ import annotations

import json
import random
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from buildstock_fetch.constants import BUILDSTOCK_RELEASES_FILE

if TYPE_CHECKING:
    pass


class State(Enum):
    """US State codes enum for BuildStock data."""

    AL = "AL"
    AK = "AK"
    AZ = "AZ"
    AR = "AR"
    CA = "CA"
    CO = "CO"
    CT = "CT"
    DE = "DE"
    FL = "FL"
    GA = "GA"
    HI = "HI"
    ID = "ID"
    IL = "IL"
    IN = "IN"
    IA = "IA"
    KS = "KS"
    KY = "KY"
    LA = "LA"
    ME = "ME"
    MD = "MD"
    MA = "MA"
    MI = "MI"
    MN = "MN"
    MS = "MS"
    MO = "MO"
    MT = "MT"
    NE = "NE"
    NV = "NV"
    NH = "NH"
    NJ = "NJ"
    NM = "NM"
    NY = "NY"
    NC = "NC"
    ND = "ND"
    OH = "OH"
    OK = "OK"
    OR = "OR"
    PA = "PA"
    RI = "RI"
    SC = "SC"
    SD = "SD"
    TN = "TN"
    TX = "TX"
    UT = "UT"
    VT = "VT"
    VA = "VA"
    WA = "WA"
    WV = "WV"
    WI = "WI"
    WY = "WY"
    DC = "DC"


def _load_releases_data() -> dict[str, dict[str, object]]:
    """Load release data from JSON file."""
    with open(BUILDSTOCK_RELEASES_FILE) as f:
        data: dict[str, dict[str, object]] = json.load(f)
        return data


# Load releases data once at module import
_RELEASES_DATA = _load_releases_data()


class BuildStockRelease(Enum):
    """BuildStock release enum with metadata attributes.

    Each release contains metadata about:
    - release_year: The year of the release
    - res_com: Whether it's 'resstock' or 'comstock'
    - weather: The weather file type (e.g., 'amy2018', 'tmy3')
    - release_number: The version number
    - upgrade_ids: List of available upgrade IDs
    - available_data: List of available data types
    """

    def __init__(self, key: str) -> None:
        self._key = key
        data = _RELEASES_DATA[key]
        self._release_year = data["release_year"]
        self._res_com = data["res_com"]
        self._weather = data["weather"]
        self._release_number = data["release_number"]
        self._upgrade_ids = data["upgrade_ids"]
        self._available_data = data["available_data"]

    @property
    def key(self) -> str:
        """The release key (e.g., 'res_2024_tmy3_2')."""
        return str(self._key)

    @property
    def release_year(self) -> str:
        """The year of the release."""
        return str(self._release_year)

    @property
    def res_com(self) -> str:
        """Whether it's 'resstock' or 'comstock'."""
        return str(self._res_com)

    @property
    def weather(self) -> str:
        """The weather file type."""
        return str(self._weather)

    @property
    def release_number(self) -> str:
        """The version number."""
        return str(self._release_number)

    @property
    def upgrade_ids(self) -> list[str]:
        """List of available upgrade IDs."""
        ids = self._upgrade_ids
        if isinstance(ids, list):
            return [str(x) for x in ids]
        return []

    @property
    def available_data(self) -> list[str]:
        """List of available data types."""
        data = self._available_data
        if isinstance(data, list):
            return [str(x) for x in data]
        return []

    # Create enum members from JSON keys
    COM_2021_AMY2018_1 = "com_2021_amy2018_1"
    COM_2021_TMY3_1 = "com_2021_tmy3_1"
    RES_2021_AMY2018_1 = "res_2021_amy2018_1"
    RES_2021_TMY3_1 = "res_2021_tmy3_1"
    RES_2022_AMY2012_1_1 = "res_2022_amy2012_1.1"
    RES_2022_AMY2012_1 = "res_2022_amy2012_1"
    RES_2022_AMY2018_1_1 = "res_2022_amy2018_1.1"
    RES_2022_AMY2018_1 = "res_2022_amy2018_1"
    RES_2022_TMY3_1_1 = "res_2022_tmy3_1.1"
    RES_2022_TMY3_1 = "res_2022_tmy3_1"
    COM_2023_AMY2018_1 = "com_2023_amy2018_1"
    COM_2023_AMY2018_2 = "com_2023_amy2018_2"
    COM_2024_AMY2018_1 = "com_2024_amy2018_1"
    COM_2024_AMY2018_2 = "com_2024_amy2018_2"
    RES_2024_AMY2018_2 = "res_2024_amy2018_2"
    RES_2024_TMY3_1 = "res_2024_tmy3_1"
    RES_2024_TMY3_2 = "res_2024_tmy3_2"
    COM_2025_AMY2012_2 = "com_2025_amy2012_2"
    COM_2025_AMY2018_1 = "com_2025_amy2018_1"
    COM_2025_AMY2018_2 = "com_2025_amy2018_2"
    COM_2025_AMY2018_3 = "com_2025_amy2018_3"
    RES_2025_AMY2012_1 = "res_2025_amy2012_1"
    RES_2025_AMY2018_1 = "res_2025_amy2018_1"


class DataNotFoundError(Exception):
    """Raised when requested data is not found on disk."""

    pass


class MetadataNotFoundError(DataNotFoundError):
    """Raised when metadata is not found on disk."""

    pass


class LoadCurveNotFoundError(DataNotFoundError):
    """Raised when load curve data is not found on disk."""

    pass


class NoUpgradesOnDiskError(DataNotFoundError):
    """Raised when no upgrades are found on disk."""

    pass


class InvalidUpgradeError(ValueError):
    """Raised when an invalid upgrade is specified."""

    pass


class SampleSizeExceedsBuildingCountError(ValueError):
    """Raised when sample_n exceeds the number of available buildings."""

    pass


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
        >>> from buildstock_fetch.io import BuildStockRead, BuildStockRelease, State
        >>> bsr = BuildStockRead(
        ...     data_path="./data",
        ...     release=BuildStockRelease.RES_2024_TMY3_2,
        ...     states=State.NY
        ... )
        >>> metadata = bsr.read_metadata(upgrades=["0", "1"])
    """

    def __init__(
        self,
        data_path: str | Path,
        release: BuildStockRelease,
        states: State | list[State] | None = None,
        sample_n: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.path = Path(data_path)
        self.release = self._validate_release(release)
        self.states = self._validate_and_resolve_states(states)
        self.sample_n = sample_n
        self.seed = seed
        self.sampled_bldgs: list[int] | None = None

        # Set up sampling if sample_n is specified
        if sample_n is not None:
            self._setup_sampling()

    def _validate_release(self, release: BuildStockRelease) -> BuildStockRelease:
        """Validate that the release is a valid BuildStockRelease enum member."""
        if not isinstance(release, BuildStockRelease):
            raise TypeError(
                f"Expected BuildStockRelease enum member, got {type(release).__name__}. "
                f"Valid releases: {[r.name for r in BuildStockRelease]}"
            )
        return release

    def _validate_and_resolve_states(self, states: State | list[State] | None) -> list[State]:
        """Validate states and resolve them if not provided.

        If states is None, detects which states are present on disk.
        """
        if states is not None:
            return self._validate_states(states)

        # Auto-detect states from disk
        detected_states = self._detect_states_on_disk()
        if not detected_states:
            raise DataNotFoundError(
                f"No states found on disk for release {self.release.key} at {self.path}. "
                "Please download data using bsf first."
            )

        print(f"Auto-detected states on disk: {[s.value for s in detected_states]}")
        return detected_states

    def _validate_states(self, states: State | list[State]) -> list[State]:
        """Validate that states are valid State enum members."""
        state_list = [states] if isinstance(states, State) else list(states)

        for state in state_list:
            if not isinstance(state, State):
                raise TypeError(
                    f"Expected State enum member, got {type(state).__name__}. Valid states: {[s.name for s in State]}"
                )
        return state_list

    def _detect_states_on_disk(self) -> list[State]:
        """Detect which states have data on disk for the release."""
        release_path = self.path / self.release.key
        if not release_path.exists():
            return []

        # Look for state directories in metadata folder first
        metadata_path = release_path / "metadata"
        if metadata_path.exists():
            return self._find_state_dirs(metadata_path)

        # Fall back to load curve directories
        for data_type in ["load_curve_15min", "load_curve_hourly", "load_curve_daily", "load_curve_annual"]:
            data_path = release_path / data_type
            if data_path.exists():
                states = self._find_state_dirs(data_path)
                if states:
                    return states

        return []

    def _find_state_dirs(self, base_path: Path) -> list[State]:
        """Find state directories in a path and return matching State enums."""
        states: list[State] = []
        if not base_path.exists():
            return states

        for item in base_path.iterdir():
            if item.is_dir() and item.name.startswith("state="):
                state_code = item.name.replace("state=", "")
                try:
                    states.append(State(state_code))
                except ValueError:
                    continue

        return sorted(states, key=lambda s: s.value)

    def _setup_sampling(self) -> None:
        """Set up building sampling based on sample_n and seed."""
        if self.sample_n is None:
            return

        # Get all unique bldg_ids from metadata
        all_bldg_ids = self._get_all_bldg_ids_from_metadata()

        if not all_bldg_ids:
            raise MetadataNotFoundError(
                f"Cannot sample buildings: no metadata found for release {self.release.key}. "
                "Please download metadata using bsf first."
            )

        if self.sample_n > len(all_bldg_ids):
            print(
                f"Warning: sample_n ({self.sample_n}) exceeds available buildings ({len(all_bldg_ids)}). "
                "Returning all buildings without sampling."
            )
            self.sampled_bldgs = None
            return

        # Sample bldg_ids
        if self.seed is not None:
            random.seed(self.seed)

        self.sampled_bldgs = random.sample(all_bldg_ids, self.sample_n)
        print(f"Sampled {len(self.sampled_bldgs)} buildings from {len(all_bldg_ids)} total.")

    def _get_all_bldg_ids_from_metadata(self) -> list[int]:
        """Get all unique bldg_ids from the first available upgrade metadata."""
        for state in self.states:
            # Find the first available upgrade on disk
            upgrades_on_disk = self._get_upgrades_on_disk("metadata", state)
            if not upgrades_on_disk:
                continue

            # Use the first available upgrade
            upgrade = upgrades_on_disk[0]
            metadata_path = (
                self.path / self.release.key / "metadata" / f"state={state.value}" / f"upgrade={upgrade.zfill(2)}"
            )

            # Find parquet files
            parquet_files = list(metadata_path.glob("*.parquet"))
            if not parquet_files:
                continue

            # Read and get unique bldg_ids
            try:
                df = pl.scan_parquet(parquet_files).select("bldg_id").collect()
                return df["bldg_id"].unique().to_list()
            except Exception:
                continue

        return []

    def _get_upgrades_on_disk(self, data_type: str, state: State) -> list[str]:
        """Get list of upgrades available on disk for a given data type and state."""
        base_path = self.path / self.release.key / data_type / f"state={state.value}"
        if not base_path.exists():
            return []

        upgrades = []
        for item in base_path.iterdir():
            if item.is_dir() and item.name.startswith("upgrade="):
                upgrade_id = item.name.replace("upgrade=", "").lstrip("0") or "0"
                upgrades.append(upgrade_id)

        return sorted(upgrades, key=lambda x: int(x))

    def _validate_upgrades(self, upgrades: str | list[str] | None, data_type: str) -> tuple[list[str], list[str]]:
        """Validate upgrades against release and disk.

        Returns:
            Tuple of (validated_upgrades, upgrades_on_disk)
        """
        # Get all upgrades available on disk across all states
        all_upgrades_on_disk: set[str] = set()
        for state in self.states:
            upgrades_on_disk = self._get_upgrades_on_disk(data_type, state)
            all_upgrades_on_disk.update(upgrades_on_disk)

        if not all_upgrades_on_disk:
            raise NoUpgradesOnDiskError(
                f"No {data_type} found on disk for release {self.release.key} "
                f"in states {[s.value for s in self.states]}. "
                "Please download data using bsf first."
            )

        upgrades_on_disk_sorted = sorted(all_upgrades_on_disk, key=lambda x: int(x))

        if upgrades is None:
            # Use all upgrades found on disk
            print(f"Using all upgrades found on disk: {upgrades_on_disk_sorted}")
            return upgrades_on_disk_sorted, upgrades_on_disk_sorted

        # Convert single upgrade to list
        upgrade_list = [upgrades] if isinstance(upgrades, str) else list(upgrades)

        # Validate each upgrade
        release_upgrade_ids = self.release.upgrade_ids
        for upgrade in upgrade_list:
            # Check if upgrade belongs to the release
            if upgrade not in release_upgrade_ids:
                raise InvalidUpgradeError(
                    f"Upgrade '{upgrade}' is not valid for release {self.release.key}. "
                    f"Valid upgrades: {release_upgrade_ids}"
                )
            # Check if upgrade is on disk
            if upgrade not in all_upgrades_on_disk:
                raise DataNotFoundError(
                    f"Upgrade '{upgrade}' is not found on disk for release {self.release.key}. "
                    f"Available upgrades on disk: {upgrades_on_disk_sorted}. "
                    "Please download the data using bsf first."
                )

        return upgrade_list, upgrades_on_disk_sorted

    def _apply_sampling_filter(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Apply sampling filter if sampled_bldgs is set."""
        if self.sampled_bldgs is not None:
            return lf.filter(pl.col("bldg_id").is_in(self.sampled_bldgs))
        return lf

    def read_metadata(self, upgrades: str | list[str] | None = None) -> pl.LazyFrame:
        """Read metadata for the specified upgrades.

        Args:
            upgrades: Single upgrade ID, list of upgrade IDs, or None.
                If None, reads metadata for all upgrades found on disk.

        Returns:
            A Polars LazyFrame containing the metadata.
            If multiple upgrades are specified, returns data for each bldg_id
            in each upgrade (multiple records per bldg_id).

        Raises:
            MetadataNotFoundError: If metadata is not available for the release.
            InvalidUpgradeError: If an invalid upgrade is specified.
            DataNotFoundError: If the requested upgrade is not found on disk.
        """
        # Check if metadata is available for this release
        if "metadata" not in self.release.available_data:
            raise MetadataNotFoundError(
                f"Metadata is not available for release {self.release.key}. "
                "Please check the release documentation or download metadata using bsf."
            )

        validated_upgrades, _ = self._validate_upgrades(upgrades, "metadata")

        # Collect all parquet files
        parquet_files: list[Path] = []
        for state in self.states:
            for upgrade in validated_upgrades:
                metadata_dir = (
                    self.path / self.release.key / "metadata" / f"state={state.value}" / f"upgrade={upgrade.zfill(2)}"
                )
                if metadata_dir.exists():
                    parquet_files.extend(metadata_dir.glob("*.parquet"))

        if not parquet_files:
            raise MetadataNotFoundError(
                f"No metadata files found on disk for release {self.release.key} "
                f"with upgrades {validated_upgrades} in states {[s.value for s in self.states]}. "
                "Please download metadata using bsf first."
            )

        # Create lazy frame from all files
        lf = pl.scan_parquet(parquet_files)

        # Apply sampling filter if set
        lf = self._apply_sampling_filter(lf)

        return lf

    def _read_load_curve(self, data_type: str, upgrades: str | list[str] | None = None) -> pl.LazyFrame:
        """Internal method to read load curve data.

        Args:
            data_type: Type of load curve (e.g., 'load_curve_15min').
            upgrades: Single upgrade ID, list of upgrade IDs, or None.

        Returns:
            A Polars LazyFrame containing the load curve data.
        """
        # Check if data type is available for this release
        if data_type not in self.release.available_data:
            raise LoadCurveNotFoundError(
                f"{data_type} is not available for release {self.release.key}. "
                f"Available data types: {self.release.available_data}. "
                "Please download the data using bsf."
            )

        validated_upgrades, _ = self._validate_upgrades(upgrades, data_type)

        # Collect all parquet files
        parquet_files: list[Path] = []
        for state in self.states:
            for upgrade in validated_upgrades:
                load_curve_dir = (
                    self.path / self.release.key / data_type / f"state={state.value}" / f"upgrade={upgrade.zfill(2)}"
                )
                if load_curve_dir.exists():
                    parquet_files.extend(load_curve_dir.glob("*.parquet"))

        if not parquet_files:
            raise LoadCurveNotFoundError(
                f"No {data_type} files found on disk for release {self.release.key} "
                f"with upgrades {validated_upgrades} in states {[s.value for s in self.states]}. "
                "Please download the data using bsf first."
            )

        # Create lazy frame from all files
        lf = pl.scan_parquet(parquet_files)

        # Apply sampling filter if set
        lf = self._apply_sampling_filter(lf)

        return lf

    def read_load_curve_15min(self, upgrades: str | list[str] | None = None) -> pl.LazyFrame:
        """Read 15-minute load curve data for the specified upgrades.

        Args:
            upgrades: Single upgrade ID, list of upgrade IDs, or None.
                If None, reads data for all upgrades found on disk.

        Returns:
            A Polars LazyFrame containing the 15-minute load curve data.

        Raises:
            LoadCurveNotFoundError: If 15-minute load curve data is not available.
            InvalidUpgradeError: If an invalid upgrade is specified.
            DataNotFoundError: If the requested upgrade is not found on disk.
        """
        return self._read_load_curve("load_curve_15min", upgrades)

    def read_load_curve_hourly(self, upgrades: str | list[str] | None = None) -> pl.LazyFrame:
        """Read hourly load curve data for the specified upgrades.

        Args:
            upgrades: Single upgrade ID, list of upgrade IDs, or None.
                If None, reads data for all upgrades found on disk.

        Returns:
            A Polars LazyFrame containing the hourly load curve data.

        Raises:
            LoadCurveNotFoundError: If hourly load curve data is not available.
            InvalidUpgradeError: If an invalid upgrade is specified.
            DataNotFoundError: If the requested upgrade is not found on disk.
        """
        return self._read_load_curve("load_curve_hourly", upgrades)

    def read_load_curve_daily(self, upgrades: str | list[str] | None = None) -> pl.LazyFrame:
        """Read daily load curve data for the specified upgrades.

        Args:
            upgrades: Single upgrade ID, list of upgrade IDs, or None.
                If None, reads data for all upgrades found on disk.

        Returns:
            A Polars LazyFrame containing the daily load curve data.

        Raises:
            LoadCurveNotFoundError: If daily load curve data is not available.
            InvalidUpgradeError: If an invalid upgrade is specified.
            DataNotFoundError: If the requested upgrade is not found on disk.
        """
        return self._read_load_curve("load_curve_daily", upgrades)

    def read_load_curve_annual(self, upgrades: str | list[str] | None = None) -> pl.LazyFrame:
        """Read annual load curve data for the specified upgrades.

        Args:
            upgrades: Single upgrade ID, list of upgrade IDs, or None.
                If None, reads data for all upgrades found on disk.

        Returns:
            A Polars LazyFrame containing the annual load curve data.

        Raises:
            LoadCurveNotFoundError: If annual load curve data is not available.
            InvalidUpgradeError: If an invalid upgrade is specified.
            DataNotFoundError: If the requested upgrade is not found on disk.
        """
        return self._read_load_curve("load_curve_annual", upgrades)
