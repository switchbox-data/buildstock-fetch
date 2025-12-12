"""Tests for the io module.

These tests use real S3 downloads (like test_main.py) to ensure BuildStockRead
is tested against the actual directory structure that bsf creates. If the S3
structure changes, these tests will fail loudly.
"""

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from buildstock_fetch.building import BuildingID
from buildstock_fetch.io import (
    BuildStockRead,
    BuildStockRelease,
    DataNotFoundError,
    InvalidUpgradeError,
    LoadCurveNotFoundError,
    NoUpgradesOnDiskError,
    State,
)
from buildstock_fetch.main import fetch_bldg_data

# =============================================================================
# Tests for State Enum
# =============================================================================


class TestStateEnum:
    """Tests for the State enum."""

    def test_state_enum_values(self):
        """Test that State enum contains expected values."""
        assert State.NY.value == "NY"
        assert State.CA.value == "CA"
        assert State.TX.value == "TX"

    def test_state_enum_all_states(self):
        """Test that State enum contains all 50 states + DC."""
        # 50 states + DC = 51
        assert len(State) == 51

    def test_state_enum_membership(self):
        """Test that we can check membership correctly."""
        assert State("NY") == State.NY
        with pytest.raises(ValueError):
            State("INVALID")


# =============================================================================
# Tests for BuildStockRelease Enum
# =============================================================================


class TestBuildStockReleaseEnum:
    """Tests for the BuildStockRelease enum."""

    def test_release_enum_values(self):
        """Test that BuildStockRelease enum contains expected values."""
        assert BuildStockRelease.RES_2024_TMY3_2.key == "res_2024_tmy3_2"
        assert BuildStockRelease.COM_2024_AMY2018_2.key == "com_2024_amy2018_2"

    def test_release_enum_attributes(self):
        """Test that BuildStockRelease enum has correct attributes."""
        release = BuildStockRelease.RES_2024_TMY3_2
        assert release.release_year == "2024"
        assert release.res_com == "resstock"
        assert release.weather == "tmy3"
        assert release.release_number == "2"
        assert isinstance(release.upgrade_ids, list)
        assert isinstance(release.available_data, list)

    def test_release_enum_upgrade_ids(self):
        """Test that upgrade_ids are correctly loaded."""
        release = BuildStockRelease.RES_2024_TMY3_2
        assert "0" in release.upgrade_ids
        assert "1" in release.upgrade_ids

    def test_release_enum_available_data(self):
        """Test that available_data is correctly loaded."""
        release = BuildStockRelease.RES_2024_TMY3_2
        assert "metadata" in release.available_data
        assert "load_curve_15min" in release.available_data


# =============================================================================
# Tests for BuildStockRead Validation
# =============================================================================


class TestBuildStockReadValidation:
    """Tests for BuildStockRead validation methods."""

    def test_validate_release_invalid_type(self):
        """Test that invalid release type raises TypeError."""
        with pytest.raises(TypeError, match="Expected BuildStockRelease enum member"):
            BuildStockRead(
                data_path="./data",
                release="res_2024_tmy3_2",  # type: ignore
                states=State.NY,
            )

    def test_validate_release_invalid_string(self):
        """Test that string release raises TypeError."""
        with pytest.raises(TypeError, match="Expected BuildStockRelease enum member"):
            BuildStockRead(
                data_path="./data",
                release="invalid_release",  # type: ignore
                states=State.NY,
            )

    def test_validate_states_invalid_type(self):
        """Test that invalid state type raises TypeError."""
        with pytest.raises(TypeError, match="Expected State enum member"):
            BuildStockRead(
                data_path="./data",
                release=BuildStockRelease.RES_2024_TMY3_2,
                states="NY",  # type: ignore
            )

    def test_validate_states_invalid_list_type(self):
        """Test that list with invalid state type raises TypeError."""
        with pytest.raises(TypeError, match="Expected State enum member"):
            BuildStockRead(
                data_path="./data",
                release=BuildStockRelease.RES_2024_TMY3_2,
                states=[State.NY, "CA"],  # type: ignore
            )

    def test_validate_states_single_state(self, cleanup_downloads):
        """Test that single State is accepted and converted to list."""
        # Download minimal data
        bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        bsr = BuildStockRead(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
        )
        assert bsr.states == [State.NY]

    def test_validate_states_list_of_states(self, cleanup_downloads):
        """Test that list of States is accepted."""
        # Download data for NY and OH
        bldg_ids = [
            BuildingID(bldg_id=7, upgrade_id="0", state="NY"),
            BuildingID(bldg_id=100000, upgrade_id="0", state="OH", release_number="2"),
        ]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        bsr = BuildStockRead(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=[State.NY, State.OH],
        )
        assert State.NY in bsr.states
        assert State.OH in bsr.states


# =============================================================================
# Tests for Auto-detection of States
# =============================================================================


class TestBuildStockReadAutoDetectStates:
    """Tests for auto-detection of states on disk."""

    def test_auto_detect_states_from_metadata(self, cleanup_downloads):
        """Test auto-detection of states from metadata directory."""
        # Download data for NY and OH
        bldg_ids = [
            BuildingID(bldg_id=7, upgrade_id="0", state="NY"),
            BuildingID(bldg_id=100000, upgrade_id="0", state="OH", release_number="2"),
        ]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        bsr = BuildStockRead(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=None,
        )

        assert State.NY in bsr.states
        assert State.OH in bsr.states

    def test_auto_detect_states_no_data(self, tmp_path):
        """Test that DataNotFoundError is raised when no data on disk."""
        with pytest.raises(DataNotFoundError, match="No states found on disk"):
            BuildStockRead(
                data_path=tmp_path / "nonexistent",
                release=BuildStockRelease.RES_2024_TMY3_2,
                states=None,
            )


# =============================================================================
# Tests for Upgrade Validation
# =============================================================================


class TestBuildStockReadUpgradeValidation:
    """Tests for upgrade validation in read methods."""

    def test_validate_upgrades_invalid_upgrade(self, cleanup_downloads):
        """Test that invalid upgrade raises InvalidUpgradeError."""
        bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        bsr = BuildStockRead(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
        )

        with pytest.raises(InvalidUpgradeError, match="is not valid for release"):
            bsr.read_metadata(upgrades="999")

    def test_validate_upgrades_not_on_disk(self, cleanup_downloads):
        """Test that upgrade not on disk raises DataNotFoundError."""
        bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        bsr = BuildStockRead(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
        )

        # Upgrade 5 is valid for the release but not downloaded
        with pytest.raises(DataNotFoundError, match="is not found on disk"):
            bsr.read_metadata(upgrades="5")

    def test_validate_upgrades_no_upgrades_on_disk(self, cleanup_downloads):
        """Test that NoUpgradesOnDiskError is raised when no upgrades found."""
        # Create empty directory structure without metadata files
        empty_dir = Path("data/res_2024_tmy3_2/metadata/state=NY")
        empty_dir.mkdir(parents=True, exist_ok=True)

        bsr = BuildStockRead(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
        )

        with pytest.raises(NoUpgradesOnDiskError, match="No metadata found on disk"):
            bsr.read_metadata()


# =============================================================================
# Tests for read_metadata()
# =============================================================================


class TestBuildStockReadMetadata:
    """Tests for read_metadata method."""

    def test_read_metadata_single_upgrade(self, cleanup_downloads):
        """Test reading metadata for a single upgrade."""
        bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        bsr = BuildStockRead(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
        )

        metadata = bsr.read_metadata(upgrades="0")
        assert isinstance(metadata, pl.LazyFrame)

        df = metadata.collect()
        assert "bldg_id" in df.columns
        assert df.height > 0  # Real metadata has many buildings

    def test_read_metadata_multiple_upgrades(self, cleanup_downloads):
        """Test reading metadata for multiple upgrades."""
        bldg_ids = [
            BuildingID(bldg_id=7, upgrade_id="0"),
            BuildingID(bldg_id=7, upgrade_id="1"),
        ]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        bsr = BuildStockRead(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
        )

        metadata = bsr.read_metadata(upgrades=["0", "1"])
        df = metadata.collect()

        # Should have records from both upgrades
        assert 0 in df["upgrade"].to_list()
        assert 1 in df["upgrade"].to_list()

    def test_read_metadata_auto_detect_upgrades(self, cleanup_downloads):
        """Test reading metadata with auto-detected upgrades."""
        bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        bsr = BuildStockRead(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
        )

        # Call without specifying upgrades - should auto-detect
        metadata = bsr.read_metadata()
        df = metadata.collect()
        assert df.height > 0


# =============================================================================
# Tests for Load Curve Reading
# =============================================================================


class TestBuildStockReadLoadCurves:
    """Tests for load curve reading methods."""

    def test_read_load_curve_15min(self, cleanup_downloads):
        """Test reading 15-minute load curve data."""
        bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
        fetch_bldg_data(bldg_ids, ("load_curve_15min",), Path("data"))

        bsr = BuildStockRead(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
        )

        load_curve = bsr.read_load_curve_15min(upgrades="0")
        assert isinstance(load_curve, pl.LazyFrame)

        df = load_curve.collect()
        assert df.height > 0
        assert "timestamp" in df.columns
        assert "bldg_id" in df.columns

    def test_read_load_curve_hourly(self, cleanup_downloads):
        """Test reading hourly load curve data."""
        bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
        fetch_bldg_data(bldg_ids, ("load_curve_hourly",), Path("data"))

        bsr = BuildStockRead(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
        )

        load_curve = bsr.read_load_curve_hourly(upgrades="0")
        assert isinstance(load_curve, pl.LazyFrame)

        df = load_curve.collect()
        assert df.height > 0
        assert "timestamp" in df.columns

    def test_read_load_curve_daily(self, cleanup_downloads):
        """Test reading daily load curve data."""
        bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
        fetch_bldg_data(bldg_ids, ("load_curve_daily",), Path("data"))

        bsr = BuildStockRead(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
        )

        load_curve = bsr.read_load_curve_daily(upgrades="0")
        assert isinstance(load_curve, pl.LazyFrame)

        df = load_curve.collect()
        assert df.height > 0
        assert "timestamp" in df.columns

    def test_read_load_curve_annual(self, cleanup_downloads):
        """Test reading annual load curve data."""
        bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
        fetch_bldg_data(bldg_ids, ("load_curve_annual",), Path("data"))

        bsr = BuildStockRead(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
        )

        load_curve = bsr.read_load_curve_annual(upgrades="0")
        assert isinstance(load_curve, pl.LazyFrame)

        df = load_curve.collect()
        assert df.height > 0
        assert "bldg_id" in df.columns

    def test_read_load_curve_not_available_for_release(self, cleanup_downloads):
        """Test that LoadCurveNotFoundError is raised for unavailable data type in release."""
        bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        bsr = BuildStockRead(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
        )

        # RES_2024_TMY3_2 should have load_curve_15min available, but not on disk
        with pytest.raises((LoadCurveNotFoundError, NoUpgradesOnDiskError)):
            bsr.read_load_curve_15min(upgrades="0")


# =============================================================================
# Tests for Sampling
# =============================================================================


class TestBuildStockReadSampling:
    """Tests for sampling functionality."""

    def test_sampling_with_seed(self, cleanup_downloads):
        """Test that sampling with seed is reproducible."""
        bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        # Create two readers with same seed
        bsr1 = BuildStockRead(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=100,
            seed=42,
        )

        bsr2 = BuildStockRead(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=100,
            seed=42,
        )

        # Should have same sampled buildings
        assert bsr1.sampled_bldgs == bsr2.sampled_bldgs

    def test_sampling_filters_metadata(self, cleanup_downloads):
        """Test that sampling correctly filters metadata."""
        bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        bsr = BuildStockRead(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=50,
            seed=42,
        )

        metadata = bsr.read_metadata(upgrades="0")
        df = metadata.collect()

        # Should only have sampled buildings
        assert df["bldg_id"].n_unique() == 50

    def test_sample_n_exceeds_buildings(self, cleanup_downloads):
        """Test warning when sample_n exceeds available buildings."""
        bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        # Request more samples than available buildings
        bsr = BuildStockRead(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=1000000,  # Way more than available
        )

        # Should not sample (sampled_bldgs should be None)
        assert bsr.sampled_bldgs is None


# =============================================================================
# Tests for Multiple States
# =============================================================================


class TestBuildStockReadMultipleStates:
    """Tests for reading data from multiple states."""

    def test_read_metadata_multiple_states(self, cleanup_downloads):
        """Test reading metadata from multiple states."""
        bldg_ids = [
            BuildingID(bldg_id=7, upgrade_id="0", state="NY"),
            BuildingID(bldg_id=100000, upgrade_id="0", state="OH", release_number="2"),
        ]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        bsr = BuildStockRead(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=[State.NY, State.OH],
        )

        metadata = bsr.read_metadata(upgrades="0")
        df = metadata.collect()

        # Should have buildings from both states
        states_in_result = df["in.state"].unique().to_list()
        assert "NY" in states_in_result
        assert "OH" in states_in_result


# =============================================================================
# Integration Tests
# =============================================================================


class TestBuildStockReadIntegration:
    """Integration tests for BuildStockRead."""

    def test_returns_lazyframe(self, cleanup_downloads):
        """Test that all read methods return LazyFrame."""
        bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        bsr = BuildStockRead(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
        )

        result = bsr.read_metadata(upgrades="0")
        assert isinstance(result, pl.LazyFrame)

    def test_path_as_string(self, cleanup_downloads):
        """Test that string paths work correctly."""
        bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        bsr = BuildStockRead(
            data_path="data",  # Pass as string
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
        )

        result = bsr.read_metadata(upgrades="0")
        assert isinstance(result, pl.LazyFrame)

    def test_read_multiple_data_types(self, cleanup_downloads):
        """Test reading multiple data types for same building."""
        bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
        fetch_bldg_data(
            bldg_ids,
            ("metadata", "load_curve_15min", "load_curve_hourly"),
            Path("data"),
        )

        bsr = BuildStockRead(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
        )

        # All should work
        metadata = bsr.read_metadata(upgrades="0")
        lc_15min = bsr.read_load_curve_15min(upgrades="0")
        lc_hourly = bsr.read_load_curve_hourly(upgrades="0")

        assert metadata.collect().height > 0
        assert lc_15min.collect().height > 0
        assert lc_hourly.collect().height > 0
