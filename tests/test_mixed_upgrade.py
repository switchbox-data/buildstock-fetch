"""Tests for mixed upgrade scenario functionality.

These tests cover scenario validation, building materialization, data reading,
and export functionality for multi-year adoption trajectories.
"""

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from buildstock_fetch.building import BuildingID
from buildstock_fetch.io import BuildStockRelease, State
from buildstock_fetch.main import fetch_bldg_data
from buildstock_fetch.mixed_upgrade import MixedUpgradeScenario, ScenarioDataNotFoundError
from buildstock_fetch.scenarios import InvalidScenarioError, uniform_adoption, validate_scenario

# =============================================================================
# Tests for Scenario Validation
# =============================================================================


class TestScenarioValidation:
    """Tests for scenario validation logic."""

    def test_uniform_adoption_basic(self):
        """Test that uniform_adoption generates correct output for simple case."""
        scenario = uniform_adoption(
            upgrade_ids=[4, 8],
            weights={4: 0.6, 8: 0.4},
            adoption_trajectory=[0.1, 0.3, 0.5],
        )

        assert scenario == {4: [0.06, 0.18, 0.30], 8: [0.04, 0.12, 0.20]}

    def test_uniform_adoption_weights_dont_sum_to_one(self):
        """Test that weights not summing to 1.0 raises error."""
        with pytest.raises(InvalidScenarioError, match="must sum to 1.0"):
            uniform_adoption(
                upgrade_ids=[4, 8],
                weights={4: 0.7, 8: 0.4},  # Sums to 1.1
                adoption_trajectory=[0.1, 0.3, 0.5],
            )

    def test_uniform_adoption_negative_weights(self):
        """Test that negative weights raise error."""
        with pytest.raises(InvalidScenarioError, match="must be in \\[0, 1\\]"):
            uniform_adoption(
                upgrade_ids=[4, 8],
                weights={4: -0.2, 8: 1.2},
                adoption_trajectory=[0.1, 0.3, 0.5],
            )

    def test_uniform_adoption_non_monotonic_trajectory(self):
        """Test that non-monotonic trajectory raises error."""
        with pytest.raises(InvalidScenarioError, match="must be non-decreasing"):
            uniform_adoption(
                upgrade_ids=[4, 8],
                weights={4: 0.6, 8: 0.4},
                adoption_trajectory=[0.3, 0.2, 0.5],  # Goes down then up
            )

    def test_uniform_adoption_mismatched_upgrades_weights(self):
        """Test that mismatched upgrade_ids and weights raises error."""
        with pytest.raises(InvalidScenarioError, match="must match upgrade_ids"):
            uniform_adoption(
                upgrade_ids=[4, 8],
                weights={4: 0.6, 9: 0.4},  # 9 not in upgrade_ids
                adoption_trajectory=[0.1, 0.3, 0.5],
            )

    def test_validate_scenario_valid(self):
        """Test that a valid scenario passes validation."""
        scenario = {4: [0.1, 0.2, 0.3], 8: [0.05, 0.10, 0.15]}
        # Should not raise
        validate_scenario(scenario)

    def test_validate_scenario_empty(self):
        """Test that empty scenario raises error."""
        with pytest.raises(InvalidScenarioError, match="cannot be empty"):
            validate_scenario({})

    def test_validate_scenario_negative_fraction(self):
        """Test that negative fraction raises error."""
        scenario = {4: [0.1, -0.1, 0.3]}
        with pytest.raises(InvalidScenarioError, match="must be in \\[0, 1\\]"):
            validate_scenario(scenario)

    def test_validate_scenario_fraction_exceeds_one(self):
        """Test that fraction > 1.0 raises error."""
        scenario = {4: [0.1, 1.2, 0.3]}
        with pytest.raises(InvalidScenarioError, match="must be in \\[0, 1\\]"):
            validate_scenario(scenario)

    def test_validate_scenario_non_monotonic(self):
        """Test that non-monotonic fractions raise error."""
        scenario = {4: [0.3, 0.2, 0.4]}  # Goes down in year 1
        with pytest.raises(InvalidScenarioError, match="must be non-decreasing"):
            validate_scenario(scenario)

    def test_validate_scenario_total_exceeds_one(self):
        """Test that total adoption > 1.0 raises error."""
        scenario = {4: [0.6, 0.7, 0.8], 8: [0.5, 0.6, 0.7]}  # Year 0: 1.1 total
        with pytest.raises(InvalidScenarioError, match="exceeds 1.0"):
            validate_scenario(scenario)

    def test_validate_scenario_mismatched_lengths(self):
        """Test that mismatched list lengths raise error."""
        scenario = {4: [0.1, 0.2, 0.3], 8: [0.05, 0.10]}  # Different lengths
        with pytest.raises(InvalidScenarioError, match="same length"):
            validate_scenario(scenario)

    def test_validate_scenario_negative_upgrade_id(self):
        """Test that negative upgrade ID raises error."""
        scenario = {-1: [0.1, 0.2, 0.3]}
        with pytest.raises(InvalidScenarioError, match="non-negative integer"):
            validate_scenario(scenario)


# =============================================================================
# Tests for Scenario Materialization
# =============================================================================


class TestScenarioMaterialization:
    """Tests for building materialization and allocation logic."""

    def test_init_samples_from_baseline(self, cleanup_downloads):
        """Test that MixedUpgradeScenario samples buildings from baseline."""
        # Download baseline data
        bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        scenario = {4: [0.1, 0.2], 8: [0.05, 0.10]}
        mus = MixedUpgradeScenario(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=50,
            seed=42,
            scenario=scenario,
        )

        assert len(mus.sampled_bldgs) == 50
        assert mus.num_years == 2

    def test_init_reproducible_with_seed(self, cleanup_downloads):
        """Test that same seed produces same materialization."""
        # Download baseline data
        bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        scenario = {4: [0.1, 0.3], 8: [0.05, 0.15]}

        mus1 = MixedUpgradeScenario(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=50,
            seed=42,
            scenario=scenario,
        )

        mus2 = MixedUpgradeScenario(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=50,
            seed=42,
            scenario=scenario,
        )

        # Same sampled buildings
        assert mus1.sampled_bldgs == mus2.sampled_bldgs

        # Same materialization
        assert mus1.materialized_scenario == mus2.materialized_scenario

    def test_materialization_correct_counts(self, cleanup_downloads):
        """Test that building allocation counts match scenario fractions."""
        # Download baseline data
        bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        scenario = {4: [0.2, 0.4], 8: [0.1, 0.2]}
        sample_n = 100

        mus = MixedUpgradeScenario(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=sample_n,
            seed=42,
            scenario=scenario,
        )

        # Check year 0
        year_0 = mus.materialized_scenario[0]
        count_upgrade_4_y0 = sum(1 for uid in year_0.values() if uid == 4)
        count_upgrade_8_y0 = sum(1 for uid in year_0.values() if uid == 8)

        assert count_upgrade_4_y0 == int(0.2 * sample_n)  # 20
        assert count_upgrade_8_y0 == int(0.1 * sample_n)  # 10

        # Check year 1
        year_1 = mus.materialized_scenario[1]
        count_upgrade_4_y1 = sum(1 for uid in year_1.values() if uid == 4)
        count_upgrade_8_y1 = sum(1 for uid in year_1.values() if uid == 8)

        assert count_upgrade_4_y1 == int(0.4 * sample_n)  # 40
        assert count_upgrade_8_y1 == int(0.2 * sample_n)  # 20

    def test_materialization_monotonicity(self, cleanup_downloads):
        """Test that buildings never change upgrades once allocated."""
        # Download baseline data
        bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        scenario = {4: [0.2, 0.4, 0.6], 8: [0.1, 0.2, 0.3]}
        mus = MixedUpgradeScenario(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=50,
            seed=42,
            scenario=scenario,
        )

        # Track buildings that adopt upgrade 4 in year 0
        year_0 = mus.materialized_scenario[0]
        upgrade_4_year_0 = {bid for bid, uid in year_0.items() if uid == 4}

        # Verify they're still in upgrade 4 in year 1 and 2
        year_1 = mus.materialized_scenario[1]
        year_2 = mus.materialized_scenario[2]

        for bldg_id in upgrade_4_year_0:
            assert year_1[bldg_id] == 4
            assert year_2[bldg_id] == 4

    def test_materialization_100_percent_adoption(self, cleanup_downloads):
        """Test edge case of 100% adoption."""
        # Download baseline data
        bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        scenario = {4: [0.5, 1.0], 8: [0.5, 0.0]}
        sample_n = 100

        mus = MixedUpgradeScenario(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=sample_n,
            seed=42,
            scenario=scenario,
        )

        # Year 1 should have 100% in upgrade 4
        year_1 = mus.materialized_scenario[1]
        count_upgrade_4 = sum(1 for uid in year_1.values() if uid == 4)
        count_baseline = sum(1 for uid in year_1.values() if uid == 0)

        assert count_upgrade_4 == sample_n
        assert count_baseline == 0

    def test_year_validation_invalid_year(self, cleanup_downloads):
        """Test that invalid year index raises ValueError."""
        # Download baseline data
        bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        scenario = {4: [0.1, 0.2]}
        mus = MixedUpgradeScenario(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=10,
            seed=42,
            scenario=scenario,
        )

        # Try to validate year 3 (out of range, only 0-1 valid)
        with pytest.raises(ValueError, match="out of range"):
            mus._validate_years([3])

    def test_year_validation_none_returns_all(self, cleanup_downloads):
        """Test that years=None returns all years."""
        # Download baseline data
        bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        scenario = {4: [0.1, 0.2, 0.3]}
        mus = MixedUpgradeScenario(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=10,
            seed=42,
            scenario=scenario,
        )

        years = mus._validate_years(None)
        assert years == [0, 1, 2]


# =============================================================================
# Tests for Data Reading
# =============================================================================


class TestMixedUpgradeReading:
    """Tests for reading data with mixed upgrade scenarios."""

    def test_read_metadata_returns_correct_schema(self, cleanup_downloads):
        """Test that read_metadata returns correct column schema."""
        # Download metadata for upgrades 0 and 4
        bldg_ids = [
            BuildingID(bldg_id=7, upgrade_id="0"),
            BuildingID(bldg_id=7, upgrade_id="4"),
        ]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        scenario = {4: [0.2, 0.4]}
        mus = MixedUpgradeScenario(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=10,
            seed=42,
            scenario=scenario,
        )

        metadata = mus.read_metadata()
        df = metadata.collect()

        # Check required columns exist
        assert "bldg_id" in df.columns
        assert "upgrade_id" in df.columns
        assert "year" in df.columns
        # Check we have data
        assert df.height > 0

    def test_read_metadata_all_years(self, cleanup_downloads):
        """Test that years=None returns all years."""
        # Download metadata for upgrades 0 and 4
        bldg_ids = [
            BuildingID(bldg_id=7, upgrade_id="0"),
            BuildingID(bldg_id=7, upgrade_id="4"),
        ]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        scenario = {4: [0.2, 0.4, 0.6]}
        mus = MixedUpgradeScenario(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=10,
            seed=42,
            scenario=scenario,
        )

        metadata = mus.read_metadata()
        df = metadata.collect()

        # Should have data for all 3 years
        years_in_data = df["year"].unique().sort().to_list()
        assert years_in_data == [0, 1, 2]

    def test_read_metadata_specific_years(self, cleanup_downloads):
        """Test that years=[1,2] filters correctly."""
        # Download metadata for upgrades 0 and 4
        bldg_ids = [
            BuildingID(bldg_id=7, upgrade_id="0"),
            BuildingID(bldg_id=7, upgrade_id="4"),
        ]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        scenario = {4: [0.2, 0.4, 0.6]}
        mus = MixedUpgradeScenario(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=10,
            seed=42,
            scenario=scenario,
        )

        metadata = mus.read_metadata(years=[1, 2])
        df = metadata.collect()

        # Should only have years 1 and 2
        years_in_data = df["year"].unique().sort().to_list()
        assert years_in_data == [1, 2]

    def test_read_load_curve_15min(self, cleanup_downloads):
        """Test that load curve reading works."""
        # Download load curves for upgrades 0 and 4
        bldg_ids = [
            BuildingID(bldg_id=7, upgrade_id="0"),
            BuildingID(bldg_id=7, upgrade_id="4"),
        ]
        fetch_bldg_data(bldg_ids, ("load_curve_15min",), Path("data"))

        scenario = {4: [0.2, 0.4]}
        mus = MixedUpgradeScenario(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=10,
            seed=42,
            scenario=scenario,
        )

        load_curves = mus.read_load_curve_15min()
        df = load_curves.collect()

        # Check required columns
        assert "bldg_id" in df.columns
        assert "upgrade_id" in df.columns
        assert "year" in df.columns
        assert "timestamp" in df.columns
        assert df.height > 0

    def test_read_load_curve_hourly(self, cleanup_downloads):
        """Test hourly load curve reading."""
        # Download load curves for upgrades 0 and 4
        bldg_ids = [
            BuildingID(bldg_id=7, upgrade_id="0"),
            BuildingID(bldg_id=7, upgrade_id="4"),
        ]
        fetch_bldg_data(bldg_ids, ("load_curve_hourly",), Path("data"))

        scenario = {4: [0.2, 0.4]}
        mus = MixedUpgradeScenario(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=10,
            seed=42,
            scenario=scenario,
        )

        load_curves = mus.read_load_curve_hourly()
        df = load_curves.collect()

        assert "bldg_id" in df.columns
        assert "year" in df.columns
        assert "timestamp" in df.columns

    def test_year_column_values_correct(self, cleanup_downloads):
        """Test that year column values match requested years."""
        # Download metadata for upgrades 0 and 4
        bldg_ids = [
            BuildingID(bldg_id=7, upgrade_id="0"),
            BuildingID(bldg_id=7, upgrade_id="4"),
        ]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        scenario = {4: [0.2, 0.4]}
        mus = MixedUpgradeScenario(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=10,
            seed=42,
            scenario=scenario,
        )

        # Read only year 0
        metadata_y0 = mus.read_metadata(years=[0])
        df_y0 = metadata_y0.collect()

        assert df_y0["year"].unique().to_list() == [0]

    def test_upgrade_data_not_on_disk(self, cleanup_downloads):
        """Test that missing upgrade data raises ScenarioDataNotFoundError."""
        # Only download upgrade 0, not upgrade 4
        bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        scenario = {4: [0.2, 0.4]}  # Requires upgrade 4
        mus = MixedUpgradeScenario(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=10,
            seed=42,
            scenario=scenario,
        )

        # Should fail when trying to read
        with pytest.raises(ScenarioDataNotFoundError, match="Missing metadata data"):
            mus.read_metadata().collect()

    def test_multiple_states(self, cleanup_downloads):
        """Test that multi-state scenarios work."""
        # Download data for NY and OH
        bldg_ids = [
            BuildingID(bldg_id=7, upgrade_id="0", state="NY"),
            BuildingID(bldg_id=7, upgrade_id="4", state="NY"),
            BuildingID(bldg_id=100000, upgrade_id="0", state="OH", release_number="2"),
            BuildingID(bldg_id=100000, upgrade_id="4", state="OH", release_number="2"),
        ]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        scenario = {4: [0.2, 0.4]}
        mus = MixedUpgradeScenario(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=[State.NY, State.OH],
            sample_n=20,
            seed=42,
            scenario=scenario,
        )

        metadata = mus.read_metadata()
        df = metadata.collect()

        # Should have buildings from both states
        states_in_result = df["in.state"].unique().to_list()
        assert "NY" in states_in_result
        assert "OH" in states_in_result


# =============================================================================
# Tests for Scenario Export
# =============================================================================


class TestScenarioExport:
    """Tests for exporting scenarios to CAIRO format."""

    def test_export_cairo_format(self, cleanup_downloads, tmp_path):
        """Test that export creates correct CSV structure."""
        # Download metadata for upgrades 0 and 4
        bldg_ids = [
            BuildingID(bldg_id=7, upgrade_id="0"),
            BuildingID(bldg_id=7, upgrade_id="4"),
        ]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        scenario = {4: [0.2, 0.4, 0.6]}
        mus = MixedUpgradeScenario(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=20,
            seed=42,
            scenario=scenario,
        )

        output_path = tmp_path / "scenario.csv"
        mus.export_scenario_to_cairo(output_path)

        # Read back the CSV
        df = pl.read_csv(output_path)

        # Check structure
        assert "bldg_id" in df.columns
        assert "year_0" in df.columns
        assert "year_1" in df.columns
        assert "year_2" in df.columns
        assert df.height == 20  # 20 buildings

    def test_export_cairo_column_names(self, cleanup_downloads, tmp_path):
        """Test that column names are human-readable (year_0, year_1, etc.)."""
        # Download metadata for upgrades 0 and 4
        bldg_ids = [
            BuildingID(bldg_id=7, upgrade_id="0"),
            BuildingID(bldg_id=7, upgrade_id="4"),
        ]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        scenario = {4: [0.2, 0.4]}
        mus = MixedUpgradeScenario(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=10,
            seed=42,
            scenario=scenario,
        )

        output_path = tmp_path / "scenario.csv"
        mus.export_scenario_to_cairo(output_path)

        df = pl.read_csv(output_path)

        # Check column names
        expected_cols = {"bldg_id", "year_0", "year_1"}
        assert expected_cols.issubset(set(df.columns))

    def test_export_cairo_row_order(self, cleanup_downloads, tmp_path):
        """Test that rows are sorted by bldg_id."""
        # Download metadata for upgrades 0 and 4
        bldg_ids = [
            BuildingID(bldg_id=7, upgrade_id="0"),
            BuildingID(bldg_id=7, upgrade_id="4"),
        ]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        scenario = {4: [0.2, 0.4]}
        mus = MixedUpgradeScenario(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=20,
            seed=42,
            scenario=scenario,
        )

        output_path = tmp_path / "scenario.csv"
        mus.export_scenario_to_cairo(output_path)

        df = pl.read_csv(output_path)

        # Check that bldg_id is sorted
        bldg_ids_col = df["bldg_id"].to_list()
        assert bldg_ids_col == sorted(bldg_ids_col)

    def test_export_cairo_values_match_scenario(self, cleanup_downloads, tmp_path):
        """Test that exported values match materialized scenario."""
        # Download metadata for upgrades 0 and 4
        bldg_ids = [
            BuildingID(bldg_id=7, upgrade_id="0"),
            BuildingID(bldg_id=7, upgrade_id="4"),
        ]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        scenario = {4: [0.2, 0.4]}
        mus = MixedUpgradeScenario(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=10,
            seed=42,
            scenario=scenario,
        )

        output_path = tmp_path / "scenario.csv"
        mus.export_scenario_to_cairo(output_path)

        df = pl.read_csv(output_path)

        # Verify first row matches materialization
        first_bldg = df.row(0, named=True)
        bldg_id = first_bldg["bldg_id"]

        # Check against materialized scenario
        assert first_bldg["year_0"] == mus.materialized_scenario[0][bldg_id]
        assert first_bldg["year_1"] == mus.materialized_scenario[1][bldg_id]

    def test_export_missing_upgrade_data(self, cleanup_downloads, tmp_path):
        """Test that export fails if upgrade data is missing."""
        # Only download upgrade 0, not upgrade 4
        bldg_ids = [BuildingID(bldg_id=7, upgrade_id="0")]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        scenario = {4: [0.2, 0.4]}
        mus = MixedUpgradeScenario(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=10,
            seed=42,
            scenario=scenario,
        )

        output_path = tmp_path / "scenario.csv"

        # Should fail because upgrade 4 data is missing
        with pytest.raises(ScenarioDataNotFoundError):
            mus.export_scenario_to_cairo(output_path)


# =============================================================================
# Integration Tests
# =============================================================================


class TestMixedUpgradeIntegration:
    """Integration tests for MixedUpgradeScenario."""

    def test_end_to_end_simple_scenario(self, cleanup_downloads, tmp_path):
        """Test full workflow with simple scenario."""
        # Download metadata and load curves for upgrades 0 and 4
        bldg_ids = [
            BuildingID(bldg_id=7, upgrade_id="0"),
            BuildingID(bldg_id=7, upgrade_id="4"),
        ]
        fetch_bldg_data(bldg_ids, ("metadata", "load_curve_15min"), Path("data"))

        # Create scenario using helper
        scenario = uniform_adoption(
            upgrade_ids=[4],
            weights={4: 1.0},
            adoption_trajectory=[0.2, 0.5],
        )

        # Initialize MixedUpgradeScenario
        mus = MixedUpgradeScenario(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=50,
            seed=42,
            scenario=scenario,
        )

        # Read metadata
        metadata = mus.read_metadata()
        df_meta = metadata.collect()
        assert df_meta.height > 0

        # Read load curves
        loads = mus.read_load_curve_15min()
        df_loads = loads.collect()
        assert df_loads.height > 0

        # Export scenario
        output_path = tmp_path / "scenario.csv"
        mus.export_scenario_to_cairo(output_path)
        assert output_path.exists()

    def test_multi_year_multi_upgrade_multi_state(self, cleanup_downloads, tmp_path):
        """Test complex scenario with multiple years, upgrades, and states."""
        # Download data for NY and OH, upgrades 0, 4, and 8
        bldg_ids = [
            BuildingID(bldg_id=7, upgrade_id="0", state="NY"),
            BuildingID(bldg_id=7, upgrade_id="4", state="NY"),
            BuildingID(bldg_id=7, upgrade_id="8", state="NY"),
            BuildingID(bldg_id=100000, upgrade_id="0", state="OH", release_number="2"),
            BuildingID(bldg_id=100000, upgrade_id="4", state="OH", release_number="2"),
            BuildingID(bldg_id=100000, upgrade_id="8", state="OH", release_number="2"),
        ]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        # Create complex scenario
        scenario = uniform_adoption(
            upgrade_ids=[4, 8],
            weights={4: 0.6, 8: 0.4},
            adoption_trajectory=[0.1, 0.3, 0.5],
        )

        mus = MixedUpgradeScenario(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=[State.NY, State.OH],
            sample_n=100,
            seed=42,
            scenario=scenario,
        )

        # Read all metadata
        metadata = mus.read_metadata()
        df = metadata.collect()

        # Verify we have data for all 3 years
        assert set(df["year"].unique().to_list()) == {0, 1, 2}

        # Verify we have both states
        states_in_data = df["in.state"].unique().to_list()
        assert "NY" in states_in_data
        assert "OH" in states_in_data

        # Verify we have both upgrades (plus baseline)
        upgrades_in_data = set(df["upgrade_id"].unique().to_list())
        assert {0, 4, 8}.issubset(upgrades_in_data)

    def test_custom_scenario_dict(self, cleanup_downloads):
        """Test using custom scenario dict (not uniform_adoption)."""
        # Download metadata for upgrades 0, 4, and 8
        bldg_ids = [
            BuildingID(bldg_id=7, upgrade_id="0"),
            BuildingID(bldg_id=7, upgrade_id="4"),
            BuildingID(bldg_id=7, upgrade_id="8"),
        ]
        fetch_bldg_data(bldg_ids, ("metadata",), Path("data"))

        # Custom staggered scenario
        custom_scenario = {
            4: [0.10, 0.25, 0.25],  # Ramps up then plateaus
            8: [0.00, 0.00, 0.15],  # Enters in year 2
        }

        mus = MixedUpgradeScenario(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=100,
            seed=42,
            scenario=custom_scenario,
        )

        # Read year 2 when both upgrades are available
        metadata_y2 = mus.read_metadata(years=[2])
        df = metadata_y2.collect()

        # Should have buildings in upgrades 0, 4, and 8
        upgrades_in_year_2 = set(df["upgrade_id"].unique().to_list())
        assert {0, 4, 8}.issubset(upgrades_in_year_2)

    def test_consistency_across_read_methods(self, cleanup_downloads):
        """Test that metadata and load curves have same buildings/years."""
        # Download metadata and load curves for upgrades 0 and 4
        bldg_ids = [
            BuildingID(bldg_id=7, upgrade_id="0"),
            BuildingID(bldg_id=7, upgrade_id="4"),
        ]
        fetch_bldg_data(bldg_ids, ("metadata", "load_curve_15min"), Path("data"))

        scenario = {4: [0.2, 0.4]}
        mus = MixedUpgradeScenario(
            data_path="data",
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=20,
            seed=42,
            scenario=scenario,
        )

        # Read metadata and load curves
        metadata = mus.read_metadata()
        df_meta = metadata.collect()

        loads = mus.read_load_curve_15min()
        df_loads = loads.collect()

        # Get unique (bldg_id, year, upgrade_id) combinations from each
        meta_combos = set(df_meta.select(["bldg_id", "year", "upgrade_id"]).unique().iter_rows())
        loads_combos = set(df_loads.select(["bldg_id", "year", "upgrade_id"]).unique().iter_rows())

        # Should be the same
        assert meta_combos == loads_combos
