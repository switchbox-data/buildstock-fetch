"""Tests for mixed upgrade scenario functionality.

These tests cover scenario validation, building materialization, data reading,
and export functionality for multi-year adoption trajectories.
"""

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from buildstock_fetch.io import BuildStockRelease, State
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

    def test_init_samples_from_baseline(self, integration_test_data):
        """Test that MixedUpgradeScenario samples buildings from baseline."""
        scenario = {4: [0.1, 0.2], 8: [0.05, 0.10]}
        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=10,
            seed=42,
            scenario=scenario,
        )

        assert len(mus.sampled_bldgs) == 10
        assert mus.num_years == 2

    def test_init_reproducible_with_seed(self, integration_test_data):
        """Test that same seed produces same materialization."""
        scenario = {4: [0.1, 0.3], 8: [0.05, 0.15]}

        mus1 = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=10,
            seed=42,
            scenario=scenario,
        )

        mus2 = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=10,
            seed=42,
            scenario=scenario,
        )

        # Same sampled buildings
        assert mus1.sampled_bldgs == mus2.sampled_bldgs

        # Same materialization
        assert mus1.materialized_scenario == mus2.materialized_scenario

    def test_materialization_correct_counts(self, integration_test_data):
        """Test that building allocation counts match scenario fractions."""
        scenario = {4: [0.2, 0.4], 8: [0.1, 0.2]}
        sample_n = 10

        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
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

        assert count_upgrade_4_y0 == int(0.2 * sample_n)  # 2
        assert count_upgrade_8_y0 == int(0.1 * sample_n)  # 1

        # Check year 1
        year_1 = mus.materialized_scenario[1]
        count_upgrade_4_y1 = sum(1 for uid in year_1.values() if uid == 4)
        count_upgrade_8_y1 = sum(1 for uid in year_1.values() if uid == 8)

        assert count_upgrade_4_y1 == int(0.4 * sample_n)  # 4
        assert count_upgrade_8_y1 == int(0.2 * sample_n)  # 2

    def test_materialization_monotonicity(self, integration_test_data):
        """Test that buildings never change upgrades once allocated."""
        scenario = {4: [0.2, 0.4, 0.6], 8: [0.1, 0.2, 0.3]}
        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=10,
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

    def test_materialization_100_percent_adoption(self, integration_test_data):
        """Test edge case of 100% adoption."""
        scenario = {4: [0.5, 1.0], 8: [0.5, 0.0]}
        sample_n = 10

        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
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

    def test_year_validation_invalid_year(self, integration_test_data):
        """Test that invalid year index raises ValueError."""
        scenario = {4: [0.1, 0.2]}
        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=10,
            seed=42,
            scenario=scenario,
        )

        # Try to validate year 3 (out of range, only 0-1 valid)
        with pytest.raises(ValueError, match="out of range"):
            mus._validate_years([3])

    def test_year_validation_none_returns_all(self, integration_test_data):
        """Test that years=None returns all years."""
        scenario = {4: [0.1, 0.2, 0.3]}
        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
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

    def test_read_metadata_returns_correct_schema(self, integration_test_data):
        """Test that read_metadata returns correct column schema."""
        scenario = {4: [0.2, 0.4]}
        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
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

    def test_read_metadata_specific_years(self, integration_test_data):
        """Test that years=[1,2] filters correctly."""
        scenario = {4: [0.2, 0.4, 0.6]}
        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
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

    def test_read_load_curve_15min(self, integration_test_data):
        """Test that load curve reading works."""
        scenario = {4: [0.2, 0.4]}
        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
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

    def test_upgrade_data_not_on_disk(self, integration_test_data):
        """Test that missing upgrade data raises ScenarioDataNotFoundError."""
        # Use upgrade 9 which is not downloaded in the fixture
        scenario = {9: [0.2, 0.4]}  # Requires upgrade 9 (not downloaded)
        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=10,
            seed=42,
            scenario=scenario,
        )

        # Should fail when trying to read
        with pytest.raises(ScenarioDataNotFoundError, match="Missing metadata data"):
            mus.read_metadata().collect()

    def test_multiple_states(self, integration_test_data):
        """Test that multi-state scenarios work."""
        scenario = {4: [0.2, 0.4]}
        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=[State.NY, State.AL],
            sample_n=10,
            seed=42,
            scenario=scenario,
        )

        metadata = mus.read_metadata()
        df = metadata.collect()

        # Should have buildings from both states
        states_in_result = df["in.state"].unique().to_list()
        assert "NY" in states_in_result
        assert "AL" in states_in_result


# =============================================================================
# Tests for Scenario Export
# =============================================================================


class TestScenarioExport:
    """Tests for exporting scenarios to CAIRO format."""

    def test_export_cairo_format(self, integration_test_data):
        """Test that export creates correct CSV structure."""
        scenario = {4: [0.2, 0.4, 0.6]}
        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=10,
            seed=42,
            scenario=scenario,
        )

        output_path = integration_test_data["outputs_path"] / "scenario.csv"
        mus.export_scenario_to_cairo(output_path)

        # Read back the CSV
        df = pl.read_csv(output_path)

        # Check structure
        assert "bldg_id" in df.columns
        assert "year_0" in df.columns
        assert "year_1" in df.columns
        assert "year_2" in df.columns
        assert df.height == 10  # 10 buildings

    def test_export_missing_upgrade_data(self, integration_test_data):
        """Test that export fails if upgrade data is missing."""
        # Use upgrade 9 which is not downloaded in the fixture
        scenario = {9: [0.2, 0.4]}
        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=10,
            seed=42,
            scenario=scenario,
        )

        output_path = integration_test_data["outputs_path"] / "scenario.csv"

        # Should fail because upgrade 9 data is missing
        with pytest.raises(ScenarioDataNotFoundError):
            mus.export_scenario_to_cairo(output_path)


# =============================================================================
# Integration Tests
# =============================================================================


class TestMixedUpgradeIntegration:
    """Integration tests for MixedUpgradeScenario."""

    def test_end_to_end_simple_scenario(self, integration_test_data):
        """Test full workflow with simple scenario."""
        # Create scenario using helper
        scenario = uniform_adoption(
            upgrade_ids=[4],
            weights={4: 1.0},
            adoption_trajectory=[0.2, 0.5],
        )

        # Initialize MixedUpgradeScenario
        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=10,
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
        output_path = integration_test_data["outputs_path"] / "scenario.csv"
        mus.export_scenario_to_cairo(output_path)
        assert output_path.exists()

    def test_multi_year_multi_upgrade_multi_state(self, integration_test_data):
        """Test complex scenario with multiple years, upgrades, and states."""
        # Create complex scenario
        scenario = uniform_adoption(
            upgrade_ids=[4, 8],
            weights={4: 0.6, 8: 0.4},
            adoption_trajectory=[0.1, 0.3, 0.5],
        )

        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=[State.NY, State.AL],
            sample_n=10,
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
        assert "AL" in states_in_data

        # Verify we have both upgrades (plus baseline)
        upgrades_in_data = set(df["upgrade_id"].unique().to_list())
        assert {0, 4, 8}.issubset(upgrades_in_data)

    def test_custom_scenario_dict(self, integration_test_data):
        """Test using custom scenario dict (not uniform_adoption)."""
        # Custom staggered scenario
        custom_scenario = {
            4: [0.10, 0.25, 0.25],  # Ramps up then plateaus
            8: [0.00, 0.00, 0.15],  # Enters in year 2
        }

        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=10,
            seed=42,
            scenario=custom_scenario,
        )

        # Read year 2 when both upgrades are available
        metadata_y2 = mus.read_metadata(years=[2])
        df = metadata_y2.collect()

        # Should have buildings in upgrades 0, 4, and 8
        upgrades_in_year_2 = set(df["upgrade_id"].unique().to_list())
        assert {0, 4, 8}.issubset(upgrades_in_year_2)

    def test_consistency_across_read_methods(self, integration_test_data):
        """Test that metadata and load curves have same buildings/years."""
        scenario = {4: [0.2, 0.4]}
        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            release=BuildStockRelease.RES_2024_TMY3_2,
            states=State.NY,
            sample_n=10,
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
