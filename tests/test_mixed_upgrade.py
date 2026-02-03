"""Tests for mixed upgrade scenario functionality."""

import polars as pl
import pytest

from buildstock_fetch.mixed_upgrade import MixedUpgradeScenario, ScenarioDataNotFoundError
from buildstock_fetch.scenarios import InvalidScenarioError, uniform_adoption

SCENARIO_NAME = "test_scenario"


# Tests that don't require data downloads
class TestScenarioInitialization:
    """Tests for MixedUpgradeScenario initialization."""

    def test_requires_scenario_parameter(self):
        """Test that scenario parameter is required."""
        with pytest.raises(ValueError, match="scenario parameter is required"):
            MixedUpgradeScenario(
                data_path="./data",
                pathway_scenario_name=SCENARIO_NAME,
                release="res_2024_tmy3_2",
                scenario=None,
            )

    def test_validates_scenario_on_init(self):
        """Test that invalid scenarios are rejected during initialization."""
        bad_scenario = {4: [0.3, 0.2, 0.4]}  # Non-monotonic
        with pytest.raises(InvalidScenarioError, match="must be non-decreasing"):
            MixedUpgradeScenario(
                data_path="./data",
                pathway_scenario_name=SCENARIO_NAME,
                release="res_2024_tmy3_2",
                scenario=bad_scenario,
            )


# Tests that use shared integration test data
class TestScenarioMaterialization:
    """Tests for building materialization and allocation logic."""

    def test_init_samples_from_baseline(self, integration_test_data):
        """Test that MixedUpgradeScenario samples buildings from baseline."""
        scenario = {4: [0.1, 0.2], 8: [0.05, 0.10]}
        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            pathway_scenario_name=SCENARIO_NAME,
            release="res_2024_tmy3_2",
            states="NY",
            sample_n=10,
            random=42,
            scenario=scenario,
        )

        assert len(mus.sampled_bldgs) <= 10  # Could be fewer than 10 if not enough buildings in NY
        assert mus.num_years == 2

    def test_init_reproducible_with_seed(self, integration_test_data):
        """Test that same seed produces same materialization."""
        scenario = {4: [0.1, 0.3], 8: [0.05, 0.15]}

        mus1 = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            pathway_scenario_name=SCENARIO_NAME,
            release="res_2024_tmy3_2",
            states="NY",
            sample_n=10,
            random=42,
            scenario=scenario,
        )
        mus2 = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            pathway_scenario_name=SCENARIO_NAME,
            release="res_2024_tmy3_2",
            states="NY",
            sample_n=10,
            random=42,
            scenario=scenario,
        )

        # Same sampled buildings and materialization
        assert mus1.sampled_bldgs == mus2.sampled_bldgs
        assert mus1.materialized_scenario == mus2.materialized_scenario

    def test_materialization_monotonicity(self, integration_test_data):
        """Test that buildings never change upgrades once allocated."""
        scenario = {4: [0.2, 0.4, 0.6], 8: [0.1, 0.2, 0.3]}
        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            pathway_scenario_name=SCENARIO_NAME,
            release="res_2024_tmy3_2",
            states="NY",
            sample_n=10,
            random=42,
            scenario=scenario,
        )

        mat = mus.materialized_scenario
        year_0_upgrade_4 = {bid for bid, uid in mat[0].items() if uid == 4}

        # Verify they stay in upgrade 4
        for year_idx in [1, 2]:
            for bldg_id in year_0_upgrade_4:
                assert mat[year_idx][bldg_id] == 4

    def test_year_validation_none_returns_all(self, integration_test_data):
        """Test that years=None returns all years."""
        scenario = {4: [0.1, 0.2, 0.3]}
        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            pathway_scenario_name=SCENARIO_NAME,
            release="res_2024_tmy3_2",
            states="NY",
            sample_n=10,
            random=42,
            scenario=scenario,
        )

        years = mus._validate_years(None)
        assert years == [0, 1, 2]


class TestMixedUpgradeReading:
    """Tests for reading data with mixed upgrade scenarios."""

    def test_read_metadata_returns_correct_schema(self, integration_test_data):
        """Test that read_metadata returns correct column schema."""
        scenario = {4: [0.2, 0.4]}
        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            pathway_scenario_name=SCENARIO_NAME,
            release="res_2024_tmy3_2",
            states="NY",
            sample_n=10,
            random=42,
            scenario=scenario,
        )

        metadata = mus.read_metadata()
        df = metadata.collect()

        assert "bldg_id" in df.columns
        assert "upgrade_id" in df.columns
        assert "year" in df.columns
        assert df.height > 0

    def test_read_metadata_specific_years(self, integration_test_data):
        """Test that years=[1,2] filters correctly."""
        scenario = {4: [0.2, 0.4, 0.6]}
        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            pathway_scenario_name=SCENARIO_NAME,
            release="res_2024_tmy3_2",
            states="NY",
            sample_n=10,
            random=42,
            scenario=scenario,
        )

        metadata = mus.read_metadata(years=[1, 2])
        df = metadata.collect()

        years_in_data = sorted(df["year"].unique().to_list())
        assert years_in_data == [1, 2]

    def test_read_load_curve_15min(self, integration_test_data):
        """Test that load curve reading works."""
        scenario = {4: [0.2, 0.4]}
        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            pathway_scenario_name=SCENARIO_NAME,
            release="res_2024_tmy3_2",
            states="NY",
            sample_n=10,
            random=42,
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

    def test_year_validation_invalid_year(self, integration_test_data):
        """Test that invalid year index raises ValueError."""
        scenario = {4: [0.1, 0.2]}
        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            pathway_scenario_name=SCENARIO_NAME,
            release="res_2024_tmy3_2",
            states="NY",
            sample_n=10,
            random=42,
            scenario=scenario,
        )

        with pytest.raises(ValueError, match="out of range"):
            mus._validate_years([3])

    def test_upgrade_data_not_on_disk(self, integration_test_data):
        """Test that missing upgrade data raises ScenarioDataNotFoundError."""
        scenario = {9: [0.2, 0.4]}  # Upgrade 9 not downloaded
        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            pathway_scenario_name=SCENARIO_NAME,
            release="res_2024_tmy3_2",
            states="NY",
            sample_n=10,
            random=42,
            scenario=scenario,
        )

        with pytest.raises(ScenarioDataNotFoundError):
            mus.read_metadata().collect()

    def test_multiple_states(self, integration_test_data):
        """Test that multi-state scenarios work."""
        scenario = {4: [0.2, 0.4]}
        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            pathway_scenario_name=SCENARIO_NAME,
            release="res_2024_tmy3_2",
            states=["NY", "AL"],
            sample_n=10,
            random=42,
            scenario=scenario,
        )

        metadata = mus.read_metadata()
        df = metadata.collect()

        # Should have buildings from both states
        states_in_result = df["in.state"].unique().to_list()
        assert "NY" in states_in_result
        assert "AL" in states_in_result


class TestScenarioExport:
    """Tests for exporting scenarios to CAIRO format."""

    def test_export_cairo_format(self, integration_test_data):
        """Test that export creates correct CSV structure."""
        scenario = {4: [0.2, 0.4, 0.6]}
        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            pathway_scenario_name=SCENARIO_NAME,
            release="res_2024_tmy3_2",
            states="NY",
            sample_n=10,
            random=42,
            scenario=scenario,
        )

        output_path = integration_test_data["outputs_path"] / "scenario.csv"
        mus.export_scenario_to_cairo(output_path)

        # Read back and validate
        df = pl.read_csv(output_path)
        assert "bldg_id" in df.columns
        assert "year_0" in df.columns
        assert "year_1" in df.columns
        assert "year_2" in df.columns
        assert df.height > 0

    def test_save_metadata_parquet_creates_year_partitions(self, integration_test_data, tmp_path):
        scenario = {4: [0.1, 0.2]}
        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            pathway_scenario_name=SCENARIO_NAME,
            release="res_2024_tmy3_2",
            states="NY",
            sample_n=5,
            random=42,
            scenario=scenario,
        )

        mus.save_metadata_parquet(path=tmp_path)

        assert (tmp_path / SCENARIO_NAME / "year=0" / "metadata.parquet").exists()
        assert (tmp_path / SCENARIO_NAME / "year=1" / "metadata.parquet").exists()

    def test_save_hourly_load_parquet_creates_files(self, integration_test_data, tmp_path):
        scenario = {4: [0.1, 0.2]}
        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            pathway_scenario_name=SCENARIO_NAME,
            release="res_2024_tmy3_2",
            states="NY",
            sample_n=5,
            random=42,
            scenario=scenario,
        )

        mus.save_hourly_load_parquet(path=tmp_path)

        year0_dir = tmp_path / SCENARIO_NAME / "year=0"
        year1_dir = tmp_path / SCENARIO_NAME / "year=1"
        assert year0_dir.exists()
        assert year1_dir.exists()
        assert list(year0_dir.glob("*.parquet"))


class TestMixedUpgradeIntegration:
    """Integration tests for MixedUpgradeScenario."""

    def test_end_to_end_simple_scenario(self, integration_test_data):
        """Test full workflow with simple scenario."""
        scenario = uniform_adoption(upgrade_ids=[4], weights={4: 1.0}, adoption_trajectory=[0.2, 0.5])

        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            pathway_scenario_name=SCENARIO_NAME,
            release="res_2024_tmy3_2",
            states="NY",
            sample_n=10,
            random=42,
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
        output_path = integration_test_data["outputs_path"] / "scenario_simple.csv"
        mus.export_scenario_to_cairo(output_path)
        assert output_path.exists()

    def test_multi_year_multi_upgrade_multi_state(self, integration_test_data):
        """Test complex scenario with multiple years, upgrades, and states."""
        scenario = uniform_adoption(upgrade_ids=[4, 8], weights={4: 0.6, 8: 0.4}, adoption_trajectory=[0.1, 0.3, 0.5])

        mus = MixedUpgradeScenario(
            data_path=str(integration_test_data["data_path"]),
            pathway_scenario_name=SCENARIO_NAME,
            release="res_2024_tmy3_2",
            states=["NY", "AL"],
            sample_n=10,
            random=42,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
