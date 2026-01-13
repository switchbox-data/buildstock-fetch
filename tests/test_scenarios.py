"""Tests for scenario validation and generation utilities."""

import pytest

from buildstock_fetch.scenarios import InvalidScenarioError, uniform_adoption, validate_scenario


class TestUniformAdoption:
    """Tests for uniform_adoption helper function."""

    def test_basic(self):
        """Test basic uniform_adoption generation."""
        scenario = uniform_adoption(
            upgrade_ids=[4, 8],
            weights={4: 0.6, 8: 0.4},
            adoption_trajectory=[0.1, 0.3, 0.5],
        )
        # Use pytest.approx for floating point comparison
        assert scenario[4] == pytest.approx([0.06, 0.18, 0.30])
        assert scenario[8] == pytest.approx([0.04, 0.12, 0.20])

    def test_single_upgrade(self):
        scenario = uniform_adoption([4], {4: 1.0}, [0.2, 0.5, 0.8])
        assert scenario == {4: [0.2, 0.5, 0.8]}

    @pytest.mark.parametrize(
        "upgrade_ids,weights,trajectory,error_match",
        [
            ([4, 8], {4: 0.7, 8: 0.4}, [0.1, 0.3], "must sum to 1.0"),
            ([4, 8], {4: -0.2, 8: 1.2}, [0.1, 0.3], "must be in \\[0, 1\\]"),
            ([4, 8], {4: 0.6, 8: 0.4}, [0.3, 0.2, 0.5], "must be non-decreasing"),
            ([4, 8], {4: 0.6, 9: 0.4}, [0.1, 0.3], "must match upgrade_ids"),
            ([], {}, [0.1, 0.3], "cannot be empty"),
            ([4, 8], {4: 0.6, 8: 0.4}, [], "cannot be empty"),
            ([-1, 8], {-1: 0.6, 8: 0.4}, [0.1, 0.3], "non-negative integer"),
        ],
    )
    def test_invalid_inputs(self, upgrade_ids, weights, trajectory, error_match):
        """Test various invalid input combinations."""
        with pytest.raises(InvalidScenarioError, match=error_match):
            uniform_adoption(upgrade_ids, weights, trajectory)


class TestScenarioValidation:
    """Tests for scenario validation logic."""

    def test_valid_scenarios(self):
        """Test that valid scenarios pass validation."""
        validate_scenario({4: [0.1, 0.2, 0.3], 8: [0.05, 0.10, 0.15]})
        validate_scenario({0: [0.0, 0.5, 1.0]})
        validate_scenario({4: [0.3, 0.4, 0.5], 8: [0.2, 0.3, 0.4]})  # Sums to 0.5, 0.7, 0.9

    @pytest.mark.parametrize(
        "scenario,error_match",
        [
            ({}, "cannot be empty"),
            ({4: [0.1, -0.1, 0.3]}, "must be in \\[0, 1\\]"),
            ({4: [0.1, 1.2, 0.3]}, "must be in \\[0, 1\\]"),
            ({4: [0.3, 0.2, 0.4]}, "must be non-decreasing"),
            ({4: [0.6, 0.7, 0.8], 8: [0.5, 0.6, 0.7]}, "exceeds 1.0"),
            ({4: [0.1, 0.2, 0.3], 8: [0.05, 0.10]}, "same length"),
            ({4: [], 8: []}, "at least one year"),
            ({-1: [0.1, 0.2, 0.3]}, "non-negative integer"),
        ],
    )
    def test_invalid_scenarios(self, scenario, error_match):
        """Test various invalid scenario configurations."""
        with pytest.raises(InvalidScenarioError, match=error_match):
            validate_scenario(scenario)

    def test_floating_point_tolerance(self):
        """Test that small floating point errors in total are tolerated."""
        # Year 1: 0.6+1e-7 + 0.4 = 1.0000001 (within 1e-6 tolerance)
        validate_scenario({4: [0.5, 0.6 + 1e-7], 8: [0.3, 0.4]})
