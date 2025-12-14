"""
Scenario definition and validation for mixed upgrade adoption trajectories.

This module provides utilities for defining and validating multi-year adoption
scenarios where buildings progressively adopt different upgrades over time.

Example:
    >>> from buildstock_fetch.scenarios import uniform_adoption
    >>> scenario = uniform_adoption(
    ...     upgrade_ids=[4, 8],
    ...     weights={4: 0.6, 8: 0.4},
    ...     adoption_trajectory=[0.1, 0.3, 0.5],
    ... )
    >>> # Returns: {4: [0.06, 0.18, 0.30], 8: [0.04, 0.12, 0.20]}
"""

from __future__ import annotations


class InvalidScenarioError(ValueError):
    """Raised when a scenario definition is invalid."""

    pass


def validate_scenario(scenario: dict[int, list[float]]) -> None:
    """Validate a mixed upgrade scenario definition.

    A valid scenario must satisfy:
    - All upgrade IDs are positive integers
    - All adoption fraction lists have the same length (num_years)
    - All fractions are in the range [0, 1]
    - Fractions are non-decreasing per upgrade (monotonic adoption)
    - Total adoption per year does not exceed 1.0

    Args:
        scenario: Dict mapping upgrade IDs to adoption fractions per year.
            Example: {4: [0.06, 0.18, 0.30], 8: [0.04, 0.12, 0.20]}

    Raises:
        InvalidScenarioError: If the scenario violates any validation rules.

    Example:
        >>> scenario = {4: [0.1, 0.2, 0.3], 8: [0.05, 0.10, 0.15]}
        >>> validate_scenario(scenario)  # Passes validation
        >>> bad_scenario = {4: [0.6, 0.4, 0.3]}  # Non-monotonic
        >>> validate_scenario(bad_scenario)  # Raises InvalidScenarioError
    """
    if not scenario:
        raise InvalidScenarioError("Scenario cannot be empty")

    # Check that all upgrade IDs are positive integers
    for upgrade_id in scenario.keys():
        if not isinstance(upgrade_id, int) or upgrade_id < 0:
            raise InvalidScenarioError(f"Upgrade ID must be a non-negative integer, got: {upgrade_id}")

    # Check that all lists have the same length
    num_years_list = [len(fractions) for fractions in scenario.values()]
    if len(set(num_years_list)) > 1:
        raise InvalidScenarioError(
            f"All upgrade adoption lists must have the same length. "
            f"Got lengths: {dict(zip(scenario.keys(), num_years_list))}"
        )

    num_years = num_years_list[0]
    if num_years == 0:
        raise InvalidScenarioError("Scenario must have at least one year")

    # Validate fractions and monotonicity per upgrade
    for upgrade_id, fractions in scenario.items():
        for year_idx, fraction in enumerate(fractions):
            # Check fraction is in [0, 1]
            if not 0 <= fraction <= 1:
                raise InvalidScenarioError(
                    f"Adoption fraction for upgrade {upgrade_id} in year {year_idx} "
                    f"must be in [0, 1], got: {fraction}"
                )

            # Check monotonicity (non-decreasing)
            if year_idx > 0 and fraction < fractions[year_idx - 1]:
                raise InvalidScenarioError(
                    f"Adoption fractions for upgrade {upgrade_id} must be non-decreasing. "
                    f"Year {year_idx - 1}: {fractions[year_idx - 1]}, Year {year_idx}: {fraction}"
                )

    # Check total adoption per year does not exceed 1.0
    for year_idx in range(num_years):
        total_adoption = sum(scenario[upgrade_id][year_idx] for upgrade_id in scenario)
        # Allow small floating point tolerance
        if total_adoption > 1.0 + 1e-6:
            raise InvalidScenarioError(
                f"Total adoption in year {year_idx} exceeds 1.0: {total_adoption:.6f}. "
                f"Per-upgrade: {dict((uid, scenario[uid][year_idx]) for uid in scenario)}"
            )


def uniform_adoption(
    upgrade_ids: list[int],
    weights: dict[int, float],
    adoption_trajectory: list[float],
) -> dict[int, list[float]]:
    """Generate a scenario from total adoption trajectory and fixed upgrade weights.

    This helper function distributes a total adoption trajectory across multiple
    upgrades according to fixed weights. For example, if 30% of buildings adopt
    in year 1, and upgrade 4 has weight 0.6, then 18% of buildings will adopt
    upgrade 4 in year 1.

    Args:
        upgrade_ids: List of upgrade IDs to include in the scenario.
        weights: Per-upgrade share of total adopters. Must sum to 1.0 (±1e-6).
            Example: {4: 0.6, 8: 0.4} means 60% choose upgrade 4, 40% choose upgrade 8.
        adoption_trajectory: Total adoption fraction per year. Must be non-decreasing.
            Example: [0.1, 0.3, 0.5] means 10%, 30%, 50% total adoption over 3 years.

    Returns:
        Scenario dict mapping upgrade IDs to per-year adoption fractions.

    Raises:
        InvalidScenarioError: If weights don't sum to 1.0, contain invalid values,
            or if adoption_trajectory is invalid.

    Example:
        >>> scenario = uniform_adoption(
        ...     upgrade_ids=[4, 8],
        ...     weights={4: 0.6, 8: 0.4},
        ...     adoption_trajectory=[0.1, 0.3, 0.5],
        ... )
        >>> scenario
        {4: [0.06, 0.18, 0.30], 8: [0.04, 0.12, 0.20]}
    """
    # Validate upgrade_ids
    if not upgrade_ids:
        raise InvalidScenarioError("upgrade_ids cannot be empty")

    for upgrade_id in upgrade_ids:
        if not isinstance(upgrade_id, int) or upgrade_id < 0:
            raise InvalidScenarioError(f"Upgrade ID must be a non-negative integer, got: {upgrade_id}")

    # Validate weights
    if set(weights.keys()) != set(upgrade_ids):
        raise InvalidScenarioError(
            f"Weights keys {set(weights.keys())} must match upgrade_ids {set(upgrade_ids)}"
        )

    for upgrade_id, weight in weights.items():
        if not 0 <= weight <= 1:
            raise InvalidScenarioError(f"Weight for upgrade {upgrade_id} must be in [0, 1], got: {weight}")

    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        raise InvalidScenarioError(f"Weights must sum to 1.0, got: {weight_sum:.6f}")

    # Validate adoption_trajectory
    if not adoption_trajectory:
        raise InvalidScenarioError("adoption_trajectory cannot be empty")

    for year_idx, adoption in enumerate(adoption_trajectory):
        if not 0 <= adoption <= 1:
            raise InvalidScenarioError(
                f"Adoption in year {year_idx} must be in [0, 1], got: {adoption}"
            )

        if year_idx > 0 and adoption < adoption_trajectory[year_idx - 1]:
            raise InvalidScenarioError(
                f"adoption_trajectory must be non-decreasing. "
                f"Year {year_idx - 1}: {adoption_trajectory[year_idx - 1]}, "
                f"Year {year_idx}: {adoption}"
            )

    # Generate scenario by multiplying trajectory by weights
    scenario: dict[int, list[float]] = {}
    for upgrade_id in upgrade_ids:
        weight = weights[upgrade_id]
        scenario[upgrade_id] = [weight * adoption for adoption in adoption_trajectory]

    # Validate the generated scenario
    validate_scenario(scenario)

    return scenario
