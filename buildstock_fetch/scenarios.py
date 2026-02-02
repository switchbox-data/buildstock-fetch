"""Scenario definition and validation for mixed upgrade adoption trajectories.

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

from typing_extensions import final, override


@final
class InvalidScenarioError(ValueError):
    """Base class for scenario validation errors."""

    pass


@final
class EmptyScenarioError(InvalidScenarioError):
    """Raised when scenario dict is empty."""

    @override
    def __str__(self) -> str:
        return "Scenario cannot be empty"


@final
class InvalidUpgradeIdError(InvalidScenarioError):
    """Raised when an upgrade ID is not a non-negative integer."""

    def __init__(self, upgrade_id: int) -> None:
        self.upgrade_id = upgrade_id
        super().__init__()

    @override
    def __str__(self) -> str:
        return f"Upgrade ID must be a non-negative integer, got: {self.upgrade_id}"


@final
class InconsistentListLengthsError(InvalidScenarioError):
    """Raised when adoption lists have different lengths."""

    def __init__(self, lengths: dict[int, int]) -> None:
        self.lengths = lengths
        super().__init__()

    @override
    def __str__(self) -> str:
        return f"All upgrade adoption lists must have the same length. Got lengths: {self.lengths}"


@final
class EmptyYearListError(InvalidScenarioError):
    """Raised when scenario has zero years."""

    @override
    def __str__(self) -> str:
        return "Scenario must have at least one year"


@final
class AdoptionFractionOutOfRangeError(InvalidScenarioError):
    """Raised when adoption fraction is not in [0, 1]."""

    def __init__(self, upgrade_id: int, year_idx: int, fraction: float) -> None:
        self.upgrade_id = upgrade_id
        self.year_idx = year_idx
        self.fraction = fraction
        super().__init__()

    @override
    def __str__(self) -> str:
        return (
            f"Adoption fraction for upgrade {self.upgrade_id} in year {self.year_idx} "
            f"must be in [0, 1], got: {self.fraction}"
        )


@final
class NonMonotonicAdoptionError(InvalidScenarioError):
    """Raised when adoption fractions decrease over time."""

    def __init__(self, upgrade_id: int, year_idx: int, prev_fraction: float, curr_fraction: float) -> None:
        self.upgrade_id = upgrade_id
        self.year_idx = year_idx
        self.prev_fraction = prev_fraction
        self.curr_fraction = curr_fraction
        super().__init__()

    @override
    def __str__(self) -> str:
        return (
            f"Adoption fractions for upgrade {self.upgrade_id} must be non-decreasing. "
            f"Year {self.year_idx - 1}: {self.prev_fraction}, Year {self.year_idx}: {self.curr_fraction}"
        )


@final
class TotalAdoptionExceedsOneError(InvalidScenarioError):
    """Raised when total adoption in a year exceeds 1.0."""

    def __init__(self, year_idx: int, total_adoption: float, per_upgrade: dict[int, float]) -> None:
        self.year_idx = year_idx
        self.total_adoption = total_adoption
        self.per_upgrade = per_upgrade
        super().__init__()

    @override
    def __str__(self) -> str:
        return (
            f"Total adoption in year {self.year_idx} exceeds 1.0: {self.total_adoption:.6f}. "
            f"Per-upgrade: {self.per_upgrade}"
        )


@final
class EmptyUpgradeIdsError(InvalidScenarioError):
    """Raised when upgrade_ids list is empty."""

    @override
    def __str__(self) -> str:
        return "upgrade_ids cannot be empty"


@final
class WeightsKeysMismatchError(InvalidScenarioError):
    """Raised when weights keys don't match upgrade_ids."""

    def __init__(self, weight_keys: set[int], upgrade_ids: set[int]) -> None:
        self.weight_keys = weight_keys
        self.upgrade_ids = upgrade_ids
        super().__init__()

    @override
    def __str__(self) -> str:
        return f"Weights keys {self.weight_keys} must match upgrade_ids {self.upgrade_ids}"


@final
class WeightOutOfRangeError(InvalidScenarioError):
    """Raised when a weight is not in [0, 1]."""

    def __init__(self, upgrade_id: int, weight: float) -> None:
        self.upgrade_id = upgrade_id
        self.weight = weight
        super().__init__()

    @override
    def __str__(self) -> str:
        return f"Weight for upgrade {self.upgrade_id} must be in [0, 1], got: {self.weight}"


@final
class WeightsSumError(InvalidScenarioError):
    """Raised when weights don't sum to 1.0."""

    def __init__(self, weight_sum: float) -> None:
        self.weight_sum = weight_sum
        super().__init__()

    @override
    def __str__(self) -> str:
        return f"Weights must sum to 1.0, got: {self.weight_sum:.6f}"


@final
class EmptyAdoptionTrajectoryError(InvalidScenarioError):
    """Raised when adoption_trajectory is empty."""

    @override
    def __str__(self) -> str:
        return "adoption_trajectory cannot be empty"


@final
class AdoptionOutOfRangeError(InvalidScenarioError):
    """Raised when adoption value is not in [0, 1]."""

    def __init__(self, year_idx: int, adoption: float) -> None:
        self.year_idx = year_idx
        self.adoption = adoption
        super().__init__()

    @override
    def __str__(self) -> str:
        return f"Adoption in year {self.year_idx} must be in [0, 1], got: {self.adoption}"


@final
class NonMonotonicTrajectoryError(InvalidScenarioError):
    """Raised when adoption_trajectory is not non-decreasing."""

    def __init__(self, year_idx: int, prev_adoption: float, curr_adoption: float) -> None:
        self.year_idx = year_idx
        self.prev_adoption = prev_adoption
        self.curr_adoption = curr_adoption
        super().__init__()

    @override
    def __str__(self) -> str:
        return (
            f"adoption_trajectory must be non-decreasing. "
            f"Year {self.year_idx - 1}: {self.prev_adoption}, "
            f"Year {self.year_idx}: {self.curr_adoption}"
        )


def _validate_upgrade_ids(scenario: dict[int, list[float]]) -> None:
    """Validate that all upgrade IDs are non-negative integers."""
    for upgrade_id in scenario:
        if not isinstance(upgrade_id, int) or upgrade_id < 0:
            raise InvalidUpgradeIdError(upgrade_id)


def _validate_list_lengths(scenario: dict[int, list[float]]) -> int:
    """Validate that all adoption lists have the same length and return num_years."""
    num_years_list = [len(fractions) for fractions in scenario.values()]
    if len(set(num_years_list)) > 1:
        raise InconsistentListLengthsError(dict(zip(scenario.keys(), num_years_list)))

    num_years = num_years_list[0]
    if num_years == 0:
        raise EmptyYearListError()

    return num_years


def _validate_fractions_and_monotonicity(scenario: dict[int, list[float]]) -> None:
    """Validate that fractions are in [0, 1] and non-decreasing per upgrade."""
    for upgrade_id, fractions in scenario.items():
        for year_idx, fraction in enumerate(fractions):
            # Check fraction is in [0, 1]
            if not 0 <= fraction <= 1:
                raise AdoptionFractionOutOfRangeError(upgrade_id, year_idx, fraction)

            # Check monotonicity (non-decreasing)
            if year_idx > 0 and fraction < fractions[year_idx - 1]:
                raise NonMonotonicAdoptionError(upgrade_id, year_idx, fractions[year_idx - 1], fraction)


def _validate_total_adoption(scenario: dict[int, list[float]], num_years: int) -> None:
    """Validate that total adoption per year does not exceed 1.0."""
    for year_idx in range(num_years):
        total_adoption = sum(scenario[upgrade_id][year_idx] for upgrade_id in scenario)
        # Allow small floating point tolerance
        if total_adoption > 1.0 + 1e-6:
            per_upgrade = {uid: scenario[uid][year_idx] for uid in scenario}
            raise TotalAdoptionExceedsOneError(year_idx, total_adoption, per_upgrade)


def validate_scenario(scenario: dict[int, list[float]]) -> None:
    """Validate a mixed upgrade scenario definition.

    A valid scenario must satisfy:
    - All upgrade IDs are non-negative integers
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
        raise EmptyScenarioError()

    _validate_upgrade_ids(scenario)
    num_years = _validate_list_lengths(scenario)
    _validate_fractions_and_monotonicity(scenario)
    _validate_total_adoption(scenario, num_years)


def _validate_upgrade_ids_param(upgrade_ids: list[int]) -> None:
    """Validate upgrade_ids parameter."""
    if not upgrade_ids:
        raise EmptyUpgradeIdsError()

    for upgrade_id in upgrade_ids:
        if not isinstance(upgrade_id, int) or upgrade_id < 0:
            raise InvalidUpgradeIdError(upgrade_id)


def _validate_weights_param(weights: dict[int, float], upgrade_ids: list[int]) -> None:
    """Validate weights parameter."""
    if set(weights.keys()) != set(upgrade_ids):
        raise WeightsKeysMismatchError(set(weights.keys()), set(upgrade_ids))

    for upgrade_id, weight in weights.items():
        if not 0 <= weight <= 1:
            raise WeightOutOfRangeError(upgrade_id, weight)

    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        raise WeightsSumError(weight_sum)


def _validate_adoption_trajectory(adoption_trajectory: list[float]) -> None:
    """Validate adoption_trajectory parameter."""
    if not adoption_trajectory:
        raise EmptyAdoptionTrajectoryError()

    for year_idx, adoption in enumerate(adoption_trajectory):
        if not 0 <= adoption <= 1:
            raise AdoptionOutOfRangeError(year_idx, adoption)

        if year_idx > 0 and adoption < adoption_trajectory[year_idx - 1]:
            raise NonMonotonicTrajectoryError(year_idx, adoption_trajectory[year_idx - 1], adoption)


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
    # Validate all parameters
    _validate_upgrade_ids_param(upgrade_ids)
    _validate_weights_param(weights, upgrade_ids)
    _validate_adoption_trajectory(adoption_trajectory)

    # Generate scenario by multiplying trajectory by weights
    scenario: dict[int, list[float]] = {}
    for upgrade_id in upgrade_ids:
        weight = weights[upgrade_id]
        scenario[upgrade_id] = [weight * adoption for adoption in adoption_trajectory]

    # Validate the generated scenario
    validate_scenario(scenario)

    return scenario
