import pytest

from utils.ev_demand import EVDemandConfig
from utils.ev_utils import (
    assign_income_midpoints,
    assign_nhts_income_bucket,
    get_census_division_for_state,
)


@pytest.fixture
def test_config():
    return EVDemandConfig(state="NY", release="resstock_tmy3_release_1")


def test_get_census_division_for_state():
    assert get_census_division_for_state("NY") == 2
    assert get_census_division_for_state("CA") == 9
    assert get_census_division_for_state("TX") == 7
    assert get_census_division_for_state("FL") == 5
    assert get_census_division_for_state("IL") == 3
    assert get_census_division_for_state("OH") == 3


def test_assign_nhts_income_bucket():
    """Test the assign_nhts_income_bucket function with key income values."""

    # Test 4 key cases covering different buckets
    assert assign_nhts_income_bucket(5000) == 1  # Low income
    assert assign_nhts_income_bucket(50000) == 6  # Middle income
    assert assign_nhts_income_bucket(150000) == 10  # High income
    assert assign_nhts_income_bucket(250000) == 11  # Very high income


def test_assign_income_midpoints():
    """Test the assign_income_midpoints function with various income range strings."""

    # Test 4 key cases covering different scenarios
    assert assign_income_midpoints("60000-69999") == 64999  # Range midpoint
    assert assign_income_midpoints("0-10000") == 5000  # Low range
    assert assign_income_midpoints("200000") == 200000  # Single value
    assert assign_income_midpoints(None) is None  # None input
