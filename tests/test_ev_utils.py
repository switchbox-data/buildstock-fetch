import pytest

from buildstock_fetch.ev_demand import EVDemandConfig
from buildstock_fetch.utils.ev_utils import (
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


# # THESE ARE COMMENTED OUT BECAUSE I HAVE NOT SET UP TESTING DATA YET - TESTS ARE HELPING WITH DEV
# def test_load_metadata(test_config):
#     metadata_df = load_metadata(test_config.metadata_path)

#     assert len(metadata_df) > 0
#     assert "bldg_id" in metadata_df.columns
#     assert "weight" in metadata_df.columns
#     assert "metro" in metadata_df.columns
#     assert "puma" in metadata_df.columns
#     assert "income" in metadata_df.columns
#     assert "occupants" in metadata_df.columns
#     assert "hhsize_bins" in metadata_df.columns


# def test_load_nhts_data(test_config):
#     print(test_config.nhts_path)
#     nhts_df = load_nhts_data(test_config.nhts_path, test_config.state)

#     assert len(nhts_df) > 0
#     assert "vehicle_id" in nhts_df.columns
#     assert "start_time" in nhts_df.columns
#     assert "end_time" in nhts_df.columns
#     assert "miles_driven" in nhts_df.columns
#     assert "weekday" in nhts_df.columns
#     assert "occupants" in nhts_df.columns
#     assert "income" in nhts_df.columns
#     assert "vehicles" in nhts_df.columns
#     assert "urban" in nhts_df.columns


# def test_load_pums_data(test_config):
#     pums_df = load_pums_data(test_config.pums_path, test_config.metadata_path)

#     assert len(pums_df) > 0
#     assert "income" in pums_df.columns
#     assert "occupants" in pums_df.columns
#     assert "vehicles" in pums_df.columns
#     assert "puma" in pums_df.columns
#     assert "hh_weight" in pums_df.columns
#     assert "metro" in pums_df.columns


# def test_load_metro_puma_map(test_config):
#     metro_pums_df = load_metro_puma_map(test_config.metadata_path)

#     assert len(metro_pums_df) > 0
#     assert "metro" in metro_pums_df.columns
#     assert "puma" in metro_pums_df.columns
#     # Check that PUMA values are 5 characters (after str.slice(-5))
#     assert all(metro_pums_df["puma"].str.len_chars() == 5)
