from buildstock_fetch.cli.main import _validate_sample, select_buildings_for_sample
from buildstock_fetch.main_new import list_buildings
from buildstock_fetch.types import normalize_upgrade_id


def test_sample_zero_maps_to_all_and_includes_last_building() -> None:
    """--sample 0 must request every building in the group, including the last row."""
    buildings = list_buildings("res_2024_amy2018_2", "CT", normalize_upgrade_id("0"), limit=5)
    assert len(buildings) == 5

    sample = _validate_sample(0)
    assert sample == "all"

    selected = select_buildings_for_sample(buildings, sample)
    assert len(selected) == len(buildings)
    assert selected[-1] == buildings[-1]
    assert {building.id for building in selected} == {building.id for building in buildings}


def test_sample_n_limits_request_list() -> None:
    buildings = list_buildings("res_2024_amy2018_2", "CT", normalize_upgrade_id("0"), limit=5)
    selected = select_buildings_for_sample(buildings, 3)
    assert selected == buildings[:3]
