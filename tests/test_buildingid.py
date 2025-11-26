import json
from pathlib import Path

import pytest

from buildstock_fetch.building import BuildingID


def _load_parameters_from_file():
    return json.loads(Path(__file__).parent.joinpath("data", "buildings_example_data.json").read_text())


def _get_parameters() -> tuple[tuple[str, ...], tuple]:
    return (
        (
            "bldg_id",
            "release_number",
            "release_year",
            "res_com",
            "weather",
            "upgrade_id",
            "state",
            "expected",
            "method_name",
        ),
        tuple(
            (
                _["bldg_id"],
                _["release_number"],
                _["release_year"],
                _["res_com"],
                _["weather"],
                _["upgrade_id"],
                _["state"],
                _["results"][method_name],
                method_name,
            )
            for _ in _load_parameters_from_file()
            for method_name in _["results"]
        ),
    )


@pytest.mark.parametrize(*_get_parameters())
def test_building_function(
    bldg_id,
    release_number,
    release_year,
    res_com,
    weather,
    upgrade_id,
    state,
    expected,
    method_name,
):
    building = BuildingID(
        bldg_id,
        release_number,
        release_year,
        res_com,
        weather,
        upgrade_id,
        state,
    )
    method = getattr(building, method_name)
    result = method()

    assert result == expected
