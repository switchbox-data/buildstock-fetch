from pathlib import Path

import polars as pl
import pytest
import typer

from buildstock_fetch.cli.main import BuildingsGroup, filter_building_groups_by_id, read_bldg_ids_file
from buildstock_fetch.main_new import list_buildings
from buildstock_fetch.types import normalize_upgrade_id


def test_read_csv_uses_bldg_id_column(tmp_path: Path) -> None:
    path = tmp_path / "ids.csv"
    pl.DataFrame({"bldg_id": [3, 1, 3, None], "county": ["a", "b", "c", "d"]}).write_csv(path)
    assert read_bldg_ids_file(path) == {1, 3}


def test_read_parquet_uses_bldg_id_column(tmp_path: Path) -> None:
    path = tmp_path / "ids.parquet"
    pl.DataFrame({"bldg_id": [10, 20], "other": [0.5, 0.7]}).write_parquet(path)
    assert read_bldg_ids_file(path) == {10, 20}


def test_read_text_one_id_per_line(tmp_path: Path) -> None:
    path = tmp_path / "ids.txt"
    path.write_text("# building IDs\n5\n\n 7 \n5\n")
    assert read_bldg_ids_file(path) == {5, 7}


def test_read_csv_without_bldg_id_column_fails(tmp_path: Path) -> None:
    path = tmp_path / "ids.csv"
    pl.DataFrame({"id": [1, 2]}).write_csv(path)
    with pytest.raises(typer.BadParameter, match="no 'bldg_id' column"):
        read_bldg_ids_file(path)


def test_read_non_integer_ids_fails(tmp_path: Path) -> None:
    path = tmp_path / "ids.txt"
    path.write_text("1\nabc\n")
    with pytest.raises(typer.BadParameter):
        read_bldg_ids_file(path)


def test_read_empty_file_fails(tmp_path: Path) -> None:
    path = tmp_path / "ids.txt"
    path.write_text("# nothing here\n")
    with pytest.raises(typer.BadParameter, match="no building IDs"):
        read_bldg_ids_file(path)


def test_filter_keeps_only_requested_ids_in_every_group() -> None:
    upgrade = normalize_upgrade_id("0")
    buildings = list_buildings("res_2024_amy2018_2", "CT", upgrade, limit=5)
    groups = [BuildingsGroup("CT", upgrade, buildings)]
    wanted = {buildings[1].id, buildings[3].id}

    filtered = filter_building_groups_by_id(groups, wanted)

    assert [building.id for building in filtered[0].buildings] == [buildings[1].id, buildings[3].id]
    assert filtered[0].state == "CT"
    assert filtered[0].upgrade_id == upgrade


def test_filter_warns_about_missing_ids(capsys: pytest.CaptureFixture[str]) -> None:
    upgrade = normalize_upgrade_id("0")
    buildings = list_buildings("res_2024_amy2018_2", "CT", upgrade, limit=5)
    groups = [BuildingsGroup("CT", upgrade, buildings)]
    missing_id = max(building.id for building in buildings) + 10_000_000

    filtered = filter_building_groups_by_id(groups, {buildings[0].id, missing_id})

    assert [building.id for building in filtered[0].buildings] == [buildings[0].id]
    assert "1 of 2 building IDs were not found" in capsys.readouterr().out
