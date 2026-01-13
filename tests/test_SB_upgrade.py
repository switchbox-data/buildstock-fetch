import json
import shutil
import sys
from pathlib import Path
from typing import Any, cast

import polars as pl
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from buildstock_fetch.building import BuildingID
from buildstock_fetch.constants import LOAD_CURVE_COLUMN_AGGREGATION, SB_ANALYSIS_UPGRADES_FILE
from buildstock_fetch.main import fetch_bldg_data
from buildstock_fetch.types import FileType, FunctionalGroup, get_args


@pytest.fixture
def SB_upgrade_load_curve_column_map():
    """Fixture that reads the load curve column map CSV file."""
    column_map_file = LOAD_CURVE_COLUMN_AGGREGATION.joinpath("data_dictionary_2024_load_curve_labeled.csv")
    return pl.read_csv(column_map_file)


@pytest.fixture
def SB_upgrade_metadata_and_annual_load_curve_column_map():
    """Fixture that reads the metadata and annual column map CSV file."""
    column_map_file = LOAD_CURVE_COLUMN_AGGREGATION.joinpath("data_dictionary_2024_metadata_and_annual_labeled.csv")
    return pl.read_csv(column_map_file)


@pytest.fixture
def SB_analysis_upgrades() -> dict[str, Any]:
    """Load SB analysis upgrades data once and cache it for subsequent calls."""
    with open(SB_ANALYSIS_UPGRADES_FILE) as f:
        SB_analysis_upgrades = json.load(f)
    return SB_analysis_upgrades


@pytest.fixture(scope="function")
def cleanup_downloads():
    """Fixture to clean up downloaded data before and after tests.

    This fixture:
    1. Removes any existing 'data' directory before the test runs
    2. Yields control to the test
    3. Removes the 'data' directory after the test completes

    This ensures each test starts with a clean slate and doesn't leave
    downloaded files behind.
    """
    # Setup - clean up any existing files before test
    data_dir = Path("data")

    if data_dir.exists():
        shutil.rmtree(data_dir)

    yield

    # Teardown - clean up downloaded files after test
    if data_dir.exists():
        shutil.rmtree(data_dir)


def test_SB_upgrade_load_curve_15min(cleanup_downloads):
    bldg_ids = [
        BuildingID(
            bldg_id=361520,
            release_number="2",
            release_year="2024",
            res_com="resstock",
            weather="tmy3",
            state="IN",
            upgrade_id="17",
        )
    ]
    file_type = ("load_curve_15min",)
    output_dir = Path("data")

    _, _ = fetch_bldg_data(bldg_ids, file_type, output_dir)
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_15min/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/{bldg_id.get_output_filename('load_curve_15min')}"
    ).exists()


def test_SB_upgrade_load_curve_hourly(cleanup_downloads):
    bldg_ids = [
        BuildingID(
            bldg_id=361520,
            release_number="2",
            release_year="2024",
            res_com="resstock",
            weather="tmy3",
            state="IN",
            upgrade_id="17",
        )
    ]
    file_type = ("load_curve_hourly",)
    output_dir = Path("data")

    _, _ = fetch_bldg_data(bldg_ids, file_type, output_dir)
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_hourly/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/{bldg_id.get_output_filename('load_curve_hourly')}"
    ).exists()


def test_SB_upgrade_load_curve_daily(cleanup_downloads):
    bldg_ids = [
        BuildingID(
            bldg_id=361520,
            release_number="2",
            release_year="2024",
            res_com="resstock",
            weather="tmy3",
            state="IN",
            upgrade_id="17",
        )
    ]
    file_type = ("load_curve_daily",)
    output_dir = Path("data")

    _, _ = fetch_bldg_data(bldg_ids, file_type, output_dir)
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_daily/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/{bldg_id.get_output_filename('load_curve_daily')}"
    ).exists()


def test_SB_upgrade_load_curve_monthly(cleanup_downloads):
    bldg_ids = [
        BuildingID(
            bldg_id=361520,
            release_number="2",
            release_year="2024",
            res_com="resstock",
            weather="tmy3",
            state="IN",
            upgrade_id="17",
        )
    ]
    file_type = ("load_curve_monthly",)
    output_dir = Path("data")

    _, _ = fetch_bldg_data(bldg_ids, file_type, output_dir)
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_monthly/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/{bldg_id.get_output_filename('load_curve_monthly')}"
    ).exists()


def test_SB_upgrade_annual_load_curve(cleanup_downloads):
    bldg_ids = [
        BuildingID(
            bldg_id=361520,
            release_number="2",
            release_year="2024",
            res_com="resstock",
            weather="tmy3",
            state="IN",
            upgrade_id="17",
        )
    ]
    file_type = ("load_curve_annual",)
    output_dir = Path("data")

    _, _ = fetch_bldg_data(bldg_ids, file_type, output_dir)
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/load_curve_annual/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/{bldg_id.get_output_filename('load_curve_annual')}"
    ).exists()


def test_SB_upgrade_metadata(cleanup_downloads):
    bldg_ids = [
        BuildingID(
            bldg_id=361520,
            release_number="2",
            release_year="2024",
            res_com="resstock",
            weather="tmy3",
            state="IN",
            upgrade_id="17",
        )
    ]
    file_type = ("metadata",)
    output_dir = Path("data")

    _, _ = fetch_bldg_data(bldg_ids, file_type, output_dir)
    bldg_id = bldg_ids[0]
    assert Path(
        f"data/{bldg_id.get_release_name()}/metadata/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/{bldg_id.get_output_filename('metadata')}"
    ).exists()


def test_SB_upgrade_column_selection(
    cleanup_downloads,
    SB_upgrade_load_curve_column_map,
    SB_upgrade_metadata_and_annual_load_curve_column_map,
    SB_analysis_upgrades,
):
    file_types = cast(
        list[FileType],
        [
            "load_curve_15min",
            "load_curve_hourly",
            "load_curve_daily",
            "load_curve_monthly",
            "metadata",
            "load_curve_annual",
        ],
    )
    SB_upgrades_2024 = ["17", "18"]
    output_dir = Path("data")
    columns_to_check = {
        "HVAC": [
            "in.hvac_system_single_speed_ac_airflow",
            "in.hvac_system_single_speed_ashp_airflow",
            "in.mechanical_ventilation",
        ],
        "HVAC_heating": [
            "in.heating_setpoint",
            "in.hvac_secondary_heating_efficiency",
            "out.electricity.heating.energy_consumption",
            "out.electricity.heating_hp_bkup.energy_consumption",
        ],
        "HVAC_cooling": [
            "in.cooling_setpoint",
            "in.hvac_cooling_efficiency",
            "out.electricity.cooling.energy_consumption",
            "out.electricity.cooling_fans_pumps.energy_consumption",
        ],
        "appliances": [
            "in.dishwasher_usage_level",
            "in.range_spot_vent_hour",
            "out.electricity.clothes_dryer.energy_consumption",
            "out.electricity.freezer.energy_consumption_intensity",
        ],
        "hot_water": [
            "in.water_heater_efficiency",
            "out.electricity.hot_water.energy_consumption",
            "out.natural_gas.hot_water.energy_consumption",
        ],
        "envelope": ["in.sqft", "in.bedrooms", "in.geometry_floor_area"],
        "demographic": ["in.income", "in.puma"],
        "metadata": ["in.simulation_control_run_period_begin_day_of_month", "in.simulation_control_timestep"],
    }

    # Checking for every file type and upgrade id above
    for file_type in file_types:
        for upgrade_id in SB_upgrades_2024:
            bldg_ids = [
                BuildingID(
                    bldg_id=361520,
                    release_number="2",
                    release_year="2024",
                    res_com="resstock",
                    weather="tmy3",
                    state="IN",
                    upgrade_id=upgrade_id,
                )
            ]
            _, _ = fetch_bldg_data(bldg_ids, (file_type,), output_dir)
            bldg_id = bldg_ids[0]
            final_bldg_df = pl.read_parquet(
                Path(
                    f"data/{bldg_id.get_release_name()}/{file_type}/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/{bldg_id.get_output_filename(file_type)}"
                )
            )
            final_bldg_columns = final_bldg_df.columns

            component_bldg_ids = bldg_id.get_SB_upgrade_component_bldg_ids()
            assert component_bldg_ids is not None
            _, _ = fetch_bldg_data(component_bldg_ids, (file_type,), output_dir)

            SB_analysis_upgrade_data = SB_analysis_upgrades[bldg_id.get_release_name()]

            column_map = (
                SB_upgrade_load_curve_column_map
                if file_type == "load_curve_15min"
                or file_type == "load_curve_hourly"
                or file_type == "load_curve_daily"
                or file_type == "load_curve_monthly"
                else SB_upgrade_metadata_and_annual_load_curve_column_map
            )

            for functional_group in get_args(FunctionalGroup):
                if functional_group in ["total", "net"] or functional_group not in SB_analysis_upgrade_data:
                    continue
                functional_group_upgrade_id = SB_analysis_upgrade_data[functional_group][bldg_id.upgrade_id]
                component_bldg_id = bldg_id.copy(upgrade_id=functional_group_upgrade_id)
                component_bldg_df = pl.read_parquet(
                    Path(
                        f"data/{component_bldg_id.get_release_name()}/{file_type}/state={component_bldg_id.state}/upgrade={str(int(component_bldg_id.upgrade_id)).zfill(2)}/{component_bldg_id.get_output_filename(file_type)}"
                    )
                )
                functional_group_column_names = (
                    column_map.filter(pl.col("functional_group") == functional_group)
                    .select("field_name")
                    .to_series()
                    .to_list()
                )
                functional_group_columns_to_check = []
                for column in functional_group_column_names:
                    if column in columns_to_check[functional_group] and column in final_bldg_columns:
                        functional_group_columns_to_check.append(column)
                if len(functional_group_columns_to_check) == 0:
                    continue
                assert verify_equal_columns(final_bldg_df, component_bldg_df, functional_group_columns_to_check)

            assert Path(
                f"data/{bldg_id.get_release_name()}/{file_type}/state={bldg_id.state}/upgrade={str(int(bldg_id.upgrade_id)).zfill(2)}/{bldg_id.get_output_filename(file_type)}"
            ).exists()


def verify_equal_columns(
    final_bldg_df: pl.DataFrame, component_bldg_df: pl.DataFrame, columns_to_check: list[str]
) -> bool:
    for column in columns_to_check:
        final_column = final_bldg_df[column].to_list()
        component_column = component_bldg_df[column].to_list()
        try:
            final_column_values = [float(value) for value in final_column]
            component_column_values = [float(value) for value in component_column]
            difference = [
                final_value - component_value
                for final_value, component_value in zip(final_column_values, component_column_values)
            ]
            if abs(sum(difference)) < 0.1:
                continue
            else:
                return False
        except ValueError:
            if final_column == component_column:
                continue
            else:
                return False
    return True
