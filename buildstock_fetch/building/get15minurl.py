import json
from typing import TYPE_CHECKING

from buildstock_fetch.constants import SB_ANALYSIS_UPGRADES_FILE

if TYPE_CHECKING:
    from . import BuildingID


def get_15min_load_curve_url(building: "BuildingID") -> str | None:
    """Generate the S3 download URL for this building."""
    if not building._validate_requested_file_type_availability("load_curve_15min"):
        return None
    if building.release_year == "2021":
        if building.upgrade_id != "0":
            return None  # This release only has baseline timeseries

        else:
            return (
                f"{building.base_url}timeseries_individual_buildings/"
                f"by_state/upgrade={building.upgrade_id}/"
                f"state={building.state}/"
                f"{building.bldg_id!s}-{int(building.upgrade_id)!s}.parquet"
            )

    elif building.release_year == "2022" or building.release_year == "2023":
        return (
            f"{building.base_url}timeseries_individual_buildings/"
            f"by_state/upgrade={building.upgrade_id}/"
            f"state={building.state}/"
            f"{building.bldg_id!s}-{int(building.upgrade_id)!s}.parquet"
        )
    elif building.release_year == "2024":
        if building.res_com == "resstock" and building.weather == "tmy3" and building.release_number == "1":
            return None

        else:
            return (
                f"{building.base_url}timeseries_individual_buildings/"
                f"by_state/upgrade={building.upgrade_id}/"
                f"state={building.state}/"
                f"{building.bldg_id!s}-{int(building.upgrade_id)!s}.parquet"
            )
    elif building.release_year == "2025":
        return (
            f"{building.base_url}timeseries_individual_buildings/"
            f"by_state/upgrade={building.upgrade_id}/"
            f"state={building.state}/"
            f"{building.bldg_id!s}-{int(building.upgrade_id)!s}.parquet"
        )
    else:
        return None


def get_SB_upgrade_load_component_bldg_ids(building: "BuildingID") -> list["BuildingID"] | None:
    with open(SB_ANALYSIS_UPGRADES_FILE) as f:
        sb_analysis_upgrades = json.load(f)
    release_name = building.get_release_name()
    if release_name not in sb_analysis_upgrades:
        return None
    sb_analysis_upgrade_data = sb_analysis_upgrades[release_name]
    upgrade_components = sb_analysis_upgrade_data["upgrade_components"][building.upgrade_id]
    bldg_id_component_list = []
    for component_id in upgrade_components:
        bldg_id_component_list.append(building.copy(upgrade_id=component_id))
    return bldg_id_component_list
