from typing import TYPE_CHECKING

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
    if (
        building.release_year == "2024"
        and building.res_com == "resstock"
        and (building.weather == "tmy3" or building.weather == "amy2018")
        and building.release_number == "2"
        and int(building.upgrade_id) >= 17
        and int(building.upgrade_id) <= 22
    ):
        bldg_id_component_list = []
        upgrade_id_int = int(building.upgrade_id)
        if upgrade_id_int == 17:
            bldg_id_component_list.append(building.copy(upgrade_id="1"))
            bldg_id_component_list.append(building.copy(upgrade_id="11"))
        elif upgrade_id_int == 18:
            bldg_id_component_list.append(building.copy(upgrade_id="2"))
            bldg_id_component_list.append(building.copy(upgrade_id="12"))
        elif upgrade_id_int == 19:
            bldg_id_component_list.append(building.copy(upgrade_id="3"))
            bldg_id_component_list.append(building.copy(upgrade_id="13"))
        elif upgrade_id_int == 20:
            bldg_id_component_list.append(building.copy(upgrade_id="4"))
            bldg_id_component_list.append(building.copy(upgrade_id="14"))
        elif upgrade_id_int == 21:
            bldg_id_component_list.append(building.copy(upgrade_id="5"))
            bldg_id_component_list.append(building.copy(upgrade_id="15"))
        elif upgrade_id_int == 22:
            bldg_id_component_list.append(building.copy(upgrade_id="0"))
            bldg_id_component_list.append(building.copy(upgrade_id="11"))
        return bldg_id_component_list
    return None
