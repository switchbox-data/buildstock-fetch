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


def get_SB_upgrade_15min_load_curve_url(building: "BuildingID") -> list[str] | None:
    if (
        building.release_year == "2024"
        and building.res_com == "resstock"
        and (building.weather == "tmy3" or building.weather == "amy2018")
        and building.release_number == "2"
        and int(building.upgrade_id) >= 17
        and int(building.upgrade_id) <= 22
    ):
        upgrade_id_mix = []
        upgrade_id_int = int(building.upgrade_id)
        if upgrade_id_int == 17:
            upgrade_id_mix.append(1)
            upgrade_id_mix.append(11)
        elif upgrade_id_int == 18:
            upgrade_id_mix.append(2)
            upgrade_id_mix.append(12)
        elif upgrade_id_int == 19:
            upgrade_id_mix.append(3)
            upgrade_id_mix.append(13)
        elif upgrade_id_int == 20:
            upgrade_id_mix.append(4)
            upgrade_id_mix.append(14)
        elif upgrade_id_int == 21:
            upgrade_id_mix.append(5)
            upgrade_id_mix.append(15)
        elif upgrade_id_int == 22:
            upgrade_id_mix.append(0)
            upgrade_id_mix.append(11)
        download_url_list = []
        for upgrade_id in upgrade_id_mix:
            download_url_list.append(
                f"{building.base_url}timeseries_individual_buildings/"
                f"by_state/upgrade={upgrade_id}/"
                f"state={building.state}/"
                f"{building.bldg_id!s}-{upgrade_id!s}.parquet"
            )
        return download_url_list
    return None
