from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import BuildingID


def get_15min_load_curve_url(building: "BuildingID") -> str | None:
    if not building.is_file_type_available("load_curve_15min"):
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
