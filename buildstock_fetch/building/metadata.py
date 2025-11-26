from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import BuildingID


def get_metadata_url(building: "BuildingID"):
    if not building.is_file_type_available("metadata"):
        return ""
    if building.release_year == "2021":
        return f"{building.base_url}metadata/metadata.parquet"
    elif building.release_year == "2022" or building.release_year == "2023":
        if building.upgrade_id == "0":
            return f"{building.base_url}metadata/baseline.parquet"
        else:
            return f"{building.base_url}metadata/upgrade{str(int(building.upgrade_id)).zfill(2)}.parquet"
    elif building.release_year == "2024":
        if building.res_com == "comstock" and building.weather == "amy2018" and building.release_number == "2":
            if building.upgrade_id == "0":
                upgrade_filename = "baseline"
            else:
                upgrade_filename = f"upgrade{str(int(building.upgrade_id)).zfill(2)}"
            return (
                f"{building.base_url}metadata_and_annual_results/by_state_and_county/full/parquet/"
                f"state={building.state}/county={building.get_county_name()}/{building.state}_{building.get_county_name()}_{upgrade_filename}.parquet"
            )
        else:
            if building.upgrade_id == "0":
                return f"{building.base_url}metadata/baseline.parquet"
            else:
                return f"{building.base_url}metadata/upgrade{str(int(building.upgrade_id)).zfill(2)}.parquet"
    elif (
        building.release_year == "2025"
        and building.res_com == "comstock"
        and building.weather == "amy2018"
        and (building.release_number == "1" or building.release_number == "2")
    ):
        return (
            f"{building.base_url}metadata_and_annual_results/by_state_and_county/full/parquet/"
            f"state={building.state}/county={building.get_county_name()}/{building.state}_{building.get_county_name()}_upgrade{building.upgrade_id}.parquet"
        )
    else:
        return ""
