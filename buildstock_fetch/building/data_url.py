from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import BuildingID


def get_building_data_url(building: "BuildingID") -> str | None:
    """Generate the S3 download URL for this building."""
    release_year = building.release_year

    if not all(map(building.is_file_type_available, ("hpxml", "schedule"))):
        return None
    if release_year in ("2021", "2023"):
        return None
    if release_year == "2022":
        return _get_building_data_url_2022(building)
    if release_year == "2024":
        return _get_building_data_url_2024(building)
    if release_year == "2025":
        return _get_building_data_url_2025(building)
    return None


def _get_building_data_url_2022(building: "BuildingID") -> str:
    return (
        f"{building.base_url}"
        f"building_energy_models/upgrade={building.upgrade_id}/"
        f"bldg{str(building.bldg_id).zfill(7)}-up{str(int(building.upgrade_id)).zfill(2)}.zip"
    )


def _get_building_data_url_2024(building: "BuildingID") -> str | None:
    if building.res_com == "comstock":
        return None
    if building.weather in ("amy2018", "tmy3") and building.release_number == "2":
        return (
            f"{building.base_url}"
            f"model_and_schedule_files/building_energy_models/upgrade={building.upgrade_id}/"
            f"bldg{str(building.bldg_id).zfill(7)}-up{str(int(building.upgrade_id)).zfill(2)}.zip"
        )
    return None


def _get_building_data_url_2025(building: "BuildingID") -> str:
    return (
        f"{building.base_url}"
        f"building_energy_models/upgrade={str(int(building.upgrade_id)).zfill(2)}/"
        f"bldg{str(building.bldg_id).zfill(7)}-up{str(int(building.upgrade_id)).zfill(2)}.zip"
    )
