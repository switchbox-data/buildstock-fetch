from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import BuildingID


def get_building_data_url(building: "BuildingID") -> str | None:
    """Generate the S3 download URL for this building."""
    if not building._validate_requested_file_type_availability(
        "hpxml"
    ) or not building._validate_requested_file_type_availability("schedule"):
        return None
    if building.release_year == "2021" or building.release_year == "2023":
        return None
    if building.release_year == "2022":
        return _get_building_data_url_2022(building)

    if building.release_year == "2024":
        return _get_building_data_url_2024(building)
    if building.release_year == "2025":
        return _get_building_data_url_2025(building)
    return None


def _get_building_data_url_2024(building: "BuildingID") -> str | None:
    """Get building data URL for 2024 releases."""
    if building.res_com == "comstock":
        return None
    if (building.weather == "amy2018" or building.weather == "tmy3") and building.release_number == "2":
        return (
            f"{building.base_url}"
            f"model_and_schedule_files/building_energy_models/upgrade={building.upgrade_id}/"
            f"bldg{str(building.bldg_id).zfill(7)}-up{str(int(building.upgrade_id)).zfill(2)}.zip"
        )
    return None


def _get_building_data_url_2025_upgrade_string(building: "BuildingID") -> str:
    """Get upgrade string for 2025 building data URLs."""
    if building.res_com == "resstock" and building.weather == "amy2018" and building.release_number == "1":
        return str(int(building.upgrade_id))
    if building.res_com == "comstock" and (
        (building.weather == "amy2012" and building.release_number == "2")
        or (building.weather == "amy2018" and building.release_number == "1")
        or (building.weather == "amy2018" and building.release_number == "2")
    ):
        return str(int(building.upgrade_id)).zfill(2)
    return str(int(building.upgrade_id)).zfill(2)


def _get_building_data_url_2025(building: "BuildingID") -> str | None:
    """Get building data URL for 2025 releases."""
    if (building.res_com == "resstock" and building.weather == "amy2012" and building.release_number == 1) or (
        building.res_com == "comstock" and building.weather == "amy2018" and building.release_number == 3
    ):
        return None
    upgrade_string = _get_building_data_url_2025_upgrade_string(building)
    if building.res_com == "resstock":
        return (
            f"{building.base_url}"
            f"building_energy_models/upgrade={upgrade_string}/"
            f"bldg{str(building.bldg_id).zfill(7)}-up{str(int(building.upgrade_id)).zfill(2)}.zip"
        )
    else:
        return (
            f"{building.base_url}"
            f"building_energy_models/upgrade={upgrade_string}/"
            f"bldg{str(building.bldg_id).zfill(7)}-up{str(int(building.upgrade_id)).zfill(2)}.osm.gz"
        )


def _get_building_data_url_2022(building: "BuildingID") -> str:
    """Get building data URL for 2022 releases."""
    return (
        f"{building.base_url}"
        f"building_energy_models/upgrade={building.upgrade_id}/"
        f"bldg{str(building.bldg_id).zfill(7)}-up{str(int(building.upgrade_id)).zfill(2)}.zip"
    )
