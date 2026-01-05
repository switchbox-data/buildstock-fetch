import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import BuildingID

from buildstock_fetch.constants import SB_ANALYSIS_UPGRADES_FILE

# Module-level cache for SB analysis upgrades data
_SB_ANALYSIS_UPGRADES_CACHE: dict[str, Any] | None = None


def _get_SB_analysis_upgrades() -> dict[str, Any] | None:
    """Load SB analysis upgrades data once and cache it for subsequent calls."""
    global _SB_ANALYSIS_UPGRADES_CACHE
    if _SB_ANALYSIS_UPGRADES_CACHE is None:
        with open(SB_ANALYSIS_UPGRADES_FILE) as f:
            _SB_ANALYSIS_UPGRADES_CACHE = json.load(f)
    return _SB_ANALYSIS_UPGRADES_CACHE


def get_metadata_url(building: "BuildingID") -> str | list[str] | None:
    """Generate the S3 download URL for this building."""

    if not building._validate_requested_file_type_availability("metadata"):
        return None
    # SB created upgrade scenarios
    if building.is_SB_upgrade():
        return _get_metadata_url_SB_upgrade(building)
    # Regular release upgrades by year
    if building.release_year == "2021":
        return _get_metadata_url_2021(building)
    if building.release_year == "2022" or building.release_year == "2023":
        return _get_metadata_url_2022_2023(building)
    if building.release_year == "2024":
        return _get_metadata_url_2024(building)
    if building.release_year == "2025":
        return _get_metadata_url_2025(building)
    return None


def _get_metadata_url_SB_upgrade(building: "BuildingID") -> list[str]:
    """Get metadata URL for SB upgrades."""
    sb_analysis_upgrades = _get_SB_analysis_upgrades()
    if sb_analysis_upgrades is None:
        msg = "SB analysis upgrades data not available"
        raise ValueError(msg)
    release_name = building.get_release_name()
    sb_analysis_upgrade_data = sb_analysis_upgrades[release_name]
    upgrade_components = sb_analysis_upgrade_data["upgrade_components"][building.upgrade_id]
    metadata_urls: list[str] = []
    for component_id in upgrade_components:
        component_building = building.copy(upgrade_id=component_id)
        component_metadata_url = component_building.get_metadata_url()
        if component_metadata_url is None:
            msg = f"Metadata URL not available for {component_building.get_release_name()}, upgrade {component_building.upgrade_id}"
            raise ValueError(msg)
        if isinstance(component_metadata_url, list):
            metadata_urls.extend(component_metadata_url)
        else:
            metadata_urls.append(component_metadata_url)
    return metadata_urls


def _get_metadata_url_2021(building: "BuildingID") -> str:
    """Get metadata URL for 2021 releases."""

    return f"{building.base_url}metadata/metadata.parquet"


def _get_metadata_url_2022_2023(building: "BuildingID") -> str:
    """Get metadata URL for 2022 or 2023 releases."""

    if building.upgrade_id == "0":
        return f"{building.base_url}metadata/baseline.parquet"
    return f"{building.base_url}metadata/upgrade{str(int(building.upgrade_id)).zfill(2)}.parquet"


def _get_metadata_url_2024_comstock_amy2018_v2(building: "BuildingID") -> str:
    """Get metadata URL for 2024 comstock amy2018 release version 2."""
    upgrade_filename = "baseline" if building.upgrade_id == "0" else f"upgrade{str(int(building.upgrade_id)).zfill(2)}"
    return (
        f"{building.base_url}metadata_and_annual_results/by_state_and_county/full/parquet/"
        f"state={building.state}/county={building._get_county_name()}/{building.state}_{building._get_county_name()}_{upgrade_filename}.parquet"
    )


def _get_metadata_url_2024(building: "BuildingID") -> str:
    """Get metadata URL for 2024 releases."""
    if building.res_com == "comstock" and building.weather == "amy2018" and building.release_number == "2":
        return _get_metadata_url_2024_comstock_amy2018_v2(building)
    if building.upgrade_id == "0":
        return f"{building.base_url}metadata/baseline.parquet"
    return f"{building.base_url}metadata/upgrade{str(int(building.upgrade_id)).zfill(2)}.parquet"


def _get_metadata_url_2025(building: "BuildingID") -> str | None:
    """Get metadata URL for 2025 releases."""
    if building.res_com == "comstock":
        return (
            f"{building.base_url}metadata_and_annual_results/by_state_and_county/full/parquet/"
            f"state={building.state}/county={building._get_county_name()}/{building.state}_{building._get_county_name()}_upgrade{building.upgrade_id}.parquet"
        )
    if building.res_com == "resstock":
        return (
            f"{building.base_url}metadata_and_annual_results/by_state/full/parquet/"
            f"state={building.state}/{building.state}_upgrade{building.upgrade_id}.parquet"
        )
    return None
