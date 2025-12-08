from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import BuildingID


def get_annual_load_curve_url(building: "BuildingID") -> str | None:
    """Generate the S3 download URL for this building."""

    if not building._validate_requested_file_type_availability("load_curve_annual"):
        return None
    if building.release_year == "2021":
        return None
    elif building.release_year == "2022" or building.release_year == "2023":
        return _build_annual_load_state_url(building)
    elif building.release_year == "2024":
        return _handle_2024_release_annual_load(building)
    elif building.release_year == "2025":
        return _handle_2025_release_annual_load(building)
    else:
        return None


def _handle_2024_release_annual_load(building: "BuildingID") -> str | None:
    """Handle the 2024 release logic for annual load curve URLs.


    Returns:

        The constructed URL or empty string if not applicable.
    """
    if building.res_com == "comstock" and building.weather == "amy2018" and building.release_number == "2":
        county = building._get_county_name()
        if not county:
            return None
        if building.upgrade_id == "0":
            return (
                f"{building.base_url}metadata_and_annual_results/"
                f"by_state_and_county/full/parquet/"
                f"state={building.state}/county={county}/"
                f"{building.state}_{county}_baseline.parquet"
            )
        else:
            return (
                f"{building.base_url}metadata_and_annual_results/"
                f"by_state_and_county/full/parquet/"
                f"state={building.state}/county={county}/"
                f"{building.state}_{county}_upgrade{str(int(building.upgrade_id)).zfill(2)}.parquet"
            )
    elif building.res_com == "resstock" and building.weather == "tmy3" and building.release_number == "1":
        return None  # This release has a different structure. Need further development
    else:
        return _build_annual_load_state_url(building)


def _build_annual_load_state_url(building: "BuildingID") -> str:
    """Build the state-level URL for annual load curve data.

    Returns:
        The constructed URL for the state-level data.
    """
    if building.upgrade_id == "0":
        return (
            f"{building.base_url}metadata_and_annual_results/"
            f"by_state/state={building.state}/parquet/"
            f"{building.state}_baseline_metadata_and_annual_results.parquet"
        )
    else:
        return (
            f"{building.base_url}metadata_and_annual_results/"
            f"by_state/state={building.state}/parquet/"
            f"{building.state}_upgrade{str(int(building.upgrade_id)).zfill(2)}_metadata_and_annual_results.parquet"
        )


def _handle_2025_release_annual_load(building: "BuildingID") -> str | None:
    """Get load curve annual URL for 2025 releases."""
    if building.res_com == "comstock":
        return (
            f"{building.base_url}metadata_and_annual_results/by_state_and_county/full/parquet/"
            f"state={building.state}/county={building._get_county_name()}/{building.state}_{building._get_county_name()}_upgrade{building.upgrade_id}.parquet"
        )
    elif building.res_com == "resstock":
        return (
            f"{building.base_url}metadata_and_annual_results/by_state/full/parquet/"
            f"state={building.state}/{building.state}_upgrade{building.upgrade_id}.parquet"
        )
    return None
