from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import BuildingID


def get_annual_load_curve_filename(building: "BuildingID") -> str | None:
    """Generate the filename for the annual load curve."""

    if building.release_year == "2021":
        return None

    if building.release_year == "2022" or building.release_year == "2023":
        return _get_annual_load_curve_filename_2022_2023(building)
    if building.release_year == "2024":
        return _get_annual_load_curve_filename_2024(building)
    if building.release_year == "2025":
        return _get_annual_load_curve_filename_2025(building)
    return None


def _get_annual_load_curve_filename_2022_2023(building: "BuildingID") -> str:
    """Get annual load curve filename for 2022 or 2023 releases."""

    return f"{building.state}_upgrade{str(int(building.upgrade_id)).zfill(2)}_metadata_and_annual_results.parquet"


def _get_annual_load_curve_filename_2024(building: "BuildingID") -> str | None:
    """Get annual load curve filename for 2024 releases."""
    if building.res_com == "comstock" and building.weather == "amy2018" and building.release_number == "2":
        county = building._get_county_name()
        if not county:
            return None
        return f"{building.state}_{county}_upgrade{str(int(building.upgrade_id)).zfill(2)}_metadata_and_annual_results.parquet"
    if building.res_com == "resstock" and building.weather == "tmy3" and building.release_number == "1":
        return None

    return f"{building.state}_upgrade{str(int(building.upgrade_id)).zfill(2)}_metadata_and_annual_results.parquet"


def _get_annual_load_curve_filename_2025(building: "BuildingID") -> str | None:
    """Get annual load curve filename for 2025 releases."""
    if building.res_com == "comstock":
        county = building._get_county_name()
        if not county:
            return None
        return f"{building.state}_{county}_upgrade{int(building.upgrade_id)!s}.parquet"
    if building.res_com == "resstock":
        return f"{building.state}_upgrade{int(building.upgrade_id)!s}.parquet"

    return None
