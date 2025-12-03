from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import BuildingID


def get_weather_file_url(building: "BuildingID") -> str | None:
    """Generate the S3 download URL for this building."""
    if building.get_weather_station_name() == "":
        return None
    return _build_weather_url(building)


def _build_weather_url(building: "BuildingID") -> str | None:
    """Build the weather file URL based on release year and weather type."""
    if building.release_year == "2021":
        return _build_2021_weather_url(building)
    elif building.release_year == "2022":
        return _build_2022_weather_url(building)
    elif building.release_year == "2023":
        return _build_2023_weather_url(building)
    elif building.release_year == "2024":
        return _build_2024_weather_url(building)
    elif building.release_year == "2025":
        return _build_2025_weather_url(building)
    else:
        return None


def _build_2021_weather_url(building: "BuildingID") -> str | None:
    """Build weather URL for 2021 release."""
    if building.weather == "tmy3":
        return f"{building.base_url}weather/{building.weather}/{building.get_weather_station_name()}_tmy3.csv"
    elif building.weather == "amy2018":
        return f"{building.base_url}weather/{building.weather}/{building.get_weather_station_name()}_2018.csv"
    elif building.weather == "amy2012":
        return f"{building.base_url}weather/{building.weather}/{building.get_weather_station_name()}_2012.csv"
    else:
        return None


def _build_2022_weather_url(building: "BuildingID") -> str | None:
    """Build weather URL for 2022 release."""
    if building.weather == "tmy3":
        return f"{building.base_url}weather/state={building.state}/{building.get_weather_station_name()}_TMY3.csv"
    elif building.weather == "amy2018":
        return f"{building.base_url}weather/state={building.state}/{building.get_weather_station_name()}_2018.csv"
    elif building.weather == "amy2012":
        return f"{building.base_url}weather/state={building.state}/{building.get_weather_station_name()}_2012.csv"
    else:
        return None


def _build_2023_weather_url(building: "BuildingID") -> str | None:
    """Build weather URL for 2023 release."""
    if building.weather == "tmy3":
        return f"{building.base_url}weather/{building.weather}/{building.get_weather_station_name()}_TMY3.csv"
    elif building.weather == "amy2018":
        return f"{building.base_url}weather/{building.weather}/{building.get_weather_station_name()}_2018.csv"
    elif building.weather == "amy2012":
        return f"{building.base_url}weather/{building.weather}/{building.get_weather_station_name()}_2012.csv"
    else:
        return None


def _build_2024_weather_url(building: "BuildingID") -> str | None:
    """Build weather URL for 2024 release."""
    if building.res_com == "comstock" and building.weather == "amy2018":
        return f"{building.base_url}weather/{building.weather}/{building.get_weather_station_name()}_2018.csv"
    else:
        if building.weather == "tmy3":
            return f"{building.base_url}weather/state={building.state}/{building.get_weather_station_name()}_TMY3.csv"
        elif building.weather == "amy2018":
            return f"{building.base_url}weather/state={building.state}/{building.get_weather_station_name()}_2018.csv"
        elif building.weather == "amy2012":
            return f"{building.base_url}weather/state={building.state}/{building.get_weather_station_name()}_2012.csv"
        else:
            return None


def _build_2025_weather_url(building: "BuildingID") -> str | None:
    """Build weather URL for 2025 release."""
    if building.weather == "tmy3":
        return f"{building.base_url}weather/{building.weather}/{building.get_weather_station_name()}_TMY3.csv"
    elif building.weather == "amy2018":
        return f"{building.base_url}weather/{building.weather}/{building.get_weather_station_name()}_2018.csv"
    elif building.weather == "amy2012":
        return f"{building.base_url}weather/{building.weather}/{building.get_weather_station_name()}_2012.csv"
    else:
        return None
