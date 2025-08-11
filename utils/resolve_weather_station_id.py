import os
import tempfile
import time
import zipfile

import polars as pl
import requests
import xmltodict
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn

from buildstock_fetch.main import BuildingID, fetch_bldg_ids


class NoXMLFileError(ValueError):
    """Raised when no XML file is found in the zip file."""

    pass


class NoWeatherStationNameError(ValueError):
    """Raised when no weather station name is found in the XML file."""

    pass


def _check_xml_files_exist(xml_files):
    """Check if XML files exist in the zip file."""
    if not xml_files:
        raise NoXMLFileError()


def _check_weather_station_found(weather_station_name):
    """Check if weather station name was found in XML."""
    if not weather_station_name:
        raise NoWeatherStationNameError()


def resolve_weather_station_id(
    product: str, release_year: str, weather_file: str, release_version: str, state: str, upgrade_id: str
) -> pl.DataFrame:
    bldg_ids = fetch_bldg_ids(product, release_year, weather_file, release_version, state, upgrade_id)

    # Prepare data for DataFrame
    data = []
    total_buildings = len(bldg_ids[:10])

    print(f"Processing {total_buildings} building IDs...")

    # Create rich progress bar with time tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("[progress.description]{task.fields[current]}"),
        TextColumn("[progress.description]{task.fields[weather_station]}"),
        expand=True,
    ) as progress:
        task = progress.add_task("Processing buildings", total=total_buildings, current="", weather_station="")

        for bldg_id in bldg_ids[:10]:
            weather_station_name = download_and_extract_weather_station(bldg_id)

            data.append({
                "bldg_id": bldg_id.bldg_id,
                "product": product,
                "release_year": release_year,
                "weather_file": weather_file,
                "release_version": release_version,
                "state": state,
                "upgrade_id": upgrade_id,
                "weather_station_name": weather_station_name,
            })

            # Update progress bar
            progress.update(
                task,
                advance=1,
                current=f"Current: {bldg_id.bldg_id}",
                weather_station=f"Weather: {weather_station_name}",
            )

    # Create Polars DataFrame
    df = pl.DataFrame(data)
    return df


def download_and_extract_weather_station(bldg_id: BuildingID) -> str:
    """
    Download building data zip file, extract XML file, process it, and clean up temporary files.

    Args:
        bldg_id: BuildingID object containing the data URL

    Returns:
        str: Processed content from the XML file, or empty string if failed
    """
    building_data_url = bldg_id.get_building_data_url()
    if building_data_url == "":
        return ""

    # Create temporary directory for all temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Download the zip file to a temporary file
            response = requests.get(building_data_url, timeout=30)
            response.raise_for_status()

            # Save zip file to temporary location
            zip_temp_path = os.path.join(temp_dir, "building_data.zip")
            with open(zip_temp_path, "wb") as zip_file:
                zip_file.write(response.content)

            # Extract XML file from zip
            xml_temp_path = os.path.join(temp_dir, "building.xml")
            with zipfile.ZipFile(zip_temp_path, "r") as zip_ref:
                # Find the first XML file in the zip
                xml_files = [f for f in zip_ref.namelist() if f.endswith(".xml")]
                _check_xml_files_exist(xml_files)

                # Extract the first (and only) XML file
                xml_filename = xml_files[0]
                with zip_ref.open(xml_filename) as xml_file, open(xml_temp_path, "wb") as temp_xml:
                    temp_xml.write(xml_file.read())

            # Read and process the XML file
            with open(xml_temp_path, encoding="utf-8") as xml_file:
                xml_content = xml_file.read()

            # Process the XML content here
            processed_content = extract_weather_station_name(xml_content)

        except Exception as e:
            print(f"Error processing building data: {e}")
            return ""
        else:
            return processed_content


def extract_weather_station_name(xml_content: str) -> str:
    """
    Process the XML content extracted from the building data file.

    Args:
        xml_content: Raw XML content as string

    Returns:
        str: Weather station name or empty string if not found
    """
    try:
        buildingXML = xmltodict.parse(xml_content)

        # Search for weather station name in the XML file
        weather_station_name = find_weather_station_name(buildingXML)
        _check_weather_station_found(weather_station_name)

    except Exception as e:
        print(f"Error parsing XML content: {e}")
        return ""

    else:
        return weather_station_name


def _extract_name_from_weather_station(weather_station):
    """Extract name from weather station data."""
    if isinstance(weather_station, dict) and "Name" in weather_station:
        return weather_station["Name"]
    elif isinstance(weather_station, list):
        for item in weather_station:
            if isinstance(item, dict) and "Name" in item:
                return item["Name"]
    return None


def _search_dict_for_weather_station(data, path=""):
    """Search dictionary for weather station."""
    if "WeatherStation" in data:
        weather_station = data["WeatherStation"]
        name = _extract_name_from_weather_station(weather_station)
        if name:
            return name

    # Recursively search all values in the dict
    for key, value in data.items():
        result = find_weather_station_name(value, f"{path}.{key}" if path else key)
        if result:
            return result
    return None


def _search_list_for_weather_station(data, path=""):
    """Search list for weather station."""
    for i, item in enumerate(data):
        result = find_weather_station_name(item, f"{path}[{i}]" if path else f"[{i}]")
        if result:
            return result
    return None


def find_weather_station_name(data, path=""):
    """
    Recursively search for WeatherStation tag and extract the Name value.

    Args:
        data: The data structure to search (dict, list, or primitive)
        path: Current path for debugging (optional)

    Returns:
        str: Weather station name if found, None otherwise
    """
    if isinstance(data, dict):
        return _search_dict_for_weather_station(data, path)
    elif isinstance(data, list):
        return _search_list_for_weather_station(data, path)

    return None


if __name__ == "__main__":
    start_time = time.time()
    product = "resstock"
    release_year = "2022"
    weather_file = "amy2012"
    release_version = "1"
    state = "NY"
    upgrade_id = "0"

    df = resolve_weather_station_id(product, release_year, weather_file, release_version, state, upgrade_id)

    # Create output directory if it doesn't exist
    output_dir = "buildstock_fetch/data/weather_station_map"
    os.makedirs(output_dir, exist_ok=True)

    # Sort by building ID for better organization
    if "bldg_id" in df.columns:
        df = df.sort("bldg_id")

    # Save as partitioned parquet file
    output_path = os.path.join(output_dir, "weather_station_map.parquet")
    df.write_parquet(
        str(output_path),  # Convert Path to string for Polars
        use_pyarrow=True,
        partition_by=["product", "release_year", "weather_file", "release_version", "state"],
    )
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"Successfully saved partitioned weather station mapping to {output_path}")
    print(f"\nDataFrame shape: {df.shape}")
