import os
import tempfile
import zipfile

import requests

from buildstock_fetch.main import BuildingID, fetch_bldg_ids


def resolve_weather_station_id(
    product: str, release_year: str, weather_file: str, release_version: str, state: str, upgrade_id: str
) -> str:
    bldg_ids = fetch_bldg_ids(product, release_year, weather_file, release_version, state, upgrade_id)
    return bldg_ids


def download_and_process_temporary_hpxml(bldg_id: BuildingID) -> str:
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
                if not xml_files:
                    print("No XML files found in zip")
                    return ""

                # Extract the first (and only) XML file
                xml_filename = xml_files[0]
                with zip_ref.open(xml_filename) as xml_file, open(xml_temp_path, "wb") as temp_xml:
                    temp_xml.write(xml_file.read())

            # Read and process the XML file
            with open(xml_temp_path, encoding="utf-8") as xml_file:
                xml_content = xml_file.read()

            # Process the XML content here
            processed_content = process_xml_content(xml_content)

        except Exception as e:
            print(f"Error processing building data: {e}")
            return ""
        else:
            return processed_content


def process_xml_content(xml_content: str) -> str:
    """
    Process the XML content extracted from the building data file.

    Args:
        xml_content: Raw XML content as string

    Returns:
        str: Processed content or extracted information
    """
    # Add your XML processing logic here
    # For example, you might want to parse the XML and extract specific data

    # For now, just return the content length as a placeholder
    # You can replace this with actual XML processing
    return f"XML content length: {len(xml_content)} characters"


if __name__ == "__main__":
    product = "resstock"
    release_year = "2022"
    weather_file = "amy2012"
    release_version = "1"
    state = "NY"
    upgrade_id = "0"

    bldg_ids = fetch_bldg_ids(product, release_year, weather_file, release_version, state, upgrade_id)
    print(bldg_ids[:10])
