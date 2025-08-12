import concurrent.futures
import os
import tempfile
import threading
import time
import zipfile
from collections import defaultdict

import polars as pl
import requests
import urllib3
import xmltodict
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn

from buildstock_fetch.main import BuildingID, fetch_bldg_ids

# Disable SSL warnings for cleaner output
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Create a session with connection pooling and SSL settings
session = requests.Session()
session.verify = False  # Disable SSL verification for problematic connections
session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=3))


class NoXMLFileError(ValueError):
    """Raised when no XML file is found in the zip file."""

    pass


class NoWeatherStationNameError(ValueError):
    """Raised when no weather station name is found in the XML file."""

    pass


class ProfilingData:
    """Class to collect and manage profiling data for the three main tasks."""

    def __init__(self):
        self.download_times = []
        self.extract_times = []
        self.process_times = []
        self.total_times = []
        self.error_counts = defaultdict(int)
        self.last_update_time = time.time()
        self.update_interval = 10  # Update every 10 seconds
        self.lock = threading.Lock()  # Thread-safe lock for data updates

    def add_timing(self, download_time, extract_time, process_time, total_time, error_type=None):
        """Add timing data for one building processing (thread-safe)."""
        with self.lock:
            self.download_times.append(download_time)
            self.extract_times.append(extract_time)
            self.process_times.append(process_time)
            self.total_times.append(total_time)

            if error_type:
                self.error_counts[error_type] += 1

            # Check if it's time for a status update (simplified to reduce lock time)
            current_time = time.time()
            if current_time - self.last_update_time >= self.update_interval:
                self.last_update_time = current_time
                # Don't print status update inside the lock - do it outside
                return True  # Signal that we should print status update
        return False  # No status update needed

    def get_current_summary(self):
        """Get summary statistics for current timing data (thread-safe)."""
        with self.lock:
            if not self.total_times:
                return "No timing data available"

            summary = {
                "total_buildings": len(self.total_times),
                "download": {
                    "total": sum(self.download_times),
                    "avg": sum(self.download_times) / len(self.download_times),
                    "min": min(self.download_times),
                    "max": max(self.download_times),
                    "pct_of_total": (sum(self.download_times) / sum(self.total_times)) * 100,
                },
                "extract": {
                    "total": sum(self.extract_times),
                    "avg": sum(self.extract_times) / len(self.extract_times),
                    "min": min(self.extract_times),
                    "max": max(self.extract_times),
                    "pct_of_total": (sum(self.extract_times) / sum(self.total_times)) * 100,
                },
                "process": {
                    "total": sum(self.process_times),
                    "avg": sum(self.process_times) / len(self.process_times),
                    "min": min(self.process_times),
                    "max": max(self.process_times),
                    "pct_of_total": (sum(self.process_times) / sum(self.total_times)) * 100,
                },
                "total_time": sum(self.total_times),
                "errors": dict(self.error_counts),
            }
            return summary

    def print_status_update(self):
        """Print a real-time status update of the profiling data."""
        summary = self.get_current_summary()
        if isinstance(summary, str):
            return

        print("\n" + "=" * 60)
        print(f"REAL-TIME PROFILING STATUS - {time.strftime('%H:%M:%S')}")
        print("=" * 60)
        print(f"Buildings processed: {summary['total_buildings']}")
        print(f"Total time elapsed: {summary['total_time']:.2f} seconds")
        print(f"Average time per building: {summary['total_time'] / summary['total_buildings']:.3f} seconds")
        print()

        print("CURRENT TASK BREAKDOWN:")
        print("-" * 40)
        print(
            f"Download:  {summary['download']['total']:.2f}s total, {summary['download']['avg']:.3f}s avg, {summary['download']['pct_of_total']:.1f}% of total"
        )
        print(
            f"Extract:   {summary['extract']['total']:.2f}s total, {summary['extract']['avg']:.3f}s avg, {summary['extract']['pct_of_total']:.1f}% of total"
        )
        print(
            f"Process:   {summary['process']['total']:.2f}s total, {summary['process']['avg']:.3f}s avg, {summary['process']['pct_of_total']:.1f}% of total"
        )
        print()

        print("TIMING RANGES:")
        print("-" * 40)
        print(f"Download:  {summary['download']['min']:.3f}s - {summary['download']['max']:.3f}s")
        print(f"Extract:   {summary['extract']['min']:.3f}s - {summary['extract']['max']:.3f}s")
        print(f"Process:   {summary['process']['min']:.3f}s - {summary['process']['max']:.3f}s")
        print()

        if summary["errors"]:
            print("ERRORS SO FAR:")
            print("-" * 40)
            for error_type, count in summary["errors"].items():
                print(f"{error_type}: {count}")
        print("=" * 60)

    def get_summary(self):
        """Get summary statistics for all timing data (thread-safe)."""
        with self.lock:
            if not self.total_times:
                return "No timing data available"

            summary = {
                "total_buildings": len(self.total_times),
                "download": {
                    "total": sum(self.download_times),
                    "avg": sum(self.download_times) / len(self.download_times),
                    "min": min(self.download_times),
                    "max": max(self.download_times),
                    "pct_of_total": (sum(self.download_times) / sum(self.total_times)) * 100,
                },
                "extract": {
                    "total": sum(self.extract_times),
                    "avg": sum(self.extract_times) / len(self.extract_times),
                    "min": min(self.extract_times),
                    "max": max(self.extract_times),
                    "pct_of_total": (sum(self.extract_times) / sum(self.total_times)) * 100,
                },
                "process": {
                    "total": sum(self.process_times),
                    "avg": sum(self.process_times) / len(self.process_times),
                    "min": min(self.process_times),
                    "max": max(self.process_times),
                    "pct_of_total": (sum(self.process_times) / sum(self.total_times)) * 100,
                },
                "total_time": sum(self.total_times),
                "errors": dict(self.error_counts),
            }
            return summary

    def print_summary(self):
        """Print a formatted summary of the profiling data."""
        summary = self.get_summary()
        if isinstance(summary, str):
            print(summary)
            return

        print("\n" + "=" * 60)
        print("FINAL PROFILING SUMMARY")
        print("=" * 60)
        print(f"Total buildings processed: {summary['total_buildings']}")
        print(f"Total time: {summary['total_time']:.2f} seconds")
        print()

        print("FINAL TASK BREAKDOWN:")
        print("-" * 40)
        print(
            f"Download:  {summary['download']['total']:.2f}s total, {summary['download']['avg']:.3f}s avg, {summary['download']['pct_of_total']:.1f}% of total"
        )
        print(
            f"Extract:   {summary['extract']['total']:.2f}s total, {summary['extract']['avg']:.3f}s avg, {summary['extract']['pct_of_total']:.1f}% of total"
        )
        print(
            f"Process:   {summary['process']['total']:.2f}s total, {summary['process']['avg']:.3f}s avg, {summary['process']['pct_of_total']:.1f}% of total"
        )
        print()

        print("FINAL TIMING RANGES:")
        print("-" * 40)
        print(f"Download:  {summary['download']['min']:.3f}s - {summary['download']['max']:.3f}s")
        print(f"Extract:   {summary['extract']['min']:.3f}s - {summary['extract']['max']:.3f}s")
        print(f"Process:   {summary['process']['min']:.3f}s - {summary['process']['max']:.3f}s")
        print()

        if summary["errors"]:
            print("TOTAL ERRORS:")
            print("-" * 40)
            for error_type, count in summary["errors"].items():
                print(f"{error_type}: {count}")
        print("=" * 60)


# Global profiling data instance
profiling_data = ProfilingData()


def _check_xml_files_exist(xml_files):
    """Check if XML files exist in the zip file."""
    if not xml_files:
        raise NoXMLFileError()


def _check_weather_station_found(weather_station_name):
    """Check if weather station name was found in XML."""
    if not weather_station_name:
        raise NoWeatherStationNameError()


def resolve_weather_station_id(
    product: str,
    release_year: str,
    weather_file: str,
    release_version: str,
    state: str,
    upgrade_id: str,
    status_update_interval: int = 10,
    progress_update_interval: int = 50,
    max_workers: int = 5,  # Reduced from 10 to 5 to prevent server overload
) -> pl.DataFrame:
    global profiling_data
    profiling_data = ProfilingData()  # Reset profiling data
    profiling_data.update_interval = status_update_interval  # Set custom update interval

    bldg_ids = fetch_bldg_ids(product, release_year, weather_file, release_version, state, upgrade_id)

    # Prepare data for DataFrame
    data = []
    failed_bldg_ids = []  # Track failed buildings for retry
    total_buildings = len(bldg_ids)

    print(f"Processing {total_buildings} building IDs with {max_workers} parallel workers...")
    print(f"Status updates every {status_update_interval} seconds or every {progress_update_interval} buildings")
    print("=" * 60)

    # Phase 1: Process all buildings, collect failures
    print("PHASE 1: Initial processing...")
    data, failed_bldg_ids = process_buildings_phase(
        bldg_ids,
        product,
        release_year,
        weather_file,
        release_version,
        state,
        upgrade_id,
        max_workers,
        status_update_interval,
        progress_update_interval,
    )

    # Phase 2: Retry failed buildings
    if failed_bldg_ids:
        print(f"\nPHASE 2: Retrying {len(failed_bldg_ids)} failed buildings...")
        print("=" * 60)

        # Reset profiling for retry phase
        profiling_data.update_interval = status_update_interval

        retry_data, still_failed = process_buildings_phase(
            failed_bldg_ids,
            product,
            release_year,
            weather_file,
            release_version,
            state,
            upgrade_id,
            max_workers // 2,  # Use fewer workers for retries to be more conservative
            status_update_interval,
            progress_update_interval,
        )

        # Add retry results to main data
        data.extend(retry_data)

        if still_failed:
            print(f"\nWARNING: {len(still_failed)} buildings still failed after retry")
            # Add empty records for permanently failed buildings
            for bldg_id in still_failed:
                data.append({
                    "bldg_id": bldg_id.bldg_id,
                    "product": product,
                    "release_year": release_year,
                    "weather_file": weather_file,
                    "release_version": release_version,
                    "state": state,
                    "upgrade_id": upgrade_id,
                    "weather_station_name": "",
                })
    else:
        print("\nNo failed buildings to retry!")

    # Create Polars DataFrame
    df = pl.DataFrame(data)

    # Print final profiling summary
    profiling_data.print_summary()

    return df


def process_buildings_phase(
    bldg_ids,
    product,
    release_year,
    weather_file,
    release_version,
    state,
    upgrade_id,
    max_workers,
    status_update_interval,
    progress_update_interval,
):
    """Process a batch of buildings and return results and failed buildings."""
    data = []
    failed_bldg_ids = []
    total_buildings = len(bldg_ids)

    print(f"Starting phase with {total_buildings} buildings, {max_workers} workers")

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

        start_time = time.time()
        completed_count = 0

        # Process buildings in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            print(f"ThreadPoolExecutor created with {max_workers} workers")

            # Submit all tasks
            future_to_bldg = {
                executor.submit(
                    process_single_building,
                    bldg_id,
                    product,
                    release_year,
                    weather_file,
                    release_version,
                    state,
                    upgrade_id,
                ): bldg_id
                for bldg_id in bldg_ids
            }

            print(f"Submitted {len(future_to_bldg)} tasks to executor")

            # Process completed futures with timeout
            try:
                for future in concurrent.futures.as_completed(future_to_bldg):  # Remove overall timeout
                    bldg_id = future_to_bldg[future]
                    completed_count += 1

                    try:
                        result = future.result(timeout=30)  # 30 second timeout per result
                        data.append(result)
                    except concurrent.futures.TimeoutError:
                        print(f"Timeout processing building {bldg_id.bldg_id}")
                        failed_bldg_ids.append(bldg_id)
                    except Exception as e:
                        print(f"Error processing building {bldg_id.bldg_id}: {e}")
                        # Collect failed building for retry
                        failed_bldg_ids.append(bldg_id)

                    elapsed_time = time.time() - start_time
                    avg_time_per_building = elapsed_time / completed_count if completed_count > 0 else 0

                    # Update progress bar
                    progress.update(
                        task,
                        advance=1,
                        current=f"Current: {bldg_id.bldg_id}",
                        weather_station=f"Weather: {result.get('weather_station_name', 'ERROR') if 'result' in locals() else 'ERROR'}",
                    )

                    # Show progress count and average time in description
                    progress.update(
                        task,
                        description=f"Processing buildings ({completed_count}/{total_buildings}) - Avg: {avg_time_per_building:.4f}s/building",
                    )

                    # Force status update every N buildings (in addition to time-based updates)
                    if completed_count % progress_update_interval == 0:
                        profiling_data.print_status_update()

            except KeyboardInterrupt:
                print("\nINTERRUPTED: Cancelling remaining tasks...")
                # Cancel remaining futures
                for future in future_to_bldg:
                    if not future.done():
                        future.cancel()
                        bldg_id = future_to_bldg[future]
                        failed_bldg_ids.append(bldg_id)
                        print(f"Cancelled building {bldg_id.bldg_id}")

    print(f"Phase completed: {len(data)} successful, {len(failed_bldg_ids)} failed")
    return data, failed_bldg_ids


def process_single_building(
    bldg_id: BuildingID,
    product: str,
    release_year: str,
    weather_file: str,
    release_version: str,
    state: str,
    upgrade_id: str,
) -> dict:
    """
    Process a single building and return the result dictionary.
    This function is designed to be called in parallel.
    """
    weather_station_name = download_and_extract_weather_station(bldg_id)

    return {
        "bldg_id": bldg_id.bldg_id,
        "product": product,
        "release_year": release_year,
        "weather_file": weather_file,
        "release_version": release_version,
        "state": state,
        "upgrade_id": upgrade_id,
        "weather_station_name": weather_station_name,
    }


def download_with_retry(url: str, max_retries: int = 3, base_delay: float = 1.0) -> requests.Response:
    """
    Download with retry logic and exponential backoff.

    Args:
        url: URL to download
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff

    Returns:
        requests.Response: Successful response

    Raises:
        requests.RequestException: If all retries fail
    """
    # Add a small delay to be respectful to the server
    time.sleep(0.1)  # 100ms delay between requests

    for attempt in range(max_retries + 1):
        try:
            # Use the session with SSL settings
            response = session.get(url, timeout=30)
            response.raise_for_status()
        except requests.ConnectionError as e:
            if attempt == max_retries:
                print(f"Connection error after {max_retries + 1} attempts: {e}")
                raise
            # Use time.time() for jitter instead of random.uniform
            jitter = time.time() % 1.0  # Get fractional part of current time
            delay = base_delay * (2**attempt) + jitter
            print(f"Connection error (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s...")
            time.sleep(delay)
        except requests.Timeout as e:
            if attempt == max_retries:
                print(f"Timeout error after {max_retries + 1} attempts: {e}")
                raise
            # Use time.time() for jitter instead of random.uniform
            jitter = time.time() % 1.0  # Get fractional part of current time
            delay = base_delay * (2**attempt) + jitter
            print(f"Timeout error (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s...")
            time.sleep(delay)
        except (requests.RequestException, requests.SSLError) as e:
            if attempt == max_retries:
                print(f"Request/SSL error after {max_retries + 1} attempts: {e}")
                raise
            # Use time.time() for jitter instead of random.uniform
            jitter = time.time() % 1.0  # Get fractional part of current time
            delay = base_delay * (2**attempt) + jitter
            print(f"Request/SSL error (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s...")
            time.sleep(delay)
        else:
            return response


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
        download_start = time.time()
        extract_start = None
        process_start = None
        error_type = None

        try:
            # TASK 1: Download the zip file to a temporary file
            response = download_with_retry(building_data_url)

            # Save zip file to temporary location
            zip_temp_path = os.path.join(temp_dir, "building_data.zip")
            with open(zip_temp_path, "wb") as zip_file:
                zip_file.write(response.content)

            download_time = time.time() - download_start
            extract_start = time.time()

            # TASK 2: Extract XML file from zip
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

            extract_time = time.time() - extract_start
            process_start = time.time()

            # TASK 3: Process the XML content here
            processed_content = extract_weather_station_name(xml_content)

            process_time = time.time() - process_start
            total_time = download_time + extract_time + process_time

            # Record timing data
            profiling_data.add_timing(download_time, extract_time, process_time, total_time)

        except requests.RequestException as e:
            error_type = "download_error"
            print(f"Download error for building {bldg_id.bldg_id}: {e}")
            return ""
        except NoXMLFileError as e:
            error_type = "no_xml_error"
            print(f"No XML file found for building {bldg_id.bldg_id}: {e}")
            return ""
        except Exception as e:
            error_type = "processing_error"
            print(f"Error processing building {bldg_id.bldg_id}: {e}")
            return ""
        else:
            return processed_content

        finally:
            # Record timing data even for errors
            if extract_start is None:
                download_time = time.time() - download_start
                extract_time = 0
                process_time = 0
            elif process_start is None:
                download_time = time.time() - download_start
                extract_time = time.time() - extract_start
                process_time = 0
            else:
                download_time = time.time() - download_start
                extract_time = time.time() - extract_start
                process_time = time.time() - process_start

            total_time = download_time + extract_time + process_time
            profiling_data.add_timing(download_time, extract_time, process_time, total_time, error_type)


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

    # You can customize the update intervals and parallel workers:
    # status_update_interval: How often to print status updates (in seconds)
    # progress_update_interval: Force status update every N buildings
    # max_workers: Number of parallel threads for downloading
    df = resolve_weather_station_id(
        product,
        release_year,
        weather_file,
        release_version,
        state,
        upgrade_id,
        status_update_interval=15,  # Update every 15 seconds
        progress_update_interval=25,  # Also update every 25 buildings
        max_workers=20,  # Use 20 parallel threads for downloading
    )

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
