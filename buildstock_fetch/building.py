import json
from dataclasses import asdict, dataclass
from typing import Literal, TypeAlias

import polars as pl

from .constants import METADATA_DIR, RELEASE_JSON_FILE, WEATHER_FILE_DIR

__all__ = [
    "BuildingID",
]

ReleaseYear: TypeAlias = Literal["2021", "2022", "2023", "2024", "2025"]
ResCom: TypeAlias = Literal["resstock", "comstock"]
Weather: TypeAlias = Literal["amy2018", "tmy3"]
weather_map_df = pl.read_parquet(WEATHER_FILE_DIR)


@dataclass
class BuildingID:
    bldg_id: int
    release_number: str = "1"
    release_year: ReleaseYear = "2022"
    res_com: ResCom = "resstock"
    weather: Weather = "tmy3"
    upgrade_id: str = "0"
    state: str = "NY"

    @property
    def base_url(self) -> str:
        if (
            self.release_year == "2024"
            and self.res_com == "resstock"
            and self.weather == "tmy3"
            and self.release_number == "1"
        ):
            return (
                f"https://oedi-data-lake.s3.amazonaws.com/"
                "nrel-pds-building-stock/"
                "end-use-load-profiles-for-us-building-stock/"
                f"{self.release_year}/"
                f"{self.res_com}_dataset_{self.release_year}.{self.release_number}/"
                f"{self.res_com}_{self.weather}/"
            )
        else:
            return (
                f"https://oedi-data-lake.s3.amazonaws.com/"
                "nrel-pds-building-stock/"
                "end-use-load-profiles-for-us-building-stock/"
                f"{self.release_year}/"
                f"{self.res_com}_{self.weather}_release_{self.release_number}/"
            )

    def _validate_requested_file_type_availability(self, file_type: str) -> bool:
        """Validate the requested file type is available for this release."""
        with open(RELEASE_JSON_FILE) as f:
            releases_data = json.load(f)
        release_name = self.get_release_name()
        if release_name not in releases_data:
            return False
        release_data = releases_data[release_name]
        return file_type in release_data["available_data"]

    def get_building_data_url(self) -> str:
        """Generate the S3 download URL for this building."""
        if not self._validate_requested_file_type_availability(
            "hpxml"
        ) or not self._validate_requested_file_type_availability("schedule"):
            return ""
        if self.release_year == "2021":
            return ""
        elif self.release_year == "2022":
            return (
                f"{self.base_url}"
                f"building_energy_models/upgrade={self.upgrade_id}/"
                f"bldg{str(self.bldg_id).zfill(7)}-up{str(int(self.upgrade_id)).zfill(2)}.zip"
            )
        elif self.release_year == "2023":
            return ""
        elif self.release_year == "2024":
            if self.res_com == "comstock":
                return ""
            elif (self.weather == "amy2018" or self.weather == "tmy3") and self.release_number == "2":
                return (
                    f"{self.base_url}"
                    f"model_and_schedule_files/building_energy_models/upgrade={self.upgrade_id}/"
                    f"bldg{str(self.bldg_id).zfill(7)}-up{str(int(self.upgrade_id)).zfill(2)}.zip"
                )
            else:
                return ""
        elif self.release_year == "2025":
            return (
                f"{self.base_url}"
                f"building_energy_models/upgrade={str(int(self.upgrade_id)).zfill(2)}/"
                f"bldg{str(self.bldg_id).zfill(7)}-up{str(int(self.upgrade_id)).zfill(2)}.zip"
            )
        else:
            return ""

    def get_metadata_url(self) -> str:
        """Generate the S3 download URL for this building."""
        if not self._validate_requested_file_type_availability("metadata"):
            return ""
        if self.release_year == "2021":
            return f"{self.base_url}metadata/metadata.parquet"
        elif self.release_year == "2022" or self.release_year == "2023":
            if self.upgrade_id == "0":
                return f"{self.base_url}metadata/baseline.parquet"
            else:
                return f"{self.base_url}metadata/upgrade{str(int(self.upgrade_id)).zfill(2)}.parquet"
        elif self.release_year == "2024":
            if self.res_com == "comstock" and self.weather == "amy2018" and self.release_number == "2":
                if self.upgrade_id == "0":
                    upgrade_filename = "baseline"
                else:
                    upgrade_filename = f"upgrade{str(int(self.upgrade_id)).zfill(2)}"
                return (
                    f"{self.base_url}metadata_and_annual_results/by_state_and_county/full/parquet/"
                    f"state={self.state}/county={self._get_county_name()}/{self.state}_{self._get_county_name()}_{upgrade_filename}.parquet"
                )
            else:
                if self.upgrade_id == "0":
                    return f"{self.base_url}metadata/baseline.parquet"
                else:
                    return f"{self.base_url}metadata/upgrade{str(int(self.upgrade_id)).zfill(2)}.parquet"
        elif (
            self.release_year == "2025"
            and self.res_com == "comstock"
            and self.weather == "amy2018"
            and (self.release_number == "1" or self.release_number == "2")
        ):
            return (
                f"{self.base_url}metadata_and_annual_results/by_state_and_county/full/parquet/"
                f"state={self.state}/county={self._get_county_name()}/{self.state}_{self._get_county_name()}_upgrade{self.upgrade_id}.parquet"
            )
        else:
            return ""

    def get_15min_load_curve_url(self) -> str:
        """Generate the S3 download URL for this building."""
        if not self._validate_requested_file_type_availability("load_curve_15min"):
            return ""
        if self.release_year == "2021":
            if self.upgrade_id != "0":
                return ""  # This release only has baseline timeseries
            else:
                return (
                    f"{self.base_url}timeseries_individual_buildings/"
                    f"by_state/upgrade={self.upgrade_id}/"
                    f"state={self.state}/"
                    f"{self.bldg_id!s}-{int(self.upgrade_id)!s}.parquet"
                )

        elif self.release_year == "2022" or self.release_year == "2023":
            return (
                f"{self.base_url}timeseries_individual_buildings/"
                f"by_state/upgrade={self.upgrade_id}/"
                f"state={self.state}/"
                f"{self.bldg_id!s}-{int(self.upgrade_id)!s}.parquet"
            )
        elif self.release_year == "2024":
            if self.res_com == "resstock" and self.weather == "tmy3" and self.release_number == "1":
                return ""
            else:
                return (
                    f"{self.base_url}timeseries_individual_buildings/"
                    f"by_state/upgrade={self.upgrade_id}/"
                    f"state={self.state}/"
                    f"{self.bldg_id!s}-{int(self.upgrade_id)!s}.parquet"
                )
        elif self.release_year == "2025":
            return (
                f"{self.base_url}timeseries_individual_buildings/"
                f"by_state/upgrade={self.upgrade_id}/"
                f"state={self.state}/"
                f"{self.bldg_id!s}-{int(self.upgrade_id)!s}.parquet"
            )
        else:
            return ""

    def get_aggregate_load_curve_url(self) -> str:
        """Generate the S3 download URL for this building. The url is the same as the 15-minute load curve url."""
        return self.get_15min_load_curve_url()

    def get_annual_load_curve_url(self) -> str:
        """Generate the S3 download URL for this building."""
        if not self._validate_requested_file_type_availability("load_curve_annual"):
            return ""
        if self.release_year == "2021":
            return ""
        elif self.release_year == "2022" or self.release_year == "2023":
            return self._build_annual_load_state_url()
        elif self.release_year == "2024":
            return self._handle_2024_release_annual_load()
        elif self.release_year == "2025":
            return self._handle_2025_release_annual_load()
        else:
            return ""

    def get_weather_file_url(self) -> str:
        """Generate the S3 download URL for this building."""
        if self.get_weather_station_name() == "":
            return ""
        return self._build_weather_url()

    def _build_weather_url(self) -> str:
        """Build the weather file URL based on release year and weather type."""
        if self.release_year == "2021":
            return self._build_2021_weather_url()
        elif self.release_year == "2022":
            return self._build_2022_weather_url()
        elif self.release_year == "2023":
            return self._build_2023_weather_url()
        elif self.release_year == "2024":
            return self._build_2024_weather_url()
        elif self.release_year == "2025":
            return self._build_2025_weather_url()
        else:
            return ""

    def _build_2021_weather_url(self) -> str:
        """Build weather URL for 2021 release."""
        if self.weather == "tmy3":
            return f"{self.base_url}weather/{self.weather}/{self.get_weather_station_name()}_tmy3.csv"
        elif self.weather == "amy2018":
            return f"{self.base_url}weather/{self.weather}/{self.get_weather_station_name()}_2018.csv"
        elif self.weather == "amy2012":
            return f"{self.base_url}weather/{self.weather}/{self.get_weather_station_name()}_2012.csv"
        else:
            return ""

    def _build_2022_weather_url(self) -> str:
        """Build weather URL for 2022 release."""
        if self.weather == "tmy3":
            return f"{self.base_url}weather/state={self.state}/{self.get_weather_station_name()}_TMY3.csv"
        elif self.weather == "amy2018":
            return f"{self.base_url}weather/state={self.state}/{self.get_weather_station_name()}_2018.csv"
        elif self.weather == "amy2012":
            return f"{self.base_url}weather/state={self.state}/{self.get_weather_station_name()}_2012.csv"
        else:
            return ""

    def _build_2023_weather_url(self) -> str:
        """Build weather URL for 2023 release."""
        if self.weather == "tmy3":
            return f"{self.base_url}weather/{self.weather}/{self.get_weather_station_name()}_TMY3.csv"
        elif self.weather == "amy2018":
            return f"{self.base_url}weather/{self.weather}/{self.get_weather_station_name()}_2018.csv"
        elif self.weather == "amy2012":
            return f"{self.base_url}weather/{self.weather}/{self.get_weather_station_name()}_2012.csv"
        else:
            return ""

    def _build_2024_weather_url(self) -> str:
        """Build weather URL for 2024 release."""
        if self.res_com == "comstock" and self.weather == "amy2018":
            return f"{self.base_url}weather/{self.weather}/{self.get_weather_station_name()}_2018.csv"
        else:
            if self.weather == "tmy3":
                return f"{self.base_url}weather/state={self.state}/{self.get_weather_station_name()}_TMY3.csv"
            elif self.weather == "amy2018":
                return f"{self.base_url}weather/state={self.state}/{self.get_weather_station_name()}_2018.csv"
            elif self.weather == "amy2012":
                return f"{self.base_url}weather/state={self.state}/{self.get_weather_station_name()}_2012.csv"
            else:
                return ""

    def _build_2025_weather_url(self) -> str:
        """Build weather URL for 2025 release."""
        if self.weather == "tmy3":
            return f"{self.base_url}weather/{self.weather}/{self.get_weather_station_name()}_TMY3.csv"
        elif self.weather == "amy2018":
            return f"{self.base_url}weather/{self.weather}/{self.get_weather_station_name()}_2018.csv"
        elif self.weather == "amy2012":
            return f"{self.base_url}weather/{self.weather}/{self.get_weather_station_name()}_2012.csv"
        else:
            return ""

    def get_annual_load_curve_filename(self) -> str:
        """Generate the filename for the annual load curve."""
        if self.release_year == "2021":
            return ""
        elif self.release_year == "2022" or self.release_year == "2023":
            return f"{self.state}_upgrade{str(int(self.upgrade_id)).zfill(2)}_metadata_and_annual_results.parquet"
        elif self.release_year == "2024":
            if self.res_com == "comstock" and self.weather == "amy2018" and self.release_number == "2":
                county = self._get_county_name()
                if county == "":
                    return ""
                else:
                    return f"{self.state}_{county}_upgrade{str(int(self.upgrade_id)).zfill(2)}_metadata_and_annual_results.parquet"
            elif self.res_com == "resstock" and self.weather == "tmy3" and self.release_number == "1":
                return ""
            else:
                return f"{self.state}_upgrade{str(int(self.upgrade_id)).zfill(2)}_metadata_and_annual_results.parquet"
        elif self.release_year == "2025":
            if self.res_com == "comstock" and self.weather == "amy2018" and self.release_number == "1":
                county = self._get_county_name()
                if county == "":
                    return ""
                else:
                    return f"{self.state}_{county}_upgrade{str(int(self.upgrade_id)).zfill(2)}_metadata_and_annual_results.parquet"
            else:
                return ""
        else:
            return ""

    def get_weather_station_name(self) -> str:
        """Get the weather station name for this building."""

        # Filter by multiple fields for a more specific match
        weather_station_map = weather_map_df.filter(
            (pl.col("product") == self.res_com)
            & (pl.col("release_year") == self.release_year)
            & (pl.col("weather_file") == self.weather)
            & (pl.col("release_version") == self.release_number)
            & (pl.col("bldg_id") == self.bldg_id)
        )

        # Check if we found a match
        if weather_station_map.height > 0:
            # Return the weather station name from the first (and should be only) match
            weather_station_name = weather_station_map.select("weather_station_name").item()
            return str(weather_station_name) if weather_station_name is not None else ""
        else:
            # No match found, return empty string
            return ""

    def _build_annual_load_state_url(self) -> str:
        """Build the state-level URL for annual load curve data.

        Returns:
            The constructed URL for the state-level data.
        """
        if self.upgrade_id == "0":
            return (
                f"{self.base_url}metadata_and_annual_results/"
                f"by_state/state={self.state}/parquet/"
                f"{self.state}_baseline_metadata_and_annual_results.parquet"
            )
        else:
            return (
                f"{self.base_url}metadata_and_annual_results/"
                f"by_state/state={self.state}/parquet/"
                f"{self.state}_upgrade{str(int(self.upgrade_id)).zfill(2)}_metadata_and_annual_results.parquet"
            )

    def _handle_2024_release_annual_load(self) -> str:
        """Handle the 2024 release logic for annual load curve URLs.

        Returns:
            The constructed URL or empty string if not applicable.
        """
        if self.res_com == "comstock" and self.weather == "amy2018" and self.release_number == "2":
            county = self._get_county_name()
            if county == "":
                return ""
            if self.upgrade_id == "0":
                return (
                    f"{self.base_url}metadata_and_annual_results/"
                    f"by_state_and_county/full/parquet/"
                    f"state={self.state}/county={county}/"
                    f"{self.state}_{county}_baseline.parquet"
                )
            else:
                return (
                    f"{self.base_url}metadata_and_annual_results/"
                    f"by_state_and_county/full/parquet/"
                    f"state={self.state}/county={county}/"
                    f"{self.state}_{county}_upgrade{str(int(self.upgrade_id)).zfill(2)}.parquet"
                )
        elif self.res_com == "resstock" and self.weather == "tmy3" and self.release_number == "1":
            return ""  # This release has a different structure. Need further development
        else:
            return self._build_annual_load_state_url()

    def _handle_2025_release_annual_load(self) -> str:
        """Handle the 2025 release logic for annual load curve URLs.

        Returns:
            The constructed URL or empty string if not applicable.
        """
        if self.res_com == "comstock" and self.weather == "amy2018" and self.release_number == "1":
            county = self._get_county_name()
            if county == "":
                return ""
            else:
                return (
                    f"{self.base_url}metadata_and_annual_results/"
                    "by_state_and_county/full/parquet/"
                    f"state={self.state}/county={county}/"
                    f"{self.state}_{county}_upgrade{int(self.upgrade_id)!s}.parquet"
                )
        else:
            return ""

    def _get_county_name(self) -> str:
        """Get the county-based URL by reading from metadata partition.

        Returns:
            The constructed URL or empty string if not found.
        """
        # Read the specific partition that matches our criteria
        partition_path = (
            METADATA_DIR
            / f"product={self.res_com}"
            / f"release_year={self.release_year}"
            / f"weather_file={self.weather}"
            / f"release_version={self.release_number}"
            / f"state={self.state}"
        )

        # Check if the partition exists
        if not partition_path.exists():
            return ""

        # Read the parquet files in the specific partition
        df = pl.read_parquet(str(partition_path))
        building_row = df.filter(pl.col("bldg_id") == self.bldg_id)

        if building_row.height == 0:
            return ""

        # Return the county value from the matching row
        county = building_row[0].select("county").item()
        return str(county)

    def get_release_name(self) -> str:
        """Generate the release name for this building."""
        res_com_str = "res" if self.res_com == "resstock" else "com"
        return f"{res_com_str}_{self.release_year}_{self.weather}_{self.release_number}"

    def to_json(self) -> str:
        """Convert the building ID object to a JSON string."""
        return json.dumps(asdict(self))
