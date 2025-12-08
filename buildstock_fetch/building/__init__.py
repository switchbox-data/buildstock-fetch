import json
from dataclasses import asdict, dataclass

import polars as pl

from buildstock_fetch.constants import METADATA_DIR, RELEASE_JSON_FILE, WEATHER_FILE_DIR
from buildstock_fetch.types import ReleaseYear, ResCom, Weather

__all__ = [
    "BuildingID",
]

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

    def _get_building_data_url_2022(self) -> str:
        """Get building data URL for 2022 releases."""
        return (
            f"{self.base_url}"
            f"building_energy_models/upgrade={self.upgrade_id}/"
            f"bldg{str(self.bldg_id).zfill(7)}-up{str(int(self.upgrade_id)).zfill(2)}.zip"
        )

    def _get_building_data_url_2024(self) -> str | None:
        """Get building data URL for 2024 releases."""
        if self.res_com == "comstock":
            return None
        if (self.weather == "amy2018" or self.weather == "tmy3") and self.release_number == "2":
            return (
                f"{self.base_url}"
                f"model_and_schedule_files/building_energy_models/upgrade={self.upgrade_id}/"
                f"bldg{str(self.bldg_id).zfill(7)}-up{str(int(self.upgrade_id)).zfill(2)}.zip"
            )
        return None

    def _get_building_data_url_2025_upgrade_string(self) -> str:
        """Get upgrade string for 2025 building data URLs."""
        if self.res_com == "resstock" and self.weather == "amy2018" and self.release_number == "1":
            return str(int(self.upgrade_id))
        if self.res_com == "comstock" and (
            (self.weather == "amy2012" and self.release_number == "2")
            or (self.weather == "amy2018" and self.release_number == "1")
            or (self.weather == "amy2018" and self.release_number == "2")
        ):
            return str(int(self.upgrade_id)).zfill(2)
        return str(int(self.upgrade_id)).zfill(2)

    def _get_building_data_url_2025(self) -> str | None:
        """Get building data URL for 2025 releases."""
        if (self.res_com == "resstock" and self.weather == "amy2012" and self.release_number == 1) or (
            self.res_com == "comstock" and self.weather == "amy2018" and self.release_number == 3
        ):
            return None
        upgrade_string = self._get_building_data_url_2025_upgrade_string()
        if self.res_com == "resstock":
            return (
                f"{self.base_url}"
                f"building_energy_models/upgrade={upgrade_string}/"
                f"bldg{str(self.bldg_id).zfill(7)}-up{str(int(self.upgrade_id)).zfill(2)}.zip"
            )
        else:
            return (
                f"{self.base_url}"
                f"building_energy_models/upgrade={upgrade_string}/"
                f"bldg{str(self.bldg_id).zfill(7)}-up{str(int(self.upgrade_id)).zfill(2)}.osm.gz"
            )

    def get_building_data_url(self) -> str | None:
        """Generate the S3 download URL for this building."""
        if not self._validate_requested_file_type_availability(
            "hpxml"
        ) or not self._validate_requested_file_type_availability("schedule"):
            return None
        if self.release_year == "2021" or self.release_year == "2023":
            return None
        if self.release_year == "2022":
            return self._get_building_data_url_2022()

        if self.release_year == "2024":
            return self._get_building_data_url_2024()
        if self.release_year == "2025":
            return self._get_building_data_url_2025()
        return None

    def _get_metadata_url_2021(self) -> str:
        """Get metadata URL for 2021 releases."""

        return f"{self.base_url}metadata/metadata.parquet"

    def _get_metadata_url_2022_2023(self) -> str:
        """Get metadata URL for 2022 or 2023 releases."""

        if self.upgrade_id == "0":
            return f"{self.base_url}metadata/baseline.parquet"
        return f"{self.base_url}metadata/upgrade{str(int(self.upgrade_id)).zfill(2)}.parquet"

    def _get_metadata_url_2024_comstock_amy2018_v2(self) -> str:
        """Get metadata URL for 2024 comstock amy2018 release version 2."""
        upgrade_filename = "baseline" if self.upgrade_id == "0" else f"upgrade{str(int(self.upgrade_id)).zfill(2)}"
        return (
            f"{self.base_url}metadata_and_annual_results/by_state_and_county/full/parquet/"
            f"state={self.state}/county={self._get_county_name()}/{self.state}_{self._get_county_name()}_{upgrade_filename}.parquet"
        )

    def _get_metadata_url_2024(self) -> str:
        """Get metadata URL for 2024 releases."""
        if self.res_com == "comstock" and self.weather == "amy2018" and self.release_number == "2":
            return self._get_metadata_url_2024_comstock_amy2018_v2()
        if self.upgrade_id == "0":
            return f"{self.base_url}metadata/baseline.parquet"
        return f"{self.base_url}metadata/upgrade{str(int(self.upgrade_id)).zfill(2)}.parquet"

    def _get_metadata_url_2025(self) -> str | None:
        """Get metadata URL for 2025 releases."""
        if self.res_com == "comstock":
            return (
                f"{self.base_url}metadata_and_annual_results/by_state_and_county/full/parquet/"
                f"state={self.state}/county={self._get_county_name()}/{self.state}_{self._get_county_name()}_upgrade{self.upgrade_id}.parquet"
            )
        if self.res_com == "resstock":
            return (
                f"{self.base_url}metadata_and_annual_results/by_state/full/parquet/"
                f"state={self.state}/{self.state}_upgrade{self.upgrade_id}.parquet"
            )
        return None

    def _handle_2025_release_annual_load(self) -> str | None:
        """Get load curve annual URL for 2025 releases."""
        if self.res_com == "comstock":
            return (
                f"{self.base_url}metadata_and_annual_results/by_state_and_county/full/parquet/"
                f"state={self.state}/county={self._get_county_name()}/{self.state}_{self._get_county_name()}_upgrade{self.upgrade_id}.parquet"
            )
        elif self.res_com == "resstock":
            return (
                f"{self.base_url}metadata_and_annual_results/by_state/full/parquet/"
                f"state={self.state}/{self.state}_upgrade{self.upgrade_id}.parquet"
            )
        return None

    def get_metadata_url(self) -> str | None:
        """Generate the S3 download URL for this building."""

        if not self._validate_requested_file_type_availability("metadata"):
            return None
        if self.release_year == "2021":
            return self._get_metadata_url_2021()
        if self.release_year == "2022" or self.release_year == "2023":
            return self._get_metadata_url_2022_2023()

        if self.release_year == "2024":
            return self._get_metadata_url_2024()
        if self.release_year == "2025":
            return self._get_metadata_url_2025()
        return None

    def get_15min_load_curve_url(self) -> str | None:
        """Generate the S3 download URL for this building."""
        if not self._validate_requested_file_type_availability("load_curve_15min"):
            return None
        if self.release_year == "2021":
            if self.upgrade_id != "0":
                return None  # This release only has baseline timeseries

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
                return None

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
            return None

    def get_aggregate_load_curve_url(self) -> str | None:
        """Generate the S3 download URL for this building. The url is the same as the 15-minute load curve url."""
        return self.get_15min_load_curve_url()

    def get_annual_load_curve_url(self) -> str | None:
        """Generate the S3 download URL for this building."""

        if not self._validate_requested_file_type_availability("load_curve_annual"):
            return None
        if self.release_year == "2021":
            return None
        elif self.release_year == "2022" or self.release_year == "2023":
            return self._build_annual_load_state_url()
        elif self.release_year == "2024":
            return self._handle_2024_release_annual_load()
        elif self.release_year == "2025":
            return self._handle_2025_release_annual_load()
        else:
            return None

    def get_weather_file_url(self) -> str | None:
        """Generate the S3 download URL for this building."""
        if not self.get_weather_station_name():
            return None
        return self._build_weather_url()

    def _build_weather_url(self) -> str | None:
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
            return None

    def _build_2021_weather_url(self) -> str | None:
        """Build weather URL for 2021 release."""

        if self.weather == "tmy3":
            return f"{self.base_url}weather/{self.weather}/{self.get_weather_station_name()}_tmy3.csv"

        elif self.weather == "amy2018":
            return f"{self.base_url}weather/{self.weather}/{self.get_weather_station_name()}_2018.csv"
        elif self.weather == "amy2012":
            return f"{self.base_url}weather/{self.weather}/{self.get_weather_station_name()}_2012.csv"
        else:
            return None

    def _build_2022_weather_url(self) -> str | None:
        """Build weather URL for 2022 release."""
        if self.weather == "tmy3":
            return f"{self.base_url}weather/state={self.state}/{self.get_weather_station_name()}_TMY3.csv"
        elif self.weather == "amy2018":
            return f"{self.base_url}weather/state={self.state}/{self.get_weather_station_name()}_2018.csv"
        elif self.weather == "amy2012":
            return f"{self.base_url}weather/state={self.state}/{self.get_weather_station_name()}_2012.csv"
        else:
            return None

    def _build_2023_weather_url(self) -> str | None:
        """Build weather URL for 2023 release."""
        if self.weather == "tmy3":
            return f"{self.base_url}weather/{self.weather}/{self.get_weather_station_name()}_TMY3.csv"
        elif self.weather == "amy2018":
            return f"{self.base_url}weather/{self.weather}/{self.get_weather_station_name()}_2018.csv"
        elif self.weather == "amy2012":
            return f"{self.base_url}weather/{self.weather}/{self.get_weather_station_name()}_2012.csv"

        else:
            return None

    def _build_2024_weather_url(self) -> str | None:
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
                return None

    def _build_2025_weather_url(self) -> str | None:
        """Build weather URL for 2025 release."""
        if self.weather == "tmy3":
            return f"{self.base_url}weather/{self.weather}/{self.get_weather_station_name()}_TMY3.csv"
        elif self.weather == "amy2018":
            return f"{self.base_url}weather/{self.weather}/{self.get_weather_station_name()}_2018.csv"
        elif self.weather == "amy2012":
            return f"{self.base_url}weather/{self.weather}/{self.get_weather_station_name()}_2012.csv"
        else:
            return None

    def _get_annual_load_curve_filename_2022_2023(self) -> str:
        """Get annual load curve filename for 2022 or 2023 releases."""

        return f"{self.state}_upgrade{str(int(self.upgrade_id)).zfill(2)}_metadata_and_annual_results.parquet"

    def _get_annual_load_curve_filename_2024(self) -> str | None:
        """Get annual load curve filename for 2024 releases."""
        if self.res_com == "comstock" and self.weather == "amy2018" and self.release_number == "2":
            county = self._get_county_name()
            if not county:
                return None
            return (
                f"{self.state}_{county}_upgrade{str(int(self.upgrade_id)).zfill(2)}_metadata_and_annual_results.parquet"
            )
        if self.res_com == "resstock" and self.weather == "tmy3" and self.release_number == "1":
            return None

        return f"{self.state}_upgrade{str(int(self.upgrade_id)).zfill(2)}_metadata_and_annual_results.parquet"

    def _get_annual_load_curve_filename_2025(self) -> str | None:
        """Get annual load curve filename for 2025 releases."""
        if self.res_com == "comstock":
            county = self._get_county_name()
            if not county:
                return None
            return f"{self.state}_{county}_upgrade{int(self.upgrade_id)!s}.parquet"
        if self.res_com == "resstock":
            return f"{self.state}_upgrade{int(self.upgrade_id)!s}.parquet"

        return None

    def get_annual_load_curve_filename(self) -> str | None:
        """Generate the filename for the annual load curve."""

        if self.release_year == "2021":
            return None

        if self.release_year == "2022" or self.release_year == "2023":
            return self._get_annual_load_curve_filename_2022_2023()
        if self.release_year == "2024":
            return self._get_annual_load_curve_filename_2024()
        if self.release_year == "2025":
            return self._get_annual_load_curve_filename_2025()
        return None

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

    def _handle_2024_release_annual_load(self) -> str | None:
        """Handle the 2024 release logic for annual load curve URLs.


        Returns:

            The constructed URL or empty string if not applicable.
        """
        if self.res_com == "comstock" and self.weather == "amy2018" and self.release_number == "2":
            county = self._get_county_name()
            if not county:
                return None
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
            return None  # This release has a different structure. Need further development
        else:
            return self._build_annual_load_state_url()

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
