import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl

from buildstock_fetch.constants import METADATA_DIR, RELEASE_JSON_FILE, SB_ANALYSIS_UPGRADES_FILE, WEATHER_FILE_DIR
from buildstock_fetch.types import FileType, ReleaseYear, ResCom, Weather

from .annualcurve import get_annual_load_curve_url
from .annualcurvefilename import get_annual_load_curve_filename
from .buildingdataurl import get_building_data_url
from .get15minurl import get_15min_load_curve_url
from .metadataurl import get_metadata_url
from .weatherurl import build_weather_url

__all__ = [
    "BuildingID",
]

weather_map_df = pl.read_parquet(WEATHER_FILE_DIR)

# Module-level cache for SB analysis upgrades data
_SB_ANALYSIS_UPGRADES_CACHE: dict[str, Any] | None = None


def _get_SB_analysis_upgrades() -> dict[str, Any] | None:
    """Load SB analysis upgrades data once and cache it for subsequent calls."""
    global _SB_ANALYSIS_UPGRADES_CACHE
    if _SB_ANALYSIS_UPGRADES_CACHE is None:
        with open(SB_ANALYSIS_UPGRADES_FILE) as f:
            _SB_ANALYSIS_UPGRADES_CACHE = json.load(f)
    return _SB_ANALYSIS_UPGRADES_CACHE


@dataclass(frozen=True)
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

    def get_metadata_url(self) -> str | list[str] | None:
        """Generate the S3 download URL for this building."""
        return get_metadata_url(self)

    def get_building_data_url(self) -> str | None:
        return get_building_data_url(self)

    def get_15min_load_curve_url(self) -> str | None:
        return get_15min_load_curve_url(self)

    def get_SB_upgrade_component_bldg_ids(self) -> list["BuildingID"] | None:
        with open(SB_ANALYSIS_UPGRADES_FILE) as f:
            sb_analysis_upgrades = json.load(f)
        release_name = self.get_release_name()
        if release_name not in sb_analysis_upgrades:
            return None
        sb_analysis_upgrade_data = sb_analysis_upgrades[release_name]
        upgrade_components = sb_analysis_upgrade_data["upgrade_components"][self.upgrade_id]
        bldg_id_component_list = []
        for component_id in upgrade_components:
            bldg_id_component_list.append(self.copy(upgrade_id=component_id))
        return bldg_id_component_list

    def get_aggregate_load_curve_url(self) -> str | None:
        """Generate the S3 download URL for this building. The url is the same as the 15-minute load curve url."""
        return self.get_15min_load_curve_url()

    def get_annual_load_curve_url(self) -> str | None:
        return get_annual_load_curve_url(self)

    def get_weather_file_url(self) -> str | None:
        """Generate the S3 download URL for this building."""
        if not self.get_weather_station_name():
            return None
        return build_weather_url(self)

    def get_annual_load_curve_filename(self) -> str | None:
        return get_annual_load_curve_filename(self)

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

    def get_output_file_path(self, file_type: FileType, output_dir: Path | None = None) -> Path:
        if output_dir is None:
            if (
                file_type == "load_curve_15min"
                or file_type == "load_curve_hourly"
                or file_type == "load_curve_daily"
                or file_type == "load_curve_monthly"
                or file_type == "load_curve_annual"
            ):
                return Path(
                    self.get_release_name(),
                    file_type,
                    f"state={self.state}",
                    f"upgrade={str(int(self.upgrade_id)).zfill(2)}",
                    f"{str(self.bldg_id)!s}-{int(self.upgrade_id)!s}.parquet",
                )
            elif file_type == "metadata":
                return Path(
                    self.get_release_name(),
                    "metadata",
                    f"state={self.state}",
                    f"upgrade={str(int(self.upgrade_id)).zfill(2)}",
                    "metadata.parquet",
                )
            elif file_type == "hpxml":
                return Path(
                    self.get_release_name(),
                    "hpxml",
                    f"state={self.state}",
                    f"upgrade={str(int(self.upgrade_id)).zfill(2)}",
                    f"bldg{str(self.bldg_id).zfill(7)}-up{self.upgrade_id.zfill(2)}.xml",
                )
            elif file_type == "schedule":
                return Path(
                    self.get_release_name(),
                    "schedule",
                    f"state={self.state}",
                    f"upgrade={str(int(self.upgrade_id)).zfill(2)}",
                    f"bldg{str(self.bldg_id).zfill(7)}-up{self.upgrade_id.zfill(2)}_schedule.csv",
                )
            else:
                return Path()
        else:
            return output_dir / self.get_output_file_path(file_type)

    def get_output_filename(self, file_type: FileType) -> str:
        if (
            file_type == "load_curve_15min"
            or file_type == "load_curve_hourly"
            or file_type == "load_curve_daily"
            or file_type == "load_curve_monthly"
        ):
            return f"{str(self.bldg_id)!s}-{int(self.upgrade_id)!s}.parquet"
        elif file_type == "load_curve_annual":
            annual_load_curve_filename = self.get_annual_load_curve_filename()
            if annual_load_curve_filename is None:
                return ""
            return annual_load_curve_filename
        elif file_type == "metadata":
            return "metadata.parquet"
        elif file_type == "hpxml":
            return f"bldg{str(self.bldg_id).zfill(7)}-up{self.upgrade_id.zfill(2)}.xml"
        elif file_type == "schedule":
            return f"bldg{str(self.bldg_id).zfill(7)}-up{self.upgrade_id.zfill(2)}_schedule.csv"
        else:
            return ""

    def to_json(self) -> str:
        """Convert the building ID object to a JSON string."""
        return json.dumps(asdict(self))

    def is_SB_upgrade(self) -> bool:
        """Check if the upgrade is a SB upgrade."""
        sb_analysis_upgrades = _get_SB_analysis_upgrades()
        if sb_analysis_upgrades is None:
            msg = "SB analysis upgrades data not available"
            raise ValueError(msg)
        release_name = self.get_release_name()
        if release_name not in sb_analysis_upgrades:
            return False
        sb_analysis_upgrade_data = sb_analysis_upgrades[release_name]
        return self.upgrade_id in sb_analysis_upgrade_data["upgrade_ids"]

    def copy(self, upgrade_id: str | None = None) -> "BuildingID":
        return BuildingID(
            bldg_id=self.bldg_id,
            release_number=self.release_number,
            release_year=self.release_year,
            res_com=self.res_com,
            weather=self.weather,
            upgrade_id=upgrade_id if upgrade_id is not None else self.upgrade_id,
            state=self.state,
        )
