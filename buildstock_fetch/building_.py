from dataclasses import dataclass
from pathlib import Path
from typing import final, override

import polars as pl

from buildstock_fetch.constants import WEATHER_FILE_DIR
from buildstock_fetch.releases import RELEASES, BuildstockRelease
from buildstock_fetch.types import FileType, ReleaseKey, UpgradeID, USStateCode

_weather_map_df: pl.DataFrame | None = None


@final
class InconsistentBuildingUpgrade(ValueError):
    def __init__(self, building: "Building") -> None:
        self.building = building
        super().__init__()

    @override
    def __str__(self) -> str:
        return f"Building {self.building} has and upgrade that does not exist in the release"


@final
class InconsistentBuildingCounty(ValueError):
    def __init__(self, building: "Building") -> None:
        self.building = building
        super().__init__()

    @override
    def __str__(self) -> str:
        return f"Building {self.building} does not have county information"


@final
class UnavailableFileTypeError(ValueError):
    def __init__(self, building: "Building", file_type: FileType) -> None:
        self.building = building
        self.file_type = file_type
        super().__init__()

    @override
    def __str__(self) -> str:
        return f"Unavailable file type for building {self.building}: {self.file_type}"


@final
class CountyUnavailableError(ValueError):
    def __init__(self, building: "Building") -> None:
        self.building = building
        super().__init__()

    @override
    def __str__(self) -> str:
        return f"County name is required for an operation but is not available for {self.building}"


@dataclass(frozen=True)
class Building:
    id: int
    release: ReleaseKey
    upgrade: UpgradeID
    state: USStateCode
    cached_county: str | None

    def __post_init__(self) -> None:
        obj = RELEASES[self.release]
        if self.upgrade not in obj.upgrades:
            raise InconsistentBuildingUpgrade(self)

    @property
    def metadata_path(self) -> str:
        obj = RELEASES[self.release]
        upgrade_str = "baseline" if self.upgrade == "0" else f"upgrade{self.upgrade.zfill(2)}"
        match obj:
            # 2021 releases
            case BuildstockRelease(year="2021"):
                return f"{self.base_path}/metadata/metadata.parquet"

            # 2022 and 2023 releases
            case BuildstockRelease(year="2022" | "2023"):
                return f"{self.base_path}/metadata_and_annual_results/by_state/state={self.state}/parquet/{self.state}_{upgrade_str}_metadata_and_annual_results.parquet"

            # 2024 releases
            case BuildstockRelease(year="2024", product="comstock", weather="amy2018", version="2"):
                return (
                    f"{self.base_path}/metadata_and_annual_results/by_state_and_county/full/parquet/"
                    f"state={self.state}/county={self.county}/{self.state}_{self.county}_{upgrade_str}.parquet"
                )
            case BuildstockRelease(year="2024", product="resstock", weather="tmy3", version="1"):
                return f"{self.base_path}/metadata/{upgrade_str}.parquet"
            case BuildstockRelease(year="2024"):
                return f"{self.base_path}/metadata_and_annual_results/by_state/state={self.state}/parquet/{self.state}_{upgrade_str}_metadata_and_annual_results.parquet"

            # 2025 releases
            case BuildstockRelease(year="2025", product="comstock"):
                return (
                    f"{self.base_path}/metadata_and_annual_results/by_state_and_county/full/parquet/"
                    f"state={self.state}/county={self.county}/{self.state}_{self.county}_upgrade{self.upgrade}.parquet"
                )
            case BuildstockRelease(year="2025"):
                return (
                    f"{self.base_path}/metadata_and_annual_results/by_state/full/parquet/"
                    f"state={self.state}/{self.state}_upgrade{self.upgrade}.parquet"
                )
            case _:
                raise ValueError(self.release)

    @property
    def load_curve_15min_path(self) -> str:
        if "load_curve_15min" not in RELEASES[self.release].file_types:
            raise UnavailableFileTypeError(self, "load_curve_15min")
        return (
            f"{self.base_path}/timeseries_individual_buildings/by_state/"
            f"upgrade={self.upgrade}/state={self.state}/{self.id}-{self.upgrade}.parquet"
        )

    @property
    def load_curve_annual_path(self) -> str:
        if "load_curve_annual" not in RELEASES[self.release].file_types:
            raise UnavailableFileTypeError(self, "load_curve_annual")
        obj = RELEASES[self.release]
        upgrade_name_str = "baseline" if self.upgrade == "0" else f"upgrade{self.upgrade.zfill(2)}"
        match obj:
            case BuildstockRelease(year="2024", product="comstock", weather="amy2018", version="2"):
                return (
                    f"{self.base_path}/metadata_and_annual_results/by_state_and_county/full/parquet/"
                    f"state={self.state}/county={self.county}/{self.state}_{self.county}_{upgrade_name_str}.parquet"
                )
            case BuildstockRelease(year="2022" | "2023" | "2024"):
                return (
                    f"{self.base_path}/metadata_and_annual_results/by_state/"
                    f"state={self.state}/parquet/{self.state}_{upgrade_name_str}_metadata_and_annual_results.parquet"
                )
            case BuildstockRelease(year="2025", product="comstock"):
                return (
                    f"{self.base_path}/metadata_and_annual_results/by_state_and_county/full/parquet/"
                    f"state={self.state}/county={self.county}/{self.state}_{self.county}_upgrade{self.upgrade}.parquet"
                )
            case BuildstockRelease(year="2025", product="resstock"):
                return (
                    f"{self.base_path}/metadata_and_annual_results/by_state/full/parquet/"
                    f"state={self.state}/{self.state}_upgrade{self.upgrade}.parquet"
                )
            case _:
                raise NotImplementedError()

    @property
    def energy_models_path(self) -> str:
        obj = RELEASES[self.release]
        if "hpxml" not in obj.file_types:
            raise UnavailableFileTypeError(self, "hpxml")
        if "schedule" not in obj.file_types:
            raise UnavailableFileTypeError(self, "schedule")

        bldg_str = f"bldg{str(self.id).zfill(7)}"
        upgrade_str = f"up{self.upgrade.zfill(2)}"
        filename_str = f"{bldg_str}-{upgrade_str}"

        match obj:
            case BuildstockRelease(year="2022"):
                return f"{self.base_path}/building_energy_models/upgrade={self.upgrade}/{filename_str}.zip"
            case BuildstockRelease(year="2024"):
                return (
                    f"{self.base_path}/model_and_schedule_files"
                    f"/building_energy_models/upgrade={self.upgrade}/{filename_str}.zip"
                )
            case BuildstockRelease(year="2025"):
                upgrade_str_ = self.upgrade if self.release == "res_2025_amy2018_1" else upgrade_str
                extension = "zip" if obj.product == "resstock" else "osm.gz"
                return f"{self.base_path}/building_energy_models/upgrade={upgrade_str_}/{filename_str}.{extension}"
            case _:
                raise ValueError(self.release)

    @property
    def base_path(self) -> str:
        if self.release == "res_2024_tmy3_1":
            return "2024/resstock_dataset_2024.1/resstock_tmy3"
        obj = RELEASES[self.release]
        return f"{obj.year}/{obj.product}_{obj.weather}_release_{obj.version}"

    @property
    def county(self) -> str:
        if self.cached_county is not None:
            return self.cached_county
        msg = "County is not set"
        raise RuntimeError(msg)

    @property
    def weather_path(self) -> str | None:  # noqa: C901
        obj = RELEASES[self.release]
        name = self.weather_station_name
        if name is None:
            return None

        match obj:
            case BuildstockRelease(year="2021", weather="tmy3"):
                return f"{self.base_path}/weather/{obj.weather}/{name}_tmy3.csv"
            case BuildstockRelease(year="2021", weather="amy2018"):
                return f"{self.base_path}/weather/{obj.weather}/{name}_2018.csv"
            case BuildstockRelease(year="2021", weather="amy2012"):
                return f"{self.base_path}/weather/{obj.weather}/{name}_2012.csv"

            case BuildstockRelease(year="2022", weather="tmy3"):
                return f"{self.base_path}/weather/state={self.state}/{name}_TMY3.csv"
            case BuildstockRelease(year="2022", weather="amy2018"):
                return f"{self.base_path}/weather/state={self.state}/{name}_2018.csv"
            case BuildstockRelease(year="2022", weather="amy2012"):
                return f"{self.base_path}/weather/state={self.state}/{name}_2012.csv"

            case BuildstockRelease(year="2023", weather="tmy3"):
                return f"{self.base_path}/weather/{obj.weather}/{name}_TMY3.csv"
            case BuildstockRelease(year="2023", weather="amy2018"):
                return f"{self.base_path}/weather/{obj.weather}/{name}_2018.csv"
            case BuildstockRelease(year="2023", weather="amy2012"):
                return f"{self.base_path}weather/{obj.weather}/{name}_2012.csv"

            case BuildstockRelease(year="2024", product="comstock", weather="amy2018"):
                return f"{self.base_path}/weather/{obj.weather}/{name}_2018.csv"
            case BuildstockRelease(year="2024", weather="tmy3"):
                return f"{self.base_path}/weather/state={self.state}/{name}_TMY3.csv"
            case BuildstockRelease(year="2024", weather="amy2018"):
                return f"{self.base_path}/weather/state={self.state}/{name}_2018.csv"
            case BuildstockRelease(year="2024", weather="amy2012"):
                return f"{self.base_path}/weather/state={self.state}/{name}_2012.csv"

            case BuildstockRelease(year="2025", weather="tmy3"):
                return f"{self.base_path}/weather/{obj.weather}/{name}_TMY3.csv"
            case BuildstockRelease(year="2025", weather="amy2018"):
                return f"{self.base_path}/weather/{obj.weather}/{name}_2018.csv"
            case BuildstockRelease(year="2025", weather="amy2012"):
                return f"{self.base_path}/weather/{obj.weather}/{name}_2012.csv"

            case _:
                return None

    @property
    def weather_station_name(self) -> str | None:
        release_obj = RELEASES[self.release]
        weather_map_df = get_weather_station_map()
        weather_station_map = weather_map_df.filter(
            (pl.col("product") == release_obj.product)
            & (pl.col("release_year") == release_obj.year)
            & (pl.col("weather_file") == release_obj.weather)
            & (pl.col("release_version") == release_obj.version)
            & (pl.col("bldg_id") == self.id)
        )
        if weather_station_map.height > 0:
            weather_station_name = weather_station_map.select("weather_station_name").item()  # pyright: ignore[reportAny]
            return str(weather_station_name) if weather_station_name is not None else None  # pyright: ignore[reportAny]
        return None

    def file_path(self, file_type: FileType) -> Path:
        url: str
        match file_type:
            case "metadata":
                filename = "metadata.parquet"
            case "load_curve_15min" | "load_curve_hourly" | "load_curve_daily" | "load_curve_monthly":
                url = self.load_curve_15min_path
                filename = url.split("/")[-1]
            case "load_curve_annual":
                url = self.load_curve_annual_path.replace("_baseline_", "_upgrade00_")
                filename = url.split("/")[-1]
            case "hpxml":
                filename = f"{str(self.id).zfill(7)}-up{self.upgrade.zfill(2)}.xml"
            case "schedule":
                filename = f"{str(self.id).zfill(7)}-up{self.upgrade.zfill(2)}_schedule.csv"
            case "trip_schedules":
                return Path(self.release) / file_type / f"state={self.state}" / "trip_schedules.parquet"
            case "weather":
                weather_station_name = self.weather_station_name
                if weather_station_name is None:
                    raise ValueError(self)
                return (
                    Path(self.release)
                    / file_type
                    / f"state={self.state}"
                    / f"upgrade={self.upgrade.zfill(2)}"
                    / f"{weather_station_name}.csv"
                )

        return Path(self.release) / file_type / f"state={self.state}" / f"upgrade={self.upgrade.zfill(2)}" / filename


def get_weather_station_map() -> pl.DataFrame:
    global _weather_map_df
    if _weather_map_df is None:
        _weather_map_df = pl.read_parquet(WEATHER_FILE_DIR)
    return _weather_map_df
