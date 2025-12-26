from dataclasses import dataclass
from typing import final, override
from urllib.parse import urljoin

from buildstock_fetch.constants import METADATA_DIR
from buildstock_fetch.releases import RELEASES, BuildstockRelease
from buildstock_fetch.types import ReleaseKey, ReleaseVersion, ReleaseYear, USStateCode, UpgradeID, Weather
import polars as pl


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


@dataclass(frozen=True)
class Building:
    id: int
    release: ReleaseKey
    upgrade: UpgradeID
    state: USStateCode
    county: str | None

    def __post_init__(self):
        obj = RELEASES[self.release]
        if self.upgrade not in obj.upgrades:
            raise InconsistentBuildingUpgrade(self)
        if obj.year == "2025" and obj.product == "comstock" and self.county is None:
            raise InconsistentBuildingCounty(self)
        if self.release == "com_2024_amy2018_2" and self.county is None:
            raise InconsistentBuildingCounty(self)

    @property
    def metadata_path(self) -> str:
        obj = RELEASES[self.release]
        upgrade_str = "baseline" if self.upgrade == "0" else f"upgrade{self.upgrade.zfill(2)}"
        match obj:
            case BuildstockRelease(year="2021"):
                return f"{self.base_path}/metadata/metadata.parquet"
            case BuildstockRelease(year="2024", product="comstock", weather="amy2018", version="2"):
                return (
                    f"{self.base_path}/metadata_and_annual_results/by_state_and_county/full/parquet/"
                    f"state={self.state}/county={self.county}/{self.state}_{self.county}_{upgrade_str}.parquet"
                )
            case BuildstockRelease(year="2022" | "2023" | "2024"):
                return f"{self.base_path}/metadata/{upgrade_str}.parquet"
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
    def base_path(self) -> str:
        if self.release == "res_2024_tmy3_1":
            return "2024/resstock_dataset_2024.1/resstock_tmy3"
        obj = RELEASES[self.release]
        return f"{obj.year}/{obj.product}_{obj.weather}_release_{obj.version}"
