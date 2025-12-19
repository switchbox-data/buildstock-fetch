from dataclasses import dataclass
from math import prod
from typing import overload
from urllib.parse import urljoin

from buildstock_fetch.releases import RELEASES, BuildstockRelease
from buildstock_fetch.types import (
    ReleaseKey,
    ReleaseKeyY2021,
    ReleaseVersion,
    ReleaseYear,
    ResCom,
    UpgradeID,
    USStateCode,
    Weather,
    is_valid_release_key,
)


class Y2021Map:
    @staticmethod
    @overload
    def release(release: ReleaseKeyY2021) -> "Y2021Release": ...

    @staticmethod
    @overload
    def release(release: BuildstockRelease) -> "Y2021Release": ...

    @staticmethod
    def release(release: ReleaseKeyY2021 | BuildstockRelease) -> "Y2021Release":  # pyright: ignore[reportInconsistentOverload, reportUnknownParameterType, reportMissingParameterType]
        if isinstance(release, BuildstockRelease):
            return Y2021Map._release_by_release_obj(release)
        return Y2021Map._release_by_release_obj(RELEASES[release])

    @staticmethod
    def release_by_keys(product: ResCom, weather: Weather, version: ReleaseVersion) -> "Y2021Release":
        return Y2021Release(product, weather, version)

    @staticmethod
    def _release_by_release_obj(release: BuildstockRelease) -> "Y2021Release":
        if release.year != "2021":
            raise ValueError("year")
        return Y2021Map.release_by_keys(
            release.product,
            release.weather,
            release.version,
        )


@dataclass(frozen=True)
class Y2021Release:
    product: ResCom
    weather: Weather
    version: ReleaseVersion

    @property
    def individual_buildings(self) -> "Y2021TimeSeriesIndividualBuildings":
        return Y2021TimeSeriesIndividualBuildings(self)

    @property
    def base_url(self) -> str:
        return (
            f"https://oedi-data-lake.s3.amazonaws.com/"
            "nrel-pds-building-stock/"
            "end-use-load-profiles-for-us-building-stock/"
            f"{self.release_year}/"
            f"{self.product}_{self.weather}_release_{self.version}/"
        )

    @property
    def release_year(self) -> ReleaseYear:
        return "2021"

    @property
    def citation_txt(self) -> str:
        return urljoin(self.base_url, "citation.txt")

    @property
    def data_dictionary_tsv(self) -> str:
        return urljoin(self.base_url, "data_dictionary.tsv")

    @property
    def enumeration_dictionary_tsv(self) -> str:
        return urljoin(self.base_url, "enumeration_dictionary.tsv")

    @property
    def metadata_parquet(self) -> str:
        return urljoin(self.base_url, "metadata/metadata.parquet")

    @property
    def timeseries_aggregates_metadata_tsv(self) -> str:
        return urljoin(self.base_url, "timeseries_aggregates_metadata/metadata.tsv")


@dataclass(frozen=True)
class Y2021TimeSeriesIndividualBuildings:
    parent: Y2021Release

    def by_state(self, upgrade: UpgradeID, state: USStateCode) -> "Y2021TimeSeriesIndividualBuildingsByState":
        return Y2021TimeSeriesIndividualBuildingsByState(self, upgrade, state)

    @property
    def base_url(self) -> str:
        return urljoin(self.parent.base_url, "timeseries_individual_buildings/")


@dataclass(frozen=True)
class Y2021TimeSeriesIndividualBuildingsByState:
    parent: Y2021TimeSeriesIndividualBuildings
    upgrade: UpgradeID
    state: USStateCode

    @property
    def base_url(self) -> str:
        return urljoin(self.parent.base_url, f"by_state/upgrade={self.upgrade}/state={self.state}/")

    def parquet(self, building_id: str) -> str:
        return urljoin(self.base_url, f"{building_id}-{self.upgrade}.parquet")
