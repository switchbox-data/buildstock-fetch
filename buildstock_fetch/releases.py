import json
from collections.abc import Collection, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, TypedDict

import typedload
from typing_extensions import Self, Unpack, final, override

from buildstock_fetch.constants import BUILDSTOCK_RELEASES_FILE, UPGRADES_LOOKUP_FILE
from buildstock_fetch.types import (
    FileType,
    ReleaseKey,
    ReleaseVersion,
    ReleaseYear,
    ResCom,
    UpgradeID,
    USStateCode,
    Weather,
    normalize_upgrade_id,
)


@final
class MoreThanOneReleaseError(Exception):
    def __init__(self, actual_number: int):
        self.actual_number = actual_number
        super().__init__()

    @override
    def __str__(self) -> str:
        return f"Expected only one release to be left, not {self.actual_number}"


@final
class NoReleasesLeftError(Exception):
    @override
    def __str__(self) -> str:
        return "No releases left"


class ReleaseFilter(TypedDict, total=False):
    product: ResCom
    year: ReleaseYear
    weather: Weather
    version: ReleaseVersion
    states: Collection[USStateCode]
    file_types: Collection[FileType]
    upgrades: Collection[UpgradeID]


@dataclass(frozen=True)
class BuildstockReleaseDefinitionRaw:
    release_year: ReleaseYear
    res_com: ResCom
    weather: Weather
    release_number: ReleaseVersion
    upgrade_ids: frozenset[str]
    available_data: frozenset[FileType]
    weather_map_available_states: frozenset[USStateCode] = field(default_factory=frozenset)
    trip_schedule_states: frozenset[USStateCode] = field(default_factory=frozenset)


class UpgradeDescriptionsRaw(TypedDict):
    upgrade_descriptions: dict[str, str]


UpgradesLookupRaw = dict[ReleaseKey, UpgradeDescriptionsRaw]


class UpgradeWithDescription(NamedTuple):
    id: UpgradeID
    description: str | None


@dataclass(frozen=True)
class BuildstockRelease:
    key: ReleaseKey
    year: ReleaseYear
    product: ResCom
    weather: Weather
    version: ReleaseVersion
    upgrades_with_descriptions: frozenset[UpgradeWithDescription]
    file_types: frozenset[FileType]
    weather_map_available_states: frozenset[USStateCode]
    trip_schedule_states: frozenset[USStateCode]

    @classmethod
    def from_definitions(
        cls,
        key: ReleaseKey,
        raw_definition: BuildstockReleaseDefinitionRaw,
        upgrade_descriptions: dict[str, str] | None,
    ) -> Self:
        if upgrade_descriptions:
            upgrades = frozenset(
                UpgradeWithDescription(normalize_upgrade_id(str(id_int)), description)
                for id_str, description in upgrade_descriptions.items()
                if (id_int := int(id_str)) is not None  # pyright: ignore[reportUnnecessaryComparison]
                if id_int in map(int, raw_definition.upgrade_ids)
            )
        else:
            upgrades = frozenset(
                UpgradeWithDescription(normalize_upgrade_id(id_), None) for id_ in raw_definition.upgrade_ids
            )

        return cls(
            key=key,
            year=raw_definition.release_year,
            product=raw_definition.res_com,
            weather=raw_definition.weather,
            version=raw_definition.release_number,
            file_types=raw_definition.available_data,
            upgrades_with_descriptions=upgrades,
            weather_map_available_states=raw_definition.weather_map_available_states,
            trip_schedule_states=raw_definition.trip_schedule_states,
        )

    @property
    def upgrades(self) -> frozenset[UpgradeID]:
        return frozenset(_.id for _ in self.upgrades_with_descriptions)

    def matches(self, **predicate: Unpack[ReleaseFilter]) -> bool:
        if (year := predicate.get("year")) is not None and year != self.year:
            return False
        if (product := predicate.get("product")) is not None and product != self.product:
            return False
        if (weather := predicate.get("weather")) is not None and weather != self.weather:
            return False
        if (version := predicate.get("version")) is not None and version != self.version:
            return False
        if (upgrades := predicate.get("upgrades")) is not None and set(upgrades) - self.upgrades:
            return False
        if (file_types := predicate.get("file_types")) is not None and set(file_types) - self.file_types:
            return False
        return True


@dataclass(frozen=True)
class BuildstockReleases:
    releases: frozenset[BuildstockRelease]

    def __iter__(self) -> Iterator[BuildstockRelease]:
        yield from self.releases

    @classmethod
    def load(cls, filepath: Path | None = None, upgrades_filepath: Path | None = None) -> Self:
        filepath = filepath or Path(BUILDSTOCK_RELEASES_FILE)
        content = filepath.read_text()
        json_ = json.loads(content)  # pyright: ignore[reportAny]
        releases_dict = typedload.load(json_, dict[ReleaseKey, BuildstockReleaseDefinitionRaw])

        upgrades_filepath = upgrades_filepath or Path(UPGRADES_LOOKUP_FILE)
        upgrades_content = upgrades_filepath.read_text()
        upgrades_json = json.loads(upgrades_content)  # pyright: ignore[reportAny]
        upgrades = typedload.load(upgrades_json, UpgradesLookupRaw)
        upgrades_dict = {
            release_key: {str(k): str(v) for k, v in value["upgrade_descriptions"].items()}
            for release_key, value in upgrades.items()
        }

        releases = frozenset(
            BuildstockRelease.from_definitions(name, definition, upgrades_dict.get(name, {}))
            for name, definition in releases_dict.items()
        )

        return cls(releases)

    def filter(self, **predicate: Unpack[ReleaseFilter]) -> "BuildstockReleases":
        return BuildstockReleases(frozenset(r for r in self.releases if r.matches(**predicate)))

    def filter_at_least_one(self, **predicate: Unpack[ReleaseFilter]) -> "BuildstockReleases":
        result = self.filter(**predicate)
        if not len(result):
            raise NoReleasesLeftError()
        return result

    def filter_one(self, **predicate: Unpack[ReleaseFilter]) -> BuildstockRelease:
        result_releases = self.filter(**predicate)
        actual_number = len(result_releases)
        if actual_number > 1:
            raise MoreThanOneReleaseError(actual_number)
        if actual_number == 0:
            raise NoReleasesLeftError()
        return next(iter(result_releases.releases))

    def __len__(self) -> int:
        return len(self.releases)

    def __getitem__(self, key: str) -> BuildstockRelease:
        return next(_ for _ in self.releases if _.key == key)

    @property
    def keys(self) -> frozenset[ReleaseKey]:
        return frozenset(_.key for _ in self)

    @property
    def years(self) -> frozenset[ReleaseYear]:
        return frozenset(_.year for _ in self)

    @property
    def products(self) -> frozenset[ResCom]:
        return frozenset(_.product for _ in self)

    @property
    def weathers(self) -> frozenset[Weather]:
        return frozenset(_.weather for _ in self)

    @property
    def versions(self) -> frozenset[ReleaseVersion]:
        return frozenset(_.version for _ in self)

    @property
    def upgrade_ids(self) -> frozenset[UpgradeID]:
        result: frozenset[UpgradeID] = frozenset()
        for item in self:
            result |= item.upgrades
        return result

    @property
    def file_types(self) -> frozenset[FileType]:
        result: frozenset[FileType] = frozenset()
        for item in self:
            result |= item.file_types
        return result


RELEASES = BuildstockReleases.load()
