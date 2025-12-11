import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, Self

import typedload

from buildstock_fetch.constants import BUILDSTOCK_RELEASES_FILE, UPGRADES_LOOKUP_FILE
from buildstock_fetch.types import (
    FileType,
    ReleaseName,
    ReleaseVersion,
    ReleaseYear,
    ResCom,
    UpgradeID,
    USStateCode,
    Weather,
)

from .inputs import InputsMaybe


class MoreThanOneReleaseError(Exception):
    def __init__(self, actual_number: int):
        self.actual_number = actual_number

    def __str__(self) -> str:
        return f"Expected only one release to be left, not {self.actual_number}"


class NoReleasesLeftError(Exception):
    def __str__(self) -> str:
        return "No releases left"


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


class Upgrade(NamedTuple):
    id: UpgradeID
    description: str | None


@dataclass(frozen=True)
class BuildstockRelease:
    name: ReleaseName
    release_year: ReleaseYear
    product: ResCom
    weather: Weather
    release_version: ReleaseVersion
    upgrades: frozenset[Upgrade]
    file_types: frozenset[FileType]
    weather_map_available_states: frozenset[USStateCode]
    trip_schedule_states: frozenset[USStateCode]

    @classmethod
    def from_definitions(
        cls,
        name: ReleaseName,
        raw_definition: BuildstockReleaseDefinitionRaw,
        upgrade_descriptions: dict[str, str] | None,
    ) -> Self:
        if upgrade_descriptions:
            upgrades = frozenset(
                Upgrade(UpgradeID(str(id_int)), description)
                for id_str, description in upgrade_descriptions.items()
                if (id_int := int(id_str)) is not None
                if id_int in map(int, raw_definition.upgrade_ids)
            )
        else:
            upgrades = frozenset(Upgrade(UpgradeID(id_), None) for id_ in raw_definition.upgrade_ids)
        return cls(
            name=name,
            release_year=raw_definition.release_year,
            product=raw_definition.res_com,
            weather=raw_definition.weather,
            release_version=raw_definition.release_number,
            file_types=raw_definition.available_data,
            upgrades=upgrades,
            weather_map_available_states=raw_definition.weather_map_available_states,
            trip_schedule_states=raw_definition.trip_schedule_states,
        )

    @property
    def upgrade_ids(self) -> frozenset[UpgradeID]:
        return frozenset(_.id for _ in self.upgrades)

    def filter(self, inputs: InputsMaybe) -> bool:
        if inputs.release_year is not None and inputs.release_year != self.release_year:
            return False
        if inputs.product is not None and inputs.product != self.product:
            return False
        if inputs.weather_file is not None and inputs.weather_file != self.weather:
            return False
        if inputs.release_version is not None and inputs.release_version != self.release_version:
            return False
        if inputs.upgrade_ids is not None and inputs.upgrade_ids - self.upgrade_ids:
            return False
        return not (inputs.file_types is not None and inputs.file_types - self.file_types)


@dataclass(frozen=True)
class BuildstockReleases:
    releases: set[BuildstockRelease]

    def __iter__(self) -> Iterator[BuildstockRelease]:
        yield from self.releases

    @classmethod
    def from_json(cls, filepath: Path | None = None, upgrades_filepath: Path | None = None) -> Self:
        filepath = filepath or Path(BUILDSTOCK_RELEASES_FILE)
        content = filepath.read_text()
        json_ = json.loads(content)
        releases_dict = typedload.load(json_, dict[ReleaseName, BuildstockReleaseDefinitionRaw])

        upgrades_filepath = upgrades_filepath or Path(UPGRADES_LOOKUP_FILE)
        upgrades_content = upgrades_filepath.read_text()
        upgrades_json = json.loads(upgrades_content)
        upgrades_dict = {
            ReleaseName(release_name): {str(k): str(v) for k, v in value["upgrade_descriptions"].items()}
            for release_name, value in upgrades_json.items()
        }

        releases = {
            BuildstockRelease.from_definitions(name, definition, upgrades_dict.get(name, {}))
            for name, definition in releases_dict.items()
        }

        return cls(releases)

    def filter(self, inputs: InputsMaybe) -> "BuildstockReleases":
        return BuildstockReleases({r for r in self.releases if r.filter(inputs)})

    def filter_at_least_one(self, inputs: InputsMaybe) -> "BuildstockReleases":
        result = self.filter(inputs)
        if not len(result):
            raise NoReleasesLeftError()
        return result

    def filter_one(self, inputs: InputsMaybe) -> BuildstockRelease:
        result_releases = self.filter(inputs)
        actual_number = len(result_releases)
        if actual_number > 1:
            raise MoreThanOneReleaseError(actual_number)
        if actual_number == 0:
            raise NoReleasesLeftError()
        return result_releases.releases.pop()

    def __len__(self) -> int:
        return len(self.releases)

    def __getitem__(self, name: str) -> BuildstockRelease:
        return next(_ for _ in self.releases if _.name == name)

    @property
    def names(self) -> frozenset[ReleaseName]:
        return frozenset(_.name for _ in self)

    @property
    def release_years(self) -> frozenset[ReleaseYear]:
        return frozenset(_.release_year for _ in self)

    @property
    def products(self) -> frozenset[ResCom]:
        return frozenset(_.product for _ in self)

    @property
    def weathers(self) -> frozenset[Weather]:
        return frozenset(_.weather for _ in self)

    @property
    def release_versions(self) -> frozenset[ReleaseVersion]:
        return frozenset(_.release_version for _ in self)

    @property
    def upgrade_ids(self) -> frozenset[UpgradeID]:
        result: frozenset[UpgradeID] = frozenset()
        for item in self:
            result |= item.upgrade_ids
        return result

    @property
    def file_types(self) -> frozenset[FileType]:
        result: frozenset[FileType] = frozenset()
        for item in self:
            result |= item.file_types
        return result
