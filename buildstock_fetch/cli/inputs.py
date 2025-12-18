from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Self, final, override

from buildstock_fetch.releases import ReleaseFilter
from buildstock_fetch.types import (
    FileType,
    ReleaseVersion,
    ReleaseYear,
    ResCom,
    UpgradeID,
    USStateCode,
    Weather,
)


@final
class InputsNotFinalizedError(Exception):
    def __init__(self, field: str):
        super().__init__()
        self.field = field

    @override
    def __str__(self) -> str:
        return f"Input not finalized: {self.field}"


@dataclass
class InputsMaybe:
    product: ResCom | None
    release_year: ReleaseYear | None
    weather_file: Weather | None
    release_version: ReleaseVersion | None
    states: set[USStateCode] | None
    file_types: set[FileType] | None
    upgrade_ids: set[UpgradeID] | None
    output_directory: Path | None

    def is_finalized(self) -> bool:
        return all((
            self.product is not None,
            self.release_year is not None,
            self.weather_file is not None,
            self.release_version is not None,
            self.states is not None,
            self.file_types is not None,
            self.upgrade_ids is not None,
            self.output_directory is not None,
        ))

    def as_filter(self) -> ReleaseFilter:
        result: ReleaseFilter = {}
        if self.product is not None:
            result["product"] = self.product
        if self.release_year is not None:
            result["year"] = self.release_year
        if self.weather_file is not None:
            result["weather"] = self.weather_file
        if self.release_version is not None:
            result["version"] = self.release_version
        if self.states is not None:
            result["states"] = self.states
        if self.file_types is not None:
            result["file_types"] = self.file_types
        if self.upgrade_ids is not None:
            result["upgrades"] = self.upgrade_ids
        return result


@dataclass(frozen=True)
class InputsFinal:
    product: ResCom
    release_year: ReleaseYear
    weather_file: Weather
    release_version: ReleaseVersion
    states: set[USStateCode]
    file_types: set[FileType]
    upgrade_ids: set[UpgradeID]
    output_directory: Path

    @classmethod
    def from_finalized_maybe(cls, inputs: InputsMaybe) -> Self:
        if inputs.product is None:
            raise InputsNotFinalizedError("product")
        if inputs.release_year is None:
            raise InputsNotFinalizedError("release_year")
        if inputs.weather_file is None:
            raise InputsNotFinalizedError("weather_file")
        if inputs.release_version is None:
            raise InputsNotFinalizedError("release_version")
        if inputs.states is None:
            raise InputsNotFinalizedError("states")
        if inputs.file_types is None:
            raise InputsNotFinalizedError("file_types")
        if inputs.upgrade_ids is None:
            raise InputsNotFinalizedError("upgrade_ids")
        if inputs.output_directory is None:
            raise InputsNotFinalizedError("output_directory")
        return cls(
            product=inputs.product,
            release_year=inputs.release_year,
            weather_file=inputs.weather_file,
            release_version=inputs.release_version,
            states=inputs.states,
            file_types=inputs.file_types,
            upgrade_ids=inputs.upgrade_ids,
            output_directory=inputs.output_directory,
        )
