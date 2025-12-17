import re
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

from typing_extensions import Self

from buildstock_fetch.releases import RELEASES, BuildstockRelease, BuildstockReleases

from .types import (
    FileType,
    ReleaseKey,
    UpgradeID,
    USStateCode,
    is_valid_file_type,
    is_valid_release_key,
    is_valid_state_code,
    is_valid_upgrade_id,
    normalize_file_type,
    normalize_release_key,
    normalize_state_code,
    normalize_upgrade_id,
)

DOWNLOADED_FILE_PATH_REGEXP = re.compile(
    r"(?P<base_path>.+?)\/(?P<release_key>[^\/]+)\/(?P<file_type>[^\/]+)\/state=(?P<state>\w+)\/upgrade=(?P<upgrade_id>\d+)\/(?P<filename>.+)"
)


T = TypeVar("T")


@dataclass(frozen=True)
class DownloadedDataInfo:
    file_path: Path
    base_path: Path
    filename: str
    release_key: ReleaseKey
    file_type: FileType
    state: USStateCode
    upgrade_id: UpgradeID
    releases: BuildstockReleases = RELEASES

    @classmethod
    def from_file_path(cls, path: Path, releases: BuildstockReleases = RELEASES) -> Self:
        posix_path = path.as_posix()
        match = DOWNLOADED_FILE_PATH_REGEXP.match(posix_path)
        if match is None:
            raise ValueError(path)
        groups = match.groupdict()

        filename = cast(str, groups["filename"])
        upgrade_id = normalize_upgrade_id(cast(str, groups["upgrade_id"]))
        state = normalize_state_code(cast(str, groups["state"]))
        file_type = normalize_file_type(cast(str, groups["file_type"]))
        release_key = normalize_release_key(cast(str, groups["release_key"]))
        base_path = Path(cast(str, groups["base_path"]))

        return cls(
            file_path=path,
            base_path=base_path,
            filename=filename,
            release_key=release_key,
            file_type=file_type,
            state=state,
            upgrade_id=upgrade_id,
            releases=releases,
        )

    @property
    def release(self) -> BuildstockRelease:
        return self.releases[self.release_key]

    def match(  # noqa: C901
        self,
        release_key: ReleaseKey | Collection[ReleaseKey] | None = None,
        file_type: FileType | Collection[FileType] | None = None,
        state: USStateCode | Collection[USStateCode] | None = None,
        upgrade_id: UpgradeID | Collection[UpgradeID] | None = None,
        suffix: str | Collection[str] | None = None,
    ) -> bool:
        if is_valid_release_key(release_key):
            release_key = (release_key,)
        if is_valid_file_type(file_type):
            file_type = ((file_type),)
        if is_valid_state_code(state):
            state = (state,)
        if is_valid_upgrade_id(upgrade_id):
            upgrade_id = (upgrade_id,)
        if isinstance(suffix, str):
            suffix = (suffix,)

        if release_key is not None and all(_ != self.release_key for _ in release_key):
            return False
        if file_type is not None and all(_ != self.file_type for _ in file_type):
            return False
        if state is not None and all(_ != self.state for _ in state):
            return False
        if upgrade_id is not None and all(_ != self.upgrade_id for _ in upgrade_id):
            return False
        if suffix is not None and all(_ != self.file_path.suffix for _ in suffix):
            return False
        return True


class DownloadedData(frozenset[DownloadedDataInfo]):
    def releases(self) -> frozenset[BuildstockRelease]:
        return frozenset(_.release for _ in self)

    def release_keys(self) -> frozenset[ReleaseKey]:
        return frozenset(_.release_key for _ in self)

    def file_types(self) -> frozenset[FileType]:
        return frozenset(_.file_type for _ in self)

    def states(self) -> frozenset[USStateCode]:
        return frozenset(_.state for _ in self)

    def upgrade_ids(self) -> frozenset[UpgradeID]:
        return frozenset(_.upgrade_id for _ in self)

    def filter(
        self,
        release_key: ReleaseKey | Collection[ReleaseKey] | None = None,
        file_type: FileType | Collection[FileType] | None = None,
        state: USStateCode | Collection[USStateCode] | None = None,
        upgrade_id: UpgradeID | Collection[UpgradeID] | None = None,
        suffix: str | Collection[str] | None = None,
    ) -> Self:
        return self.__class__(_ for _ in self if _.match(release_key, file_type, state, upgrade_id, suffix))


def filter_downloads(
    path: Path,
    release_key: Collection[ReleaseKey] | ReleaseKey | None = None,
    file_type: Collection[FileType] | FileType | None = None,
    state: Collection[USStateCode] | USStateCode | None = None,
    upgrade_id: Collection[UpgradeID] | UpgradeID | None = None,
    releases: BuildstockReleases = RELEASES,
) -> Iterator[DownloadedDataInfo]:
    if is_valid_release_key(release_key):
        release_key = (release_key,)
    if is_valid_file_type(file_type):
        file_type = (file_type,)
    if is_valid_state_code(state):
        state = (state,)
    if is_valid_upgrade_id(upgrade_id):
        upgrade_id = (upgrade_id,)

    return (
        DownloadedDataInfo.from_file_path(file_, releases)
        for release_key_path in path.iterdir()
        if (release_key_value := noraise(normalize_release_key, release_key_path.name)) is not None
        if release_key is None or release_key_value in release_key
        if release_key_path.is_dir()
        for file_type_path in release_key_path.iterdir()
        if (file_type_value := noraise(normalize_file_type, file_type_path.name)) is not None
        if file_type is None or file_type_value in file_type
        if file_type_path.is_dir()
        for state_path in file_type_path.iterdir()
        if (state_value := noraise(normalize_state_code, state_path.name.replace("state=", ""))) is not None
        if state is None or state_value in state
        if state_path.is_dir()
        for upgrade_id_path in state_path.iterdir()
        if (upgrade_id_value := noraise(normalize_upgrade_id, upgrade_id_path.name.replace("upgrade=", ""))) is not None
        if upgrade_id is None or upgrade_id_value in upgrade_id
        if upgrade_id_path.is_dir()
        for file_ in upgrade_id_path.iterdir()
        if not file_.is_dir()
    )


def noraise(func: Callable[[Any], T], value: T) -> T | None:  # pyright: ignore[reportExplicitAny]
    try:
        return func(value)
    except Exception as _:
        return None
