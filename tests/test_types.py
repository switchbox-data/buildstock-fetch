import json
from importlib.resources import files
from typing import get_args

import pytest
import typedload

from buildstock_fetch import types
from buildstock_fetch.releases import BuildstockReleasesRaw
from buildstock_fetch.types import (
    FileType,
    ReleaseKey,
    ReleaseKeyCom,
    ReleaseKeyRes,
    ReleaseVersion,
    ReleaseYear,
    Weather,
    normalize_upgrade_id,
)


@pytest.fixture(scope="function")
def releases_json() -> BuildstockReleasesRaw:
    releases_file = files("buildstock_fetch") / "data" / "buildstock_releases.json"
    return typedload.load(json.loads(releases_file.read_text()), BuildstockReleasesRaw)


def test_release_keys_total_match(releases_json: BuildstockReleasesRaw):
    """All values defined in ReleaseKey match keys defined in releases file"""
    keys = set(releases_json.keys())
    args = get_args(ReleaseKey)
    assert set(keys) == set(args)


def test_release_keys_res(releases_json: BuildstockReleasesRaw):
    """All resstock release keys must match resstock keys in releases file"""
    keys = {k for k, v in releases_json.items() if v.res_com == "resstock"}
    args = get_args(ReleaseKeyRes)
    assert set(keys) == set(args)


def test_release_keys_com(releases_json: BuildstockReleasesRaw):
    """All resstock release keys must match comstock keys in releases file"""
    keys = {k for k, v in releases_json.items() if v.res_com == "comstock"}
    args = get_args(ReleaseKeyCom)
    assert set(keys) == set(args)


@pytest.mark.parametrize("year", get_args(ReleaseYear))
def test_release_keys_by_year(releases_json: BuildstockReleasesRaw, year: ReleaseYear):
    member = getattr(types, f"ReleaseKeyY{year}")  # pyright: ignore[reportAny]
    keys = {k for k, v in releases_json.items() if v.release_year == year}
    args = get_args(member)
    assert set(keys) == set(args)


@pytest.mark.parametrize("year", get_args(ReleaseYear))
def test_release_keys_resstock_by_year(releases_json: BuildstockReleasesRaw, year: ReleaseYear):
    member = getattr(types, f"ReleaseKeyResY{year}", None)  # pyright: ignore[reportAny]
    keys = {k for k, v in releases_json.items() if v.release_year == year and v.res_com == "resstock"}
    args = get_args(member) if member else ()
    assert set(keys) == set(args)


@pytest.mark.parametrize("year", get_args(ReleaseYear))
def test_release_keys_comstock_by_year(releases_json: BuildstockReleasesRaw, year: ReleaseYear):
    member = getattr(types, f"ReleaseKeyComY{year}", None)  # pyright: ignore[reportAny]
    keys = {k for k, v in releases_json.items() if v.release_year == year and v.res_com == "comstock"}
    args = get_args(member) if member else ()
    assert set(keys) == set(args)


def test_years(releases_json: BuildstockReleasesRaw):
    """Check if ReleaseYear matches years defined in the releases file"""
    keys = {v.release_year for v in releases_json.values()}
    args = get_args(ReleaseYear)
    assert set(keys) == set(args)


def test_weathers(releases_json: BuildstockReleasesRaw):
    """Check if Weather matches weathers defined in the releases file"""
    keys = {v.weather for v in releases_json.values()}
    args = get_args(Weather)
    assert set(keys) == set(args)


def test_release_version(releases_json: BuildstockReleasesRaw):
    """Check if release versions match release versions defined in the releases file"""
    keys = {v.release_number for v in releases_json.values()}
    args = get_args(ReleaseVersion)
    assert set(keys) == set(args)


def test_file_types(releases_json: BuildstockReleasesRaw):
    """Check if FileType matches file types defined in the releases file"""
    keys = frozenset[str]()
    for v in releases_json.values():
        keys |= v.available_data
    args = get_args(FileType)
    assert set(keys) == set(args)


def test_upgrade_id_conversion(releases_json: BuildstockReleasesRaw):
    """Check if upgrades defined in the releases file convert successfully to UpgradeID"""
    keys = frozenset[str]()
    for v in releases_json.values():
        keys |= v.upgrade_ids
    for upgrade_id in keys:
        _ = normalize_upgrade_id(upgrade_id)

