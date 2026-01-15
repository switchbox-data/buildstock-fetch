from urllib.parse import urljoin

import httpx
import pytest

from buildstock_fetch.building_ import Building, UnavailableFileTypeError
from buildstock_fetch.constants import OEDI_WEB_URL
from buildstock_fetch.main_new import list_buildings
from buildstock_fetch.releases import RELEASES
from buildstock_fetch.types import ReleaseKey, UpgradeID


def building_id_str(building: Building) -> str:
    return f"{building.release}-{building.upgrade}-{building.state}"


@pytest.mark.vcr
# @pytest.mark.block_network
@pytest.mark.network
@pytest.mark.parametrize(
    "building, release, upgrade",
    (
        (building, release.key, upgrade)
        for release in list(RELEASES)
        for upgrade in sorted(release.upgrades)
        for building in list_buildings(release.key, "NY", upgrade, 1)
    ),
)
def test_metadata_path(building: Building, release: ReleaseKey, upgrade: UpgradeID):  # pyright: ignore[reportUnusedParameter]
    url = urljoin(OEDI_WEB_URL, building.metadata_path)
    response = httpx.head(url)
    _ = response.raise_for_status()
    headers = response.headers
    assert headers["Content-Type"] in ("binary/octet-stream", "application/octet-stream")
    assert headers["Accept-Ranges"] == "bytes"


@pytest.mark.vcr
@pytest.mark.network
@pytest.mark.parametrize(
    "building",
    [
        building
        for release in RELEASES
        if "load_curve_15min" in release.file_types
        for upgrade in release.upgrades
        for building in list_buildings(release.key, "NY", upgrade, 1)
    ],
    ids=building_id_str,
)
def test_15min_load_curve(building: Building):
    url = urljoin(OEDI_WEB_URL, building.load_curve_15min_path)
    response = httpx.head(url)
    _ = response.raise_for_status()
    headers = response.headers
    assert headers["Content-Type"] in ("binary/octet-stream", "application/octet-stream")
    assert headers["Accept-Ranges"] == "bytes"


@pytest.mark.parametrize(
    "building",
    [
        building
        for release in RELEASES
        if "load_curve_15min" not in release.file_types
        for upgrade in release.upgrades
        for building in list_buildings(release.key, "NY", upgrade, 1)
    ],
    ids=building_id_str,
)
@pytest.mark.network
def test_15min_load_curve_file_type_unavailable(building: Building):
    with pytest.raises(UnavailableFileTypeError):
        _ = building.load_curve_15min_path


@pytest.mark.parametrize(
    "building",
    [
        building
        for release in RELEASES
        if "load_curve_annual" in release.file_types
        for upgrade in release.upgrades
        for building in list_buildings(release.key, "NY", upgrade, 1)
    ],
    ids=building_id_str,
)
@pytest.mark.vcr
@pytest.mark.network
def test_annual_load_curve(building: Building):
    url = urljoin(OEDI_WEB_URL, building.load_curve_annual_path)
    response = httpx.head(url)
    _ = response.raise_for_status()
    headers = response.headers
    assert headers["Content-Type"] in ("binary/octet-stream", "application/octet-stream")
    assert headers["Accept-Ranges"] == "bytes"


@pytest.mark.parametrize(
    "building",
    [
        building
        for release in RELEASES
        if "load_curve_annual" not in release.file_types
        for upgrade in release.upgrades
        for building in list_buildings(release.key, "NY", upgrade, 1)
    ],
    ids=building_id_str,
)
def test_annual_load_curve_file_type_unavailable(building: Building):
    with pytest.raises(UnavailableFileTypeError):
        _ = building.load_curve_annual_path


@pytest.mark.parametrize(
    "building",
    [
        building
        for release in RELEASES
        if "hpxml" in release.file_types and "schedule" in release.file_types
        for upgrade in release.upgrades
        for building in list_buildings(release.key, "NY", upgrade, 1)
    ],
    ids=building_id_str,
)
@pytest.mark.vcr
def test_energy_models_path(building: Building):
    url = urljoin(OEDI_WEB_URL, building.energy_models_path)
    response = httpx.head(url)
    _ = response.raise_for_status()


@pytest.mark.parametrize(
    "building",
    [
        building
        for release in RELEASES
        if not ("hpxml" in release.file_types and "schedule" in release.file_types)
        for upgrade in release.upgrades
        for building in list_buildings(release.key, "NY", upgrade, 1)
    ],
    ids=building_id_str,
)
def test_energy_models_path_unavailable(building: Building):
    with pytest.raises(UnavailableFileTypeError):
        _ = building.energy_models_path
