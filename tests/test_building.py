from urllib.parse import urljoin

import httpx
import pytest

from buildstock_fetch.building_ import Building
from buildstock_fetch.constants import OEDI_WEB_URL
from buildstock_fetch.main_new import list_buildings
from buildstock_fetch.releases import RELEASES
from buildstock_fetch.types import ReleaseKey, UpgradeID


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
