from pathlib import Path
from urllib.parse import urljoin

import httpx
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from buildstock_fetch import building_ as building_module
from buildstock_fetch.building_ import Building, UnavailableFileTypeError
from buildstock_fetch.constants import METADATA_DIR, OEDI_WEB_URL
from buildstock_fetch.main_new import list_buildings
from buildstock_fetch.releases import RELEASES
from buildstock_fetch.types import ReleaseKey, UpgradeID

_BUCKET_UNIFORMITY_EXPECTATIONS = {
    "max_deviation_pct": 35.0,
    "max_empty_buckets": 0,
}


def building_id_str(building: Building) -> str:
    return f"{building.release}-{building.upgrade}-{building.state}"


def _release_metadata_path(release: ReleaseKey) -> Path:
    release_obj = RELEASES[release]
    return (
        METADATA_DIR
        / f"product={release_obj.product}"
        / f"release_year={release_obj.year}"
        / f"weather_file={release_obj.weather}"
        / f"release_version={release_obj.version}"
    )


def _bucket_stats_for_release(release: ReleaseKey, bucket_count: int) -> tuple[int, int, float]:
    counts = [0] * bucket_count
    dataset = ds.dataset(_release_metadata_path(release), format="parquet", partitioning="hive")
    building = Building(0, release, sorted(RELEASES[release].upgrades, key=int)[0], "NY", None)
    total = 0

    for batch in dataset.to_batches(columns=["bldg_id"], batch_size=65_536):
        building_ids = batch.column(0).to_pylist()
        total += len(building_ids)
        for bldg_id in building_ids:
            object.__setattr__(building, "id", bldg_id)
            counts[building.bucket] += 1

    expected = total / bucket_count
    max_deviation_pct = max(abs(count - expected) for count in counts) / expected * 100
    empty_buckets = sum(1 for count in counts if count == 0)
    return total, empty_buckets, max_deviation_pct


def _sample_building(release: ReleaseKey, upgrade: UpgradeID | None = None) -> Building:
    release_obj = RELEASES[release]
    chosen_upgrade = upgrade or sorted(release_obj.upgrades, key=int)[0]
    buildings = list_buildings(release, "NY", chosen_upgrade, 1)
    assert buildings, f"No sample building found for {release} upgrade {chosen_upgrade}"
    return buildings[0]


@pytest.mark.parametrize(
    ("release", "expected_fragment"),
    [
        ("res_2021_tmy3_1", "metadata/metadata.parquet"),
        ("res_2022_tmy3_1", "metadata_and_annual_results/by_state/state=NY/parquet/NY_baseline_metadata_and_annual_results.parquet"),
        ("res_2024_tmy3_1", "metadata/baseline.parquet"),
        ("res_2024_amy2018_2", "metadata_and_annual_results/by_state/state=NY/parquet/NY_baseline_metadata_and_annual_results.parquet"),
        ("com_2024_amy2018_2", "metadata_and_annual_results/by_state_and_county/full/parquet/state=NY/county="),
        ("res_2025_amy2018_1", "metadata_and_annual_results/by_state/full/parquet/state=NY/NY_upgrade0.parquet"),
        ("com_2025_amy2018_3", "metadata_and_annual_results/by_state_and_county/full/parquet/state=NY/county="),
    ],
)
def test_metadata_path_patterns(release: ReleaseKey, expected_fragment: str):
    building = _sample_building(release)
    assert expected_fragment in building.metadata_path


@pytest.mark.parametrize(
    "release",
    sorted(release.key for release in RELEASES if "load_curve_15min" in release.file_types),
)
def test_15min_load_curve_path_uses_bucketed_filename(release: ReleaseKey):
    building = _sample_building(release)
    expected_filename = f"bucket_{building.bucket:04d}_up{int(building.upgrade):02d}.parquet"
    path = Path(building.file_path("load_curve_15min"))

    assert path.parts[-3] == f"state={building.state}"
    assert path.parts[-2] == f"upgrade={int(building.upgrade):02d}"
    assert path.name == expected_filename


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize(
    "release",
    sorted(release.key for release in RELEASES if "load_curve_15min" in release.file_types),
)
def test_15min_load_curve_remote_schema_includes_bldg_id_for_all_releases(release: ReleaseKey):
    building = _sample_building(release)
    url = urljoin(OEDI_WEB_URL, building.load_curve_15min_path)

    with httpx.Client(timeout=60, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()

    schema = pq.read_schema(pa.BufferReader(response.content))
    assert "bldg_id" in schema.names, f"{release} 15min parquet is missing bldg_id"


@pytest.mark.parametrize(
    "release",
    sorted(release.key for release in RELEASES if "load_curve_15min" not in release.file_types),
)
def test_15min_load_curve_file_type_unavailable(release: ReleaseKey):
    building = _sample_building(release)
    with pytest.raises(UnavailableFileTypeError):
        _ = building.load_curve_15min_path


@pytest.mark.parametrize(
    ("release", "expected_subpath"),
    [
        ("res_2022_tmy3_1", "metadata_and_annual_results/by_state/state=NY/parquet/"),
        ("com_2024_amy2018_2", "metadata_and_annual_results/by_state_and_county/full/parquet/state=NY/"),
        ("res_2025_amy2018_1", "metadata_and_annual_results/by_state/full/parquet/state=NY/"),
        ("com_2025_amy2018_3", "metadata_and_annual_results/by_state_and_county/full/parquet/state=NY/"),
    ],
)
def test_annual_load_curve_path_patterns(release: ReleaseKey, expected_subpath: str):
    building = _sample_building(release)
    assert expected_subpath in building.load_curve_annual_path
    assert building.load_curve_annual_path.endswith(".parquet")


@pytest.mark.parametrize(
    "release",
    sorted(release.key for release in RELEASES if "load_curve_annual" not in release.file_types),
)
def test_annual_load_curve_file_type_unavailable(release: ReleaseKey):
    building = _sample_building(release)
    with pytest.raises(UnavailableFileTypeError):
        _ = building.load_curve_annual_path


@pytest.mark.parametrize(
    ("release", "expected_suffix"),
    [
        ("res_2022_tmy3_1", ".zip"),
        ("res_2024_tmy3_2", ".zip"),
        ("res_2025_amy2018_1", ".zip"),
    ],
)
def test_energy_models_path_patterns(release: ReleaseKey, expected_suffix: str):
    building = _sample_building(release)
    assert building.energy_models_path.endswith(expected_suffix)


@pytest.mark.parametrize(
    "release",
    sorted(release.key for release in RELEASES if not ("hpxml" in release.file_types and "schedule" in release.file_types)),
)
def test_energy_models_path_unavailable(release: ReleaseKey):
    building = _sample_building(release)
    with pytest.raises(UnavailableFileTypeError):
        _ = building.energy_models_path


@pytest.mark.parametrize("release", sorted(RELEASES.keys))
def test_bucket_distribution_stays_uniform_against_256_bucket_baseline(release: ReleaseKey):
    max_deviation_pct = _BUCKET_UNIFORMITY_EXPECTATIONS["max_deviation_pct"]
    max_empty_buckets = _BUCKET_UNIFORMITY_EXPECTATIONS["max_empty_buckets"]

    total, empty_buckets, deviation_pct = _bucket_stats_for_release(release, building_module._NUM_BUCKETS)

    assert total > 0
    assert empty_buckets <= max_empty_buckets, (
        f"{release} had {empty_buckets} empty buckets using Building.bucket; "
        f"expected at most {max_empty_buckets}"
    )
    assert deviation_pct <= max_deviation_pct, (
        f"{release} bucket distribution deviated by {deviation_pct:.2f}% using Building.bucket; "
        f"expected at most {max_deviation_pct:.2f}%"
    )


@pytest.mark.vcr
@pytest.mark.network
@pytest.mark.parametrize(
    "building",
    [
        _sample_building("res_2021_tmy3_1"),
        _sample_building("res_2024_tmy3_1"),
        _sample_building("com_2024_amy2018_2"),
    ],
    ids=building_id_str,
)
def test_metadata_path_smoke(building: Building):
    response = httpx.head(urljoin(OEDI_WEB_URL, building.metadata_path))
    _ = response.raise_for_status()


@pytest.mark.vcr
@pytest.mark.network
@pytest.mark.parametrize(
    "building",
    [
        _sample_building("res_2024_tmy3_2"),
        _sample_building("com_2024_amy2018_2"),
    ],
    ids=building_id_str,
)
def test_load_curve_path_smoke(building: Building):
    response = httpx.head(urljoin(OEDI_WEB_URL, building.load_curve_15min_path))
    _ = response.raise_for_status()
