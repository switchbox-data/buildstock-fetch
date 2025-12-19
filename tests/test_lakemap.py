import json
from collections.abc import Collection
from pathlib import Path
from typing import cast, get_args

import polars as pl
import pytest

from buildstock_fetch.lakemap import Y2021Map
from buildstock_fetch.releases import RELEASES
from buildstock_fetch.types import ReleaseKeyY2021, UpgradeID, USStateCode, normalize_upgrade_id


def assert_df_contains_columns(df: pl.LazyFrame, columns: Collection[str]):
    df_columns = df.collect_schema().names()
    for column in columns:
        assert column in df_columns
    assert len(columns) == len(df_columns)


BUILDING_IDS: dict[ReleaseKeyY2021, dict[UpgradeID, dict[USStateCode, list[int]]]] = {
    "com_2021_amy2018_1": {UpgradeID("0"): {"NY": [100007]}},
    "com_2021_tmy3_1": {UpgradeID("0"): {"NY": [100007]}},
    "res_2021_amy2018_1": {UpgradeID("0"): {"NY": [100075]}},
    "res_2021_tmy3_1": {UpgradeID("0"): {"NY": [100075]}},
}


@pytest.mark.network
class TestY2021Network:
    @pytest.mark.parametrize(
        "release",
        get_args(ReleaseKeyY2021),
    )
    def test_metadata(self, release: ReleaseKeyY2021):
        """Metadata must have all the expected fields"""
        expected_fields_path = Path(__file__).parent / "data" / "y2021_metadata_columns.json"
        expected_fields = cast(list[str], json.loads(expected_fields_path.read_text())[release])
        url = Y2021Map.release(release).metadata_parquet
        df = pl.scan_parquet(url)
        assert_df_contains_columns(df, expected_fields)

    @pytest.mark.parametrize(
        "release",
        get_args(ReleaseKeyY2021),
    )
    def test_timeseries_individual_buildings_by_state(self, release: ReleaseKeyY2021):
        """Individual building timeseries by state must must have all the expected fields"""
        release_obj = RELEASES[release]
        for upgrade in release_obj.upgrades:
            upgrade_dict = BUILDING_IDS[release][upgrade]
            if not upgrade_dict:
                raise RuntimeError("empty_buildings_list")
            for state, buildings in upgrade_dict.items():
                for building in buildings:
                    url = (
                        Y2021Map.release(release)
                        .individual_buildings.by_state(normalize_upgrade_id(upgrade), state)
                        .parquet(str(building))
                    )
                    expected_fields = cast(
                        dict[ReleaseKeyY2021, list[str]],
                        json.loads(
                            (
                                Path(__file__).parent
                                / "data"
                                / "y2021_expected_individual_buildings_timeseries_columns.json"
                            ).read_text()
                        ),
                    )
                    df = pl.scan_parquet(url)
                    assert_df_contains_columns(df, expected_fields[release])
