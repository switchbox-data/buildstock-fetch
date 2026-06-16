from pathlib import Path

import polars as pl
import pytest

from utils.ev_adoption import (
    NATIONAL_EV_OWNERSHIP_RATE,
    add_ev_ownership_probability,
    load_ev_ownership_lookup,
    resstock_puma_dependency,
    sample_ev_ownership,
    summarize_ev_adoption,
)

LOOKUP_PATH = (
    Path(__file__).resolve().parents[1]
    / "utils/ev_data/inputs/resstock_ev_reference/Electric_Vehicle_Ownership.tsv"
)
METADATA_PATH = Path(
    "/ebs/data/nrel/resstock/res_2024_amy2018_2_sb/metadata/state=MD/upgrade=00/metadata-sb.parquet"
)


@pytest.fixture
def ev_lookup() -> pl.DataFrame:
    if not LOOKUP_PATH.exists():
        pytest.skip(f"Missing {LOOKUP_PATH}; run just download-resstock-ev-reference")
    return load_ev_ownership_lookup(LOOKUP_PATH)


def test_resstock_puma_dependency():
    assert resstock_puma_dependency("MD", "G24000901") == "MD, 00901"
    assert resstock_puma_dependency("CA", "G06037101") == "CA, 37101"


def test_lookup_join_for_md_sample(ev_lookup: pl.DataFrame):
    if not METADATA_PATH.exists():
        pytest.skip(f"Missing local metadata at {METADATA_PATH}")

    metadata = pl.read_parquet(METADATA_PATH)
    with_probs = add_ev_ownership_probability(metadata, lookup=ev_lookup)

    occupied = with_probs.filter(pl.col("in.vacancy_status") == "Occupied")
    assert occupied["ev_ownership_probability"].null_count() == 0
    assert occupied["ev_ownership_probability"].min() >= 0.0
    assert occupied["ev_ownership_probability"].max() <= 1.0
    assert occupied["ev_ownership_probability"].mean() == pytest.approx(0.02, abs=0.02)


def test_sample_ev_ownership_reproducible(ev_lookup: pl.DataFrame):
    metadata = pl.DataFrame({
        "bldg_id": [1, 2, 3, 4],
        "weight": [1.0, 1.0, 1.0, 1.0],
        "in.state": ["MD", "MD", "MD", "MD"],
        "in.puma": ["G24000901", "G24000901", "G24000901", "G24000901"],
        "in.federal_poverty_level": ["400%+", "0-100%", "400%+", "200-300%"],
        "in.geometry_building_type_recs": [
            "Single-Family Detached",
            "Multi-Family with 5+ Units",
            "Single-Family Detached",
            "Single-Family Detached",
        ],
        "in.tenure": ["Owner", "Renter", "Owner", "Owner"],
        "in.vacancy_status": ["Occupied", "Occupied", "Occupied", "Vacant"],
    })

    sampled_a = sample_ev_ownership(metadata, seed=7, lookup=ev_lookup)
    sampled_b = sample_ev_ownership(metadata, seed=7, lookup=ev_lookup)
    assert sampled_a["has_ev"].to_list() == sampled_b["has_ev"].to_list()
    assert sampled_a.filter(pl.col("is_vacant"))["ev_count"].to_list() == [0]


def test_summarize_ev_adoption(ev_lookup: pl.DataFrame):
    metadata = pl.DataFrame({
        "bldg_id": [1, 2],
        "weight": [10.0, 5.0],
        "in.state": ["MD", "MD"],
        "in.puma": ["G24000901", "G24000901"],
        "in.federal_poverty_level": ["400%+", "0-100%"],
        "in.geometry_building_type_recs": ["Single-Family Detached", "Mobile Home"],
        "in.tenure": ["Owner", "Renter"],
        "in.vacancy_status": ["Occupied", "Occupied"],
    })
    sampled = sample_ev_ownership(metadata, seed=0, lookup=ev_lookup)
    summary = summarize_ev_adoption(sampled)
    assert summary.filter(pl.col("metric") == "buildings")["value"][0] == 2.0


def test_national_fallback_rate_is_documented():
    assert NATIONAL_EV_OWNERSHIP_RATE == pytest.approx(0.0145, abs=0.001)
