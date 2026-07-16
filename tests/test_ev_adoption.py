"""Tests for NREL ResStock EV adoption lookup and EVAdoptionSampler.

Uses a tiny fixture TSV (tests/fixtures/ev_ownership_lookup_sample.tsv) so tests
do not require downloading the full ~19 MB national lookup.
"""
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from utils.EVAdoptionSampler import EVAdoptionSampler
from utils.ev_utils import (
    load_ev_ownership_lookup,
    resstock_puma_dependency,
    state_ev_ownership_rate,
)

FIXTURE_LOOKUP = Path(__file__).parent / "fixtures/ev_ownership_lookup_sample.tsv"


@pytest.fixture
def ev_lookup():
    return load_ev_ownership_lookup(FIXTURE_LOOKUP, "MD")


@pytest.fixture
def md_ev_metadata():
    return pl.DataFrame({
        "bldg_id": [1, 2, 3, 4],
        "occupants": [2, 3, 4, 1],
        "income_bucket": [6, 8, 10, 4],
        "metro": ["urban", "urban", "suburban", "urban"],
        "weight": [100.0, 200.0, 150.0, 50.0],
        "fpl": ["400%+", "0-100%", "400%+", "400%+"],
        "building_type": [
            "Single-Family Detached",
            "Single-Family Attached",
            "Multi-Family with 5+ Units",
            "Single-Family Detached",
        ],
        "tenure": ["Owner", "Renter", "Renter", "Not Available"],
        "puma_dependency": ["MD, 00805", "MD, 00506", "MD, 01202", "MD, 00805"],
        "is_vacant": [False, False, False, True],
    })


@pytest.fixture
def ev_sampler(ev_lookup):
    return EVAdoptionSampler(ev_ownership_df=ev_lookup, random_state=42)


def test_resstock_puma_dependency():
    assert resstock_puma_dependency("MD", "G24000805") == "MD, 00805"
    assert resstock_puma_dependency("NY", "G36006101") == "NY, 06101"


def test_load_ev_ownership_lookup(ev_lookup):
    assert ev_lookup.height == 4
    assert set(ev_lookup.columns) == {
        "fpl",
        "building_type",
        "puma_dependency",
        "tenure",
        "ev_ownership_probability",
        "source_weight",
    }


def test_state_ev_ownership_rate(ev_lookup):
    rate = state_ev_ownership_rate(ev_lookup, "MD")
    assert 0 < rate < 0.05


def test_predict_num_evs_assigns_probabilities(ev_sampler, md_ev_metadata):
    result = ev_sampler.sample(md_ev_metadata)

    owner_row = result.filter(pl.col("bldg_id") == 1).row(0, named=True)
    renter_row = result.filter(pl.col("bldg_id") == 2).row(0, named=True)
    vacant_row = result.filter(pl.col("bldg_id") == 4).row(0, named=True)

    assert owner_row["ev_ownership_probability"] == pytest.approx(0.0241447)
    assert renter_row["ev_ownership_probability"] == pytest.approx(0.0013446)
    assert vacant_row["ev_ownership_probability"] == 0.0
    assert vacant_row["evs"] == 0


def test_predict_num_evs_reproducible(ev_sampler, md_ev_metadata):
    result1 = ev_sampler.sample(md_ev_metadata)
    result2 = ev_sampler.sample(md_ev_metadata)
    assert result1["evs"].to_list() == result2["evs"].to_list()


def test_predict_num_evs_max_one_per_household(ev_sampler, md_ev_metadata):
    result = ev_sampler.sample(md_ev_metadata)
    assert result["evs"].max() <= 1
    assert set(result.filter(~pl.col("is_vacant"))["evs"].unique().to_list()).issubset({0, 1})


def test_predict_num_evs_missing_columns(ev_sampler):
    incomplete = pl.DataFrame({"bldg_id": [1], "occupants": [2]})
    with pytest.raises(ValueError, match="Missing EV adoption metadata columns"):
        ev_sampler.sample(incomplete)


def test_predict_num_evs_bernoulli_sampling(ev_sampler):
    """High P(EV) row should be sampled as EV with fixed seed."""
    high_prob_metadata = pl.DataFrame({
        "bldg_id": [99],
        "occupants": [2],
        "income_bucket": [10],
        "metro": ["urban"],
        "weight": [1.0],
        "fpl": ["400%+"],
        "building_type": ["Single-Family Detached"],
        "tenure": ["Owner"],
        "puma_dependency": ["MD, 00805"],
        "is_vacant": [False],
    })

    rng = np.random.default_rng(42)
    draws = rng.random(1)
    expected_evs = int(draws[0] < 0.0241447)

    result = ev_sampler.sample(high_prob_metadata)
    assert result["evs"][0] == expected_evs


def test_predict_num_evs_join_miss_raises(ev_sampler):
    """Occupied buildings with no lookup segment should raise."""
    unmatched_metadata = pl.DataFrame({
        "bldg_id": [99],
        "occupants": [2],
        "income_bucket": [10],
        "metro": ["urban"],
        "weight": [1.0],
        "fpl": ["400%+"],
        "building_type": ["Mobile Home"],
        "tenure": ["Owner"],
        "puma_dependency": ["MD, 00805"],
        "is_vacant": [False],
    })

    with pytest.raises(ValueError, match="lookup join missed"):
        ev_sampler.sample(unmatched_metadata)


def test_predict_num_evs_requires_matching_puma(ev_sampler):
    """Join keys include puma_dependency, matching ev_adoption.ipynb."""
    metadata = pl.DataFrame({
        "bldg_id": [1],
        "occupants": [2],
        "income_bucket": [6],
        "metro": ["urban"],
        "weight": [100.0],
        "fpl": ["400%+"],
        "building_type": ["Single-Family Detached"],
        "tenure": ["Owner"],
        "puma_dependency": ["MD, 99999"],
        "is_vacant": [False],
    })

    with pytest.raises(ValueError, match="lookup join missed"):
        ev_sampler.sample(metadata)
