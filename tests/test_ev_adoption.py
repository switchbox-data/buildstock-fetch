"""Tests for NREL ResStock EV adoption lookup and predict_num_EVs().

Uses a tiny fixture TSV (tests/fixtures/ev_ownership_lookup_sample.tsv) so tests
do not require downloading the full ~19 MB national lookup.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from utils.ev_demand import EVDemandCalculator, MetadataDataFrameError
from utils.ev_utils import (
    load_ev_ownership_lookup,
    resstock_puma_dependency,
    state_ev_ownership_rate,
)

FIXTURE_LOOKUP = Path(__file__).parent / "fixtures/ev_ownership_lookup_sample.tsv"


@pytest.fixture
def ev_lookup():
    return load_ev_ownership_lookup(FIXTURE_LOOKUP)


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
def ev_calculator(md_ev_metadata, ev_lookup):
    return EVDemandCalculator(
        metadata_df=md_ev_metadata,
        nhts_df=pl.DataFrame(),
        pums_df=pl.DataFrame(),
        ev_ownership_df=ev_lookup,
        state_ev_rate=state_ev_ownership_rate(ev_lookup, "MD"),
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 1, 7),
        random_state=42,
    )


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


def test_predict_num_evs_assigns_probabilities(ev_calculator, md_ev_metadata):
    result = ev_calculator.predict_num_EVs()

    owner_row = result.filter(pl.col("bldg_id") == 1).row(0, named=True)
    renter_row = result.filter(pl.col("bldg_id") == 2).row(0, named=True)
    vacant_row = result.filter(pl.col("bldg_id") == 4).row(0, named=True)

    assert owner_row["ev_ownership_probability"] == pytest.approx(0.0241447)
    assert renter_row["ev_ownership_probability"] == pytest.approx(0.0013446)
    assert vacant_row["ev_ownership_probability"] == 0.0
    assert vacant_row["evs"] == 0
    assert vacant_row["has_ev"] is False


def test_predict_num_evs_reproducible(ev_calculator):
    result1 = ev_calculator.predict_num_EVs()
    result2 = ev_calculator.predict_num_EVs()
    assert result1["evs"].to_list() == result2["evs"].to_list()
    assert result1["has_ev"].to_list() == result2["has_ev"].to_list()


def test_predict_num_evs_max_one_per_household(ev_calculator):
    result = ev_calculator.predict_num_EVs()
    assert result["evs"].max() <= 1
    assert set(result.filter(~pl.col("is_vacant"))["evs"].unique().to_list()).issubset({0, 1})


def test_predict_num_evs_missing_columns(ev_calculator):
    incomplete = pl.DataFrame({"bldg_id": [1], "occupants": [2]})
    with pytest.raises(ValueError, match="Missing EV adoption metadata columns"):
        ev_calculator.predict_num_EVs(incomplete)


def test_predict_num_evs_without_metadata(ev_calculator):
    ev_calculator.metadata_df = None
    with pytest.raises(MetadataDataFrameError):
        ev_calculator.predict_num_EVs(None)


def test_predict_num_evs_bernoulli_sampling(ev_calculator):
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
    expected_has_ev = draws[0] < 0.0241447

    result = ev_calculator.predict_num_EVs(high_prob_metadata)
    assert result["has_ev"][0] == expected_has_ev
    assert result["evs"][0] == int(expected_has_ev)
