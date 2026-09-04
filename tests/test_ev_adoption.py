"""Tests for NREL ResStock EV adoption lookup and EVAdoptionSampler.

Uses a tiny fixture TSV (tests/fixtures/ev_ownership_lookup_sample.tsv) so tests
do not require downloading the full ~19 MB national lookup.
"""
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from utils.EVs.EVAdoptionSampler import EVAdoptionSampler, assign_evs_by_priority_sampling
from utils.EVs.ev_utils import (
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


def test_assign_evs_by_priority_sampling_hits_weight_target():
    probs = np.array([0.04, 0.01])
    weights = np.array([1.0, 9.0])
    draws = np.array([0.5, 0.5])  # equal draws ⇒ key order follows p
    # target 0.1 of total weight 10 → need weight 1 → only the higher-p building
    evs = assign_evs_by_priority_sampling(probs, weights, draws, target=0.1)
    assert evs.tolist() == [1, 0]
    stock = float(np.dot(evs, weights) / weights.sum())
    assert stock == pytest.approx(0.1)


def test_assign_evs_by_priority_sampling_higher_p_wins_on_equal_draws():
    probs = np.array([0.01, 0.05, 0.0])
    weights = np.array([1.0, 1.0, 1.0])
    draws = np.array([0.5, 0.5, 0.5])
    evs = assign_evs_by_priority_sampling(probs, weights, draws, target=1.0 / 3.0)
    assert evs.tolist() == [0, 1, 0]


def test_assign_evs_by_priority_sampling_low_p_can_beat_high_p():
    """No all-or-nothing cascade: a lucky low-p building outranks an unlucky high-p one."""
    probs = np.array([0.5, 0.1])
    weights = np.array([1.0, 1.0])
    draws = np.array([0.9, 0.05])  # keys: 1.8 vs 0.5
    evs = assign_evs_by_priority_sampling(probs, weights, draws, target=0.5)
    assert evs.tolist() == [0, 1]


def test_assign_evs_by_priority_sampling_rates_scale_with_p():
    """Segment adoption rates stay ordered by p instead of saturating top-down."""
    rng = np.random.default_rng(0)
    n = 2000
    probs = np.concatenate([np.full(n, 0.02), np.full(n, 0.08)])
    weights = np.ones(2 * n)
    draws = rng.random(2 * n)
    evs = assign_evs_by_priority_sampling(probs, weights, draws, target=0.25)
    low_rate = evs[:n].mean()
    high_rate = evs[n:].mean()
    # Both segments participate, and the high-p segment adopts several times faster.
    assert 0.0 < low_rate < high_rate < 1.0
    assert high_rate / low_rate == pytest.approx(4.0, rel=0.35)


def test_assign_evs_by_priority_sampling_zero_p_can_adopt_late():
    """Unlike s*p scaling, p==0 buildings enter after all p>0 stock."""
    probs = np.array([0.05, 0.0])
    weights = np.array([1.0, 1.0])
    draws = np.array([0.5, 0.5])
    half = assign_evs_by_priority_sampling(probs, weights, draws, target=0.5)
    assert half.tolist() == [1, 0]
    full = assign_evs_by_priority_sampling(probs, weights, draws, target=1.0)
    assert full.tolist() == [1, 1]


def test_assign_evs_by_priority_sampling_nests_across_targets():
    rng = np.random.default_rng(7)
    probs = rng.random(500) * 0.2
    weights = np.ones(500)
    draws = rng.random(500)
    prev = assign_evs_by_priority_sampling(probs, weights, draws, target=0.1)
    for target in [0.2, 0.4, 0.6, 0.9]:
        cur = assign_evs_by_priority_sampling(probs, weights, draws, target=target)
        assert np.all(cur[prev == 1] == 1)
        prev = cur


def test_assign_evs_by_priority_sampling_invalid():
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        assign_evs_by_priority_sampling(np.array([0.1]), np.array([1.0]), np.array([0.2]), 1.5)
    with pytest.raises(ValueError, match="weights must be non-negative"):
        assign_evs_by_priority_sampling(np.array([0.1]), np.array([-1.0]), np.array([0.2]), 0.5)


def test_target_adoption_rate_hits_stock_share(ev_lookup, md_ev_metadata):
    # Equal weights ⇒ stock share is exact at 1/N granularity (no overshoot ambiguity).
    metadata = md_ev_metadata.with_columns(pl.lit(1.0).alias("weight"))
    target = 2.0 / 3.0  # 2 of 3 occupied buildings
    result = EVAdoptionSampler(
        ev_ownership_df=ev_lookup,
        random_state=42,
        target_adoption_rate=target,
    ).sample(metadata)
    occupied = result.filter(~pl.col("is_vacant"))
    stock = (occupied["evs"] * occupied["weight"]).sum() / occupied["weight"].sum()
    assert stock == pytest.approx(target, abs=1e-12)
    assert occupied["evs"].sum() == 2


def test_target_adoption_rate_may_overshoot_one_building_with_unequal_weights(
    ev_lookup, md_ev_metadata
):
    """Prefix rule includes the building that crosses τ, so stock may exceed τ slightly."""
    target = 0.5
    result = EVAdoptionSampler(
        ev_ownership_df=ev_lookup,
        random_state=42,
        target_adoption_rate=target,
    ).sample(md_ev_metadata)
    occupied = result.filter(~pl.col("is_vacant"))
    total_w = occupied["weight"].sum()
    stock = (occupied["evs"] * occupied["weight"]).sum() / total_w
    max_w = occupied["weight"].max()
    assert stock >= target - 1e-12
    assert stock <= target + max_w / total_w + 1e-12


def test_target_adoption_requires_weight(ev_lookup):
    metadata = pl.DataFrame({
        "bldg_id": [1],
        "occupants": [2],
        "income_bucket": [10],
        "metro": ["urban"],
        "fpl": ["400%+"],
        "building_type": ["Single-Family Detached"],
        "tenure": ["Owner"],
        "puma_dependency": ["MD, 00805"],
        "is_vacant": [False],
    })
    sampler = EVAdoptionSampler(
        ev_ownership_df=ev_lookup,
        target_adoption_rate=0.25,
    )
    with pytest.raises(ValueError, match="Missing EV adoption metadata columns: weight"):
        sampler.sample(metadata)


def test_target_adoption_preserves_high_p_first(ev_lookup):
    """At 50% stock with equal weights, only the higher-p building gets an EV."""
    metadata = pl.DataFrame({
        "bldg_id": [1, 2],
        "occupants": [2, 2],
        "income_bucket": [10, 4],
        "metro": ["urban", "urban"],
        "weight": [1.0, 1.0],
        "fpl": ["400%+", "0-100%"],
        "building_type": ["Single-Family Detached", "Single-Family Attached"],
        "tenure": ["Owner", "Renter"],
        "puma_dependency": ["MD, 00805", "MD, 00506"],
        "is_vacant": [False, False],
    })
    result = EVAdoptionSampler(
        ev_ownership_df=ev_lookup,
        random_state=0,
        target_adoption_rate=0.5,
    ).sample(metadata)
    owner = result.filter(pl.col("bldg_id") == 1).row(0, named=True)
    renter = result.filter(pl.col("bldg_id") == 2).row(0, named=True)
    assert owner["ev_ownership_probability"] > renter["ev_ownership_probability"]
    assert owner["evs"] == 1
    assert renter["evs"] == 0


def test_target_adoption_vacants_always_zero(ev_lookup, md_ev_metadata):
    result = EVAdoptionSampler(
        ev_ownership_df=ev_lookup,
        random_state=42,
        target_adoption_rate=1.0,
    ).sample(md_ev_metadata)
    vacant = result.filter(pl.col("is_vacant"))
    assert vacant["evs"].to_list() == [0]
    occupied = result.filter(~pl.col("is_vacant"))
    assert occupied["evs"].to_list() == [1, 1, 1]


def test_target_adoption_nested_across_rates(ev_lookup, md_ev_metadata):
    low = EVAdoptionSampler(
        ev_ownership_df=ev_lookup,
        random_state=42,
        target_adoption_rate=0.2,
    ).sample(md_ev_metadata)
    high = EVAdoptionSampler(
        ev_ownership_df=ev_lookup,
        random_state=42,
        target_adoption_rate=0.8,
    ).sample(md_ev_metadata)
    low_ids = set(low.filter(pl.col("evs") == 1)["bldg_id"].to_list())
    high_ids = set(high.filter(pl.col("evs") == 1)["bldg_id"].to_list())
    assert low_ids <= high_ids


def test_target_adoption_rate_invalid_raises(ev_lookup):
    with pytest.raises(ValueError, match="target_adoption_rate must be in"):
        EVAdoptionSampler(ev_ownership_df=ev_lookup, target_adoption_rate=1.5)
