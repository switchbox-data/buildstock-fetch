"""Tests for ResStock 2025 EV charger (L1/L2) assignment."""

import polars as pl
import pytest

from tests.conftest import RESSTOCK_EV_REFERENCE_DIR
from utils.EVs.EVChargerAssigner import (
    RESSTOCK_LEVEL1_CHARGER_KW,
    RESSTOCK_LEVEL2_CHARGER_KW,
    EVChargerAssigner,
)
from utils.EVs.ev_utils import load_ev_charger_lookup

REF_DIR = RESSTOCK_EV_REFERENCE_DIR
CHARGER_TSV = REF_DIR / "Electric_Vehicle_Charger.tsv"


@pytest.fixture
def charger_lookup() -> pl.DataFrame:
    """Yes-ownership L1/L2 probabilities from Electric_Vehicle_Charger.tsv."""
    return load_ev_charger_lookup(CHARGER_TSV)


def test_load_ev_charger_lookup_yes_rows_only(charger_lookup):
    # The thin loader drops No-ownership rows but retains Void like the ownership loader.
    assert {"fpl", "building_type", "tenure", "p_level1", "p_level2", "p_void"} <= set(
        charger_lookup.columns
    )
    assert charger_lookup.height > 0

    # Only non-Void Yes rows are valid L1/L2 distributions and must sum to one.
    usable = charger_lookup.filter(pl.col("p_void") == 0)
    row_sum = usable["p_level1"] + usable["p_level2"]
    assert row_sum.to_numpy() == pytest.approx(1.0, abs=1e-3)

    # Void rows remain available so the assigner—not the loader—owns row validation.
    assert charger_lookup.filter(pl.col("p_void") == 1).height > 0


def test_assign_is_reproducible(charger_lookup):
    # Same seed → identical L1/L2 draws; different seed generally differs.
    vehicles = pl.DataFrame({
        "bldg_id": list(range(40)),
        "vehicle_id": [1] * 40,
        "fpl": ["400%+"] * 40,
        "building_type": ["Single-Family Detached"] * 40,
        "tenure": ["Owner"] * 40,
    })
    a = EVChargerAssigner(charger_lookup, random_state=42).assign(vehicles)
    b = EVChargerAssigner(charger_lookup, random_state=42).assign(vehicles)
    c = EVChargerAssigner(charger_lookup, random_state=7).assign(vehicles)

    assert a.select("charger_level").equals(b.select("charger_level"))
    assert not a.select("charger_level").equals(c.select("charger_level"))
    assert set(a["charger_level"].unique().to_list()) <= {"Level 1", "Level 2"}
    assert set(a["charger_power_kw"].unique().to_list()) <= {
        RESSTOCK_LEVEL1_CHARGER_KW,
        RESSTOCK_LEVEL2_CHARGER_KW,
    }


def test_assign_maps_level_to_resstock_kw(charger_lookup):
    # SFD owner 400%+ has ~49% L2 in the TSV — either level is valid; powers must match TRG.
    vehicles = pl.DataFrame({
        "bldg_id": [1, 2],
        "vehicle_id": [1, 1],
        "fpl": ["400%+", "400%+"],
        "building_type": ["Single-Family Detached", "Single-Family Detached"],
        "tenure": ["Owner", "Owner"],
    })
    result = EVChargerAssigner(charger_lookup, random_state=0).assign(vehicles)
    for level, power in zip(
        result["charger_level"].to_list(),
        result["charger_power_kw"].to_list(),
        strict=True,
    ):
        if level == "Level 1":
            assert power == RESSTOCK_LEVEL1_CHARGER_KW
        else:
            assert level == "Level 2"
            assert power == RESSTOCK_LEVEL2_CHARGER_KW


def test_assign_uses_custom_power_kw(charger_lookup):
    # Scenario overrides replace ResStock TRG defaults after the L1/L2 draw.
    vehicles = pl.DataFrame({
        "bldg_id": [1, 2, 3, 4],
        "vehicle_id": [1, 1, 1, 1],
        "fpl": ["400%+"] * 4,
        "building_type": ["Single-Family Detached"] * 4,
        "tenure": ["Owner"] * 4,
    })
    result = EVChargerAssigner(
        charger_lookup,
        random_state=0,
        level1_power_kw=1.4,
        level2_power_kw=7.2,
    ).assign(vehicles)
    for level, power in zip(
        result["charger_level"].to_list(),
        result["charger_power_kw"].to_list(),
        strict=True,
    ):
        if level == "Level 1":
            assert power == 1.4
        else:
            assert power == 7.2


def test_assign_rejects_negative_power_kw(charger_lookup):
    with pytest.raises(ValueError, match="level2_power_kw"):
        EVChargerAssigner(charger_lookup, level2_power_kw=-1.0)

def test_assign_empty_slots(charger_lookup):
    vehicles = pl.DataFrame(
        schema={
            "bldg_id": pl.Int64,
            "vehicle_id": pl.Int64,
            "fpl": pl.Utf8,
            "building_type": pl.Utf8,
            "tenure": pl.Utf8,
        }
    )
    result = EVChargerAssigner(charger_lookup, random_state=0).assign(vehicles)
    assert result.height == 0
    assert "charger_power_kw" in result.columns


def test_assign_raises_on_unmatched_demographics(charger_lookup):
    # Garbage join keys must fail fast rather than silently drop chargers.
    vehicles = pl.DataFrame({
        "bldg_id": [1],
        "vehicle_id": [1],
        "fpl": ["not-a-real-bin"],
        "building_type": ["Single-Family Detached"],
        "tenure": ["Owner"],
    })
    with pytest.raises(ValueError, match="charger lookup join missed"):
        EVChargerAssigner(charger_lookup, random_state=0).assign(vehicles)


def test_assign_raises_on_void_demographics(charger_lookup):
    # Real FPL + unavailable tenure is an impossible occupied/vacant combination.
    vehicles = pl.DataFrame({
        "bldg_id": [1],
        "vehicle_id": [1],
        "fpl": ["0-100%"],
        "building_type": ["Single-Family Detached"],
        "tenure": ["Not Available"],
    })

    # The loader retains this row, but assignment must reject it before sampling.
    with pytest.raises(ValueError, match="matched a Void row"):
        EVChargerAssigner(charger_lookup, random_state=0).assign(vehicles)


def test_high_p_level2_cell_favors_level2(charger_lookup):
    # SFD owner 400%+: p_level2 ≈ 0.495 — many draws should include both levels.
    row = charger_lookup.filter(
        (pl.col("fpl") == "400%+")
        & (pl.col("building_type") == "Single-Family Detached")
        & (pl.col("tenure") == "Owner")
    )
    assert row.height == 1
    p_l2 = float(row["p_level2"][0])
    assert 0.4 < p_l2 < 0.6

    n = 500
    vehicles = pl.DataFrame({
        "bldg_id": list(range(n)),
        "vehicle_id": [1] * n,
        "fpl": ["400%+"] * n,
        "building_type": ["Single-Family Detached"] * n,
        "tenure": ["Owner"] * n,
    })
    result = EVChargerAssigner(charger_lookup, random_state=1).assign(vehicles)
    share_l2 = (result["charger_level"] == "Level 2").mean()
    assert abs(float(share_l2) - p_l2) < 0.08
