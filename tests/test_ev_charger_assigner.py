"""Tests for ResStock 2025 EV charger (L1/L2) assignment."""

import numpy as np
import polars as pl
import pytest

from tests.conftest import RESSTOCK_EV_REFERENCE_DIR
from utils.EVs.EVChargerAssigner import (
    DEFAULT_CHARGER_BUFFER_FRACTION,
    RESSTOCK_LEVEL1_CHARGER_KW,
    RESSTOCK_LEVEL2_CHARGER_KW,
    EVChargerAssigner,
)
from utils.EVs.charging import is_home_charging_soc_feasible
from utils.EVs.ev_utils import load_ev_charger_lookup

REF_DIR = RESSTOCK_EV_REFERENCE_DIR
CHARGER_TSV = REF_DIR / "Electric_Vehicle_Charger.tsv"

# Short always-home / zero-discharge schedules keep both L1 and L2 SOC-feasible.
_NUM_HOURS = 24
_CAPACITY_KWH = 60.0


def _always_home_presence(n: int = _NUM_HOURS) -> pl.DataFrame:
    return pl.DataFrame({"at_home": [True] * n})


def _zero_discharge(n: int = _NUM_HOURS) -> np.ndarray:
    return np.zeros(n, dtype=np.float64)


def _vehicles_frame(
    n: int,
    *,
    fpl: str = "400%+",
    building_type: str = "Single-Family Detached",
    tenure: str = "Owner",
    capacity_kwh: float = _CAPACITY_KWH,
) -> pl.DataFrame:
    return pl.DataFrame({
        "bldg_id": list(range(n)),
        "vehicle_id": [1] * n,
        "fpl": [fpl] * n,
        "building_type": [building_type] * n,
        "tenure": [tenure] * n,
        "battery_capacity_kwh": [capacity_kwh] * n,
    })


def _easy_schedules(vehicles: pl.DataFrame) -> tuple[dict, dict]:
    """Presence/discharge where both charger levels are feasible (home, no trips)."""
    presence = {
        (row["bldg_id"], int(row["vehicle_id"])): _always_home_presence()
        for row in vehicles.iter_rows(named=True)
    }
    discharge = {
        (row["bldg_id"], int(row["vehicle_id"])): _zero_discharge()
        for row in vehicles.iter_rows(named=True)
    }
    return presence, discharge


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
    vehicles = _vehicles_frame(40)
    presence, discharge = _easy_schedules(vehicles)
    kwargs = dict(
        presence_by_vehicle=presence,
        discharge_kwh_by_vehicle=discharge,
        buffer_fraction=0.0,
    )
    a = EVChargerAssigner(charger_lookup, random_state=42).assign(vehicles, **kwargs)
    b = EVChargerAssigner(charger_lookup, random_state=42).assign(vehicles, **kwargs)
    c = EVChargerAssigner(charger_lookup, random_state=7).assign(vehicles, **kwargs)

    assert a.select("charger_level").equals(b.select("charger_level"))
    assert not a.select("charger_level").equals(c.select("charger_level"))
    assert set(a["charger_level"].unique().to_list()) <= {"Level 1", "Level 2"}
    assert set(a["charger_power_kw"].unique().to_list()) <= {
        RESSTOCK_LEVEL1_CHARGER_KW,
        RESSTOCK_LEVEL2_CHARGER_KW,
    }


def test_assign_maps_level_to_resstock_kw(charger_lookup):
    # SFD owner 400%+ has ~49% L2 in the TSV — either level is valid; powers must match TRG.
    vehicles = _vehicles_frame(2)
    presence, discharge = _easy_schedules(vehicles)
    result = EVChargerAssigner(charger_lookup, random_state=0).assign(
        vehicles,
        presence_by_vehicle=presence,
        discharge_kwh_by_vehicle=discharge,
        buffer_fraction=0.0,
    )
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
    vehicles = _vehicles_frame(4)
    presence, discharge = _easy_schedules(vehicles)
    result = EVChargerAssigner(
        charger_lookup,
        random_state=0,
        level1_power_kw=1.4,
        level2_power_kw=7.2,
    ).assign(
        vehicles,
        presence_by_vehicle=presence,
        discharge_kwh_by_vehicle=discharge,
        buffer_fraction=0.0,
    )
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
            "battery_capacity_kwh": pl.Float64,
        }
    )
    result = EVChargerAssigner(charger_lookup, random_state=0).assign(
        vehicles,
        presence_by_vehicle={},
        discharge_kwh_by_vehicle={},
    )
    assert result.height == 0
    assert "charger_power_kw" in result.columns


def test_assign_raises_on_unmatched_demographics(charger_lookup):
    # Garbage join keys must fail fast rather than silently drop chargers.
    vehicles = _vehicles_frame(1, fpl="not-a-real-bin")
    presence, discharge = _easy_schedules(vehicles)
    with pytest.raises(ValueError, match="charger lookup join missed"):
        EVChargerAssigner(charger_lookup, random_state=0).assign(
            vehicles,
            presence_by_vehicle=presence,
            discharge_kwh_by_vehicle=discharge,
        )


def test_assign_raises_on_void_demographics(charger_lookup):
    # Real FPL + unavailable tenure is an impossible occupied/vacant combination.
    vehicles = _vehicles_frame(1, fpl="0-100%", tenure="Not Available")
    presence, discharge = _easy_schedules(vehicles)

    # The loader retains this row, but assignment must reject it before sampling.
    with pytest.raises(ValueError, match="matched a Void row"):
        EVChargerAssigner(charger_lookup, random_state=0).assign(
            vehicles,
            presence_by_vehicle=presence,
            discharge_kwh_by_vehicle=discharge,
        )


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
    vehicles = _vehicles_frame(n)
    presence, discharge = _easy_schedules(vehicles)
    result = EVChargerAssigner(charger_lookup, random_state=1).assign(
        vehicles,
        presence_by_vehicle=presence,
        discharge_kwh_by_vehicle=discharge,
        buffer_fraction=0.0,
    )
    share_l2 = (result["charger_level"] == "Level 2").mean()
    assert abs(float(share_l2) - p_l2) < 0.08


def test_assign_forces_level2_when_l1_infeasible(charger_lookup):
    """After a large first trip, a short home window recovers enough for L2 but not L1."""
    # Start full at 40 kWh.
    # Hours 0-3 away: draw 30 kWh → SOC=10.
    # Hours 4-5 home (2h): L1 adds 3.2 → SOC=13.2; L2 adds 11.38 → SOC=21.38.
    # Hours 6-8 away: draw 15 kWh → L1 underflows (13.2<15), L2 ok (21.38≥15).
    n_hours = 10
    at_home = np.array(
        [False, False, False, False, True, True, False, False, False, True]
    )
    discharge = np.zeros(n_hours, dtype=np.float64)
    discharge[0:4] = 30.0 / 4.0
    discharge[6:9] = 15.0 / 3.0
    capacity = 40.0

    assert not is_home_charging_soc_feasible(
        at_home,
        discharge,
        battery_capacity_kwh=capacity,
        charger_power_kw=RESSTOCK_LEVEL1_CHARGER_KW,
        buffer_fraction=0.0,
    )
    assert is_home_charging_soc_feasible(
        at_home,
        discharge,
        battery_capacity_kwh=capacity,
        charger_power_kw=RESSTOCK_LEVEL2_CHARGER_KW,
        buffer_fraction=0.0,
    )

    vehicles = _vehicles_frame(20, capacity_kwh=capacity)
    presence = {
        (row["bldg_id"], int(row["vehicle_id"])): pl.DataFrame({"at_home": at_home.tolist()})
        for row in vehicles.iter_rows(named=True)
    }
    discharge_by = {
        (row["bldg_id"], int(row["vehicle_id"])): discharge.copy()
        for row in vehicles.iter_rows(named=True)
    }
    result = EVChargerAssigner(charger_lookup, random_state=0).assign(
        vehicles,
        presence_by_vehicle=presence,
        discharge_kwh_by_vehicle=discharge_by,
        buffer_fraction=0.0,
    )
    assert (result["charger_level"] == "Level 2").all()
    assert (result["charger_power_kw"] == RESSTOCK_LEVEL2_CHARGER_KW).all()


def test_assign_raises_when_neither_level_feasible(charger_lookup):
    # Always away with trip draw exceeding pack size → no home charging can help.
    n_hours = 8
    at_home = np.zeros(n_hours, dtype=bool)
    discharge = np.full(n_hours, 5.0, dtype=np.float64)  # 40 kWh total > 20 kWh pack
    vehicles = _vehicles_frame(1, capacity_kwh=20.0)
    key = (0, 1)
    with pytest.raises(ValueError, match="No ResStock EV charger level"):
        EVChargerAssigner(charger_lookup, random_state=0).assign(
            vehicles,
            presence_by_vehicle={key: pl.DataFrame({"at_home": at_home.tolist()})},
            discharge_kwh_by_vehicle={key: discharge},
            buffer_fraction=0.0,
        )


def test_buffer_fraction_can_exclude_level1(charger_lookup):
    """A schedule feasible at L1 with buffer=0 becomes L2-only once buffer is applied."""
    # Same two-trip skeleton as the L2-force test, but tune the second draw so L1
    # barely works at buffer=0 and fails at buffer=0.2.
    # After trip1 (30 kWh) SOC=10; 3 home hours: L1 → SOC=14.8.
    # Second trip 14 kWh: ok at buffer=0; buffered 16.8 kWh → L1 fails, L2 ok.
    n_hours = 12
    at_home = np.array(
        [False, False, False, False, True, True, True, False, False, False, True, True]
    )
    discharge = np.zeros(n_hours, dtype=np.float64)
    discharge[0:4] = 30.0 / 4.0
    discharge[7:10] = 14.0 / 3.0
    capacity = 40.0

    assert is_home_charging_soc_feasible(
        at_home,
        discharge,
        battery_capacity_kwh=capacity,
        charger_power_kw=RESSTOCK_LEVEL1_CHARGER_KW,
        buffer_fraction=0.0,
    )
    assert not is_home_charging_soc_feasible(
        at_home,
        discharge,
        battery_capacity_kwh=capacity,
        charger_power_kw=RESSTOCK_LEVEL1_CHARGER_KW,
        buffer_fraction=DEFAULT_CHARGER_BUFFER_FRACTION,
    )
    assert is_home_charging_soc_feasible(
        at_home,
        discharge,
        battery_capacity_kwh=capacity,
        charger_power_kw=RESSTOCK_LEVEL2_CHARGER_KW,
        buffer_fraction=DEFAULT_CHARGER_BUFFER_FRACTION,
    )

    vehicles = _vehicles_frame(15, capacity_kwh=capacity)
    presence = {
        (row["bldg_id"], int(row["vehicle_id"])): pl.DataFrame({"at_home": at_home.tolist()})
        for row in vehicles.iter_rows(named=True)
    }
    discharge_by = {
        (row["bldg_id"], int(row["vehicle_id"])): discharge.copy()
        for row in vehicles.iter_rows(named=True)
    }
    result = EVChargerAssigner(charger_lookup, random_state=3).assign(
        vehicles,
        presence_by_vehicle=presence,
        discharge_kwh_by_vehicle=discharge_by,
        buffer_fraction=DEFAULT_CHARGER_BUFFER_FRACTION,
    )
    assert (result["charger_level"] == "Level 2").all()


def test_is_home_charging_soc_feasible_rejects_negative_buffer():
    with pytest.raises(ValueError, match="buffer_fraction"):
        is_home_charging_soc_feasible(
            np.ones(4, dtype=bool),
            np.zeros(4),
            battery_capacity_kwh=10.0,
            charger_power_kw=1.6,
            buffer_fraction=-0.1,
        )
