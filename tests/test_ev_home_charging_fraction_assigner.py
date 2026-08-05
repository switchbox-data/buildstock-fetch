"""Tests for ResStock 2025 EV home-charging fraction assignment."""

import polars as pl
import pytest

from tests.conftest import RESSTOCK_EV_REFERENCE_DIR
from utils.EVs.EVHomeChargingFractionAssigner import (
    CHARGE_AT_HOME_BIN_TO_FRACTION,
    CHARGE_AT_HOME_BINS,
    EVHomeChargingFractionAssigner,
)
from utils.EVs.ev_utils import load_ev_charge_at_home_lookup

REF_DIR = RESSTOCK_EV_REFERENCE_DIR
CHARGE_AT_HOME_TSV = REF_DIR / "Electric_Vehicle_Charge_At_Home.tsv"


def _vehicles_frame(
    n: int,
    *,
    fpl: str = "400%+",
    building_type: str = "Single-Family Detached",
) -> pl.DataFrame:
    return pl.DataFrame({
        "bldg_id": list(range(n)),
        "vehicle_id": [1] * n,
        "fpl": [fpl] * n,
        "building_type": [building_type] * n,
    })


@pytest.fixture
def charge_at_home_lookup() -> pl.DataFrame:
    """FPL × building-type bin probabilities from Electric_Vehicle_Charge_At_Home.tsv."""
    return load_ev_charge_at_home_lookup(CHARGE_AT_HOME_TSV)


def test_load_ev_charge_at_home_lookup_shape(charge_at_home_lookup):
    # Thin loader keeps join keys + one probability column per RECS bin.
    expected_cols = {
        "fpl",
        "building_type",
        "p_0_19",
        "p_20_39",
        "p_40_59",
        "p_60_79",
        "p_80_99",
        "p_100",
    }
    assert expected_cols <= set(charge_at_home_lookup.columns)
    assert charge_at_home_lookup.height > 0

    # Each row is a valid multinomial over the six bins.
    row_sum = (
        charge_at_home_lookup["p_0_19"]
        + charge_at_home_lookup["p_20_39"]
        + charge_at_home_lookup["p_40_59"]
        + charge_at_home_lookup["p_60_79"]
        + charge_at_home_lookup["p_80_99"]
        + charge_at_home_lookup["p_100"]
    )
    assert row_sum.to_numpy() == pytest.approx(1.0, abs=1e-3)


def test_bin_midpoints_match_resstock_options_lookup():
    # options_lookup: Electric Vehicle Charge At Home → ev_fraction_charged_home.
    assert CHARGE_AT_HOME_BIN_TO_FRACTION == {
        "0-19%": 0.10,
        "20-39%": 0.30,
        "40-59%": 0.50,
        "60-79%": 0.70,
        "80-99%": 0.90,
        "100%": 1.00,
    }
    assert CHARGE_AT_HOME_BINS == tuple(CHARGE_AT_HOME_BIN_TO_FRACTION.keys())


def test_assign_is_reproducible(charge_at_home_lookup):
    # Same seed → identical draws; different seed generally differs.
    vehicles = _vehicles_frame(40)
    a = EVHomeChargingFractionAssigner(charge_at_home_lookup, random_state=42).assign(
        vehicles
    )
    b = EVHomeChargingFractionAssigner(charge_at_home_lookup, random_state=42).assign(
        vehicles
    )
    c = EVHomeChargingFractionAssigner(charge_at_home_lookup, random_state=7).assign(
        vehicles
    )

    assert a.select("charge_at_home_bin", "fraction_charged_home").equals(
        b.select("charge_at_home_bin", "fraction_charged_home")
    )
    assert not a.select("charge_at_home_bin").equals(c.select("charge_at_home_bin"))
    assert set(a["charge_at_home_bin"].unique().to_list()) <= set(CHARGE_AT_HOME_BINS)
    assert set(a["fraction_charged_home"].unique().to_list()) <= set(
        CHARGE_AT_HOME_BIN_TO_FRACTION.values()
    )


def test_assign_maps_bin_to_midpoint(charge_at_home_lookup):
    # Every drawn bin must map to its ResStock midpoint scalar.
    vehicles = _vehicles_frame(20)
    result = EVHomeChargingFractionAssigner(charge_at_home_lookup, random_state=0).assign(
        vehicles
    )
    for bin_label, fraction in zip(
        result["charge_at_home_bin"].to_list(),
        result["fraction_charged_home"].to_list(),
        strict=True,
    ):
        assert fraction == CHARGE_AT_HOME_BIN_TO_FRACTION[bin_label]


def test_assign_empty_vehicles(charge_at_home_lookup):
    empty = pl.DataFrame(
        schema={
            "bldg_id": pl.Int64,
            "vehicle_id": pl.Int64,
            "fpl": pl.Utf8,
            "building_type": pl.Utf8,
        }
    )
    result = EVHomeChargingFractionAssigner(charge_at_home_lookup).assign(empty)
    assert result.is_empty()
    assert set(result.columns) == {
        "bldg_id",
        "vehicle_id",
        "charge_at_home_bin",
        "fraction_charged_home",
    }


def test_assign_raises_on_unmatched_keys(charge_at_home_lookup):
    vehicles = _vehicles_frame(1, fpl="not-a-real-fpl")
    with pytest.raises(ValueError, match="lookup join missed"):
        EVHomeChargingFractionAssigner(charge_at_home_lookup).assign(vehicles)


def test_discharge_scales_by_home_fraction():
    """Residential discharge = temp_scaled_miles × kwh/mi × fraction_charged_home."""
    from utils.EVs.ChargingSimulator import ChargingSimulator

    # One vehicle-hour: 10 duty miles at 0.2 kWh/mi with 50% home charging.
    hourly = pl.DataFrame({
        "bldg_id": [1],
        "vehicle_id": [1],
        "hour_index": [0],
        "temp_scaled_miles": [10.0],
    })
    efficiency = pl.DataFrame({
        "bldg_id": [1],
        "vehicle_id": [1],
        "kwh_per_mile": [0.2],
        "fraction_charged_home": [0.5],
    })
    out = ChargingSimulator._discharge_kwh_from_temp_scaled_miles(hourly, efficiency)
    assert out["discharge_kwh"].to_list() == pytest.approx([1.0])  # 10 * 0.2 * 0.5


def test_discharge_defaults_to_full_home_without_fraction_column():
    """Missing fraction_charged_home preserves legacy full-home discharge."""
    from utils.EVs.ChargingSimulator import ChargingSimulator

    hourly = pl.DataFrame({
        "bldg_id": [1],
        "vehicle_id": [1],
        "hour_index": [0],
        "temp_scaled_miles": [10.0],
    })
    efficiency = pl.DataFrame({
        "bldg_id": [1],
        "vehicle_id": [1],
        "kwh_per_mile": [0.2],
    })
    out = ChargingSimulator._discharge_kwh_from_temp_scaled_miles(hourly, efficiency)
    assert out["discharge_kwh"].to_list() == pytest.approx([2.0])  # 10 * 0.2 * 1.0
