"""Tests for ResStock 2025 EV battery assignment."""

import numpy as np
import polars as pl
import pytest

from utils.EVBatteryAssigner import (
    DEFAULT_RESSTOCK_EV_REFERENCE_DIR,
    EVBatteryAssigner,
    load_autonomie_vehicle_params,
    load_ev_battery_option_probabilities,
    load_ev_battery_option_probabilities_from_saturations,
    parse_ev_option_name,
    vehicle_slots_from_building_evs,
)


# Local checkout of ResStock reference files (TSV + Autonomie CSV).
REF_DIR = DEFAULT_RESSTOCK_EV_REFERENCE_DIR


@pytest.fixture
def option_probabilities() -> pl.DataFrame:
    """National BEV option shares from Electric_Vehicle_Battery.tsv."""
    return load_ev_battery_option_probabilities(REF_DIR / "Electric_Vehicle_Battery.tsv")


@pytest.fixture
def autonomie_params() -> pl.DataFrame:
    """Usable kWh + kWh/mile lookup keyed by the same option names."""
    return load_autonomie_vehicle_params(REF_DIR / "resstock_autonomie_2022_vehicle_params.csv")


def test_parse_ev_option_name():
    # Option strings encode both body class and EPA-style range bin.
    assert parse_ev_option_name("Compact, Battery Electric Vehicle, 200 mile range") == ("Compact", 200)
    assert parse_ev_option_name("SUV, Battery Electric Vehicle, 300 mile range") == ("SUV", 300)
    with pytest.raises(ValueError):
        parse_ev_option_name("not a real option")


def test_battery_tsv_probabilities_sum_to_one(option_probabilities):
    # ResStock defines eight Compact/Midsize/SUV/Pickup × 200/300-mi options.
    assert option_probabilities.height == 8
    assert option_probabilities["probability"].sum() == pytest.approx(1.0, abs=1e-5)


def test_saturations_fallback_matches_tsv(option_probabilities):
    # saturations CSV should be an alternate packaging of the same shares.
    from_saturations = load_ev_battery_option_probabilities_from_saturations(
        REF_DIR / "resstock_options_saturations.csv"
    )
    merged = option_probabilities.join(from_saturations, on="ev_option_name", suffix="_sat")
    assert merged.height == 8
    assert (
        (merged["probability"] - merged["probability_sat"]).abs().max() == pytest.approx(0.0, abs=1e-8)
    )


def test_autonomie_join_covers_all_options(option_probabilities, autonomie_params):
    # Constructing the assigner fails if any TSV option lacks Autonomie params.
    assigner = EVBatteryAssigner(
        option_probabilities=option_probabilities,
        autonomie_params=autonomie_params,
        random_state=0,
    )
    assert set(assigner.option_probabilities["ev_option_name"]) <= set(
        assigner.autonomie_params["ev_option_name"]
    )


def test_from_paths_and_from_resstock_reference():
    # Explicit file paths (config-style) and directory convenience API both work.
    assigner_paths = EVBatteryAssigner.from_paths(
        battery_path=REF_DIR / "Electric_Vehicle_Battery.tsv",
        autonomie_path=REF_DIR / "resstock_autonomie_2022_vehicle_params.csv",
        random_state=1,
    )
    assigner_dir = EVBatteryAssigner.from_resstock_reference(REF_DIR, random_state=1)
    assert assigner_paths.option_probabilities.height == 8
    assert assigner_dir.autonomie_params.height == 8
    assert assigner_paths.option_probabilities.equals(assigner_dir.option_probabilities)

def test_assign_is_reproducible(option_probabilities, autonomie_params):
    slots = pl.DataFrame({"bldg_id": list(range(50)), "vehicle_id": [1] * 50})
    # Same seed → identical multinomial draws; different seed → generally different.
    a = EVBatteryAssigner(option_probabilities, autonomie_params, random_state=42).assign(slots)
    b = EVBatteryAssigner(option_probabilities, autonomie_params, random_state=42).assign(slots)
    c = EVBatteryAssigner(option_probabilities, autonomie_params, random_state=7).assign(slots)

    assert a.select("ev_option_name").equals(b.select("ev_option_name"))
    assert not a.select("ev_option_name").equals(c.select("ev_option_name"))
    # Join must always populate physical parameters.
    assert a["battery_capacity_kwh"].null_count() == 0
    assert a["kwh_per_mile"].null_count() == 0
    assert set(a["body_class"].unique().to_list()) <= {"Compact", "Midsize", "Pickup", "SUV"}


def test_assign_empty_slots(option_probabilities, autonomie_params):
    # No EVs → empty typed frame (safe for concat / parquet write).
    slots = pl.DataFrame(schema={"bldg_id": pl.Int64, "vehicle_id": pl.Int64})
    result = EVBatteryAssigner(option_probabilities, autonomie_params, random_state=0).assign(slots)
    assert result.height == 0
    assert "battery_capacity_kwh" in result.columns


def test_vehicle_slots_from_building_evs():
    # vehicles=0 buildings are dropped; vehicles=2 expands to vehicle_id 1 and 2.
    buildings = pl.DataFrame({
        "bldg_id": ["a", "b", "c"],
        "vehicles": [1, 0, 2],
    })
    slots = vehicle_slots_from_building_evs(buildings)
    assert slots.sort(["bldg_id", "vehicle_id"]).to_dicts() == [
        {"bldg_id": "a", "vehicle_id": 1},
        {"bldg_id": "c", "vehicle_id": 1},
        {"bldg_id": "c", "vehicle_id": 2},
    ]


def test_assigner_rejects_probability_sum_not_one(autonomie_params):
    # Guardrail against a corrupted / truncated probability table.
    bad = pl.DataFrame({
        "ev_option_name": autonomie_params["ev_option_name"].head(2),
        "probability": [0.5, 0.6],
    })
    with pytest.raises(ValueError, match="sum to"):
        EVBatteryAssigner(bad, autonomie_params, random_state=0)
