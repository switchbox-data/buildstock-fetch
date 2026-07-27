"""Tests for ResStock 2025 EV battery assignment."""

import polars as pl
import pytest

from tests.conftest import RESSTOCK_EV_REFERENCE_DIR
from utils.EVs.EVBatteryAssigner import EVBatteryAssigner
from utils.EVs.ev_utils import load_ev_autonomie_params, load_ev_battery_lookup, parse_ev_option_name


REF_DIR = RESSTOCK_EV_REFERENCE_DIR


@pytest.fixture
def option_probabilities() -> pl.DataFrame:
    """National BEV option shares from Electric_Vehicle_Battery.tsv."""
    return load_ev_battery_lookup(REF_DIR / "Electric_Vehicle_Battery.tsv")


@pytest.fixture
def autonomie_params() -> pl.DataFrame:
    """Usable kWh + kWh/mile lookup keyed by the same option names."""
    return load_ev_autonomie_params(REF_DIR / "resstock_autonomie_2022_vehicle_params.csv")


def _duty(slots: pl.DataFrame, max_daily_miles: float | list[float]) -> pl.DataFrame:
    """Attach constant or per-row max_daily_miles for assign()."""
    if isinstance(max_daily_miles, list):
        return slots.with_columns(pl.Series("max_daily_miles", max_daily_miles))
    return slots.with_columns(pl.lit(max_daily_miles).alias("max_daily_miles"))


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


def test_assign_is_reproducible(option_probabilities, autonomie_params):
    duty = _duty(pl.DataFrame({"bldg_id": list(range(50)), "vehicle_id": [1] * 50}), 30.0)
    # Same seed → identical multinomial draws; different seed → generally different.
    a = EVBatteryAssigner(option_probabilities, autonomie_params, random_state=42).assign(duty)
    b = EVBatteryAssigner(option_probabilities, autonomie_params, random_state=42).assign(duty)
    c = EVBatteryAssigner(option_probabilities, autonomie_params, random_state=7).assign(duty)

    assert a.select("ev_option_name").equals(b.select("ev_option_name"))
    assert not a.select("ev_option_name").equals(c.select("ev_option_name"))
    # Join must always populate physical parameters.
    assert a["battery_capacity_kwh"].null_count() == 0
    assert a["kwh_per_mile"].null_count() == 0
    assert set(a["body_class"].unique().to_list()) <= {"Compact", "Midsize", "Pickup", "SUV"}


def test_assign_empty_slots(option_probabilities, autonomie_params):
    # No EVs → empty typed frame (safe for concat / parquet write).
    duty = pl.DataFrame(
        schema={"bldg_id": pl.Int64, "vehicle_id": pl.Int64, "max_daily_miles": pl.Float64}
    )
    result = EVBatteryAssigner(option_probabilities, autonomie_params, random_state=0).assign(duty)
    assert result.height == 0
    assert "battery_capacity_kwh" in result.columns


def test_assigner_rejects_probability_sum_not_one(autonomie_params):
    # Guardrail against a corrupted / truncated probability table.
    bad = pl.DataFrame({
        "ev_option_name": autonomie_params["ev_option_name"].head(2),
        "probability": [0.5, 0.6],
    })
    with pytest.raises(ValueError, match="sum to"):
        EVBatteryAssigner(bad, autonomie_params, random_state=0)


def test_assign_filters_to_feasible_options(option_probabilities, autonomie_params):
    assigner = EVBatteryAssigner(option_probabilities, autonomie_params, random_state=0)
    # ~180 peak miles: small/ inefficient packs drop out; larger packs remain.
    duty = _duty(pl.DataFrame({"bldg_id": ["x"], "vehicle_id": [1]}), 180.0)
    result = assigner.assign(duty)
    assert result.height == 1
    # Feasible <=> capacity >= miles * kwh_per_mile * 1.2
    assert float(result["battery_capacity_kwh"][0]) >= 180.0 * float(result["kwh_per_mile"][0]) * 1.2


def test_assign_raises_when_no_option_feasible(option_probabilities, autonomie_params):
    assigner = EVBatteryAssigner(option_probabilities, autonomie_params, random_state=0)
    duty = _duty(pl.DataFrame({"bldg_id": ["x"], "vehicle_id": [1]}), 10_000.0)
    with pytest.raises(ValueError, match="No ResStock EV battery option"):
        assigner.assign(duty)


def test_high_miles_never_draws_infeasible_small_pack(option_probabilities, autonomie_params):
    # Many draws at high duty cycle should never land on Compact 200-mi (too small).
    assigner = EVBatteryAssigner(option_probabilities, autonomie_params, random_state=1)
    duty = _duty(
        pl.DataFrame({"bldg_id": list(range(200)), "vehicle_id": [1] * 200}),
        170.0,
    )
    result = assigner.assign(duty)
    assert "Compact, Battery Electric Vehicle, 200 mile range" not in result["ev_option_name"].to_list()
