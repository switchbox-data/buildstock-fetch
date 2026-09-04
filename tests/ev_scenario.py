"""Helpers for EVDemandConfig / YAML tests (no production defaults)."""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from utils.EVs.charging import DEFAULT_TOU_SUMMER_MONTHS
from utils.EVs.EVBatteryAssigner import DEFAULT_CAPACITY_BUFFER_FRACTION
from utils.EVs.EVChargerAssigner import DEFAULT_CHARGER_BUFFER_FRACTION
from utils.EVs.ev_demand import EVDemandConfig
from utils.EVs.TripScheduleGenerator import (
    DEFAULT_MAX_ARRIVAL_HOUR,
    DEFAULT_MAX_DEPARTURE_HOUR,
    DEFAULT_MILES_NOISE_STD_FRACTION,
    DEFAULT_MIN_TRIP_AWAY_HOURS,
    DEFAULT_TIME_OFFSET_PROBABILITIES,
    DEFAULT_TIME_OFFSETS,
)

# Nested YAML document with every required scenario key set explicitly.
MINIMAL_SCENARIO: dict[str, Any] = {
    "state": "MD",
    "release": "res_2024_tmy3_2",
    "start_date": "2024-01-01T04:00:00",
    "end_date": "2024-01-03T03:00:00",
    "sampling": {
        "ev_assignment": "resstock_adoption",
        "random_state": 42,
        "include_zero_driving_days_in_match_pool": True,
        "nhts_daily_miles_percentile_low": 0,
        "nhts_daily_miles_percentile_high": 100,
        "nhts_feasibility_temperature_f": 0,
    },
    "trips": {
        "min_trip_away_hours": 1,
        "max_departure_hour": 27,
        "max_arrival_hour": 28,
        "time_offsets": [-2, -1, 0, 1, 2],
        "time_offset_probabilities": [0.05, 0.10, 0.70, 0.10, 0.05],
        "miles_noise_std_fraction": 0.1,
    },
    "battery": {"capacity_buffer_fraction": 0.2},
    "temperature": {"temperature_adjustment": "none"},
    "home_charging": {"home_charging_fraction_assignment": "none"},
    "pipeline": {"max_workers": 8, "batch_size": 20000, "upload_s3": False},
    "charging": {
        "charging_strategy": "immediate",
        "charger_assignment": "fixed",
        "charger_power_kw": 7.2,
        "charger_buffer_fraction": 0.2,
        "tou_summer_months": [6, 7, 8, 9],
        "tou_weekends_off_peak": False,
        "tou_holidays_off_peak": False,
        "allow_emergency_peak_charging": False,
    },
}

# Explicit constructor kwargs matching former EVDemandConfig defaults.
SCENARIO_KWARGS: dict[str, Any] = {
    "state": "MD",
    "release": "res_2024_tmy3_2",
    "start_date": datetime(2024, 1, 1, 4),
    "end_date": datetime(2024, 1, 2, 3),
    "ev_assignment": "resstock_adoption",
    "random_state": 42,
    "nhts_daily_miles_percentile_low": 0.0,
    "nhts_daily_miles_percentile_high": 100.0,
    "include_zero_driving_days_in_match_pool": True,
    "nhts_feasibility_temperature_f": 0.0,
    "min_trip_away_hours": DEFAULT_MIN_TRIP_AWAY_HOURS,
    "max_departure_hour": DEFAULT_MAX_DEPARTURE_HOUR,
    "max_arrival_hour": DEFAULT_MAX_ARRIVAL_HOUR,
    "time_offsets": DEFAULT_TIME_OFFSETS,
    "time_offset_probabilities": DEFAULT_TIME_OFFSET_PROBABILITIES,
    "miles_noise_std_fraction": DEFAULT_MILES_NOISE_STD_FRACTION,
    "capacity_buffer_fraction": DEFAULT_CAPACITY_BUFFER_FRACTION,
    "temperature_adjustment": "none",
    "max_workers": 8,
    "batch_size": 20000,
    "upload_s3": False,
    "charging_strategy": "immediate",
    "charger_assignment": "fixed",
    "charger_power_kw": 7.2,
    "charger_buffer_fraction": DEFAULT_CHARGER_BUFFER_FRACTION,
    "home_charging_fraction_assignment": "none",
    "tou_summer_months": DEFAULT_TOU_SUMMER_MONTHS,
    "tou_weekends_off_peak": False,
    "tou_holidays_off_peak": False,
    "allow_emergency_peak_charging": False,
}

CALCULATOR_SCENARIO_KWARGS: dict[str, Any] = {
    "ev_assignment": "resstock_adoption",
    "charger_assignment": "fixed",
    "charger_power_kw": 7.2,
    "home_charging_fraction_assignment": "none",
    "temperature_adjustment": "none",
}


def make_ev_demand_config(**overrides: Any) -> EVDemandConfig:
    """Build an ``EVDemandConfig`` with every required scenario knob set."""
    kwargs = {**SCENARIO_KWARGS, **overrides}
    if kwargs.get("charger_assignment") == "resstock":
        kwargs.setdefault("level1_charger_power_kw", 1.6)
        kwargs.setdefault("level2_charger_power_kw", 7.2)
        if "charger_power_kw" not in overrides:
            kwargs["charger_power_kw"] = None
    return EVDemandConfig(**kwargs)


def write_scenario_yaml(path: Path, **section_updates: Any) -> Path:
    """Write a complete scenario YAML, overlaying nested section updates."""
    data = copy.deepcopy(MINIMAL_SCENARIO)
    for key, value in section_updates.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key].update(value)
        else:
            data[key] = value
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path
