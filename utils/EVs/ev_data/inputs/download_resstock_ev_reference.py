#!/usr/bin/env python3
"""
Download ResStock-published EV reference parameters for EDA / comparison work.

Fetches:
- options_lookup.tsv (EV battery option -> HPXML argument mapping)
- Electric_Vehicle_Ownership.tsv (conditional P(EV) by FPL / type / tenure / PUMA)
- Electric_Vehicle_Battery.tsv (national BEV class × range option shares)
- Electric_Vehicle_Charger.tsv (conditional L1/L2/None by ownership / FPL / type / tenure)

    Also writes a small summary CSV of Autonomie 2022 vehicle parameters and the
    constant average speed used in ResStock (from NREL/TP-5500-93766 Table 208 and
    ResStock Technical Reference Guide 2025 §8.9.2).
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import requests

RESSTOCK_BASE = "https://raw.githubusercontent.com/NREL/resstock/develop"
LOOKUP_URL = f"{RESSTOCK_BASE}/resources/options_lookup.tsv"
EV_OWNERSHIP_URL = f"{RESSTOCK_BASE}/project_national/housing_characteristics/Electric%20Vehicle%20Ownership.tsv"
# National BEV stock mix used by EVBatteryAssigner.
EV_BATTERY_URL = (
    f"{RESSTOCK_BASE}/project_national/housing_characteristics/Electric%20Vehicle%20Battery.tsv"
)
# Conditional L1/L2 charger shares used by EVChargerAssigner (RECS 2020 EVCHRGTYPE).
# Same file lives under NatLabRockies/resstock; NREL/resstock develop tracks it.
EV_CHARGER_URL = (
    f"{RESSTOCK_BASE}/project_national/housing_characteristics/Electric%20Vehicle%20Charger.tsv"
)
# Dependency tables ResStock uses to assign FPL, tenure, and building type to each unit.
# Not required for predict_num_EVs() (which only needs Electric_Vehicle_Ownership.tsv),
# but included here for EDA and cross-checking against metadata values.
FPL_URL = f"{RESSTOCK_BASE}/project_national/housing_characteristics/Federal%20Poverty%20Level.tsv"
TENURE_URL = f"{RESSTOCK_BASE}/project_national/housing_characteristics/Tenure.tsv"
BUILDING_TYPE_URL = f"{RESSTOCK_BASE}/project_national/housing_characteristics/Geometry%20Building%20Type%20RECS.tsv"

# NREL/TP-5500-93766 Table 208 (Autonomie 2022 via TEMPO)
AUTONOMIE_2022_VEHICLES: list[dict[str, object]] = [
    {
        "option_name": "Compact, Battery Electric Vehicle, 200 mile range",
        "vehicle_battery_usable_capacity_kwh": 40.168,
        "vehicle_fuel_economy_combined_kwh_per_mile": 0.209901,
        "ev_average_mph": 22,
    },
    {
        "option_name": "Compact, Battery Electric Vehicle, 300 mile range",
        "vehicle_battery_usable_capacity_kwh": 63.433,
        "vehicle_fuel_economy_combined_kwh_per_mile": 0.220020,
        "ev_average_mph": 22,
    },
    {
        "option_name": "Midsize, Battery Electric Vehicle, 200 mile range",
        "vehicle_battery_usable_capacity_kwh": 41.978,
        "vehicle_fuel_economy_combined_kwh_per_mile": 0.219174,
        "ev_average_mph": 22,
    },
    {
        "option_name": "Midsize, Battery Electric Vehicle, 300 mile range",
        "vehicle_battery_usable_capacity_kwh": 65.441,
        "vehicle_fuel_economy_combined_kwh_per_mile": 0.229449,
        "ev_average_mph": 22,
    },
    {
        "option_name": "Pickup, Battery Electric Vehicle, 200 mile range",
        "vehicle_battery_usable_capacity_kwh": 67.738,
        "vehicle_fuel_economy_combined_kwh_per_mile": 0.357648,
        "ev_average_mph": 22,
    },
    {
        "option_name": "Pickup, Battery Electric Vehicle, 300 mile range",
        "vehicle_battery_usable_capacity_kwh": 105.946,
        "vehicle_fuel_economy_combined_kwh_per_mile": 0.373794,
        "ev_average_mph": 22,
    },
    {
        "option_name": "SUV, Battery Electric Vehicle, 200 mile range",
        "vehicle_battery_usable_capacity_kwh": 53.503,
        "vehicle_fuel_economy_combined_kwh_per_mile": 0.267513,
        "ev_average_mph": 22,
    },
    {
        "option_name": "SUV, Battery Electric Vehicle, 300 mile range",
        "vehicle_battery_usable_capacity_kwh": 83.680,
        "vehicle_fuel_economy_combined_kwh_per_mile": 0.278934,
        "ev_average_mph": 22,
    },
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _fetch_text(url: str) -> str:
    logger.info("Fetching %s", url)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.text


def download_resstock_ev_reference(output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    lookup_text = _fetch_text(LOOKUP_URL)
    ev_ownership_text = _fetch_text(EV_OWNERSHIP_URL)
    # Housing-characteristic battery option shares (input for EVBatteryAssigner).
    ev_battery_text = _fetch_text(EV_BATTERY_URL)
    # Housing-characteristic charger L1/L2 shares (input for EVChargerAssigner).
    ev_charger_text = _fetch_text(EV_CHARGER_URL)
    # Housing-characteristic dependency tables (for reference / EDA)
    fpl_text = _fetch_text(FPL_URL)
    tenure_text = _fetch_text(TENURE_URL)
    building_type_text = _fetch_text(BUILDING_TYPE_URL)

    lookup_path = output_dir / "resstock_options_lookup.tsv"
    ev_ownership_path = output_dir / "Electric_Vehicle_Ownership.tsv"
    ev_battery_path = output_dir / "Electric_Vehicle_Battery.tsv"
    ev_charger_path = output_dir / "Electric_Vehicle_Charger.tsv"
    fpl_path = output_dir / "Federal_Poverty_Level.tsv"
    tenure_path = output_dir / "Tenure.tsv"
    building_type_path = output_dir / "Geometry_Building_Type_RECS.tsv"
    lookup_path.write_text(lookup_text)
    ev_ownership_path.write_text(ev_ownership_text)
    ev_battery_path.write_text(ev_battery_text)
    ev_charger_path.write_text(ev_charger_text)
    fpl_path.write_text(fpl_text)
    tenure_path.write_text(tenure_text)
    building_type_path.write_text(building_type_text)

    autonomie_path = output_dir / "resstock_autonomie_2022_vehicle_params.csv"
    with open(autonomie_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "option_name",
                "vehicle_battery_usable_capacity_kwh",
                "vehicle_fuel_economy_combined_kwh_per_mile",
                "ev_average_mph",
                "hours_driven_per_week_at_11000_mi",
                "notes",
            ],
        )
        writer.writeheader()
        for row in AUTONOMIE_2022_VEHICLES:
            hours_per_week = round(11_000 / (row["ev_average_mph"] * 52), 1)
            writer.writerow({
                **row,
                "hours_driven_per_week_at_11000_mi": hours_per_week,
                "notes": (
                    "Autonomie 2022 via TEMPO; ev_average_mph=22 from 2017 NHTS summary "
                    "(ResStock TRG §8.9.2). hours_driven_per_week example uses 11,000 mi/yr."
                ),
            })

    return {
        "resstock_options_lookup.tsv": str(lookup_path),
        "Electric_Vehicle_Ownership.tsv": str(ev_ownership_path),
        "Electric_Vehicle_Battery.tsv": str(ev_battery_path),
        "Electric_Vehicle_Charger.tsv": str(ev_charger_path),
        "Federal_Poverty_Level.tsv": str(fpl_path),
        "Tenure.tsv": str(tenure_path),
        "Geometry_Building_Type_RECS.tsv": str(building_type_path),
        "resstock_autonomie_2022_vehicle_params.csv": str(autonomie_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Download ResStock EV reference parameters")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=str(Path(__file__).parent / "resstock_ev_reference"),
        help="Output directory (default: utils/EVs/ev_data/inputs/resstock_ev_reference)",
    )
    args = parser.parse_args()

    try:
        paths = download_resstock_ev_reference(Path(args.output_dir))
        print("Successfully downloaded ResStock EV reference files:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
