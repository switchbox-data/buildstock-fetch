import json
from itertools import product
from pathlib import Path

from buildstock_fetch.building import BuildingID
from buildstock_fetch.statecode import STATE_CODES


def main():
    data = []
    prod = list(
        product(
            (1235,),
            ("1", "2", "3"),
            ("2021", "2022", "2023", "2024", "2025"),
            ("resstock", "comstock"),
            ("tmy3", "amy2018", "amy2012"),
            map(str, range(10)),
            STATE_CODES[:1],
        )
    )
    print(len(prod))
    for i, (
        bldg_id,
        release_number,
        release_year,
        res_com,
        weather,
        upgrade_id,
        state,
    ) in enumerate(prod):
        print(i)
        building = BuildingID(bldg_id, release_number, release_year, res_com, weather, upgrade_id, state)
        data.append({
            "bldg_id": bldg_id,
            "release_number": release_number,
            "release_year": release_year,
            "res_com": res_com,
            "weather": weather,
            "upgrade_id": upgrade_id,
            "state": state,
            "results": {
                "get_building_data_url": building.get_building_data_url(),
                "get_metadata_url": building.get_metadata_url(),
                "get_15min_load_curve_url": building.get_15min_load_curve_url(),
                "get_aggregate_load_curve_url": (building.get_aggregate_load_curve_url()),
                "get_annual_load_curve_url": building.get_annual_load_curve_url(),
                "get_weather_file_url": building.get_weather_file_url(),
                "get_annual_load_curve_filename": (building.get_annual_load_curve_filename()),
                "get_weather_station_name": building.get_weather_station_name(),
                "get_release_name": building.get_release_name(),
                "to_json": building.to_json(),
            },
        })
    Path("tests/data/buildings_example_data.json").write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
