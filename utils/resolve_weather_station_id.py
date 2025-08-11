from buildstock_fetch.main import fetch_bldg_ids


def resolve_weather_station_id(
    product: str, release_year: str, weather_file: str, release_version: str, state: str, upgrade_id: str
) -> str:
    bldg_ids = fetch_bldg_ids(product, release_year, weather_file, release_version, state, upgrade_id)
    return bldg_ids


if __name__ == "__main__":
    product = "resstock"
    release_year = "2022"
    weather_file = "amy2012"
    release_version = "1"
    state = "NY"
    upgrade_id = "0"

    bldg_ids = fetch_bldg_ids(product, release_year, weather_file, release_version, state, upgrade_id)
    print(bldg_ids[:10])
