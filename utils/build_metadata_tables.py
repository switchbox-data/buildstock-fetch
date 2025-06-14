import json
import os
from pathlib import Path

import requests

with open("buildstock_releases.json") as file:
    data = json.load(file)

data_dir = Path(__file__).parent.parent / "data"
downloaded_paths = []
os.makedirs(data_dir, exist_ok=True)

for release_name, release in data.items():
    release_year = release["release_year"]
    res_com = release["res_com"]
    weather = release["weather"]
    release_number = release["release_number"]

    url = (
        f"https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/"
        f"end-use-load-profiles-for-us-building-stock/{release_year}/"
        f"{res_com}_{weather}_release_{release_number}/metadata/metadata.parquet"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    output_path = data_dir / f"{release_name}_metadata.parquet"
    with open(output_path, "wb") as file:
        file.write(response.content)

    downloaded_paths.append(output_path)
    print(url)
