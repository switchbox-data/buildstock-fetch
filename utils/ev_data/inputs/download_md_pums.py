#!/usr/bin/env python3
"""Download Maryland ACS 2021 1-year PUMS for EV vehicle-ownership modeling.

Fetches Census csv_hmd.zip (Maryland housing microdata, psam_h24.csv) and writes
a slim CSV with the columns expected by utils.ev_utils.load_pums_data():
HINCP, VEH, NP, PUMA, WGTP.

Source: https://www2.census.gov/programs-surveys/acs/data/pums/2021/1-Year/csv_hmd.zip
"""

from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path

import polars as pl
import requests

logger = logging.getLogger(__name__)

CENSUS_MD_PUMS_URL = (
    "https://www2.census.gov/programs-surveys/acs/data/pums/2021/1-Year/csv_hmd.zip"
)
PUMS_COLUMNS = ["HINCP", "VEH", "NP", "PUMA", "WGTP"]
DEFAULT_OUTPUT = Path(__file__).parent / "MD_2021_pums_PUMA_HINCP_VEH_NP.csv"


def download_md_pums(output_path: Path = DEFAULT_OUTPUT) -> Path:
    """Download and extract Maryland PUMS; return path to the slim CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    zip_path = output_path.parent / "csv_hmd.zip"
    logger.info("Downloading Maryland PUMS from %s", CENSUS_MD_PUMS_URL)
    response = requests.get(CENSUS_MD_PUMS_URL, stream=True, timeout=120)
    response.raise_for_status()
    zip_path.write_bytes(response.content)

    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            msg = f"No CSV found in {CENSUS_MD_PUMS_URL}"
            raise FileNotFoundError(msg)
        csv_name = csv_names[0]
        logger.info("Extracting %s", csv_name)
        with zf.open(csv_name) as csv_file:
            pums = pl.read_csv(
                csv_file,
                columns=PUMS_COLUMNS,
                null_values=["b", "B", ""],
                schema_overrides={"PUMA": pl.Utf8},
            )

    pums = pums.select(PUMS_COLUMNS).with_columns(
        pl.col("HINCP").cast(pl.Int64, strict=False),
        pl.col("VEH").cast(pl.Int64, strict=False),
        pl.col("NP").cast(pl.Int64, strict=False),
        pl.col("WGTP").cast(pl.Int64, strict=False),
    ).filter(
        pl.col("HINCP").is_not_null(),
        pl.col("HINCP") > 0,
        pl.col("VEH").is_not_null(),
        pl.col("NP").is_not_null(),
        pl.col("WGTP") > 0,
    )

    pums.write_csv(output_path)
    logger.info("Wrote %s rows to %s", f"{pums.height:,}", output_path)

    zip_path.unlink(missing_ok=True)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Download Maryland ACS 2021 PUMS for EV modeling")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path",
    )
    args = parser.parse_args()
    path = download_md_pums(args.output)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
