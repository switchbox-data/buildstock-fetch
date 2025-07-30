#!/usr/bin/env python3
"""
Utility script to download NHTS (National Household Travel Survey) data.

Downloads the zip file from https://nhts.ornl.gov/media/2022/download/csv.zip,
extracts the tripv2pub.csv file, and saves it as NHTS_v2_1_trip_surveys.csv.
Cleans up temporary files after extraction.
"""

import argparse
import logging
import zipfile
from pathlib import Path
from typing import Optional

import requests


class NHTSDownloadError(Exception):
    """Custom exception for NHTS download errors."""

    pass


class NHTSFileNotFoundError(NHTSDownloadError):
    """Raised when required file is not found in downloaded zip."""

    pass


class NHTSFileExtractionError(NHTSDownloadError):
    """Raised when file extraction fails."""

    pass


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _handle_file_not_found(all_files: list[str]) -> None:
    """Handle case when required file is not found in zip."""
    logger.error(f"tripv2pub.csv not found in zip. Available files: {all_files}")
    raise NHTSFileNotFoundError()


def _handle_extraction_failure() -> None:
    """Handle case when file extraction fails."""
    raise NHTSFileExtractionError()


def download_nhts_data(output_dir: Optional[str] = None) -> str:
    """
    Download NHTS data from the official website and extract the trip survey file.

    Args:
        output_dir: Directory to save the extracted file. If None, uses current directory.

    Returns:
        Path to the extracted NHTS_v2_1_trip_surveys.csv file
    """
    # NHTS data URL
    url = "https://nhts.ornl.gov/media/2022/download/csv.zip"

    # Set output directory
    output_dir = Path(__file__).parent if output_dir is None else Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # File paths
    zip_path = output_dir / "nhts_temp.zip"
    target_file = output_dir / "NHTS_v2_1_trip_surveys.csv"

    try:
        logger.info(f"Downloading NHTS data from {url}")

        # Download the zip file
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Downloaded {zip_path}")

        # Extract the specific file
        logger.info("Extracting tripv2pub.csv from zip file")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Look for tripv2pub.csv in the zip
            csv_files = [f for f in zip_ref.namelist() if f.endswith("tripv2pub.csv")]

            if not csv_files:
                # List all files in zip for debugging
                all_files = zip_ref.namelist()
                _handle_file_not_found(all_files)

            csv_file = csv_files[0]
            logger.info(f"Found {csv_file} in zip file")

            # Extract the file
            zip_ref.extract(csv_file, output_dir)

            # Rename to desired filename
            extracted_path = output_dir / csv_file
            if extracted_path.exists():
                extracted_path.rename(target_file)
                logger.info(f"Successfully extracted and renamed to {target_file}")
            else:
                _handle_extraction_failure()

        # Clean up temporary zip file
        if zip_path.exists():
            zip_path.unlink()
            logger.info("Cleaned up temporary zip file")

        return str(target_file)

    except requests.RequestException:
        logger.exception("Failed to download file")
        raise
    except zipfile.BadZipFile:
        logger.exception("Invalid zip file")
        raise
    except Exception:
        logger.exception("Unexpected error")
        raise
    finally:
        # Clean up temporary files in case of error
        if zip_path.exists():
            zip_path.unlink()
            logger.info("Cleaned up temporary zip file")


def main():
    """Main function to run the download process."""
    parser = argparse.ArgumentParser(description="Download NHTS (National Household Travel Survey) data")
    parser.add_argument(
        "--output-dir", "-o", type=str, help="Output directory for the downloaded file (default: script directory)"
    )

    args = parser.parse_args()

    try:
        output_file = download_nhts_data(output_dir=args.output_dir)
        print(f"Successfully downloaded NHTS data to: {output_file}")
    except Exception as e:
        print(f"Error downloading NHTS data: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
