# BuildStock Fetch

A Python CLI tool for downloading building energy simulation data from the National Renewable Energy Laboratory (NREL) ResStock and ComStock projects.

## Table of Contents

- [Background](#background)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Command Line Interface](#command-line-interface)
- [Data Sources](#data-sources)
- [Contributing](#contributing)
- [References](#references)

## Background

NREL's ResStock and ComStock projects, collectively known as BuildStock, offer a statistically representative snapshot of buildings across the United States. These realistic-but-not-real buildings are then loaded into building energy modeling (BEM) software EnergyPlus, in order to estimate their weather-dependent energy consumption and water use by simulating heating, cooling, lighting, ventilation, and other energy flows.

The simulations output [end-use load profiles](https://www.nrel.gov/buildings/end-use-load-profiles), which NREL releases as open data. But it's also possible to do your own Buildstock simulations using the same building and weather file that NREL uses.

**BuildStock Fetch** is a python library that makes it easy to download NREL's [end-use load profiles](https://www.nrel.gov/buildings/end-use-load-profiles), as well as the [hpxml](https://hpxml-guide.readthedocs.io/en/latest/), weather, and other input files needed to run your simulations using [OCHRE](https://ochre-nrel.readthedocs.io/en/latest/Introduction.html) a lightweight BEM library also developed by NREL.

## Key Features

- **Comprehensive Data Access**: Download physical building characteristics, load and occupancy schedules, simulated load curves, and weather files.
- **Flexible Selection**: Choose specific states, upgrade scenarios, and building types.
- **Organized Output**: Structured file organization by release version, state, and building type.

## Data source

BuildStock Fetch (`bsf`) makes it easier to download the files available on [NREL's public aws s3 buckets](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F).

Besides weather files, all the data that `bsf` allows you to download is on **building units**, either residential dwellings or commercial establishments.

### Buildstock Releases

These units are grouped into **Buildstock releases**, which are defined by:

- **Product**: either ResStock (for residential buildings) or ComStock (for commercial buildings).

- **Release Year**: The year of release. NREL's first release was in 2021, and each subsequent year has seen its own set of new releases.

- **Weather File**: Building energy simulations require weather files, which represent temperature, moisture, solar radiation, and other environmental condition. NREL has usef three different weather files for ComStock and ResStock: `tmy3`, `amy2012`, and `amy2018`.
  - `tmy3` stands for Typical Meteorological Year 3. This is a synthetic weather dataset that represent a "typical" annual weather conditions and were built using historical weather data from 1991-2005.
  - `amy2012` stands for Actual Meteorological Year 2012, and represents actual weather data from 2012.
  - `amy2018` stands for Actual Meteorological Year 2018, and represents actual weather data from 2018.

- **Release Version**: Most releases have only one version, which is version 1. NREL will occasionally correct or update a release, these releases will have versions 1.1 or 2.

- **Upgrades**: that main value of ResStock and ComStock is the ability to run "what if?" scenarios, where building shells and equipment are altered in order to compare the resulting simulated load curve to the current baseline building stock. Each release has its own distinct list of numbered upgrade scenarios, representing various building envelope and equipment upgrades, including weatherization packages, efficient electrical appliances, and so on.

## Available datasets

`bsf` makes it easier to download three kinds of files:

- **Building Unit Metadata**: The metadata table offers information each building unit for each upgrade within each release. This information includes the building unit's `bldg_id`, overall physical characteristics (square footage, number of rooms, etc.), the building envelope, all energy consuming devices, air infilation levels, approximate physical location, income range, and so on. These building units are synthetic, drawn from a stastistical distribution of the US building stock assembled from dozens of surveys and other datasets.

- **End Use Load Profiles**: NREL maps these synthetic buildings to geometric models in EnergyPlus, and runs physics simulations with particular weather files in order to generate energy consumption load curves for every appliances in every building unit in a given release. NREL then publishes these simulation outputs under the [End Use Load Profiles](https://www.nrel.gov/buildings/end-use-load-profiles) data product. These datasets provide detailed simulation input parameters and load outputs, such as fuel consumption by usage type (e.g. space heating/cooling, water heating, lights, etc.), and building conditions (e.g. indoor temperature and humidity, hot water temperature, etc.). NREL publishes these profiles at 15-min time granularity for each building unit in a particular release, as well as yearly aggregates. Load are often needed at the hourly or monthly level for anaysis, so `bsf` plans to offer these pre-aggregated datasets in the future to save you time.

- **Building Simulation Input Data**: NREL also makes available the input files they use to run these simulations, which are the building metadata in [HPXML format](https://hpxml.nrel.gov/), equipment and occupancy schedule files, and weather files. You can use these input files to run your own simulations of Buildstock building units, either in EnergyPlus (via [buildstock-batch](https://buildstockbatch.readthedocs.io/en/stable/)) or in [OCHRE](https://www.nrel.gov/grid/ochre), a new building energy simulation tool developed by NREL. OCHRE is meant to be lighter-weight alternative to other numerical simulation softwares such as EnergyPlus or TRNSYS. OCHRE requires as input a building HPXML file that provides information on the building envelope and HVAC and water heating equipment, the schedule files that provide load profiles for non-controllable appliance use such as lights or dishwasher and occupancy levels, and weather files. Each building HPXML file has a corresponding weather station ID based on its location, and `bsf` can fetch this for you too.

## Installation

### Prerequisites

- Python 3.9 or higher
- `pip` or `uv` package manager

### Install from Source

BuildStock Fetch is not yet available as a published library on PyPI. In the meantime, the library can be installed directly from the GitHub repository using:

```bash
pip3 install git+https://github.com/switchbox-data/buildstock-fetch.git@main
```

## Quick Start

To download data with BuildStock Fetch, you must specify the following variables: product type, release year, weather file type, release version, upgrade scenarios, state, types of files to download, and output directory. Once these inputs have been specified, BuildStock Fetch will first resolved the individual `bldg_ids` correspond to this  selection, and then download the requested files pertaining to each building unit to the specified output directory.

### Interactive Mode

The easiest way to get started is using the interactive CLI:

```bash
bsf
```

This will guide you through:

1. Selecting product type (ResStock/ComStock)
2. Choosing release year
3. Choosing the weather file type
4. Choosing the release version
5. Selecting the upgrade scenarios (you can choose multiple)
6. Selecting the state (you can choose multiple)
7. Selecting the file types to download (you can choose multiple)
8. Specifying the output directory for the downloaded files.

The interactive CLI mode only shows valid release options and file types, based on what's currently available on NREL's s3 bucket.

#### Downloading a sample

Once all the inputs have been collected, BuildStock Fetch will ask whether to download all files for the release, or just a sample. Each release has around 10,000 to 30,000 building units associated with it. If the user wishes to download just a small sample, they can specify the number of files to download for each state-upgrade version pair.

You can also specify the sample size via `--sample` command-line argument. Use `0` to download all files.

### Direct Command Line

You can also use specify the buildstock release, file types, and geography in one go, using direct command line arguments:

```bash
bsf \
  --product resstock \
  --release_year 2022 \
  --weather_file tmy3 \
  --release_version 1 \
  --states "CA NY" \
  --file_type "metadata load_curve_15min" \
  --upgrade_id "0 1" \
  --output_directory ./data \
  --sample 0
```

`bsf` will warn you if you've specified an invalid buildstock release, state, file type, upgrade, or output directory.

You can provide only some of the arguments, in which case the app will launch in interactive mode, and ask for parameters you did not provide yet.

### Specifying the number of concurrent downloads

By default, `bsf` will limit the number of concurrent downloads to 15. You may increase or decrease that number by providing `--threads <number>` command-line argument (i.e. `--threads: 5` will limit concurrent downloads to 5 files at a time)

### Output Structure

The `--output_directory` option specifies the top directory where the files will be downloaded. From there, the files will be organized as:

```
output_directory/
└── release_name/
    └── file_type/
        └── state/
            └── bldgid1_file
            └── bldgid2_file
            └── ...
```

## Usage Examples

### Example 1: Download Metadata for California

```bash
bsf \
  --product resstock \
  --release_year 2022 \
  --weather_file tmy3 \
  --release_version 1 \
  --states CA \
  --file_type metadata \
  --upgrade_id 0 \
  --output_directory ./california_data
```

### Example 2: Download Sample Building Files

```bash
bsf \
  --product comstock \
  --release_year 2024 \
  --weather_file amy2018 \
  --release_version 1 \
  --states "CA TX NY" \
  --file_type "hpxml schedule" \
  --upgrade_id "0 1 2" \
  --output_directory ./sample_buildings
```

**Output Structure:**

```
sample_buildings/
└── com_2024_amy2018_1/
    ├── hpxml/
    │   ├── CA/
    │   │   ├── bldg_id1_up_0.xml
    │   │   ├── bldg_id1_up_1.xml
    │   │   ├── bldg_id1_up_2.xml
    │   │   ├── bldg_id2_up_0.xml
    │   │   ├── ...
    │   ├── TX/
    │   │   ├── bldg_id1_up_0.xml
    │   │   ├── bldg_id1_up_1.xml
    │   │   ├── ...
    │   └── NY/
    │   │   ├── bldg_id1_up_0.xml
    │   │   ├── bldg_id1_up_1.xml
    │   │   ├── ...
    └── schedule/
        ├── CA/
        │   ├── bldg_id1_up_0_schedule.csv
        │   ├── bldg_id1_up_1_schedule.csv
        │   ├── bldg_id1_up_2_schedule.csv
        │   ├── bldg_id2_up_0_schedule.csv
        │   ├── ...
        ├── ...
```

### Example 3: Download Load Curves for Analysis

```bash
bsf \
  --product resstock \
  --release_year 2022 \
  --weather_file tmy3 \
  --release_version 1 \
  --states "CA MA NY" \
  --file_type "load_curve_15min" \
  --upgrade_id "0 1" \
  --output_directory ./load_analysis
```

## Command Line Interface

### Interactive Mode

When run without arguments, BuildStock Fetch launches an interactive interface:

```
BuildStock Fetch Interactive CLI
Welcome to the BuildStock Fetch CLI!
This tool allows you to fetch data from the NREL BuildStock API.
Please select the release information and file type you would like to fetch:

Select product type: [resstock, comstock]
Select release year: [2021, 2022, 2023, 2024, ...]
Select weather file: [tmy3, amy2012, amy2018]
Select release version: [1, 1.1, 2]
Select upgrade ids: [0, 1, 2, 3, 4, 5, 6, 7, ...]
Select states: [AL, AZ, AR, CA, ...]
Select file type: [hpxml, schedule, metadata, load_curve_15min, ...]
Select output directory: ./data
```

### Command Line Options

| Option | Short | Description | Example |
|--------|-------|-------------|---------|
| `--product` | `-p` | Product type (resstock/comstock) | `--product resstock` |
| `--release_year` | `-y` | Release year | `--release_year 2022` |
| `--weather_file` | `-w` | Weather file type | `--weather_file tmy3` |
| `--release_version` | `-r` | Release version | `--release_version 1` |
| `--states` | `-s` | States (space-separated) | `--states "CA NY TX"` |
| `--file_type` | `-f` | File types (space-separated) | `--file_type "metadata hpxml"` |
| `--upgrade_id` | `-u` | Upgrade IDs (space-separated) | `--upgrade_id "0 1 2"` |
| `--output_directory` | `-o` | Output directory | `--output_directory ./data` |

You can also run

```bash
bsf --help
```

to see these options

### Available File Types

- **`hpxml`**: Building energy model files in HPXML format
- **`schedule`**: Occupancy and equipment schedules
- **`metadata`**: Building characteristics and metadata
- **`load_curve_15min`**: 15-minute resolution load profiles
- **`load_curve_hourly`**: Hourly resolution load profiles
- **`load_curve_daily`**: Daily resolution load profiles
- **`load_curve_monthly`**: Monthly resolution load profiles
- **`load_curve_annual`**: Annual resolution load profiles

### Python API

You can also use BuildStock Fetch programmatically:

```python
from buildstock_fetch.main import fetch_bldg_ids, fetch_bldg_data
from pathlib import Path

# Get building IDs for a specific state
bldg_ids = fetch_bldg_ids(
    product="resstock",
    release_year="2022",
    weather_file="tmy3",
    release_version="1",
    state="CA",
    upgrade_id="0"
)

# Download data
downloaded_paths, failed_downloads = fetch_bldg_data(
    bldg_ids=bldg_ids,
    file_type=("metadata", "load_curve_15min"),
    output_dir=Path("./data")
)
```
