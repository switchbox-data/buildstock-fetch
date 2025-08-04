# BuildStock Fetch

A Python CLI tool for downloading and managing building energy data from the National Renewable Energy Laboratory (NREL) ResStock and ComStock database.

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

**BuildStock Fetch** is a python-based library built for fetching building energy files provided by NREL's ResStock and ComStock projects. These include data files used for building energy simulation, pre-run simulation results, weather data, and more. The results from these simulations represent detailed energy simulations of the U.S. building stock, enabling researchers, policymakers, and energy analysts to:

- **Analyze building energy consumption patterns** across different climate zones and building types
- **Evaluate energy efficiency measures** and their potential impact
- **Support energy policy development** with data-driven insights
- **Enable building energy research** with standardized, high-quality datasets

The BuildStock project, which encompasses the ResStock and ComStock projects, simulates energy usage for millions of buildings across the United States, providing granular data on energy consumption, load profiles, and building characteristics. This tool makes this valuable data accessible to the broader energy research community.

### Key Features

- **Comprehensive Data Access**: Download building envelope and HVAC equipment files, load and occupancy schedules, metadata, load curves, and weather files.
- **Flexible Selection**: Choose specific states, upgrade scenarios, and building types
- **Organized Output**: Structured file organization by release version, state, and building type

### Data source

Majority of the files available for download via BuildStock Fetch is provided by [NREL's public aws s3 buckets](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F). Here, each file is cateogrized by grouped by its release name, which is comprised of four elements:

- **Product**: The release is categorized as either ResStock (for simulating residential buildings) or ComStock (for simulating commercial buildings).

- **Release Year**: The year of release. First set of available releases was in 2021, and each subsequent year has its own set of new releases.

- **Weather File**: There are three types of weather files used for simulations, tmy3, amy2012, and amy2018. The tmy3 weather files stand for Typical Meteorological Year 3. They are a standardized weather dataset representing a "typical" annual weather conditions and were built using historical weather data from 1991-2005. The amy2012 weather files stand for Actual Meteorological Year 2012 and they are a set of real weather data from the year 2012. Similarly, amy2018 stands for Actual Meteorological Year 2018, and they are a set of weather data files from the year 2018.

- **Release Version**: The first of each type of release has an associated version, which starts from 1. If there are subsequent updated releases for the same release type (i.e. same product, release year, and weather files), then they will have release versions 1.1 or 2.

For each release, there are a list of upgrade scenarios associated with it. These scenarios represent various building envelope and equipment upgrades, such as building envelope weatherization, more efficient HVAC equipments, etc.

### Available File Types

There are largely three classes of file types available for download:

- **End Use Load Profiles**: The end use load profiles are building energy simulation results published by NREL. These files provide simulation results that have already been completed by NREL using EnergyPlus. These files provide detailed simulation input parameters and final results, such as fuel consumption by usage type (e.g. space heating/cooling, water heating, lights, etc.), building conditions (e.g. indoor temperature and humidity, hot water temperature, etc.). The end use load profiles provided by NREL are in 15-min time granularity.


- **Building Simulation Input Data**: The input files needed to run a simulation on OCHRE, which are the building HPXML files, load and occupancy schedule files, and weather files. [OCHRE](https://www.nrel.gov/grid/ochre) is a python-based simulation tool developed by NREL. They are meant to be a lighter-weight alternative to other numerical simulation softwares such as EnergyPlus or TRNSYS. OCHRE requires as input a building HPXML file that provides information on the building envelope and HVAC and water heating equipment, the schedule files that provide load profiles for non-controllable appliance use such as lights or dishwasher and occupancy levels, and weather files. Each building HPXML file has a corresponding weather station ID based on its location.

- **Metadata**: The metadata table provides attribute information for each building in a given release. These include the building ID that specifies the building HPXML and schedule file, the location of the building, the utility district of the building, and more.

## Installation

### Prerequisites

- Python 3.9 or higher
- pip or uv package manager

### Install from Source

BuildStock Fetch is not yet available as a published library on PyPI. In the meantime, the library can be installed directly from the GitHub repository using:

```bash
git clone https://github.com/your-org/buildstock-fetch.git
cd buildstock-fetch
pip install -e .
```

## Quick Start

The inputs required for BuildStock Fetch are: product type, release year, weather file type, release version, upgrade scenarios, state, types of files to download, and output directory. Once these inputs have been specified, BuildStock Fetch will retrieve the building ID's that correspond to the selection, and download the files requested to the specified output directory.

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
5. Selecting the upgrade scenarios (can choose multiple)
6. Selecting the state (can choose multiple)
7. Selecting the file types to download (can choose multiple)
8. Specifying the output directory for the downloaded files.

The interactive CLI mode only show valid release options and file types based on what's currently available on NREL's public database.

### Direct Command Line

You can also use direct command line arguments, for example:

```bash
buildstock-fetch \
  --product resstock \
  --release_year 2022 \
  --weather_file tmy3 \
  --release_version 1 \
  --states "CA NY" \
  --file_type "metadata load_curve_15min" \
  --upgrade_id "0 1" \
  --output_directory ./data
```

### Output Structure

The `--output_directory` option specifies the top directory where the files will be downloaded. From there, the files will be organized as:

```
output_directory/
└── release_name/
    └── file_type/
        └── state/
            └── file1
            └── file2
            └── ...
```

### Download All Files vs Sample

Once all the inputs have been collected, BuildStock Fetch will ask whether to download all files for the release or a sample. Each release has around 10,000 to 30,000 building ID's associated with it. If the user wishes to download just a small sample, they can specify the number of files to download for each state-upgrade version pair.

## Usage Examples

### Example 1: Download Metadata for California

```bash
buildstock-fetch \
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
    │   ├── TX/
    │   └── NY/
    └── schedule/
        ├── CA/
        ├── TX/
        └── NY/
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
Select release version: [1, 2]
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
