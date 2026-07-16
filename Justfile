
# =============================================================================
# 🔍 CODE QUALITY & TESTING
# =============================================================================
# These commands check your code quality and run tests

# Run code quality tools
check:
    echo "🚀 Checking lock file consistency with 'pyproject.toml'"
    uv lock --locked
    echo "🚀 Linting, formatting, and type checking code"
    prek run -a
    echo "🚀 Static type checking: Running mypy"
    uv run mypy

# Check for obsolete dependencies
check-deps:
    echo "🚀 Checking for obsolete dependencies: Running deptry"
    uv run deptry .

# Test the code with pytest
test:
    echo "🚀 Testing code: Running pytest"
    uv run python -m pytest --doctest-modules

# =============================================================================
# 📚 DOCUMENTATION
# =============================================================================
# These commands help you build and serve project documentation

# Test if documentation can be built without warnings or errors
docs-test:
    uv run mkdocs build -s

# Build and serve the documentation
docs:
    uv run mkdocs serve

# =============================================================================
# 📦 BUILD & RELEASE
# =============================================================================
# These commands build your package and publish it to PyPI

# Clean build artifacts
clean-build:
    echo "🚀 Removing build artifacts"
    uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

# Build wheel file
build: clean-build
    echo "🚀 Creating wheel file"
    uvx --from build pyproject-build --installer uv

# Publish a release to PyPI
publish:
    echo "🚀 Publishing."
    uvx twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

# Build and publish
build-and-publish: build publish

# =============================================================================
# 🏗️  DEVELOPMENT ENVIRONMENT SETUP
# =============================================================================
# These commands help you set up your development environment

# Install the virtual environment and install the pre-commit hooks
install:
    echo "🚀 Creating virtual environment using uv, install pre-commit hooks with prek"
    .devcontainer/postCreateCommand.sh

# =============================================================================
# 📊 DATA DOWNLOAD
# =============================================================================
# These commands help you download and prepare data files

# Download NHTS (National Household Travel Survey) data
download-nhts:
    echo "🚀 Downloading NHTS data from https://nhts.ornl.gov/media/2022/download/csv.zip"
    uv run python utils/EVs/ev_data/inputs/download_2022_nhts_data.py --output-dir utils/EVs/ev_data/inputs

# Download PUMS (Public Use Microdata Sample) data for EV demand calculations
download-pums:
    echo "🚀 Downloading PUMS data from https://buildstock-fetch.s3.amazonaws.com/ev_demand/pums/NY_2021_pums_PUMA_HINCP_VEH_NP.csv"
    mkdir -p utils/EVs/ev_data/inputs
    curl -L -o utils/EVs/ev_data/inputs/NY_2021_pums_PUMA_HINCP_VEH_NP.csv \
        https://buildstock-fetch.s3.amazonaws.com/ev_demand/pums/NY_2021_pums_PUMA_HINCP_VEH_NP.csv
    echo "✅ PUMS data downloaded successfully"

# Download ResStock EV adoption reference data (ownership lookup + dependency tables)
download-resstock-ev-reference:
    echo "🚀 Downloading ResStock EV reference data from NREL/resstock"
    uv run python utils/EVs/ev_data/inputs/download_resstock_ev_reference.py
    echo "✅ ResStock EV reference data downloaded successfully"

# Download Maryland ACS 2021 PUMS from Census (for EV vehicle-ownership model)
download-pums-md:
    echo "🚀 Downloading Maryland PUMS from Census ACS 2021 1-year microdata"
    uv run python utils/EVs/ev_data/inputs/download_md_pums.py
    echo "✅ Maryland PUMS downloaded successfully"

# Run EV demand from a YAML scenario config
# just ev-demand utils/EVs/configs/md_2024.yaml
# just ev-demand utils/EVs/configs/md_2024.yaml --upload-s3
ev-demand config *args="":
    uv run python -m utils.EVs.ev_demand --config {{config}} {{args}}

# =============================================================================
# Utility functions
# =============================================================================
# These commands help you create internal mapping and metadata tables

# Create bldg_id to weather station id map
build-weather-station-map:
    echo "🔧 Starting interactive weather station mapping tool"
    uv run python -c "from utils.resolve_weather_station_id import _interactive_mode; _interactive_mode()"
