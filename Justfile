
# =============================================================================
# 🔍 CODE QUALITY & TESTING
# =============================================================================
# These commands check your code quality and run tests

# Run code quality tools
check:
    echo "🚀 Checking lock file consistency with 'pyproject.toml'"
    uv lock --locked
    echo "🚀 Linting code: Running pre-commit"
    uv run pre-commit run -a
    echo "🚀 Static type checking: Running mypy"
    uv run mypy
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
    echo "🚀 Creating virtual environment using uv"
    uv sync
    uv run pre-commit install

# =============================================================================
# 📊 DATA DOWNLOAD
# =============================================================================
# These commands help you download and prepare data files

# Download NHTS (National Household Travel Survey) data
download-nhts:
    echo "🚀 Downloading NHTS data from https://nhts.ornl.gov/media/2022/download/csv.zip"
    uv run python utils/ev_data/inputs/download_2022_nhts_data.py --output-dir utils/ev_data/inputs

# Download PUMS (Public Use Microdata Sample) data for EV demand calculations
download-pums:
    echo "🚀 Downloading PUMS data from https://buildstock-fetch.s3.amazonaws.com/ev_demand/pums/NY_2021_pums_PUMA_HINCP_VEH_NP.csv"
    mkdir -p utils/ev_data/inputs
    curl -L -o utils/ev_data/inputs/NY_2021_pums_PUMA_HINCP_VEH_NP.csv \
        https://buildstock-fetch.s3.amazonaws.com/ev_demand/pums/NY_2021_pums_PUMA_HINCP_VEH_NP.csv
    echo "✅ PUMS data downloaded successfully"

# Save locally (default)
# just ev-demand NY res_2022_tmy3_1.1 2018-01-01 2018-01-02
# Upload to S3 (requires --upload-s3 flag)
# just ev-demand NY res_2022_tmy3_1.1 2018-01-01 2018-01-02 --upload-s3
ev-demand state release start_date end_date *args="":
    uv run python utils/ev_demand.py --state {{state}} --release {{release}} --start-date {{start_date}} --end-date {{end_date}} {{args}}