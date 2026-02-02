# Mixed Upgrade Scenarios

This guide demonstrates how to use `BuildStockRead` and `MixedUpgradeScenario` to model building upgrade adoption over time.

## Overview

The mixed upgrade scenario functionality allows you to:

- Model heterogeneous upgrade mixes where buildings adopt different upgrades
- Simulate multi-year adoption trajectories
- Ensure monotonic adoption (buildings can only move from baseline to upgraded, never backwards)
- Read and analyze data across different upgrade scenarios and time periods

## Quick Start

### 1. Define a Scenario

Use the `uniform_adoption` helper to create a scenario where buildings progressively adopt upgrades over time:

```python
from buildstock_fetch.scenarios import uniform_adoption

# Define a scenario with 2 upgrades over 3 years
# 60% of adopters choose upgrade 4, 40% choose upgrade 8
# Total adoption grows from 10% to 30% to 50% over 3 years
scenario = uniform_adoption(
    upgrade_ids=[4, 8],
    weights={4: 0.6, 8: 0.4},
    adoption_trajectory=[0.1, 0.3, 0.5],
)

# This creates:
# {4: [0.06, 0.18, 0.30], 8: [0.04, 0.12, 0.20]}
# Year 0: 6% upgrade 4, 4% upgrade 8, 90% baseline
# Year 1: 18% upgrade 4, 12% upgrade 8, 70% baseline
# Year 2: 30% upgrade 4, 20% upgrade 8, 50% baseline
```

### 2. Create a Mixed Upgrade Scenario

```python
from buildstock_fetch.mixed_upgrade import MixedUpgradeScenario

mus = MixedUpgradeScenario(
    data_path="./data",                  # Path to downloaded data
    release="res_2024_tmy3_2",           # BuildStock release
    states="NY",                         # State(s) to analyze
    sample_n=1000,                       # Sample 1000 buildings (optional)
    random=42,                           # Random seed for reproducibility
    scenario=scenario,                   # Adoption scenario
)
```

### 3. Read and Analyze Data

```python
# Read metadata for all years
metadata = mus.read_metadata().collect()
print(metadata.head())

# Read metadata for specific years only
metadata_years_0_1 = mus.read_metadata(years=[0, 1]).collect()

# Read 15-minute load curves
load_curves = mus.read_load_curve_15min().collect()

# Export scenario to CAIRO-compatible CSV format
mus.export_scenario_to_cairo("./scenario.csv")
```

## Reading BuildStock Data with BuildStockRead

Before using `MixedUpgradeScenario`, you can use `BuildStockRead` to explore individual releases:

```python
from buildstock_fetch.read import BuildStockRead

# Initialize reader for a specific release and state
bsr = BuildStockRead(
    data_path="./data",
    release="res_2024_tmy3_2",
    states="NY",
    sample_n=500,
    random=42,
)

# Read baseline metadata (upgrade 0)
baseline = bsr.read_metadata(upgrades="0").collect()

# Read specific upgrade
upgrade_4 = bsr.read_metadata(upgrades="4").collect()

# Read load curves for a specific upgrade
load_curves = bsr.read_load_curve_15min(upgrades="4").collect()

# Filter by building IDs
specific_buildings = bsr.read_metadata(
    upgrades="0",
    building_ids=[123456, 234567, 345678]
).collect()
```

## Advanced Scenario Configuration

### Custom Adoption Trajectories

You can manually define scenarios with custom adoption patterns:

```python
# Complex scenario with 3 upgrades
scenario = {
    4: [0.05, 0.15, 0.25, 0.35],   # Upgrade 4: grows from 5% to 35%
    8: [0.03, 0.10, 0.18, 0.25],   # Upgrade 8: grows from 3% to 25%
    12: [0.02, 0.05, 0.07, 0.10],  # Upgrade 12: grows from 2% to 10%
}

mus = MixedUpgradeScenario(
    data_path="./data",
    release="res_2024_tmy3_2",
    states=["NY", "CA"],            # Multiple states
    scenario=scenario,
)
```

### Multiple States

Analyze adoption across multiple states:

```python
mus = MixedUpgradeScenario(
    data_path="./data",
    release="res_2024_tmy3_2",
    states=["NY", "CA", "TX"],      # Analyze 3 states
    sample_n=2000,                   # Sample across all states
    scenario=scenario,
)

metadata = mus.read_metadata().collect()

# Group by state to see state-specific adoption
state_summary = metadata.group_by(["in.state", "year", "upgrade_id"]).agg(
    pl.count("bldg_id").alias("count")
)
print(state_summary)
```

### Accessing Materialized Allocations

Get the exact building-to-upgrade mapping for each year:

```python
# Get materialized scenario (cached property)
materialized = mus.materialized_scenario

# materialized is a dict: {year_idx: {bldg_id: upgrade_id}}
# Example:
# {
#   0: {405821: 0, 612547: 4, 789234: 0, ...},
#   1: {405821: 0, 612547: 4, 789234: 8, ...},
#   2: {405821: 4, 612547: 4, 789234: 8, ...}
# }

# Check which upgrade a specific building has in year 1
building_id = 612547
year_1_upgrade = materialized[1][building_id]
print(f"Building {building_id} has upgrade {year_1_upgrade} in year 1")
```

## Reading Different Data Types

`MixedUpgradeScenario` supports multiple temporal resolutions:

```python
# 15-minute resolution
load_15min = mus.read_load_curve_15min(years=[0, 1, 2]).collect()

# Hourly resolution
load_hourly = mus.read_load_curve_hourly().collect()

# Daily resolution
load_daily = mus.read_load_curve_daily().collect()

# Annual totals
load_annual = mus.read_load_curve_annual().collect()

# All methods return Polars LazyFrames - call .collect() to execute
```

## Exporting for CAIRO

Export your scenario to a CSV format compatible with CAIRO (or other analysis tools):

```python
mus.export_scenario_to_cairo("./my_scenario.csv")
```

Output format:
```
bldg_id,year_0,year_1,year_2
405821,0,0,0
612547,0,4,4
789234,0,0,8
```

Each row is a building, each column is a year, and values are upgrade IDs.

## Data Requirements

Before running mixed upgrade scenarios, ensure you have downloaded the required data:

```bash
# Download metadata and load curves for all required upgrades
bsf \
  --product resstock \
  --release_year 2024 \
  --weather_file tmy3 \
  --release_version 2 \
  --states "NY CA" \
  --file_type "metadata load_curve_15min" \
  --upgrade_id "0 4 8" \
  --output_directory ./data \
  --sample 100
```

If scenario data is missing, `MixedUpgradeScenario` will raise a `ScenarioDataNotFoundError` indicating which upgrades are missing.

## Performance Tips

1. **Sampling**: Use `sample_n` to work with a representative subset of buildings
2. **Lazy Evaluation**: `read_*` methods return LazyFrames - only call `.collect()` when you need results
3. **Specific Years**: Read only needed years: `read_metadata(years=[0, 1])` instead of all years
4. **State Filtering**: Download and analyze only relevant states

## Example: Complete Analysis Workflow

```python
from buildstock_fetch.mixed_upgrade import MixedUpgradeScenario
from buildstock_fetch.scenarios import uniform_adoption
import polars as pl

# 1. Define adoption scenario
scenario = uniform_adoption(
    upgrade_ids=[4, 8],
    weights={4: 0.7, 8: 0.3},
    adoption_trajectory=[0.1, 0.2, 0.3, 0.4, 0.5],
)

# 2. Initialize scenario reader
mus = MixedUpgradeScenario(
    data_path="./data",
    release="res_2024_tmy3_2",
    states=["NY", "CA"],
    sample_n=5000,
    random=42,
    scenario=scenario,
)

# 3. Read and analyze metadata
metadata = mus.read_metadata().collect()

# 4. Compute adoption statistics by year
adoption_stats = metadata.group_by(["year", "upgrade_id"]).agg([
    pl.count("bldg_id").alias("building_count"),
    pl.col("bldg_id").count().truediv(pl.len()).alias("fraction"),
])

print("Adoption over time:")
print(adoption_stats.sort(["year", "upgrade_id"]))

# 5. Analyze energy consumption by upgrade and year
load_annual = mus.read_load_curve_annual().collect()

energy_by_upgrade = load_annual.group_by(["year", "upgrade_id"]).agg([
    pl.col("out.electricity.total.energy_consumption").sum().alias("total_electricity_kwh"),
    pl.col("out.natural_gas.total.energy_consumption").sum().alias("total_gas_therm"),
])

print("\nEnergy consumption by upgrade scenario:")
print(energy_by_upgrade.sort(["year", "upgrade_id"]))

# 6. Export scenario for CAIRO
mus.export_scenario_to_cairo("./scenario.csv")

print("\nScenario exported to scenario.csv")
```

## See Also

- [BuildStock Fetch Usage Guide](usage.md) - General usage and data downloading
- [API Reference](modules.md) - Complete API documentation
- [NREL End-Use Load Profiles](https://www.nrel.gov/buildings/end-use-load-profiles) - Data source information
