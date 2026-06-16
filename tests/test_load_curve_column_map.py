"""Tests for load curve column map CSVs and aggregation rules."""

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from buildstock_fetch.constants import LOAD_CURVE_COLUMN_AGGREGATION
from buildstock_fetch.loadcurves import _load_aggregation_rules

VALID_AGGREGATE_FUNCTIONS = {"sum", "mean", "first"}

ALL_COLUMN_MAP_CSVS = sorted(LOAD_CURVE_COLUMN_AGGREGATION.glob("*.csv"))
ALL_RELEASE_CSVS = [
    p for p in ALL_COLUMN_MAP_CSVS if not p.name[0].isdigit()
]


# ---------------------------------------------------------------------------
# CSV structure tests — run against every column map file
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("csv_path", ALL_COLUMN_MAP_CSVS, ids=lambda p: p.name)
def test_csv_has_required_columns(csv_path: Path):
    df = pl.read_csv(csv_path)
    assert "name" in df.columns
    assert "Aggregate_function" in df.columns


@pytest.mark.parametrize("csv_path", ALL_COLUMN_MAP_CSVS, ids=lambda p: p.name)
def test_csv_no_empty_aggregate_functions(csv_path: Path):
    df = pl.read_csv(csv_path)
    empty = df.filter(pl.col("Aggregate_function").str.strip_chars() == "")
    assert empty.is_empty(), f"Rows with empty Aggregate_function:\n{empty}"


@pytest.mark.parametrize("csv_path", ALL_COLUMN_MAP_CSVS, ids=lambda p: p.name)
def test_csv_only_valid_aggregate_functions(csv_path: Path):
    df = pl.read_csv(csv_path)
    invalid = df.filter(~pl.col("Aggregate_function").is_in(list(VALID_AGGREGATE_FUNCTIONS)))
    assert invalid.is_empty(), f"Invalid aggregate functions:\n{invalid}"


@pytest.mark.parametrize("csv_path", ALL_COLUMN_MAP_CSVS, ids=lambda p: p.name)
def test_csv_no_duplicate_column_names(csv_path: Path):
    df = pl.read_csv(csv_path)
    duplicates = df.filter(pl.col("name").is_duplicated())
    assert duplicates.is_empty(), f"Duplicate column names:\n{duplicates}"


@pytest.mark.parametrize("csv_path", ALL_COLUMN_MAP_CSVS, ids=lambda p: p.name)
def test_energy_consumption_columns_are_sum(csv_path: Path):
    df = pl.read_csv(csv_path)
    energy_cols = df.filter(pl.col("name").str.contains("energy_consumption"))
    wrong = energy_cols.filter(pl.col("Aggregate_function") != "sum")
    assert wrong.is_empty(), f"Energy consumption columns not using sum:\n{wrong}"


@pytest.mark.parametrize("csv_path", ALL_COLUMN_MAP_CSVS, ids=lambda p: p.name)
def test_energy_delivered_columns_are_sum(csv_path: Path):
    df = pl.read_csv(csv_path)
    delivered_cols = df.filter(pl.col("name").str.contains("energy_delivered"))
    if delivered_cols.is_empty():
        pytest.skip("No energy_delivered columns in this file")
    wrong = delivered_cols.filter(pl.col("Aggregate_function") != "sum")
    assert wrong.is_empty(), f"energy_delivered columns not using sum:\n{wrong}"


@pytest.mark.parametrize("csv_path", ALL_COLUMN_MAP_CSVS, ids=lambda p: p.name)
def test_temperature_columns_are_mean(csv_path: Path):
    df = pl.read_csv(csv_path)
    temp_cols = df.filter(pl.col("name").str.contains("temperature") | pl.col("name").str.contains("_temp"))
    if temp_cols.is_empty():
        pytest.skip("No temperature columns in this file")
    wrong = temp_cols.filter(pl.col("Aggregate_function") != "mean")
    assert wrong.is_empty(), f"Temperature columns not using mean:\n{wrong}"


@pytest.mark.parametrize("csv_path", ALL_COLUMN_MAP_CSVS, ids=lambda p: p.name)
def test_bldg_id_is_first(csv_path: Path):
    df = pl.read_csv(csv_path)
    bldg_rows = df.filter(pl.col("name") == "bldg_id")
    if bldg_rows.is_empty():
        pytest.skip("No bldg_id column in this file")
    assert bldg_rows["Aggregate_function"][0] == "first"


# ---------------------------------------------------------------------------
# 2025-specific CSV tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "csv_name",
    [
        "2025_resstock_load_curve_columns.csv",
        "res_2025_amy2012_1.csv",
        "res_2025_amy2018_1.csv",
    ],
)
def test_2025_csv_exists(csv_name: str):
    assert (LOAD_CURVE_COLUMN_AGGREGATION / csv_name).exists()


@pytest.mark.parametrize(
    "csv_name",
    ["2025_resstock_load_curve_columns.csv", "res_2025_amy2012_1.csv"],
)
def test_2025_csv_has_expected_new_columns(csv_name: str):
    """Columns present in 2025 schema but not in 2024 should be present."""
    df = pl.read_csv(LOAD_CURVE_COLUMN_AGGREGATION / csv_name)
    names = set(df["name"].to_list())
    expected_new = {
        "out.emissions.electricity.total.aer_midcase_avg..co2e_kg",
        "component_load__cooling__ceilings__kbtu",
        "out.schedules.electric_vehicle_charging",
    }
    for col in expected_new:
        assert col in names, f"Expected 2025 column '{col}' missing from {csv_name}"


def test_2025_amy2018_csv_has_expected_columns():
    """res_2025_amy2018_1.csv should contain the columns derived from its parquets."""
    df = pl.read_csv(LOAD_CURVE_COLUMN_AGGREGATION / "res_2025_amy2018_1.csv")
    names = set(df["name"].to_list())
    expected = {
        "bldg_id",
        "timestamp",
        "in.sqft",
        "out.electricity.cooling.energy_consumption..kwh",
        "out.emissions.electricity.total.aer_midcase_avg..co2e_kg",
        "out.schedules.electric_vehicle_charging",
        "out.indoor_temperature.conditioned_space..c",
        "out.outdoor_air_drybulb_temp..c",
    }
    for col in expected:
        assert col in names, f"Expected column '{col}' missing from res_2025_amy2018_1.csv"


def test_2025_amy2018_csv_uses_plain_sqft():
    """res_2025_amy2018_1.csv should use 'in.sqft', not 'in.sqft..ft2'."""
    df = pl.read_csv(LOAD_CURVE_COLUMN_AGGREGATION / "res_2025_amy2018_1.csv")
    names = set(df["name"].to_list())
    assert "in.sqft" in names
    assert "in.sqft..ft2" not in names


def test_2025_amy2018_sqft_is_first():
    df = pl.read_csv(LOAD_CURVE_COLUMN_AGGREGATION / "res_2025_amy2018_1.csv")
    row = df.filter(pl.col("name") == "in.sqft")
    assert row["Aggregate_function"][0] == "first"


# ---------------------------------------------------------------------------
# loadcurves.py aggregation rules loading (per-release path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("release", ["res_2025_amy2012_1", "res_2025_amy2018_1"])
def test_load_aggregation_rules_2025(release: str):
    rules = _load_aggregation_rules(release)
    assert len(rules) > 0


@pytest.mark.parametrize("release", ["res_2025_amy2012_1", "res_2025_amy2018_1"])
def test_load_aggregation_rules_2025_is_cached(release: str):
    rules_first = _load_aggregation_rules(release)
    rules_second = _load_aggregation_rules(release)
    assert rules_first is rules_second


def test_load_aggregation_rules_res_2025_amy2012_1():
    rules = _load_aggregation_rules("res_2025_amy2012_1")
    assert len(rules) > 0


def test_load_aggregation_rules_res_2025_amy2012_1_is_cached():
    rules_first = _load_aggregation_rules("res_2025_amy2012_1")
    rules_second = _load_aggregation_rules("res_2025_amy2012_1")
    assert rules_first is rules_second


# ---------------------------------------------------------------------------
# main.py aggregation path (year-level CSV)
# ---------------------------------------------------------------------------


def _make_2025_load_curve_df(columns: list[str]) -> pl.DataFrame:
    """Build a minimal synthetic 15-min load curve DataFrame with 2025 column names."""
    n = 4
    base = datetime(2025, 1, 1, 0, 15)
    timestamps = [base + timedelta(minutes=15 * i) for i in range(n)]
    data: dict[str, list] = {"timestamp": timestamps}
    for col in columns:
        if col in ("bldg_id",):
            data[col] = [1] * n
        elif col == "timestamp":
            continue
        else:
            data[col] = [1.0] * n
    return pl.DataFrame(data)


def test_aggregate_load_curve_2025_year_path():
    """_aggregate_load_curve_aggregate correctly loads the 2025 CSV and aggregates."""
    from buildstock_fetch.main import _aggregate_load_curve_aggregate

    csv_path = LOAD_CURVE_COLUMN_AGGREGATION / "2025_resstock_load_curve_columns.csv"
    csv_cols = pl.read_csv(csv_path)["name"].to_list()
    df = _make_2025_load_curve_df(csv_cols)

    result = _aggregate_load_curve_aggregate(df, "hourly", "2025")

    assert "timestamp" in result.columns
    assert result.height > 0


def test_aggregate_load_curve_2025_sum_columns_are_summed():
    """Sum columns should add up correctly across 15-min intervals."""
    from buildstock_fetch.main import _aggregate_load_curve_aggregate

    csv_path = LOAD_CURVE_COLUMN_AGGREGATION / "2025_resstock_load_curve_columns.csv"
    csv_cols = pl.read_csv(csv_path)["name"].to_list()

    # Use only sum + first columns to keep the DataFrame simple
    sum_cols = (
        pl.read_csv(csv_path)
        .filter(pl.col("Aggregate_function").is_in(["sum", "first"]))["name"]
        .to_list()
    )
    df = _make_2025_load_curve_df(sum_cols)

    result = _aggregate_load_curve_aggregate(df, "hourly", "2025")

    # Each hour has 4 × 15-min rows all with value 1.0 → sum should be 4.0
    energy_col = next(c for c in result.columns if c.startswith("out.electricity.") and c != "timestamp")
    assert result[energy_col][0] == pytest.approx(4.0)


def test_aggregate_load_curve_2025_mean_columns_are_averaged():
    """Mean columns should be averaged, not summed, across 15-min intervals."""
    from buildstock_fetch.main import _aggregate_load_curve_aggregate

    csv_path = LOAD_CURVE_COLUMN_AGGREGATION / "2025_resstock_load_curve_columns.csv"
    mean_cols = (
        pl.read_csv(csv_path)
        .filter(pl.col("Aggregate_function").is_in(["mean", "first"]))["name"]
        .to_list()
    )
    df = _make_2025_load_curve_df(mean_cols)

    result = _aggregate_load_curve_aggregate(df, "hourly", "2025")

    temp_col = next(c for c in result.columns if "temperature" in c or "_temp" in c)
    assert result[temp_col][0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# res_2025_amy2018_1 per-release aggregation (loadcurves.py path)
# ---------------------------------------------------------------------------


def _make_amy2018_load_curve_df() -> pl.DataFrame:
    """Build a synthetic 15-min load curve DataFrame matching res_2025_amy2018_1 columns."""
    csv_cols = pl.read_csv(LOAD_CURVE_COLUMN_AGGREGATION / "res_2025_amy2018_1.csv")["name"].to_list()
    n = 4
    base = datetime(2025, 1, 1, 0, 15)
    timestamps = [base + timedelta(minutes=15 * i) for i in range(n)]
    data: dict[str, list] = {"timestamp": timestamps}
    for col in csv_cols:
        if col == "timestamp":
            continue
        elif col == "bldg_id":
            data[col] = [1] * n
        else:
            data[col] = [1.0] * n
    return pl.DataFrame(data)


def test_amy2018_aggregation_rules_produce_correct_expressions():
    """Rules loaded from res_2025_amy2018_1.csv should produce one expression per non-timestamp column."""
    csv = pl.read_csv(LOAD_CURVE_COLUMN_AGGREGATION / "res_2025_amy2018_1.csv")
    rules = _load_aggregation_rules("res_2025_amy2018_1")
    # One expression per column except timestamp (handled separately in the aggregation loop)
    non_timestamp_cols = [r for r in csv["name"].to_list() if r != "timestamp"]
    assert len(rules) == len(non_timestamp_cols)


def test_amy2018_per_release_sum_columns_are_summed():
    """Sum columns should total across 4 × 15-min rows using the per-release rules."""
    rules = _load_aggregation_rules("res_2025_amy2018_1")
    df = _make_amy2018_load_curve_df()

    lf = df.lazy()
    lf = lf.with_columns((pl.col("timestamp").cast(pl.Datetime) - timedelta(minutes=15)).alias("timestamp"))
    lf = lf.with_columns(pl.col("timestamp").dt.truncate("1h").alias("_bucket"))
    result = lf.group_by("_bucket").agg(rules).collect()

    # All 4 rows fall in the same hour → energy sum should be 4.0
    energy_col = next(
        c for c in result.columns
        if c.startswith("out.electricity.") and c not in ("_bucket", "timestamp")
    )
    assert result[energy_col][0] == pytest.approx(4.0)


def test_amy2018_per_release_mean_columns_are_averaged():
    """Mean columns should be averaged across 4 × 15-min rows using the per-release rules."""
    rules = _load_aggregation_rules("res_2025_amy2018_1")
    df = _make_amy2018_load_curve_df()

    lf = df.lazy()
    lf = lf.with_columns((pl.col("timestamp").cast(pl.Datetime) - timedelta(minutes=15)).alias("timestamp"))
    lf = lf.with_columns(pl.col("timestamp").dt.truncate("1h").alias("_bucket"))
    result = lf.group_by("_bucket").agg(rules).collect()

    temp_col = next(c for c in result.columns if "temperature" in c)
    assert result[temp_col][0] == pytest.approx(1.0)
