import bisect
import statistics
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from random import Random
from typing import Annotated

import polars as pl
import typer
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn

from buildstock_fetch.building_ import get_bucket_boundaries


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    file_type: str
    upgrade_mode: str
    building_ids: tuple[int, ...] | None
    result_mode: str
    notes: str


@dataclass(frozen=True)
class CaseResult:
    case_name: str
    data_path: str
    file_type: str
    upgrade_mode: str
    result_mode: str
    building_count_filter: int | None
    rows: int
    columns: int
    unique_buildings: int | None
    first_seconds: float
    median_seconds: float
    min_seconds: float
    max_seconds: float
    stdev_seconds: float
    metric_column: str | None
    metric_sum: float | None
    min_timestamp: str | None
    max_timestamp: str | None
    notes: str


app = typer.Typer(add_completion=False)


def dataset_root(data_path: str, release: str, file_type: str) -> str:
    return f"{data_path.rstrip('/')}/{release}/{file_type}/"


def bucket_for_id(release: str, bldg_id: int) -> int:
    boundaries = get_bucket_boundaries(release)
    return min(bisect.bisect_left(boundaries, bldg_id), len(boundaries) - 1)


def load_metadata_building_ids(data_path: str, release: str, state: str, upgrade: str) -> list[int]:
    metadata_root = dataset_root(data_path, release, "metadata")
    df = (
        pl.scan_parquet(metadata_root)
        .filter(pl.col("state") == state)
        .filter(pl.col("upgrade") == int(upgrade))
        .select("bldg_id")
        .unique()
        .sort("bldg_id")
        .collect()
    )
    if df.is_empty():
        msg = f"No metadata rows found for release={release} state={state} upgrade={upgrade} at {data_path}"
        raise ValueError(msg)
    return df["bldg_id"].to_list()


def group_ids_by_bucket(release: str, building_ids: Sequence[int]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for bldg_id in building_ids:
        grouped[bucket_for_id(release, bldg_id)].append(bldg_id)
    return dict(sorted(grouped.items()))


def pick_same_bucket_ids(bucket_to_ids: dict[int, list[int]], count: int) -> tuple[int, ...]:
    eligible = [(bucket, ids) for bucket, ids in bucket_to_ids.items() if len(ids) >= count]
    if not eligible:
        msg = f"No bucket contains at least {count} buildings"
        raise ValueError(msg)
    bucket, ids = max(eligible, key=lambda item: len(item[1]))
    midpoint = len(ids) // 2
    start = max(0, midpoint - count // 2)
    selected = ids[start : start + count]
    if len(selected) < count:
        selected = ids[:count]
    if len(selected) < count:
        msg = f"Bucket {bucket} only yielded {len(selected)} buildings, expected {count}"
        raise ValueError(msg)
    return tuple(selected)


def pick_spread_bucket_ids(bucket_to_ids: dict[int, list[int]], count: int, seed: int) -> tuple[int, ...]:
    non_empty_buckets = [bucket for bucket, ids in bucket_to_ids.items() if ids]
    if not non_empty_buckets:
        raise ValueError("No non-empty buckets available")

    rng = Random(seed)
    position_by_bucket = {bucket: 0 for bucket in non_empty_buckets}
    selected: list[int] = []

    if count <= len(non_empty_buckets):
        indexes = [round(i * (len(non_empty_buckets) - 1) / max(count - 1, 1)) for i in range(count)]
        chosen_buckets = [non_empty_buckets[i] for i in indexes]
    else:
        chosen_buckets = []
        while len(chosen_buckets) < count:
            shuffled = non_empty_buckets[:]
            rng.shuffle(shuffled)
            chosen_buckets.extend(shuffled)
        chosen_buckets = chosen_buckets[:count]

    for bucket in chosen_buckets:
        ids = bucket_to_ids[bucket]
        idx = position_by_bucket[bucket] % len(ids)
        selected.append(ids[idx])
        position_by_bucket[bucket] += 1

    return tuple(sorted(selected))


def build_cases(
    release: str,
    building_ids: Sequence[int],
    upgrade: str,
    seed: int,
) -> list[BenchmarkCase]:
    bucket_to_ids = group_ids_by_bucket(release, building_ids)
    first_id = building_ids[len(building_ids) // 2]

    return [
        BenchmarkCase(
            name="hourly_full_state_single_upgrade",
            file_type="load_curve_hourly",
            upgrade_mode=upgrade,
            building_ids=None,
            result_mode="aggregate",
            notes="Primary success metric for state-wide hourly reads.",
        ),
        BenchmarkCase(
            name="monthly_full_state_single_upgrade",
            file_type="load_curve_monthly",
            upgrade_mode=upgrade,
            building_ids=None,
            result_mode="aggregate",
            notes="Checks the smaller monthly cadence against the same layout.",
        ),
        BenchmarkCase(
            name="hourly_full_state_all_upgrades",
            file_type="load_curve_hourly",
            upgrade_mode="all",
            building_ids=None,
            result_mode="aggregate",
            notes="Exercises partition pruning when all upgrade partitions are included.",
        ),
        BenchmarkCase(
            name="hourly_single_building",
            file_type="load_curve_hourly",
            upgrade_mode=upgrade,
            building_ids=(first_id,),
            result_mode="collect",
            notes="Worst-case targeted lookup after moving away from one file per building.",
        ),
        BenchmarkCase(
            name="hourly_ten_buildings_same_bucket",
            file_type="load_curve_hourly",
            upgrade_mode=upgrade,
            building_ids=pick_same_bucket_ids(bucket_to_ids, 10),
            result_mode="collect",
            notes="Small targeted read concentrated within one bucket/chunk.",
        ),
        BenchmarkCase(
            name="hourly_ten_buildings_spread_buckets",
            file_type="load_curve_hourly",
            upgrade_mode=upgrade,
            building_ids=pick_spread_bucket_ids(bucket_to_ids, 10, seed),
            result_mode="collect",
            notes="Small targeted read spread across many buckets/chunks.",
        ),
        BenchmarkCase(
            name="hourly_hundred_buildings_spread_buckets",
            file_type="load_curve_hourly",
            upgrade_mode=upgrade,
            building_ids=pick_spread_bucket_ids(bucket_to_ids, min(100, len(building_ids)), seed + 1),
            result_mode="aggregate",
            notes="Medium subset read spread across the state without materializing all rows.",
        ),
        BenchmarkCase(
            name="hourly_thousand_buildings_spread_buckets",
            file_type="load_curve_hourly",
            upgrade_mode=upgrade,
            building_ids=pick_spread_bucket_ids(bucket_to_ids, min(1000, len(building_ids)), seed + 2),
            result_mode="aggregate",
            notes="Large subset read to locate the crossover with full-state reads without exhausting RAM.",
        ),
    ]


def query_for_case(data_path: str, release: str, state: str, case: BenchmarkCase) -> pl.LazyFrame:
    lf = pl.scan_parquet(dataset_root(data_path, release, case.file_type))
    lf = lf.filter(pl.col("state") == state)
    if case.upgrade_mode != "all":
        lf = lf.filter(pl.col("upgrade") == int(case.upgrade_mode))
    if case.building_ids is not None:
        lf = lf.filter(pl.col("bldg_id").is_in(list(case.building_ids)))
    return lf


def choose_metric_column(lf: pl.LazyFrame) -> str | None:
    schema = lf.collect_schema()
    for column, dtype in schema.items():
        if column in {"bldg_id", "upgrade"}:
            continue
        if dtype in pl.NUMERIC_DTYPES:
            return column
    return None


def aggregate_summary(lf: pl.LazyFrame) -> tuple[pl.DataFrame, str | None]:
    schema = lf.collect_schema()
    metric_column = choose_metric_column(lf)
    expressions: list[pl.Expr] = [pl.len().alias("rows")]
    if "bldg_id" in schema:
        expressions.append(pl.col("bldg_id").n_unique().alias("unique_bldgs"))
    if "timestamp" in schema:
        expressions.extend(
            [
                pl.col("timestamp").min().cast(pl.String).alias("min_ts"),
                pl.col("timestamp").max().cast(pl.String).alias("max_ts"),
            ]
        )
    if metric_column is not None:
        expressions.append(pl.col(metric_column).sum().alias("metric_sum"))
    return lf.select(expressions).collect(), metric_column


def summarize_frame(df: pl.DataFrame) -> tuple[int | None, str | None, str | None, str | None, float | None]:
    unique_buildings = df["bldg_id"].n_unique() if "bldg_id" in df.columns else None

    min_timestamp: str | None = None
    max_timestamp: str | None = None
    if "timestamp" in df.columns and df.height > 0:
        timestamp_summary = df.select(
            pl.col("timestamp").min().cast(pl.String).alias("min_ts"),
            pl.col("timestamp").max().cast(pl.String).alias("max_ts"),
        ).row(0, named=True)
        min_timestamp = timestamp_summary["min_ts"]
        max_timestamp = timestamp_summary["max_ts"]

    metric_column = next(
        (
            column
            for column, dtype in zip(df.columns, df.dtypes, strict=True)
            if column not in {"bldg_id", "upgrade"} and dtype in pl.NUMERIC_DTYPES
        ),
        None,
    )
    metric_sum = float(df[metric_column].sum()) if metric_column is not None else None

    return unique_buildings, min_timestamp, max_timestamp, metric_column, metric_sum


def summarize_aggregate_result(df: pl.DataFrame) -> tuple[int, int | None, str | None, str | None, float | None]:
    row = df.row(0, named=True)
    rows = int(row["rows"])
    unique_buildings = int(row["unique_bldgs"]) if "unique_bldgs" in row and row["unique_bldgs"] is not None else None
    min_timestamp = row["min_ts"] if "min_ts" in row else None
    max_timestamp = row["max_ts"] if "max_ts" in row else None
    metric_sum = float(row["metric_sum"]) if "metric_sum" in row and row["metric_sum"] is not None else None
    return rows, unique_buildings, min_timestamp, max_timestamp, metric_sum


def benchmark_case(
    data_path: str,
    release: str,
    state: str,
    case: BenchmarkCase,
    repeats: int,
) -> CaseResult:
    timings: list[float] = []
    final_df: pl.DataFrame | None = None
    aggregate_df: pl.DataFrame | None = None
    metric_column: str | None = None

    for _ in range(repeats):
        lf = query_for_case(data_path, release, state, case)
        started_at = time.perf_counter()
        if case.result_mode == "aggregate":
            aggregate_df, metric_column = aggregate_summary(lf)
        else:
            final_df = lf.collect()
        timings.append(time.perf_counter() - started_at)

    if case.result_mode == "aggregate":
        if aggregate_df is None:
            raise RuntimeError(f"Aggregate benchmark case {case.name} did not produce a summary")
        rows, unique_buildings, min_timestamp, max_timestamp, metric_sum = summarize_aggregate_result(aggregate_df)
        columns = 0
    else:
        if final_df is None:
            raise RuntimeError(f"Collect benchmark case {case.name} did not produce a frame")
        rows = final_df.height
        columns = final_df.width
        unique_buildings, min_timestamp, max_timestamp, metric_column, metric_sum = summarize_frame(final_df)

    stdev_seconds = statistics.stdev(timings) if len(timings) > 1 else 0.0

    return CaseResult(
        case_name=case.name,
        data_path=data_path,
        file_type=case.file_type,
        upgrade_mode=case.upgrade_mode,
        result_mode=case.result_mode,
        building_count_filter=None if case.building_ids is None else len(case.building_ids),
        rows=rows,
        columns=columns,
        unique_buildings=unique_buildings,
        first_seconds=timings[0],
        median_seconds=statistics.median(timings),
        min_seconds=min(timings),
        max_seconds=max(timings),
        stdev_seconds=stdev_seconds,
        metric_column=metric_column,
        metric_sum=metric_sum,
        min_timestamp=min_timestamp,
        max_timestamp=max_timestamp,
        notes=case.notes,
    )


def print_selected_ids(cases: Iterable[BenchmarkCase]) -> None:
    print("\nSelected real building IDs from metadata:")
    for case in cases:
        if case.building_ids is None:
            continue
        preview = ", ".join(str(bldg_id) for bldg_id in case.building_ids[:10])
        suffix = "" if len(case.building_ids) <= 10 else ", ..."
        print(f"  {case.name}: [{preview}{suffix}]")


def print_results(results: Sequence[CaseResult]) -> None:
    df = pl.DataFrame(
        {
            "case": [result.case_name for result in results],
            "data_path": [result.data_path for result in results],
            "file_type": [result.file_type for result in results],
            "upgrade": [result.upgrade_mode for result in results],
            "result_mode": [result.result_mode for result in results],
            "bldg_filter_n": [result.building_count_filter for result in results],
            "rows": [result.rows for result in results],
            "cols": [result.columns for result in results],
            "unique_bldgs": [result.unique_buildings for result in results],
            "first_s": [round(result.first_seconds, 3) for result in results],
            "median_s": [round(result.median_seconds, 3) for result in results],
            "min_s": [round(result.min_seconds, 3) for result in results],
            "max_s": [round(result.max_seconds, 3) for result in results],
            "stdev_s": [round(result.stdev_seconds, 3) for result in results],
            "metric_col": [result.metric_column for result in results],
            "metric_sum": [None if result.metric_sum is None else round(result.metric_sum, 3) for result in results],
            "min_ts": [result.min_timestamp for result in results],
            "max_ts": [result.max_timestamp for result in results],
        }
    )
    print("\nBenchmark results:")
    print(df)

    print("\nCase notes:")
    seen: set[str] = set()
    for result in results:
        if result.case_name in seen:
            continue
        seen.add(result.case_name)
        print(f"  {result.case_name}: {result.notes}")


def print_case_result(result: CaseResult) -> None:
    metric_sum = "n/a" if result.metric_sum is None else round(result.metric_sum, 3)
    print(
        f"completed case={result.case_name} data_path={result.data_path} "
        f"mode={result.result_mode} rows={result.rows} unique_bldgs={result.unique_buildings} "
        f"first_s={result.first_seconds:.3f} median_s={result.median_seconds:.3f} metric_sum={metric_sum}"
    )


@app.command()
def main(
    data_path: Annotated[
        list[str],
        typer.Option(
            "--data-path",
            help=(
                "Base path containing release directories. Pass multiple times to compare "
                "old vs new layouts. Example: s3://data.sb/nrel/resstock"
            ),
        ),
    ],
    release: Annotated[str, typer.Option("--release")] = "res_2024_amy2018_2",
    state: Annotated[str, typer.Option("--state")] = "NY",
    upgrade: Annotated[str, typer.Option("--upgrade", help="Upgrade used for metadata-driven sampling.")] = "0",
    repeats: Annotated[int, typer.Option("--repeats")] = 5,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    metadata_source = data_path[0]
    building_ids = load_metadata_building_ids(metadata_source, release, state, upgrade)
    cases = build_cases(release, building_ids, upgrade, seed)

    print(
        "Metadata sample source:",
        metadata_source,
        f"| release={release} state={state} upgrade={upgrade} buildings={len(building_ids)}",
    )
    print_selected_ids(cases)

    results: list[CaseResult] = []
    total_cases = len(data_path) * len(cases)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task_id = progress.add_task("Running benchmark cases", total=total_cases)
        for one_data_path in data_path:
            for case in cases:
                progress.update(task_id, description=f"Running {case.name} on {one_data_path}")
                result = benchmark_case(one_data_path, release, state, case, repeats)
                results.append(result)
                print_case_result(result)
                progress.advance(task_id)

    print_results(results)


if __name__ == "__main__":
    app()
