import statistics
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from random import Random
from typing import Annotated, Literal, Protocol, cast
from urllib.parse import urlparse

import boto3
import polars as pl
import typer
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn

from buildstock_fetch.explore import filter_downloads
from buildstock_fetch.read import BuildStockRead
from buildstock_fetch.releases import BuildstockReleases
from buildstock_fetch.types import (
    FileType,
    ReleaseKey,
    USStateCode,
    UpgradeID,
    normalize_file_type,
    normalize_state_code,
    normalize_upgrade_id,
)

app = typer.Typer(add_completion=False)

BenchmarkMode = Literal["schema", "head0", "full", "request_overhead"]


class ReadMethod(Protocol):
    def __call__(self, upgrades: str | None = None, building_ids: list[int] | None = None) -> pl.LazyFrame: ...


class S3Body(Protocol):
    def read(self) -> bytes: ...


class S3GetObjectResponse(Protocol):
    def __getitem__(self, key: str) -> S3Body: ...


class S3Client(Protocol):
    def get_object(self, *, Bucket: str, Key: str, Range: str) -> S3GetObjectResponse: ...


def get_reader_method(reader: BuildStockRead, file_type: FileType) -> ReadMethod:
    methods: dict[FileType, ReadMethod] = {
        "metadata": reader.read_metadata,
        "load_curve_15min": reader.read_load_curve_15min,
        "load_curve_hourly": reader.read_load_curve_hourly,
        "load_curve_daily": reader.read_load_curve_daily,
        "load_curve_monthly": reader.read_load_curve_monthly,
        "load_curve_annual": reader.read_load_curve_annual,
    }
    try:
        return methods[file_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported file type for BuildStockRead benchmark: {file_type}") from exc


def iter_s3_pages(bucket: str, prefix: str) -> Iterable[Mapping[str, object]]:
    s3_client = boto3.client("s3")  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    paginator = s3_client.get_paginator("list_objects_v2")  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    return cast(
        Iterable[Mapping[str, object]],
        paginator.paginate(Bucket=bucket, Prefix=prefix),  # pyright: ignore[reportUnknownMemberType]
    )


def count_s3_objects(base_path: str, release: str, file_type: str, state: str, upgrade: str | None) -> int:
    parsed = urlparse(base_path)
    bucket = parsed.netloc
    base_prefix = parsed.path.lstrip("/")

    prefix = f"{base_prefix}/{release}/{file_type}/state={state}/"
    if upgrade is not None:
        prefix += f"upgrade={int(upgrade):02d}/"

    total = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task_id = progress.add_task("Listing S3 objects", total=None)
        for page in iter_s3_pages(bucket, prefix):
            contents_obj = page.get("Contents", [])
            contents = cast(list[Mapping[str, object]], contents_obj)
            matched = sum(1 for obj in contents if str(obj.get("Key", "")).endswith(".parquet"))
            total += matched
            progress.update(task_id, advance=len(contents), description=f"Listing S3 objects ({total} parquet files)")
    return total


def count_local_files(base_path: str, release: str, file_type: str, state: str, upgrade: str | None) -> int:
    prefix = Path(base_path) / release / file_type / f"state={state}"
    if upgrade is not None:
        prefix = prefix / f"upgrade={int(upgrade):02d}"
    return sum(1 for path in prefix.rglob("*.parquet") if path.is_file())


def count_matching_files(base_path: str, release: str, file_type: str, state: str, upgrade: str | None) -> int:
    if base_path.startswith("s3://"):
        return count_s3_objects(base_path, release, file_type, state, upgrade)
    return count_local_files(base_path, release, file_type, state, upgrade)


def get_matching_parquet_paths(
    reader: BuildStockRead,
    release: ReleaseKey,
    file_type: FileType,
    state: USStateCode,
    upgrade: UpgradeID | None,
) -> list[str]:
    paths = [
        str(file_info.file_path)
        for file_info in filter_downloads(
            reader.data_path,
            release_key=release,
            file_type=file_type,
            state=state,
            upgrade=upgrade,
        )
        if str(file_info.file_path).endswith(".parquet")
    ]
    return sorted(paths)


def sample_paths(paths: list[str], max_files: int | None, random_seed: int) -> list[str]:
    if max_files is None or len(paths) <= max_files:
        return paths
    rng = Random(random_seed)
    return sorted(rng.sample(paths, max_files))


def s3_get_range(s3_client: S3Client, bucket: str, key: str, byte_range: str) -> None:
    response = s3_client.get_object(Bucket=bucket, Key=key, Range=byte_range)
    body = response["Body"]
    _ = body.read()


def benchmark_request_overhead(paths: list[str], byte_range: str) -> float:
    s3_client = cast(S3Client, boto3.client("s3"))  # pyright: ignore[reportUnknownMemberType]
    started_at = time.perf_counter()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task_id = progress.add_task("Benchmarking request overhead", total=len(paths))
        for path in paths:
            parsed = urlparse(path)
            s3_get_range(s3_client, parsed.netloc, parsed.path.lstrip("/"), byte_range)
            progress.advance(task_id)
    return time.perf_counter() - started_at


def run_benchmark(lazy_frame: pl.LazyFrame, mode: BenchmarkMode) -> tuple[float, int | None, int | None]:
    started_at = time.perf_counter()

    if mode == "schema":
        schema = lazy_frame.collect_schema()
        elapsed = time.perf_counter() - started_at
        return elapsed, None, len(schema)

    if mode == "head0":
        frame = lazy_frame.head(0).collect()
        elapsed = time.perf_counter() - started_at
        return elapsed, frame.height, frame.width

    frame = lazy_frame.collect()
    elapsed = time.perf_counter() - started_at
    return elapsed, frame.height, frame.width


@app.command()
def main(
    data_path: Annotated[
        str,
        typer.Option(
            "--data-path",
            help=(
                "Base directory containing release folders. Example local: ./data. "
                "Example S3: s3://data.sb/nrel/resstock"
            ),
        ),
    ],
    release: Annotated[str, typer.Option("--release", help="Release key, for example res_2024_amy2018_2")],
    state: Annotated[str, typer.Option("--state", help="Two-letter state code, for example NY")],
    file_type_input: Annotated[
        str,
        typer.Option(
            "--file-type",
            help="File type to benchmark. Defaults to load_curve_hourly.",
        ),
    ] = "load_curve_hourly",
    upgrade_input: Annotated[
        str,
        typer.Option(
            "--upgrade",
            help="Upgrade to read. Defaults to 0. Pass 'all' to read all available upgrades.",
        ),
    ] = "0",
    repeats: Annotated[
        int,
        typer.Option("--repeats", help="Number of benchmark runs to execute. Defaults to 3."),
    ] = 3,
    mode: Annotated[
        BenchmarkMode,
        typer.Option(
            "--mode",
            help=(
                "Benchmark mode: schema for metadata/footer overhead, head0 for zero-row execution, "
                "full for full read, request_overhead for tiny S3 range GETs on the matching files."
            ),
        ),
    ] = "full",
    max_files: Annotated[
        int | None,
        typer.Option("--max-files", help="Optional max number of matching files to benchmark."),
    ] = None,
    random_seed: Annotated[
        int,
        typer.Option("--random-seed", help="Random seed used when sampling matching files."),
    ] = 42,
    byte_range: Annotated[
        str,
        typer.Option("--byte-range", help="HTTP Range header value for request_overhead mode."),
    ] = "bytes=0-0",
    count_files: Annotated[
        bool,
        typer.Option("--count-files", help="Count matching parquet objects before benchmarking."),
    ] = False,
) -> None:
    release = cast(ReleaseKey, release)
    state = normalize_state_code(state)
    file_type = normalize_file_type(file_type_input)
    upgrade = None if upgrade_input == "all" else normalize_upgrade_id(upgrade_input)

    reader = BuildStockRead(
        data_path=data_path,
        release=release,
        states=state,
    )

    release_def = BuildstockReleases.load()[release]
    if file_type not in release_def.file_types:
        raise ValueError(f"{file_type} is not available for release {release}")

    if count_files:
        file_count = count_matching_files(data_path, release, file_type, state, upgrade)
        print(f"matching_parquet_files={file_count}")

    read_method = get_reader_method(reader, file_type)
    upgrades = "all" if upgrade is None else upgrade
    matching_paths: list[str] = []
    if mode == "request_overhead":
        if not data_path.startswith("s3://"):
            raise ValueError("request_overhead mode currently requires an s3:// data path")
        matching_paths = get_matching_parquet_paths(reader, release, file_type, state, upgrade)
        matching_paths = sample_paths(matching_paths, max_files, random_seed)
        print(f"request_overhead_files={len(matching_paths)}")

    timings: list[float] = []
    row_count: int | None = None
    column_count: int | None = None

    for run_idx in range(1, repeats + 1):
        if mode == "request_overhead":
            elapsed = benchmark_request_overhead(matching_paths, byte_range)
        else:
            lazy_frame = read_method(upgrades=None if upgrade is None else upgrade)
            elapsed, row_count, column_count = run_benchmark(lazy_frame, mode)
        timings.append(elapsed)

        print(f"run_{run_idx}_seconds={elapsed:.3f}")

    print(f"mode={mode}")
    print(f"rows={row_count}")
    print(f"cols={column_count}")
    print(f"repeats={repeats}")
    print(f"upgrade={upgrades}")
    print(f"mean_seconds={statistics.mean(timings):.3f}")
    print(f"median_seconds={statistics.median(timings):.3f}")
    print(f"min_seconds={min(timings):.3f}")
    print(f"max_seconds={max(timings):.3f}")


if __name__ == "__main__":
    app()
