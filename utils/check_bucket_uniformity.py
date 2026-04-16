import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import pyarrow.dataset as ds
import typer
from rich.console import Console
from rich.table import Table

from buildstock_fetch.building_ import _NUM_BUCKETS, Building
from buildstock_fetch.constants import METADATA_DIR
from buildstock_fetch.releases import RELEASES
from buildstock_fetch.types import ReleaseKey, UpgradeID

app = typer.Typer(add_completion=False)
console = Console()


@dataclass(frozen=True)
class BucketUniformityStats:
    release: ReleaseKey
    upgrades: tuple[UpgradeID, ...]
    total_buildings: int
    bucket_count: int
    min_bucket_id: int
    max_bucket_id: int
    min_bucket_size: int
    max_bucket_size: int
    mean_bucket_size: float
    median_bucket_size: float
    stddev_bucket_size: float
    empty_buckets: int
    max_abs_deviation: float
    max_deviation_pct: float
    min_bucket_ids: tuple[int, ...]
    max_bucket_ids: tuple[int, ...]
    bucket_sizes: tuple[int, ...]
    buckets_with_overlap: int
    overlapping_bucket_pairs: int
    overlapping_bucket_pair_pct: float
    max_overlap_depth: int


@dataclass(frozen=True)
class BucketRange:
    bucket_id: int
    min_building_id: int
    max_building_id: int
    count: int


def _compute_overlap_stats(bucket_ranges: list[BucketRange]) -> tuple[int, int, float, int]:
    if len(bucket_ranges) <= 1:
        return 0, 0, 0.0, len(bucket_ranges)

    overlapping_pairs = 0
    buckets_with_overlap: set[int] = set()
    events: list[tuple[int, int]] = []

    for index, current in enumerate(bucket_ranges):
        events.append((current.min_building_id, 1))
        events.append((current.max_building_id + 1, -1))
        for other in bucket_ranges[index + 1 :]:
            if current.min_building_id <= other.max_building_id and other.min_building_id <= current.max_building_id:
                overlapping_pairs += 1
                buckets_with_overlap.add(current.bucket_id)
                buckets_with_overlap.add(other.bucket_id)

    active_ranges = 0
    max_overlap_depth = 0
    for _, delta in sorted(events):
        active_ranges += delta
        if active_ranges > max_overlap_depth:
            max_overlap_depth = active_ranges

    total_pairs = len(bucket_ranges) * (len(bucket_ranges) - 1) // 2
    overlapping_pair_pct = overlapping_pairs / total_pairs * 100 if total_pairs else 0.0
    return len(buckets_with_overlap), overlapping_pairs, overlapping_pair_pct, max_overlap_depth


def _release_metadata_path(release: ReleaseKey) -> Path:
    release_obj = RELEASES[release]
    return (
        METADATA_DIR
        / f"product={release_obj.product}"
        / f"release_year={release_obj.year}"
        / f"weather_file={release_obj.weather}"
        / f"release_version={release_obj.version}"
    )


def _build_bucket_probe(release: ReleaseKey) -> Building:
    release_obj = RELEASES[release]
    first_upgrade = sorted(release_obj.upgrades, key=int)[0]
    return Building(id=0, release=release, upgrade=first_upgrade, state="NY", cached_county=None)


def _compute_release_stats(release: ReleaseKey, bucket_count: int) -> BucketUniformityStats:
    _ = bucket_count
    dataset = ds.dataset(_release_metadata_path(release), format="parquet", partitioning="hive")
    building = _build_bucket_probe(release)
    total_buildings = 0
    counts_by_bucket: dict[int, int] = {}
    min_id_by_bucket: dict[int, int] = {}
    max_id_by_bucket: dict[int, int] = {}

    for batch in dataset.to_batches(columns=["bldg_id"], batch_size=65_536):
        building_ids = batch.column(0).to_pylist()
        total_buildings += len(building_ids)
        for building_id in building_ids:
            building_id_int = int(building_id)
            object.__setattr__(building, "id", building_id_int)
            bucket_id = building.bucket
            counts_by_bucket[bucket_id] = counts_by_bucket.get(bucket_id, 0) + 1
            min_id_by_bucket[bucket_id] = min(building_id_int, min_id_by_bucket.get(bucket_id, building_id_int))
            max_id_by_bucket[bucket_id] = max(building_id_int, max_id_by_bucket.get(bucket_id, building_id_int))

    if not counts_by_bucket:
        msg = f"No building ids found for release {release}"
        raise ValueError(msg)

    sorted_bucket_ids = sorted(counts_by_bucket)
    bucket_sizes = tuple(counts_by_bucket[bucket_id] for bucket_id in sorted_bucket_ids)
    bucket_ranges = [
        BucketRange(
            bucket_id=bucket_id,
            min_building_id=min_id_by_bucket[bucket_id],
            max_building_id=max_id_by_bucket[bucket_id],
            count=counts_by_bucket[bucket_id],
        )
        for bucket_id in sorted_bucket_ids
    ]
    observed_bucket_count = len(bucket_sizes)
    mean_bucket_size = total_buildings / observed_bucket_count
    min_bucket_size = min(bucket_sizes)
    max_bucket_size = max(bucket_sizes)
    max_abs_deviation = max(abs(count - mean_bucket_size) for count in bucket_sizes)
    max_deviation_pct = max_abs_deviation / mean_bucket_size * 100
    min_bucket_ids = tuple(bucket_id for bucket_id, count in counts_by_bucket.items() if count == min_bucket_size)
    max_bucket_ids = tuple(bucket_id for bucket_id, count in counts_by_bucket.items() if count == max_bucket_size)
    buckets_with_overlap, overlapping_bucket_pairs, overlapping_bucket_pair_pct, max_overlap_depth = (
        _compute_overlap_stats(bucket_ranges)
    )

    return BucketUniformityStats(
        release=release,
        upgrades=tuple(sorted(RELEASES[release].upgrades, key=int)),
        total_buildings=total_buildings,
        bucket_count=observed_bucket_count,
        min_bucket_id=sorted_bucket_ids[0],
        max_bucket_id=sorted_bucket_ids[-1],
        min_bucket_size=min_bucket_size,
        max_bucket_size=max_bucket_size,
        mean_bucket_size=mean_bucket_size,
        median_bucket_size=statistics.median(bucket_sizes),
        stddev_bucket_size=statistics.pstdev(bucket_sizes),
        empty_buckets=max(0, sorted_bucket_ids[-1] - sorted_bucket_ids[0] + 1 - observed_bucket_count),
        max_abs_deviation=max_abs_deviation,
        max_deviation_pct=max_deviation_pct,
        min_bucket_ids=tuple(sorted(min_bucket_ids)),
        max_bucket_ids=tuple(sorted(max_bucket_ids)),
        bucket_sizes=bucket_sizes,
        buckets_with_overlap=buckets_with_overlap,
        overlapping_bucket_pairs=overlapping_bucket_pairs,
        overlapping_bucket_pair_pct=overlapping_bucket_pair_pct,
        max_overlap_depth=max_overlap_depth,
    )


def _format_upgrades(upgrades: tuple[UpgradeID, ...]) -> str:
    if len(upgrades) <= 6:
        return ",".join(upgrades)
    return f"{upgrades[0]}..{upgrades[-1]} ({len(upgrades)} total)"


@app.command()
def main(
    release_filter: Annotated[
        str | None,
        typer.Option("--release", help="Only include releases whose key contains this substring."),
    ] = None,
    bucket_count: Annotated[
        int,
        typer.Option(
            "--bucket-count",
            help="Legacy option retained for compatibility. Observed bucket ids now determine the actual bucket count.",
        ),
    ] = _NUM_BUCKETS,
    show_bucket_sizes: Annotated[
        bool,
        typer.Option("--show-bucket-sizes", help="Include the full per-bucket counts in text output."),
    ] = False,
    show_bucket_id_lists: Annotated[
        bool,
        typer.Option("--show-bucket-id-lists", help="Include the min_bucket_ids and max_bucket_ids lists in text output."),
    ] = False,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON instead of text."),
    ] = False,
) -> None:
    release_keys = sorted(
        release.key for release in RELEASES if release_filter is None or release_filter in release.key
    )
    stats = [_compute_release_stats(release, bucket_count) for release in release_keys]

    if as_json:
        typer.echo(json.dumps([asdict(item) for item in stats], indent=2))
        return

    console.print(f"Building.bucket uniformity check across {len(stats)} releases and their upgrades")
    console.print(f"Requested bucket count hint: {bucket_count}")

    stats_table = Table(show_header=True, header_style="bold")
    stats_table.add_column("Release", no_wrap=True)
    stats_table.add_column("Upgrades")
    stats_table.add_column("Buildings", justify="right")
    stats_table.add_column("Bucket IDs", justify="right")
    stats_table.add_column("Min", justify="right")
    stats_table.add_column("Max", justify="right")
    stats_table.add_column("Mean", justify="right")
    stats_table.add_column("Median", justify="right")
    stats_table.add_column("Stddev", justify="right")
    stats_table.add_column("Empty", justify="right")
    stats_table.add_column("Max Dev", justify="right")
    stats_table.add_column("Overlap")

    for item in stats:
        stats_table.add_row(
            item.release,
            _format_upgrades(item.upgrades),
            str(item.total_buildings),
            f"{item.min_bucket_id}..{item.max_bucket_id} ({item.bucket_count})",
            str(item.min_bucket_size),
            str(item.max_bucket_size),
            f"{item.mean_bucket_size:.2f}",
            f"{item.median_bucket_size:.2f}",
            f"{item.stddev_bucket_size:.2f}",
            str(item.empty_buckets),
            f"{item.max_abs_deviation:.2f} ({item.max_deviation_pct:.2f}%)",
            (
                f"b={item.buckets_with_overlap}/{item.bucket_count}, "
                f"p={item.overlapping_bucket_pairs} ({item.overlapping_bucket_pair_pct:.2f}%), "
                f"d={item.max_overlap_depth}"
            ),
        )

    console.print(stats_table)
    console.print("Legend: `Bucket IDs` = observed bucket id range and count. `Max Dev` = max absolute deviation from mean bucket size.")
    console.print("Legend: `Overlap` uses `b` = buckets with overlap, `p` = overlapping bucket pairs, `d` = max overlap depth.")

    if show_bucket_id_lists:
        bucket_id_table = Table(show_header=True, header_style="bold")
        bucket_id_table.add_column("Release", no_wrap=True)
        bucket_id_table.add_column("Min Bucket IDs")
        bucket_id_table.add_column("Max Bucket IDs")
        for item in stats:
            bucket_id_table.add_row(
                item.release,
                str(list(item.min_bucket_ids)),
                str(list(item.max_bucket_ids)),
            )
        console.print(bucket_id_table)

    if show_bucket_sizes:
        bucket_sizes_table = Table(show_header=True, header_style="bold")
        bucket_sizes_table.add_column("Release", no_wrap=True)
        bucket_sizes_table.add_column("Bucket Sizes")
        for item in stats:
            bucket_sizes_table.add_row(item.release, str(list(item.bucket_sizes)))
        console.print(bucket_sizes_table)

    if not stats:
        return

    worst_release = max(stats, key=lambda item: item.max_deviation_pct)
    highest_bucket = max(stats, key=lambda item: item.max_bucket_size)
    lowest_bucket = min(stats, key=lambda item: item.min_bucket_size)
    worst_overlap_release = max(stats, key=lambda item: item.overlapping_bucket_pair_pct)

    overall_table = Table(title="Overall", show_header=True, header_style="bold")
    overall_table.add_column("Metric")
    overall_table.add_column("Value")
    overall_table.add_row(
        "Worst max deviation",
        f"{worst_release.release} ({worst_release.max_deviation_pct:.2f}%)",
    )
    overall_table.add_row(
        "Lowest min bucket",
        f"{lowest_bucket.release} ({lowest_bucket.min_bucket_size})",
    )
    overall_table.add_row(
        "Highest max bucket",
        f"{highest_bucket.release} ({highest_bucket.max_bucket_size})",
    )
    overall_table.add_row(
        "Worst overlap",
        (
            f"{worst_overlap_release.release} "
            f"({worst_overlap_release.overlapping_bucket_pair_pct:.2f}% of bucket pairs, "
            f"max_depth={worst_overlap_release.max_overlap_depth})"
        ),
    )
    console.print(overall_table)


if __name__ == "__main__":
    app()
