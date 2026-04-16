import json
from pathlib import Path
from typing import Annotated

import pyarrow.dataset as ds
import typer
from rich.console import Console
from rich.table import Table

from buildstock_fetch.building_ import _NUM_BUCKETS
from buildstock_fetch.constants import BUCKET_BOUNDARIES_FILE, METADATA_DIR
from buildstock_fetch.releases import RELEASES
from buildstock_fetch.types import ReleaseKey

app = typer.Typer(add_completion=False)
console = Console()


def _release_metadata_path(release: ReleaseKey) -> Path:
    release_obj = RELEASES[release]
    return (
        METADATA_DIR
        / f"product={release_obj.product}"
        / f"release_year={release_obj.year}"
        / f"weather_file={release_obj.weather}"
        / f"release_version={release_obj.version}"
    )


def _read_sorted_building_ids(release: ReleaseKey) -> list[int]:
    dataset = ds.dataset(_release_metadata_path(release), format="parquet", partitioning="hive")
    building_ids = dataset.to_table(columns=["bldg_id"]).column(0).to_pylist()
    sorted_ids = sorted(int(building_id) for building_id in building_ids)
    if not sorted_ids:
        msg = f"No building ids found for release {release}"
        raise ValueError(msg)
    return sorted_ids


def _compute_boundaries(sorted_ids: list[int], bucket_count: int) -> list[int]:
    total = len(sorted_ids)
    if total < bucket_count:
        msg = f"Release only has {total} building ids, fewer than {bucket_count} buckets"
        raise ValueError(msg)

    boundaries: list[int] = []
    for bucket_index in range(bucket_count):
        end_index = ((bucket_index + 1) * total + bucket_count - 1) // bucket_count - 1
        boundaries.append(sorted_ids[end_index])
    return boundaries


@app.command()
def main(
    release_filter: Annotated[
        str | None,
        typer.Option("--release", help="Only include releases whose key contains this substring."),
    ] = None,
    bucket_count: Annotated[
        int,
        typer.Option("--bucket-count", help="Number of buckets to generate boundaries for."),
    ] = _NUM_BUCKETS,
    output_path: Annotated[
        str,
        typer.Option("--output", help="Path to the bucket boundary JSON file."),
    ] = BUCKET_BOUNDARIES_FILE,
) -> None:
    """Regenerate the derived bucket boundary index from the packaged metadata.

    This file must be rebuilt whenever the underlying combined metadata changes.
    """
    release_keys = sorted(
        release.key for release in RELEASES if release_filter is None or release_filter in release.key
    )
    boundaries_by_release: dict[str, list[int]] = {}

    summary_table = Table(show_header=True, header_style="bold")
    summary_table.add_column("Release", no_wrap=True)
    summary_table.add_column("Buildings", justify="right")
    summary_table.add_column("Buckets", justify="right")
    summary_table.add_column("First Boundary", justify="right")
    summary_table.add_column("Last Boundary", justify="right")

    for release in release_keys:
        sorted_ids = _read_sorted_building_ids(release)
        boundaries = _compute_boundaries(sorted_ids, bucket_count)
        boundaries_by_release[release] = boundaries
        summary_table.add_row(
            release,
            str(len(sorted_ids)),
            str(len(boundaries)),
            str(boundaries[0]),
            str(boundaries[-1]),
        )

    output_file = Path(output_path)
    output_file.write_text(json.dumps(boundaries_by_release, indent=2))

    console.print(summary_table)
    console.print(f"Wrote bucket boundaries to {output_file}")
    console.print("Note: regenerate this file whenever the combined metadata changes.")


if __name__ == "__main__":
    app()
