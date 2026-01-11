import gc
import json
import os
from pathlib import Path
from typing import Any, get_args

import polars as pl

from buildstock_fetch.building import BuildingID
from buildstock_fetch.constants import LOAD_CURVE_COLUMN_AGGREGATION, SB_ANALYSIS_UPGRADES_FILE
from buildstock_fetch.types import FileType, FunctionalGroup, LoadCurveFuelType, ReleaseYear

# Module-level cache for SB analysis upgrades data
_SB_ANALYSIS_UPGRADES_CACHE: dict[str, Any] | None = None


def _get_SB_analysis_upgrades() -> dict[str, Any]:
    """Load SB analysis upgrades data once and cache it for subsequent calls."""
    global _SB_ANALYSIS_UPGRADES_CACHE
    if _SB_ANALYSIS_UPGRADES_CACHE is None:
        with open(SB_ANALYSIS_UPGRADES_FILE) as f:
            _SB_ANALYSIS_UPGRADES_CACHE = json.load(f)
    if _SB_ANALYSIS_UPGRADES_CACHE is None:
        msg = "SB analysis upgrades data not available"
        raise ValueError(msg)
    return _SB_ANALYSIS_UPGRADES_CACHE


# Module-level cache for SB analysis upgrades column map
_SB_ANALYSIS_UPGRADES_LOAD_CURVE_COLUMN_MAP_CACHE: pl.DataFrame | None = None
_SB_ANALYSIS_UPGRADES_METADATA_COLUMN_MAP_CACHE: pl.DataFrame | None = None


def _get_SB_analysis_upgrades_column_map(release_year: ReleaseYear, file_type: FileType) -> pl.DataFrame:
    """Load SB analysis upgrades column map once and cache it for subsequent calls."""
    global _SB_ANALYSIS_UPGRADES_LOAD_CURVE_COLUMN_MAP_CACHE, _SB_ANALYSIS_UPGRADES_METADATA_COLUMN_MAP_CACHE

    if release_year == "2024":
        if (
            file_type == "load_curve_15min"
            or file_type == "load_curve_hourly"
            or file_type == "load_curve_daily"
            or file_type == "load_curve_monthly"
        ):
            if _SB_ANALYSIS_UPGRADES_LOAD_CURVE_COLUMN_MAP_CACHE is None:
                column_map_file = LOAD_CURVE_COLUMN_AGGREGATION.joinpath("data_dictionary_2024_load_curve_labeled.csv")
                _SB_ANALYSIS_UPGRADES_LOAD_CURVE_COLUMN_MAP_CACHE = pl.read_csv(column_map_file)
                if _SB_ANALYSIS_UPGRADES_LOAD_CURVE_COLUMN_MAP_CACHE.is_empty():
                    msg = "SB analysis upgrades load curve column map not available"
                    raise ValueError(msg)
            return _SB_ANALYSIS_UPGRADES_LOAD_CURVE_COLUMN_MAP_CACHE
        elif file_type == "metadata" or file_type == "load_curve_annual":
            if _SB_ANALYSIS_UPGRADES_METADATA_COLUMN_MAP_CACHE is None:
                column_map_file = LOAD_CURVE_COLUMN_AGGREGATION.joinpath(
                    "data_dictionary_2024_metadata_and_annual_labeled.csv"
                )
                _SB_ANALYSIS_UPGRADES_METADATA_COLUMN_MAP_CACHE = pl.read_csv(column_map_file)
                if _SB_ANALYSIS_UPGRADES_METADATA_COLUMN_MAP_CACHE.is_empty():
                    msg = "SB analysis upgrades metadata column map not available"
                    raise ValueError(msg)
            return _SB_ANALYSIS_UPGRADES_METADATA_COLUMN_MAP_CACHE
    else:
        msg = f"Release year {release_year} not supported"
        raise ValueError(msg)
    return pl.DataFrame()


def _get_SB_upgrade_component_filename(
    final_bldg_id: BuildingID, component_bldg_id: BuildingID, output_dir: Path, file_type: FileType
) -> Path:
    if file_type == "metadata":
        return (
            output_dir
            / final_bldg_id.get_release_name()
            / file_type
            / f"state={final_bldg_id.state}"
            / f"upgrade={str(int(final_bldg_id.upgrade_id)).zfill(2)}"
            / f"upgrade{str(int(component_bldg_id.upgrade_id)).zfill(2)}.parquet"
        )
    elif (
        file_type == "load_curve_15min"
        or file_type == "load_curve_hourly"
        or file_type == "load_curve_daily"
        or file_type == "load_curve_monthly"
    ):
        return (
            output_dir
            / final_bldg_id.get_release_name()
            / file_type
            / f"state={final_bldg_id.state}"
            / f"upgrade={str(int(final_bldg_id.upgrade_id)).zfill(2)}"
            / component_bldg_id.get_output_filename(file_type)
        )
    elif file_type == "load_curve_annual":
        component_output_filename = component_bldg_id.get_annual_load_curve_filename()
        if component_output_filename is None:
            msg = f"Annual load curve filename is not available for {component_bldg_id.get_release_name()}, upgrade {component_bldg_id.upgrade_id}"
            raise ValueError(msg)
        return (
            output_dir
            / final_bldg_id.get_release_name()
            / file_type
            / f"state={final_bldg_id.state}"
            / f"upgrade={str(int(final_bldg_id.upgrade_id)).zfill(2)}"
            / component_output_filename
        )
    else:
        msg = f"File type {file_type} not supported"
        raise ValueError(msg)


def _initialize_SB_upgrade_dataframe(first_upgrade_component_filename: Path, file_type: FileType) -> pl.DataFrame:
    SB_upgrade_df = pl.DataFrame()
    if file_type == "metadata" or file_type == "load_curve_annual":
        SB_upgrade_df = pl.read_parquet(first_upgrade_component_filename).select("bldg_id")
    elif (
        file_type == "load_curve_15min"
        or file_type == "load_curve_hourly"
        or file_type == "load_curve_daily"
        or file_type == "load_curve_monthly"
    ):
        SB_upgrade_df = pl.read_parquet(first_upgrade_component_filename).select("timestamp")
    else:
        msg = f"File type {file_type} not supported"
        raise ValueError(msg)
    return SB_upgrade_df


def _add_SB_upgrade_load_curve_total_columns(SB_upgrade_load_curve_df: pl.DataFrame) -> pl.DataFrame:
    """Add total load curve columns for SB upgrade scenarios."""
    # Add total load curve columns for each fuel type
    for fuel_type in get_args(LoadCurveFuelType):
        # Extract all columns that start with the fuel type
        fuel_type_consumption_columns = [
            col
            for col in SB_upgrade_load_curve_df.columns
            if col.startswith(f"out.{fuel_type}")
            and "consumption" in col
            and "consumption_intensity" not in col
            and "savings" not in col
        ]
        fuel_type_consumption_intensity_columns = [
            col
            for col in SB_upgrade_load_curve_df.columns
            if col.startswith(f"out.{fuel_type}") and "consumption_intensity" in col and "savings" not in col
        ]

        # Skip if no columns found for this fuel type
        if not fuel_type_consumption_columns and not fuel_type_consumption_intensity_columns:
            continue

        # Create expressions for the total columns
        expressions = []
        if fuel_type_consumption_columns:
            expressions.append(
                pl.sum_horizontal(fuel_type_consumption_columns).alias(f"out.{fuel_type}.total.energy_consumption")
            )
        if fuel_type_consumption_intensity_columns:
            expressions.append(
                pl.sum_horizontal(fuel_type_consumption_intensity_columns).alias(
                    f"out.{fuel_type}.total.energy_consumption_intensity"
                )
            )

        # Add the total fuel type columns to the dataframe
        SB_upgrade_load_curve_df = SB_upgrade_load_curve_df.with_columns(expressions)

    return SB_upgrade_load_curve_df


def _get_SB_upgrade_component_columns(
    file_type: FileType, functional_group_column_names: list[str]
) -> tuple[str, list[str]]:
    if file_type == "metadata" or file_type == "load_curve_annual":
        join_column_name = "bldg_id"
        return join_column_name, [join_column_name, *functional_group_column_names]
    elif (
        file_type == "load_curve_15min"
        or file_type == "load_curve_hourly"
        or file_type == "load_curve_daily"
        or file_type == "load_curve_monthly"
    ):
        join_column_name = "timestamp"
        return join_column_name, [join_column_name, *functional_group_column_names]
    else:
        msg = f"File type {file_type} not supported"
        raise ValueError(msg)


def _is_valid_parquet_file(file_path: Path) -> bool:
    """Check if a parquet file exists, is non-empty, and is valid.

    Args:
        file_path: Path to the parquet file to validate

    Returns:
        True if the file is a valid parquet file, False otherwise
    """
    if not file_path.exists():
        return False

    # Check file size (must be > 0)
    if file_path.stat().st_size == 0:
        return False

    # Try to read the parquet file to validate it's complete and valid
    try:
        # Use lazy reading to avoid loading the entire file into memory
        _ = pl.scan_parquet(file_path).schema
    except Exception:
        # File is corrupted, incomplete, or not a valid parquet file
        return False
    else:
        return True


def _get_required_component_files_for_SB_upgrade(
    bldg_id: BuildingID, output_dir: Path, file_type: FileType
) -> list[Path]:
    """Get all required component file paths for an SB upgrade scenario."""
    sb_analysis_upgrades = _get_SB_analysis_upgrades()
    release_name = bldg_id.get_release_name()
    if release_name not in sb_analysis_upgrades:
        return []

    sb_analysis_upgrade_data = sb_analysis_upgrades[release_name]
    upgrade_components = sb_analysis_upgrade_data["upgrade_components"][bldg_id.upgrade_id]

    required_files: list[Path] = []

    for upgrade_id in upgrade_components:
        component_bldg_id = bldg_id.copy(upgrade_id=upgrade_id)
        component_filename = _get_SB_upgrade_component_filename(bldg_id, component_bldg_id, output_dir, file_type)
        required_files.append(component_filename)

    return required_files


def _process_SB_upgrade_scenario(
    bldg_id: BuildingID,
    output_dir: Path,
    output_file: Path,
    file_type: FileType,
    aggregate_time_step: str | None = None,
) -> None:
    """Process a SwitchBox Analysis upgrade load curve."""
    # Load SB analysis upgrades data json file
    sb_analysis_upgrades = _get_SB_analysis_upgrades()
    # Load SB analysis upgrades column map csv file (maps which columns are HVAC and which are appliances)
    sb_analysis_upgrades_column_map = _get_SB_analysis_upgrades_column_map(bldg_id.release_year, file_type)
    # Extract out this specific release's data
    release_name = bldg_id.get_release_name()
    if release_name not in sb_analysis_upgrades:
        msg = f"Release name {release_name} not found in SB analysis upgrades data"
        raise ValueError(msg)
    sb_analysis_upgrade_data = sb_analysis_upgrades[release_name]
    upgrade_components = sb_analysis_upgrade_data["upgrade_components"][bldg_id.upgrade_id]

    # Initialize the final SB upgrade load curve dataframe. Load the timestamp from the first upgrade component.
    SB_upgrade_df = pl.DataFrame()
    first_upgrade_component = upgrade_components[0]
    first_upgrade_component_bldg_id = bldg_id.copy(upgrade_id=first_upgrade_component)
    first_upgrade_component_filename = _get_SB_upgrade_component_filename(
        bldg_id, first_upgrade_component_bldg_id, output_dir, file_type
    )
    SB_upgrade_df = _initialize_SB_upgrade_dataframe(first_upgrade_component_filename, file_type)

    # Save component file names to delete later
    component_file_names_to_delete = []

    for functional_group in get_args(FunctionalGroup):
        if functional_group in ["total", "net"] or functional_group not in sb_analysis_upgrade_data:
            continue
        # Each functional group comes from a different upgrade
        # There are also column names associated with each functional group
        functional_group_upgrade_id = sb_analysis_upgrade_data[functional_group][bldg_id.upgrade_id]
        functional_group_column_names = (
            sb_analysis_upgrades_column_map.filter(pl.col("functional_group") == functional_group)
            .select("field_name")
            .to_series()
            .to_list()
        )
        if not functional_group_column_names:
            continue
        bldg_id_component = bldg_id.copy(upgrade_id=functional_group_upgrade_id)
        component_filename = _get_SB_upgrade_component_filename(bldg_id, bldg_id_component, output_dir, file_type)
        # Read the file for this component and add to the final SB upgrade dataframe
        join_column_name, columns_to_select = _get_SB_upgrade_component_columns(
            file_type, functional_group_column_names
        )

        # Get available columns from the parquet file and filter to only select existing columns
        available_columns = set(pl.scan_parquet(component_filename).collect_schema().names())
        columns_to_select_filtered = [col for col in columns_to_select if col in available_columns]

        # Ensure join column exists, otherwise skip this component
        if join_column_name not in columns_to_select_filtered:
            continue

        component_df = pl.read_parquet(component_filename).select(columns_to_select_filtered)
        SB_upgrade_df = SB_upgrade_df.join(component_df, on=join_column_name)
        # Add component file name to list of files to delete
        component_file_names_to_delete.append(component_filename)

    # Convert to set of strings for deletion
    component_file_names_to_delete_set = {str(path) for path in component_file_names_to_delete}

    # Delete the component load curve dataframes
    for component_filename_str in component_file_names_to_delete_set:
        os.remove(component_filename_str)
    gc.collect()

    if (
        file_type == "load_curve_15min"
        or file_type == "load_curve_hourly"
        or file_type == "load_curve_daily"
        or file_type == "load_curve_monthly"
    ):
        # Add total load curve columns for SB upgrade scenarios
        SB_upgrade_df = _add_SB_upgrade_load_curve_total_columns(SB_upgrade_df)

    # Save the final SB upgrade load curve dataframe to a parquet file
    SB_upgrade_df.write_parquet(output_file)
