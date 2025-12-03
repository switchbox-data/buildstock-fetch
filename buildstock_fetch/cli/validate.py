from pathlib import Path

from .availability import get_all_available_releases, get_state_options
from .inputs import Inputs


class InvalidProductError(Exception):
    """Exception raised when an invalid product is provided."""

    pass


def validate_output_directory(output_directory: Path | str) -> bool | str:
    """Validate that the path format is correct for a directory"""
    try:
        path = Path(output_directory)
        path.resolve()
    except (OSError, ValueError):
        return "Please enter a valid directory path"
    else:
        return True


def validate_release_name(inputs: Inputs) -> str | bool:
    """Validate the release name."""
    available_releases = get_all_available_releases()

    if inputs["product"] == "resstock":
        product_short_name = "res"
    elif inputs["product"] == "comstock":
        product_short_name = "com"
    else:
        raise InvalidProductError

    release_name = f"{product_short_name}_{inputs['release_year']}_{inputs['weather_file']}_{inputs['release_version']}"
    if release_name not in available_releases:
        return f"Invalid release name: {release_name}"
    return True


def validate_upgrade_ids(inputs: Inputs, release_name: str) -> str | bool:
    """Validate upgrade IDs."""
    available_releases = get_all_available_releases()
    for upgrade_id in inputs["upgrade_ids"]:
        if int(upgrade_id) not in [
            int(upgrade_id_val) for upgrade_id_val in available_releases[release_name]["upgrade_ids"]
        ]:
            return f"Invalid upgrade id: {upgrade_id}"
    return True


def validate_file_types(inputs: Inputs, release_name: str) -> str | bool:
    """Validate file types."""
    available_releases = get_all_available_releases()
    for file_type in inputs["file_type"]:
        # TODO: Validate EV related files
        if file_type not in available_releases[release_name]["available_data"]:
            return f"Invalid file type: {file_type}"
    return True


def validate_states(inputs: Inputs) -> str | bool:
    """Validate states."""
    for state in inputs["states"]:
        if state not in get_state_options():
            return f"Invalid state: {state}"
    return True
