from pathlib import Path
from typing import Any, Literal, NewType, TypeAlias, get_args

from cloudpathlib import S3Path
from typing_extensions import TypeIs

ReleaseKey = Literal[
    "res_2021_amy2018_1",
    "res_2021_tmy3_1",
    "com_2021_amy2018_1",
    "com_2021_tmy3_1",
    "res_2022_amy2012_1",
    "res_2022_amy2012_1.1",
    "res_2022_amy2018_1",
    "res_2022_amy2018_1.1",
    "res_2022_tmy3_1",
    "res_2022_tmy3_1.1",
    "com_2023_amy2018_1",
    "com_2023_amy2018_2",
    "res_2024_amy2018_2",
    "res_2024_tmy3_1",
    "res_2024_tmy3_2",
    "com_2024_amy2018_1",
    "com_2024_amy2018_2",
    "res_2025_amy2012_1",
    "res_2025_amy2018_1",
    "com_2025_amy2012_2",
    "com_2025_amy2018_1",
    "com_2025_amy2018_2",
    "com_2025_amy2018_3",
]

UpgradeID = NewType("UpgradeID", str)
ReleaseYear: TypeAlias = Literal["2021", "2022", "2023", "2024", "2025"]
ResCom: TypeAlias = Literal["resstock", "comstock"]
Weather: TypeAlias = Literal["amy2012", "amy2018", "tmy3"]
ReleaseVersion = Literal["1", "1.1", "2", "3"]
Sample = int | Literal["all"]
FileType = Literal[
    "hpxml",
    "schedule",
    "metadata",
    "load_curve_15min",
    "load_curve_hourly",
    "load_curve_daily",
    "load_curve_monthly",
    "load_curve_annual",
    "trip_schedules",
    "weather",
]


USStateCode = Literal[
    "AL",
    # "AK",  - Too Russian
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
]


def is_s3_path(path: Path | S3Path | str) -> bool:
    if isinstance(path, S3Path):
        return True
    if isinstance(path, Path):
        return False
    return path.startswith("s3:/") or path.startswith("data.sb")


def is_valid_state_code(value: Any) -> TypeIs[USStateCode]:  # pyright: ignore[reportAny, reportExplicitAny]
    if not isinstance(value, str):
        return False
    return value in get_args(USStateCode)


def is_valid_upgrade_id(value: Any) -> TypeIs[UpgradeID]:  # pyright: ignore[reportAny, reportExplicitAny]
    if not isinstance(value, str):
        return False
    return value.isdigit() and str(int(value)) == value


def is_valid_release_key(value: Any) -> TypeIs[ReleaseKey]:  # pyright: ignore[reportAny, reportExplicitAny]
    if not isinstance(value, str):
        return False
    return value in get_args(ReleaseKey)


def is_valid_rescom(value: Any) -> TypeIs[ResCom]:  # pyright: ignore[reportAny, reportExplicitAny]
    if not isinstance(value, str):
        return False
    return value in get_args(ResCom)


def is_valid_file_type(value: Any) -> TypeIs[FileType]:  # pyright: ignore[reportAny, reportExplicitAny]
    if not isinstance(value, str):
        return False
    return value in get_args(FileType)


def normalize_upgrade_id(value: str) -> UpgradeID:
    return UpgradeID(str(int(value)))


def normalize_state_code(value: str) -> USStateCode:
    value = value.upper().strip()
    if not is_valid_state_code(value):
        raise ValueError(value)
    return value


def normalize_rescom(value: str) -> ResCom:
    value = value.strip().lower()
    if not is_valid_rescom(value):
        raise ValueError(value)
    return value


def normalize_release_key(value: str) -> ReleaseKey:
    value = value.strip().lower()
    if not is_valid_release_key(value):
        raise ValueError(value)
    return value


def normalize_file_type(value: str) -> FileType:
    value = value.strip().lower()
    if not is_valid_file_type(value):
        raise ValueError(value)
    return value
