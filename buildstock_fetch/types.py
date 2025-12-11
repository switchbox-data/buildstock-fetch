from typing import Literal, NewType, TypeAlias, TypeGuard, get_args

__all__ = [
    "ReleaseYear",
    "ResCom",
    "Weather",
]


ReleaseName = NewType("ReleaseName", str)
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


def is_valid_state_code(value: str) -> TypeGuard[USStateCode]:
    return value in get_args(USStateCode)
