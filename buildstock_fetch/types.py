from typing import Any, Literal, NewType, TypeAlias, get_args

from typing_extensions import TypeIs

ReleaseKeyResY2021 = Literal[
    "res_2021_amy2018_1",
    "res_2021_tmy3_1",
]

ReleaseKeyComY2021 = Literal[
    "com_2021_amy2018_1",
    "com_2021_tmy3_1",
]

ReleaseKeyY2021 = Literal[
    ReleaseKeyResY2021,
    ReleaseKeyComY2021,
]

ReleaseKeyResY2022 = Literal[
    "res_2022_amy2012_1",
    "res_2022_amy2012_1.1",
    "res_2022_amy2018_1",
    "res_2022_amy2018_1.1",
    "res_2022_tmy3_1",
    "res_2022_tmy3_1.1",
]

ReleaseKeyY2022 = ReleaseKeyResY2022

ReleaseKeyComY2023 = Literal[
    "com_2023_amy2018_1",
    "com_2023_amy2018_2",
]

ReleaseKeyY2023 = ReleaseKeyComY2023

ReleaseKeyResY2024 = Literal[
    "res_2024_amy2018_2",
    "res_2024_tmy3_1",
    "res_2024_tmy3_2",
]

ReleaseKeyComY2024 = Literal[
    "com_2024_amy2018_1",
    "com_2024_amy2018_2",
]

ReleaseKeyY2024 = Literal[ReleaseKeyResY2024, ReleaseKeyComY2024]

ReleaseKeyResY2025 = Literal[
    "res_2025_amy2012_1",
    "res_2025_amy2018_1",
]

ReleaseKeyComY2025 = Literal[
    "com_2025_amy2012_2",
    "com_2025_amy2018_1",
    "com_2025_amy2018_2",
    "com_2025_amy2018_3",
]

ReleaseKeyY2025 = Literal[
    ReleaseKeyResY2025,
    ReleaseKeyComY2025,
]

ReleaseKeyRes = Literal[
    ReleaseKeyResY2021,
    ReleaseKeyY2022,
    ReleaseKeyResY2024,
    ReleaseKeyResY2025,
]

ReleaseKeyCom = Literal[
    ReleaseKeyComY2021,
    ReleaseKeyY2023,
    ReleaseKeyComY2024,
    ReleaseKeyComY2025,
]

ReleaseKey = Literal[
    ReleaseKeyRes,
    ReleaseKeyCom,
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

ReleaseKeyAmy2012 = Literal[
    "res_2022_amy2012_1",
    "res_2022_amy2012_1.1",
    "com_2025_amy2012_2",
]

ReleaseKeyAmy2018 = Literal[
    "res_2021_amy2018_1",
    "com_2021_amy2018_1",
    "res_2022_amy2018_1",
    "res_2022_amy2018_1.1",
    "com_2023_amy2018_1",
    "com_2023_amy2018_2",
    "res_2024_amy2018_2",
    "com_2024_amy2018_1",
    "com_2024_amy2018_2",
    "res_2025_amy2012_1",
    "res_2025_amy2018_1",
    "com_2025_amy2018_1",
    "com_2025_amy2018_2",
    "com_2025_amy2018_3",
]

ReleaseKeyTMY3 = Literal[
    "res_2021_tmy3_1",
    "com_2021_tmy3_1",
    "res_2022_tmy3_1",
    "res_2022_tmy3_1.1",
    "res_2024_tmy3_1",
    "res_2024_tmy3_2",
]

ReleaseKeyV1 = Literal[
    "res_2021_amy2018_1",
    "res_2021_tmy3_1",
    "com_2021_amy2018_1",
    "com_2021_tmy3_1",
    "res_2022_amy2012_1",
    "res_2022_amy2018_1",
    "res_2022_tmy3_1",
    "com_2023_amy2018_1",
    "res_2024_tmy3_1",
    "com_2024_amy2018_1",
    "res_2025_amy2012_1",
    "res_2025_amy2018_1",
    "com_2025_amy2018_1",
]

ReleaseKeyV1_1 = Literal[
    "res_2022_amy2012_1.1",
    "res_2022_amy2018_1.1",
    "res_2022_tmy3_1.1",
]

ReleaseKeyV2 = Literal[
    "com_2023_amy2018_2",
    "res_2024_amy2018_2",
    "res_2024_tmy3_2",
    "com_2024_amy2018_2",
    "com_2025_amy2012_2",
    "com_2025_amy2018_2",
]

ReleaseKeyV3 = Literal["com_2025_amy2018_3",]


def is_year_2021(release_key: ReleaseKey) -> TypeIs[ReleaseKeyY2021]:
    return "_2021_" in release_key


def is_year_2022(release_key: ReleaseKey) -> TypeIs[ReleaseKeyY2022]:
    return "_2022_" in release_key


def is_year_2023(release_key: ReleaseKey) -> TypeIs[ReleaseKeyY2023]:
    return "_2023_" in release_key


def is_year_2024(release_key: ReleaseKey) -> TypeIs[ReleaseKeyY2024]:
    return "_2024_" in release_key


def is_year_2025(release_key: ReleaseKey) -> TypeIs[ReleaseKeyY2025]:
    return "_2025_" in release_key


def is_amy2012(release_key: ReleaseKey) -> TypeIs[ReleaseKeyAmy2012]:
    return "amy2012" in release_key


def is_amy2018(release_key: ReleaseKey) -> TypeIs[ReleaseKeyAmy2018]:
    return "amy2018" in release_key


def is_tmy3(release_key: ReleaseKey) -> TypeIs[ReleaseKeyTMY3]:
    return "tmy3" in release_key


def is_version_1(release_key: ReleaseKey) -> TypeIs[ReleaseKeyV1]:
    return release_key.endswith("_1")


def is_version_1_1(release_key: ReleaseKey) -> TypeIs[ReleaseKeyV1_1]:
    return release_key.endswith("_1.1")


def is_version_2(release_key: ReleaseKey) -> TypeIs[ReleaseKeyV2]:
    return release_key.endswith("_2")


def is_version_3(release_key: ReleaseKey) -> TypeIs[ReleaseKeyV3]:
    return release_key.endswith("_3")


def is_resstock(release_key: ReleaseKey) -> TypeIs[ReleaseKeyRes]:
    return release_key.startswith("res_")


def is_comstock(release_key: ReleaseKey) -> TypeIs[ReleaseKeyCom]:
    return release_key.startswith("com_")


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
