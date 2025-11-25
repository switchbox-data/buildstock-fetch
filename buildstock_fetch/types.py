from typing import Literal, TypeAlias

__all__ = [
    "ReleaseYear",
    "ResCom",
    "Weather",
]


ReleaseYear: TypeAlias = Literal["2021", "2022", "2023", "2024", "2025"]
ResCom: TypeAlias = Literal["resstock", "comstock"]
Weather: TypeAlias = Literal["amy2018", "tmy3"]
