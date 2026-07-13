"""Assign ResStock 2025 EV battery class, usable capacity, and efficiency.

ResStock samples each dwelling's EV type from a national BEV stock distribution
(Experian 2023 registrations via TEMPO), then looks up Autonomie parameters for
usable battery capacity (kWh) and combined fuel economy (kWh/mile). This module
reproduces that assignment for our post-hoc EV demand pipeline on ResStock buildings.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

# Default local cache of ResStock housing-characteristic / Autonomie reference files
# (populated by `just download-resstock-ev-reference`).
DEFAULT_RESSTOCK_EV_REFERENCE_DIR = (
    Path(__file__).resolve().parent / "ev_data" / "inputs" / "resstock_ev_reference"
)
DEFAULT_BATTERY_TSV = "Electric_Vehicle_Battery.tsv"  # national option shares
DEFAULT_AUTONOMIE_CSV = "resstock_autonomie_2022_vehicle_params.csv"  # kWh + kWh/mi by option
DEFAULT_SATURATIONS_CSV = "resstock_options_saturations.csv"  # fallback source for the same shares

# Housing-characteristic TSV headers look like: Option=Compact, Battery Electric Vehicle, 200 mile range
_OPTION_HEADER_RE = re.compile(r"^Option=(.+)$")
# Canonical option names used by both the TSV and Autonomie CSV.
_OPTION_PARSE_RE = re.compile(
    r"^(?P<body_class>Compact|Midsize|Pickup|SUV), "
    r"Battery Electric Vehicle, "
    r"(?P<range_miles>\d+) mile range$"
)


def parse_ev_option_name(option_name: str) -> tuple[str, int]:
    """Extract body class and range miles from a ResStock EV battery option name."""
    match = _OPTION_PARSE_RE.match(option_name.strip())
    if match is None:
        raise ValueError(f"Unrecognized EV battery option name: {option_name!r}")
    return match.group("body_class"), int(match.group("range_miles"))


def load_ev_battery_option_probabilities(path: Path) -> pl.DataFrame:
    """
    Load option probabilities from ResStock ``Electric Vehicle Battery.tsv``.

    The housing-characteristic file has one data row of option shares and a
    trailing ``sampling_probability`` column that should equal 1.
    """
    if not path.exists():
        raise FileNotFoundError(f"EV battery options file not found: {path}")

    # Skip blank lines and ResStock comment lines that start with '#'.
    data_rows: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            data_rows.append(stripped)

    # Expect exactly: header row with Option=... columns, then one probability row.
    if len(data_rows) < 2:
        raise ValueError(f"EV battery TSV must have a header and data row: {path}")

    header_cells = data_rows[0].split("\t")
    value_cells = data_rows[1].split("\t")
    if len(header_cells) != len(value_cells):
        raise ValueError(
            f"EV battery TSV header/value length mismatch in {path}: "
            f"{len(header_cells)} vs {len(value_cells)}"
        )

    # Build (option_name, probability) pairs; ignore the trailing sampling_probability=1 column.
    option_names: list[str] = []
    probabilities: list[float] = []
    for header, value in zip(header_cells, value_cells, strict=True):
        option_match = _OPTION_HEADER_RE.match(header)
        if option_match is None:
            if header == "sampling_probability":
                continue  # ResStock QA column; not an EV option
            raise ValueError(f"Unexpected EV battery TSV column: {header!r}")
        option_names.append(option_match.group(1))
        probabilities.append(float(value))

    probs = pl.DataFrame({
        "ev_option_name": option_names,
        "probability": probabilities,
    })
    # Soft check before multinomial sampling (numpy also requires sum≈1).
    total = float(probs["probability"].sum())
    if not np.isclose(total, 1.0, atol=1e-5):
        raise ValueError(f"EV battery option probabilities sum to {total}, expected 1.0")
    return probs


def load_ev_battery_option_probabilities_from_saturations(path: Path) -> pl.DataFrame:
    """Fallback loader using ``options_saturations.csv`` Electric Vehicle Battery rows.

    Same national shares as the housing-characteristic TSV, stored in the flat
    saturations table ResStock uses for other option parameters.
    """
    if not path.exists():
        raise FileNotFoundError(f"ResStock options saturations file not found: {path}")

    probs = (
        pl.read_csv(path)
        .filter(pl.col("Parameter") == "Electric Vehicle Battery")
        .select(
            pl.col("Option").alias("ev_option_name"),
            pl.col("Saturation").cast(pl.Float64).alias("probability"),
        )
    )
    if probs.is_empty():
        raise ValueError(f"No Electric Vehicle Battery rows found in {path}")
    total = float(probs["probability"].sum())
    if not np.isclose(total, 1.0, atol=1e-5):
        raise ValueError(f"EV battery option saturations sum to {total}, expected 1.0")
    return probs


def load_autonomie_vehicle_params(path: Path) -> pl.DataFrame:
    """Load Autonomie usable capacity and efficiency keyed by EV option name.

    These are the physical model inputs that ResStock maps into HPXML / EnergyPlus
    (usable kWh pack size and combined kWh per mile).
    """
    if not path.exists():
        raise FileNotFoundError(f"Autonomie vehicle params file not found: {path}")

    params = (
        pl.read_csv(path)
        .select(
            pl.col("option_name").alias("ev_option_name"),
            # Usable capacity is what SOC / charging constraints should use (not nominal).
            pl.col("vehicle_battery_usable_capacity_kwh").cast(pl.Float64).alias("battery_capacity_kwh"),
            # Base vehicle efficiency before any temperature adjustment.
            pl.col("vehicle_fuel_economy_combined_kwh_per_mile")
            .cast(pl.Float64)
            .alias("kwh_per_mile"),
        )
        # Derive body_class / range_miles from the option string for matching / diagnostics.
        .with_columns(
            pl.col("ev_option_name")
            .map_elements(lambda name: parse_ev_option_name(name)[0], return_dtype=pl.Utf8)
            .alias("body_class"),
            pl.col("ev_option_name")
            .map_elements(lambda name: parse_ev_option_name(name)[1], return_dtype=pl.Int64)
            .alias("range_miles"),
        )
    )
    if params.is_empty():
        raise ValueError(f"Autonomie vehicle params file is empty: {path}")
    return params


def vehicle_slots_from_building_evs(bldg_veh_df: pl.DataFrame) -> pl.DataFrame:
    """
    Expand buildings with ``vehicles`` > 0 into one row per ``(bldg_id, vehicle_id)``.

    ``vehicle_id`` is 1-based within each building, matching NHTS / trip schedule slots.
    In the current max-1-EV adoption model, ``vehicles`` is usually 0 or 1.
    """
    if "bldg_id" not in bldg_veh_df.columns or "vehicles" not in bldg_veh_df.columns:
        raise ValueError("bldg_veh_df must include bldg_id and vehicles columns")

    # Only EV-owning (or multi-vehicle) buildings get battery attributes.
    occupied = bldg_veh_df.filter(pl.col("vehicles") > 0)
    if occupied.is_empty():
        # Preserve bldg_id dtype for empty result schema compatibility.
        return pl.DataFrame(
            schema={
                "bldg_id": bldg_veh_df.schema.get("bldg_id", pl.Int64),
                "vehicle_id": pl.Int64,
            }
        )

    # int_ranges(1, vehicles+1) -> [1], [1,2], ... then explode to one row per slot.
    return (
        occupied.select("bldg_id", "vehicles")
        .with_columns(
            pl.int_ranges(1, pl.col("vehicles") + 1).alias("vehicle_id"),
        )
        .explode("vehicle_id")
        .select("bldg_id", pl.col("vehicle_id").cast(pl.Int64))
    )


@dataclass
class EVBatteryAssigner:
    """
    Sample ResStock 2025 EV battery options and attach Autonomie capacity / efficiency.

    Mirrors ResStock: national BEV stock shares with no household covariates
    (income, PUMA, miles, etc. do not affect the draw).
    """

    # Columns: ev_option_name, probability (must sum to ~1)
    option_probabilities: pl.DataFrame
    # Columns: ev_option_name, battery_capacity_kwh, kwh_per_mile, body_class, range_miles
    autonomie_params: pl.DataFrame
    random_state: int = 42
    # Seeded RNG created in __post_init__; not part of construction args.
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Validate probability table schema.
        required_prob_cols = {"ev_option_name", "probability"}
        missing_prob = required_prob_cols - set(self.option_probabilities.columns)
        if missing_prob:
            raise ValueError(f"option_probabilities missing columns: {sorted(missing_prob)}")

        # Validate Autonomie params schema (capacity + efficiency for SOC discharge).
        required_param_cols = {
            "ev_option_name",
            "battery_capacity_kwh",
            "kwh_per_mile",
            "body_class",
            "range_miles",
        }
        missing_params = required_param_cols - set(self.autonomie_params.columns)
        if missing_params:
            raise ValueError(f"autonomie_params missing columns: {sorted(missing_params)}")

        # Every sampled option must have Autonomie parameters (join would otherwise null out).
        autonomie_names = set(self.autonomie_params["ev_option_name"].to_list())
        missing_options = [
            name
            for name in self.option_probabilities["ev_option_name"].to_list()
            if name not in autonomie_names
        ]
        if missing_options:
            raise ValueError(
                "EV battery options missing Autonomie parameters: " + ", ".join(missing_options)
            )

        total = float(self.option_probabilities["probability"].sum())
        if not np.isclose(total, 1.0, atol=1e-5):
            raise ValueError(f"EV battery option probabilities sum to {total}, expected 1.0")

        self._rng = np.random.default_rng(self.random_state)

    @classmethod
    def from_paths(
        cls,
        battery_path: Path | str,
        autonomie_path: Path | str,
        *,
        random_state: int = 42,
        saturations_fallback_path: Path | str | None = None,
    ) -> EVBatteryAssigner:
        """
        Load probabilities from an EV battery TSV and Autonomie params from a CSV.

        Mirrors how EV ownership is configured with an explicit file path
        (``ev_ownership_path``), rather than requiring a whole reference directory.
        """
        battery_tsv = Path(battery_path)
        autonomie_csv = Path(autonomie_path)

        if battery_tsv.exists():
            probs = load_ev_battery_option_probabilities(battery_tsv)
        elif saturations_fallback_path is not None and Path(saturations_fallback_path).exists():
            # Older caches may only have options_saturations.csv.
            logging.warning(
                "EV battery TSV not found at %s; falling back to %s",
                battery_tsv,
                saturations_fallback_path,
            )
            probs = load_ev_battery_option_probabilities_from_saturations(Path(saturations_fallback_path))
        else:
            raise FileNotFoundError(
                f"EV battery options file not found: {battery_tsv}. "
                "Run `just download-resstock-ev-reference` to download the data."
            )

        if not autonomie_csv.exists():
            raise FileNotFoundError(
                f"Autonomie vehicle params not found: {autonomie_csv}. "
                "Run `just download-resstock-ev-reference` to download the data."
            )

        params = load_autonomie_vehicle_params(autonomie_csv)
        return cls(
            option_probabilities=probs,
            autonomie_params=params,
            random_state=random_state,
        )

    @classmethod
    def from_resstock_reference(
        cls,
        reference_dir: Path | str | None = None,
        *,
        random_state: int = 42,
    ) -> EVBatteryAssigner:
        """Load from a ResStock EV reference directory (convenience wrapper around ``from_paths``)."""
        ref_dir = Path(reference_dir) if reference_dir is not None else DEFAULT_RESSTOCK_EV_REFERENCE_DIR
        return cls.from_paths(
            battery_path=ref_dir / DEFAULT_BATTERY_TSV,
            autonomie_path=ref_dir / DEFAULT_AUTONOMIE_CSV,
            saturations_fallback_path=ref_dir / DEFAULT_SATURATIONS_CSV,
            random_state=random_state,
        )

    def assign(self, vehicle_slots: pl.DataFrame) -> pl.DataFrame:
        """
        Draw an EV battery option for each vehicle slot and join Autonomie parameters.

        Args:
            vehicle_slots: DataFrame with ``bldg_id`` and ``vehicle_id`` columns

        Returns:
            One row per vehicle with option name, body class, range, capacity, and efficiency
        """
        required = {"bldg_id", "vehicle_id"}
        missing = required - set(vehicle_slots.columns)
        if missing:
            raise ValueError(f"vehicle_slots missing columns: {sorted(missing)}")

        # Empty input → typed empty output (keeps downstream concat / write happy).
        if vehicle_slots.is_empty():
            return pl.DataFrame(
                schema={
                    "bldg_id": vehicle_slots.schema.get("bldg_id", pl.Int64),
                    "vehicle_id": pl.Int64,
                    "ev_option_name": pl.Utf8,
                    "body_class": pl.Utf8,
                    "range_miles": pl.Int64,
                    "battery_capacity_kwh": pl.Float64,
                    "kwh_per_mile": pl.Float64,
                }
            )

        option_names = self.option_probabilities["ev_option_name"].to_list()
        probabilities = np.asarray(self.option_probabilities["probability"].to_numpy(), dtype=np.float64)
        # ResStock shares can sum to 1.000000057; numpy choice requires exact renormalization.
        probabilities = probabilities / probabilities.sum()
        # Independent draws with replacement = ResStock national stock mix (no covariates).
        drawn = self._rng.choice(option_names, size=vehicle_slots.height, replace=True, p=probabilities)

        assigned = vehicle_slots.select("bldg_id", "vehicle_id").with_columns(
            pl.Series("ev_option_name", drawn),
        )
        # Attach capacity / efficiency / body_class / range_miles from Autonomie.
        return assigned.join(self.autonomie_params, on="ev_option_name", how="left").select(
            "bldg_id",
            "vehicle_id",
            "ev_option_name",
            "body_class",
            "range_miles",
            "battery_capacity_kwh",
            "kwh_per_mile",
        )
