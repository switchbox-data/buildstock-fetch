import argparse
import logging
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass, fields
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final, Literal, cast

import numpy as np
import polars as pl
import yaml

# Repo root on sys.path when run as `python utils/EVs/ev_demand.py`
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from utils.EVs import ev_utils
from utils.EVs.ChargingSimulator import DEFAULT_LEVEL2_CHARGER_KW, ChargingSimulator
from utils.EVs.EVAdoptionSampler import EVAdoptionSampler
from utils.EVs.EVBatteryAssigner import DEFAULT_CAPACITY_BUFFER_FRACTION, EVBatteryAssigner
from utils.EVs.EVChargerAssigner import (
    EVChargerAssigner,
    RESSTOCK_LEVEL1_CHARGER_KW,
    RESSTOCK_LEVEL2_CHARGER_KW,
)
from utils.EVs.NHTSProfileSampler import NHTSProfileSampler, VehicleProfile
from utils.EVs.TripScheduleGenerator import (
    DEFAULT_MAX_ARRIVAL_HOUR,
    DEFAULT_MAX_DEPARTURE_HOUR,
    DEFAULT_MILES_NOISE_STD_FRACTION,
    DEFAULT_MIN_TRIP_AWAY_HOURS,
    DEFAULT_TIME_OFFSET_PROBABILITIES,
    DEFAULT_TIME_OFFSETS,
    DEFAULT_TRAVEL_DAY_START_HOUR,
    TripScheduleGenerator,
)
from utils.EVs.VehicleOwnershipModel import VehicleOwnershipModel
from utils.EVs.charging import (
    ChargingStrategy,
    DEFAULT_PEAK_CLOCK_HOURS,
    DEFAULT_SHED_LOAD_PENALTY_USD_PER_KWH,
    DEFAULT_SOC_MIN_FRACTION,
    DEFAULT_SOC_SAFETY_BUFFER_FRACTION,
    build_hours_base,
)

# How EVs (or vehicle slots treated as EVs) are assigned to ResStock buildings.
# - pums_vehicles: PUMS multinomial vehicle-count model; treat all as EVs; match NHTS on vehicles
# - resstock_adoption: ResStock P(EV) Bernoulli sampler; at most one EV; do not match NHTS on vehicles
EvAssignmentMode = Literal["pums_vehicles", "resstock_adoption"]
EV_ASSIGNMENT_MODES: Final[frozenset[str]] = frozenset({"pums_vehicles", "resstock_adoption"})

# How home charger power is assigned to each EV slot.
# - fixed: every vehicle uses charging.charger_power_kw (legacy / scenario override)
# - resstock: L1 vs L2 from Electric_Vehicle_Charger.tsv; powers via
#   level1_charger_power_kw / level2_charger_power_kw (default ResStock 1.6 / 5.69)
ChargerAssignmentMode = Literal["fixed", "resstock"]
CHARGER_ASSIGNMENT_MODES: Final[frozenset[str]] = frozenset({"fixed", "resstock"})

# Outdoor-temp adjustment applied to discharge kWh (ResStock / OpenStudio-HPXML curve).
TemperatureAdjustmentMode = Literal["none", "resstock"]
TEMPERATURE_ADJUSTMENT_MODES: Final[frozenset[str]] = frozenset({"none", "resstock"})

# Package directory (utils/EVs); data lives under ev_data/
EVS_DIR: Final[Path] = Path(__file__).resolve().parent
EV_DATA_DIR: Final[Path] = EVS_DIR / "ev_data"
CONFIGS_DIR: Final[Path] = EVS_DIR / "configs"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class InvalidDateFormatError(ValueError):
    """Raised when a config datetime string is not a valid ISO datetime with clock hour."""

    def __init__(self, date_str: str):
        super().__init__(
            f"Invalid datetime format: {date_str!r}. "
            "Use an ISO datetime with clock hour, e.g. 2024-01-01T04:00:00 "
            "(date-only values are not allowed)."
        )


# Simulation windows must align to NHTS travel days: start at 4am, end at 3am
# (last included hour slot covering 03:00–03:59).
REQUIRED_SIM_START_HOUR = DEFAULT_TRAVEL_DAY_START_HOUR  # 4
REQUIRED_SIM_END_HOUR = (DEFAULT_TRAVEL_DAY_START_HOUR - 1) % 24  # 3


@dataclass
class EVDemandConfig:
    """All parameters for an EV demand run (typically loaded from YAML)."""

    state: str
    release: str
    start_date: datetime | None = None
    end_date: datetime | None = None

    metadata_path: str | None = None
    # Required when ev_assignment=pums_vehicles; ignored for resstock_adoption.
    pums_path: str | None = None
    nhts_path: str = str(EV_DATA_DIR / "inputs" / "NHTS_v2_1_trip_surveys.csv")
    # Required when ev_assignment=resstock_adoption; ignored for pums_vehicles.
    ev_ownership_path: str | None = None
    # ResStock national BEV class × range shares (housing-characteristic TSV).
    ev_battery_path: str = str(
        EV_DATA_DIR / "inputs" / "resstock_ev_reference" / "Electric_Vehicle_Battery.tsv"
    )
    # Autonomie usable capacity (kWh) + efficiency (kWh/mi) keyed by the same option names.
    ev_autonomie_path: str = str(
        EV_DATA_DIR
        / "inputs"
        / "resstock_ev_reference"
        / "resstock_autonomie_2022_vehicle_params.csv"
    )
    # Required when charger_assignment=resstock; ignored for fixed.
    ev_charger_path: str | None = None
    output_dir: Path | None = None
    # ResStock OEDI weather CSVs named ``{station}.csv`` (required when
    # temperature_adjustment=resstock; ignored otherwise).
    weather_dir: str | None = None

    # Sampling / matching
    # Default: ResStock max-1-EV adoption (no NHTS vehicle-count matching).
    ev_assignment: EvAssignmentMode = "resstock_adoption"
    random_state: int = 42
    # Required when ev_assignment=pums_vehicles; ignored for resstock_adoption.
    max_vehicles: int | None = None
    # Applied in load_nhts_data: keep trip profiles whose daily miles fall in [low, high].
    # 0–100 = full pool; e.g. 0–95 keeps empties and drops the heaviest ~5% of survey days.
    nhts_daily_miles_percentile_low: float = 0.0
    nhts_daily_miles_percentile_high: float = 100.0

    # Trip schedule perturbation / packing
    min_trip_away_hours: int = DEFAULT_MIN_TRIP_AWAY_HOURS
    max_departure_hour: int = DEFAULT_MAX_DEPARTURE_HOUR
    max_arrival_hour: int = DEFAULT_MAX_ARRIVAL_HOUR
    time_offsets: tuple[int, ...] = DEFAULT_TIME_OFFSETS
    time_offset_probabilities: tuple[float, ...] = DEFAULT_TIME_OFFSET_PROBABILITIES
    miles_noise_std_fraction: float = DEFAULT_MILES_NOISE_STD_FRACTION

    # Battery assignment: pack must cover peak daily discharge duty × (1 + buffer).
    # With temperature_adjustment=resstock, duty miles are Σ(miles_share * power_mult(T)).
    capacity_buffer_fraction: float = DEFAULT_CAPACITY_BUFFER_FRACTION

    # Discharge temperature dependence (ResStock curve on Autonomie kWh/mi).
    # none: miles * kwh_per_mile; resstock: × power_mult(T_outdoor) using weather_dir CSVs.
    # Also used for battery sizing duty miles when enabled.
    temperature_adjustment: TemperatureAdjustmentMode = "none"

    # Pipeline
    max_workers: int | None = 8
    batch_size: int = 20000
    upload_s3: bool = False

    # Charging — strategy-specific knobs are required only for the strategy that uses them:
    #   immediate / off_peak / …: see _validate_charging_strategy_inputs
    # Charger power:
    #   charger_assignment=fixed → charger_power_kw (defaults to 7.2 if omitted);
    #     level1/level2_charger_power_kw ignored
    #   charger_assignment=resstock → sample L1 vs L2 from TSV; powers from
    #     level1_charger_power_kw / level2_charger_power_kw (default ResStock 1.6 / 5.69);
    #     charger_power_kw ignored
    charging_strategy: ChargingStrategy = "immediate"
    # fixed: use charger_power_kw for every EV; resstock: sample L1/L2 from TSV.
    charger_assignment: ChargerAssignmentMode = "fixed"
    # Global charger kW when charger_assignment=fixed; None when resstock (per-vehicle).
    charger_power_kw: float | None = None
    # Per-level rated powers when charger_assignment=resstock (None → ResStock defaults).
    level1_charger_power_kw: float | None = None
    level2_charger_power_kw: float | None = None
    # None = each vehicle starts at full battery capacity.
    initial_soc_kwh: float | None = None
    # off_peak only
    soc_min_fraction: float | None = None
    soc_safety_buffer_fraction: float | None = None
    # off_peak and off_peak_immediate
    peak_clock_hours: tuple[int, ...] | None = None
    # off_peak_immediate only (default False = pure TOU Immediate)
    allow_emergency_peak_charging: bool = False
    # cost_minimizing only
    shed_load_penalty_usd_per_kwh: float | None = None
    # Prices for cost_minimizing (exactly one of these):
    # - hourly_price_path: CSV/parquet/npy with one value per simulation hour
    # - flat_price_usd_per_kwh: constant $/kWh broadcast to all hours
    # - daily_price_usd_per_kwh: length-24 profile indexed by clock hour (element h = hour h)
    hourly_price_path: str | None = None
    flat_price_usd_per_kwh: float | None = None
    daily_price_usd_per_kwh: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.start_date is not None and self.end_date is not None:
            validate_travel_day_simulation_window(self.start_date, self.end_date)

        if self.ev_assignment not in EV_ASSIGNMENT_MODES:
            raise ValueError(
                f"ev_assignment must be one of {sorted(EV_ASSIGNMENT_MODES)}; "
                f"got {self.ev_assignment!r}"
            )
        if self.charger_assignment not in CHARGER_ASSIGNMENT_MODES:
            raise ValueError(
                f"charger_assignment must be one of {sorted(CHARGER_ASSIGNMENT_MODES)}; "
                f"got {self.charger_assignment!r}"
            )
        if self.temperature_adjustment not in TEMPERATURE_ADJUSTMENT_MODES:
            raise ValueError(
                f"temperature_adjustment must be one of {sorted(TEMPERATURE_ADJUSTMENT_MODES)}; "
                f"got {self.temperature_adjustment!r}"
            )

        if self.ev_assignment == "pums_vehicles":
            if self.max_vehicles is None:
                raise ValueError(
                    "sampling.max_vehicles is required when ev_assignment=pums_vehicles"
                )
            if self.max_vehicles < 1:
                raise ValueError(f"max_vehicles must be >= 1; got {self.max_vehicles}")
        elif self.max_vehicles is not None:
            logging.warning(
                "sampling.max_vehicles=%s is ignored when ev_assignment=resstock_adoption "
                "(at most one EV per household)",
                self.max_vehicles,
            )

        if self.metadata_path is None:
            self.metadata_path = str(
                EV_DATA_DIR / "inputs" / self.release / "metadata" / self.state / "metadata.parquet"
            )
        if self.output_dir is None:
            self.output_dir = EV_DATA_DIR / "outputs" / f"{self.state}_{self.release}"
        elif not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)

        # Mode-specific path defaults: only require files the active mode uses.
        if self.ev_assignment == "pums_vehicles":
            if self.pums_path is None:
                self.pums_path = str(
                    EV_DATA_DIR / "inputs" / f"{self.state}_2021_pums_PUMA_HINCP_VEH_NP.csv"
                )
            if self.ev_ownership_path is not None:
                logging.warning(
                    "paths.ev_ownership_path is ignored when ev_assignment=pums_vehicles"
                )
        else:
            if self.ev_ownership_path is None:
                self.ev_ownership_path = str(
                    EV_DATA_DIR
                    / "inputs"
                    / "resstock_ev_reference"
                    / "Electric_Vehicle_Ownership.tsv"
                )
            if self.pums_path is not None:
                logging.warning(
                    "paths.pums_path is ignored when ev_assignment=resstock_adoption"
                )

        # Charger path: only required for resstock L1/L2 sampling.
        if self.charger_assignment == "resstock":
            if self.ev_charger_path is None:
                self.ev_charger_path = str(
                    EV_DATA_DIR
                    / "inputs"
                    / "resstock_ev_reference"
                    / "Electric_Vehicle_Charger.tsv"
                )
        elif self.ev_charger_path is not None:
            logging.warning(
                "paths.ev_charger_path is ignored when charger_assignment=fixed"
            )

        if self.temperature_adjustment == "resstock":
            if self.weather_dir is None:
                self.weather_dir = str(
                    EV_DATA_DIR / "inputs" / self.release / "weather" / self.state
                )
            if not self.weather_dir:
                raise ValueError(
                    "temperature_adjustment=resstock requires paths.weather_dir "
                    "(or the default under ev_data/inputs/<release>/weather/<state>)"
                )
        elif self.weather_dir is not None:
            logging.warning(
                "paths.weather_dir is ignored when temperature_adjustment=%s",
                self.temperature_adjustment,
            )

        if not (0.0 <= self.nhts_daily_miles_percentile_low <= self.nhts_daily_miles_percentile_high <= 100.0):
            raise ValueError(
                "nhts_daily_miles_percentile_low/high must satisfy "
                f"0 <= low <= high <= 100; got "
                f"[{self.nhts_daily_miles_percentile_low}, {self.nhts_daily_miles_percentile_high}]"
            )

        if self.capacity_buffer_fraction < 0:
            raise ValueError(
                f"capacity_buffer_fraction must be >= 0; got {self.capacity_buffer_fraction}"
            )

        # Charger power is mode-gated:
        #   fixed    → scalar charger_power_kw; ignore level1/level2 overrides
        #   resstock → sample L1/L2; apply level1/level2_charger_power_kw (ResStock defaults)
        if self.charger_assignment == "fixed":
            if self.charger_power_kw is None:
                # Preserve historical default for fixed-rate scenarios / unit tests.
                self.charger_power_kw = DEFAULT_LEVEL2_CHARGER_KW
            if self.charger_power_kw < 0:
                raise ValueError(
                    f"charger_power_kw must be >= 0; got {self.charger_power_kw}"
                )
            if self.level1_charger_power_kw is not None or self.level2_charger_power_kw is not None:
                logging.warning(
                    "level1_charger_power_kw / level2_charger_power_kw are ignored when "
                    "charger_assignment=fixed (use charger_power_kw for the global rate)"
                )
                self.level1_charger_power_kw = None
                self.level2_charger_power_kw = None
        else:
            if self.charger_power_kw is not None:
                logging.warning(
                    "charging.charger_power_kw=%s is ignored when charger_assignment=resstock "
                    "(use level1_charger_power_kw / level2_charger_power_kw instead)",
                    self.charger_power_kw,
                )
                self.charger_power_kw = None
            if self.level1_charger_power_kw is None:
                self.level1_charger_power_kw = RESSTOCK_LEVEL1_CHARGER_KW
            if self.level2_charger_power_kw is None:
                self.level2_charger_power_kw = RESSTOCK_LEVEL2_CHARGER_KW
            if self.level1_charger_power_kw < 0:
                raise ValueError(
                    f"level1_charger_power_kw must be >= 0; got {self.level1_charger_power_kw}"
                )
            if self.level2_charger_power_kw < 0:
                raise ValueError(
                    f"level2_charger_power_kw must be >= 0; got {self.level2_charger_power_kw}"
                )

        self._validate_charging_strategy_inputs()

    def _validate_charging_strategy_inputs(self) -> None:
        """Require strategy-specific knobs; warn when unused knobs are set."""
        strategy = self.charging_strategy
        valid = {"immediate", "off_peak", "off_peak_immediate", "cost_minimizing"}
        if strategy not in valid:
            raise ValueError(
                f"charging_strategy must be one of {sorted(valid)}; got {strategy!r}"
            )

        price_sources = sum(
            1
            for src in (
                self.hourly_price_path,
                self.flat_price_usd_per_kwh,
                self.daily_price_usd_per_kwh,
            )
            if src is not None
        )
        if price_sources > 1:
            raise ValueError(
                "Provide at most one of hourly_price_path, flat_price_usd_per_kwh, "
                "or daily_price_usd_per_kwh"
            )
        if self.daily_price_usd_per_kwh is not None and len(self.daily_price_usd_per_kwh) != 24:
            raise ValueError(
                f"daily_price_usd_per_kwh must have length 24; got {len(self.daily_price_usd_per_kwh)}"
            )

        soc_target_knobs = {
            "soc_min_fraction": self.soc_min_fraction,
            "soc_safety_buffer_fraction": self.soc_safety_buffer_fraction,
        }
        cost_min_knobs = {
            "shed_load_penalty_usd_per_kwh": self.shed_load_penalty_usd_per_kwh,
        }
        has_prices = price_sources > 0

        def _validate_peak_clock_hours() -> None:
            if self.peak_clock_hours is None:
                raise ValueError(
                    f"charging_strategy={strategy} requires peak_clock_hours"
                )
            if not self.peak_clock_hours:
                raise ValueError(
                    f"peak_clock_hours must be a non-empty list for {strategy}"
                )
            if any(h < 0 or h > 23 for h in self.peak_clock_hours):
                raise ValueError(
                    f"peak_clock_hours must be clock hours in 0–23; got {self.peak_clock_hours}"
                )

        if strategy == "off_peak":
            missing = [name for name, value in soc_target_knobs.items() if value is None]
            if self.peak_clock_hours is None:
                missing.append("peak_clock_hours")
            if missing:
                raise ValueError(
                    "charging_strategy=off_peak requires " + ", ".join(missing)
                )
            assert self.soc_min_fraction is not None
            assert self.soc_safety_buffer_fraction is not None
            _validate_peak_clock_hours()
            if not 0.0 <= self.soc_min_fraction <= 1.0:
                raise ValueError(
                    f"soc_min_fraction must be within [0, 1]; got {self.soc_min_fraction}"
                )
            if not 0.0 <= self.soc_safety_buffer_fraction <= 1.0:
                raise ValueError(
                    "soc_safety_buffer_fraction must be within [0, 1]; "
                    f"got {self.soc_safety_buffer_fraction}"
                )
            unused = [name for name, value in cost_min_knobs.items() if value is not None]
            if has_prices:
                unused.append("prices")
            if self.allow_emergency_peak_charging:
                unused.append("allow_emergency_peak_charging")
            if unused:
                logging.warning(
                    "charging_strategy=off_peak ignores %s "
                    "(off_peak uses peak_clock_hours, not $/kWh prices)",
                    ", ".join(unused),
                )

        elif strategy == "off_peak_immediate":
            _validate_peak_clock_hours()
            unused = [
                name
                for name, value in {**soc_target_knobs, **cost_min_knobs}.items()
                if value is not None
            ]
            if has_prices:
                unused.append("prices")
            if unused:
                logging.warning(
                    "charging_strategy=off_peak_immediate ignores %s "
                    "(fills to full on off-peak home hours; does not use SOC_req)",
                    ", ".join(unused),
                )

        elif strategy == "cost_minimizing":
            if not has_prices:
                raise ValueError(
                    "charging_strategy=cost_minimizing requires one of: "
                    "hourly_price_path, flat_price_usd_per_kwh, or daily_price_usd_per_kwh"
                )
            if self.shed_load_penalty_usd_per_kwh is None:
                raise ValueError(
                    "charging_strategy=cost_minimizing requires shed_load_penalty_usd_per_kwh"
                )
            if self.shed_load_penalty_usd_per_kwh < 0:
                raise ValueError(
                    "shed_load_penalty_usd_per_kwh must be >= 0; "
                    f"got {self.shed_load_penalty_usd_per_kwh}"
                )
            unused = [
                name
                for name, value in {
                    **soc_target_knobs,
                    "peak_clock_hours": self.peak_clock_hours,
                }.items()
                if value is not None
            ]
            if self.allow_emergency_peak_charging:
                unused.append("allow_emergency_peak_charging")
            if unused:
                logging.warning(
                    "charging_strategy=cost_minimizing ignores %s",
                    ", ".join(unused),
                )

        else:  # immediate
            unused = [
                name
                for name, value in {
                    **soc_target_knobs,
                    **cost_min_knobs,
                    "peak_clock_hours": self.peak_clock_hours,
                }.items()
                if value is not None
            ]
            if has_prices:
                unused.append("prices")
            if self.allow_emergency_peak_charging:
                unused.append("allow_emergency_peak_charging")
            if unused:
                logging.warning(
                    "charging_strategy=immediate ignores %s",
                    ", ".join(unused),
                )

    @property
    def match_on_vehicles(self) -> bool:
        """True only for the PUMS multi-vehicle assignment mode."""
        return self.ev_assignment == "pums_vehicles"

    def num_simulation_hours(self) -> int:
        """Inclusive hourly count from ``start_date`` through ``end_date`` (hour-aligned)."""
        if self.start_date is None or self.end_date is None:
            raise ValueError("start_date and end_date are required to compute simulation hours")
        start_hour = self.start_date.replace(minute=0, second=0, microsecond=0)
        end_hour = self.end_date.replace(minute=0, second=0, microsecond=0)
        if end_hour < start_hour:
            raise ValueError("end_date must be on or after start_date")
        return int((end_hour - start_hour).total_seconds() // 3600) + 1


def parse_date(date_str: str) -> datetime:
    """Parse an ISO datetime string that includes a clock hour.

    Date-only values (``YYYY-MM-DD``) are rejected. Accepted examples:
    ``2024-01-01T04:00:00``, ``2024-01-01 04:00``, ``2025-01-01T03:00:00``.

    Args:
        date_str: Datetime string to parse

    Returns:
        datetime: Parsed datetime

    Raises:
        InvalidDateFormatError: If the string is not a datetime with clock hour
    """
    raw = date_str.strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M",
    ):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    raise InvalidDateFormatError(date_str)


def _coerce_config_date(value: Any) -> datetime:
    """Accept ISO datetime strings or ``datetime`` values from YAML (time required)."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return parse_date(value)
    if isinstance(value, date):
        raise ValueError(
            f"Config dates must include a clock hour (got date-only {value!r}). "
            f"Use e.g. {value}T{REQUIRED_SIM_START_HOUR:02d}:00:00 for start or "
            f"…T{REQUIRED_SIM_END_HOUR:02d}:00:00 for end."
        )
    raise TypeError(f"Cannot parse config date from {type(value)}: {value!r}")


def validate_travel_day_simulation_window(start: datetime, end: datetime) -> None:
    """Require an NHTS-aligned window: start at 4am, end at 3am (hour slots).

    Minutes/seconds are allowed (e.g. ``03:59``) but only the clock hour is checked;
    the hourly calendar floors to ``:00`` of that hour.

    Args:
        start (datetime): Start of the simulation window
        end (datetime): End of the simulation window

    Raises:
        ValueError: If the start or end date is not a valid ISO datetime with clock hour
        ValueError: If the start date is not at 04:00
        ValueError: If the end date is not at 03:00
        ValueError: If the end date is before the start date
    """
    if end < start:
        raise ValueError(f"end_date {end} must be on or after start_date {start}")
    if start.hour != REQUIRED_SIM_START_HOUR:
        raise ValueError(
            f"start_date must be at {REQUIRED_SIM_START_HOUR:02d}:00 "
            f"(NHTS travel-day start); got {start.isoformat(sep=' ')}"
        )
    if end.hour != REQUIRED_SIM_END_HOUR:
        raise ValueError(
            f"end_date must be at {REQUIRED_SIM_END_HOUR:02d}:00 "
            f"(last hour of an NHTS travel day, covering "
            f"{REQUIRED_SIM_END_HOUR:02d}:00–{REQUIRED_SIM_END_HOUR:02d}:59); "
            f"got {end.isoformat(sep=' ')}"
        )


def _load_hourly_price_file(path: str | Path, *, num_hours: int) -> np.ndarray:
    """Load a length-``num_hours`` price series from CSV, parquet, or npy."""
    price_path = Path(path)
    if not price_path.exists():
        raise FileNotFoundError(f"hourly_price_path not found: {price_path}")

    suffix = price_path.suffix.lower()
    if suffix == ".npy":
        prices = np.asarray(np.load(price_path), dtype=np.float64).reshape(-1)
    elif suffix in {".csv", ".parquet"}:
        df = pl.read_csv(price_path) if suffix == ".csv" else pl.read_parquet(price_path)
        if "usd_per_kwh" in df.columns:
            prices = df["usd_per_kwh"].to_numpy().astype(np.float64)
        elif "price" in df.columns:
            prices = df["price"].to_numpy().astype(np.float64)
        elif df.width == 1:
            prices = df.to_series(0).to_numpy().astype(np.float64)
        else:
            raise ValueError(
                f"Price file {price_path} must have column 'usd_per_kwh' or 'price', "
                f"or a single column; got columns {df.columns}"
            )
    else:
        raise ValueError(
            f"Unsupported hourly_price_path suffix {suffix!r}; use .csv, .parquet, or .npy"
        )

    if len(prices) != num_hours:
        raise ValueError(
            f"hourly_price_path length {len(prices)} does not match simulation hours {num_hours}"
        )
    return prices


def resolve_hourly_prices(config: EVDemandConfig) -> np.ndarray | None:
    """Build the simulation-length hourly price array from config, or ``None`` if unset."""
    num_hours = config.num_simulation_hours()
    if config.hourly_price_path is not None:
        return _load_hourly_price_file(config.hourly_price_path, num_hours=num_hours)
    if config.flat_price_usd_per_kwh is not None:
        return np.full(num_hours, float(config.flat_price_usd_per_kwh), dtype=np.float64)
    if config.daily_price_usd_per_kwh is not None:
        # Index by clock hour, not simulation-hour offset: element h is the price for
        # clock hour h, so a profile stays aligned to midnight even though the
        # simulation window starts at 04:00.
        assert config.start_date is not None  # guaranteed by num_simulation_hours() above
        daily = np.asarray(config.daily_price_usd_per_kwh, dtype=np.float64)
        clock_hours = (np.arange(num_hours) + config.start_date.hour) % 24
        return daily[clock_hours]
    return None


def load_ev_demand_config(path: str | Path) -> EVDemandConfig:
    """Load an ``EVDemandConfig`` from a YAML scenario file.

    Nested sections ``paths``, ``sampling``, ``trips``, ``battery``, ``temperature``,
    ``pipeline``, and ``charging`` are flattened into dataclass fields. Dates must be
    ISO datetimes with clock hour (e.g. ``2024-01-01T04:00:00``); date-only values are
    rejected. ``start_date`` must be at 04:00 and ``end_date`` at 03:00 (NHTS travel day).
    """
    config_path = Path(path)
    with config_path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    flat: dict[str, Any] = {}
    for key in ("state", "release", "start_date", "end_date"):
        if key in raw:
            flat[key] = raw[key]

    for section in ("paths", "sampling", "trips", "battery", "temperature", "pipeline", "charging"):
        section_data = raw.get(section) or {}
        if not isinstance(section_data, dict):
            raise ValueError(f"YAML section '{section}' must be a mapping, got {type(section_data)}")
        flat.update(section_data)

    if "start_date" in flat and flat["start_date"] is not None:
        flat["start_date"] = _coerce_config_date(flat["start_date"])
    if "end_date" in flat and flat["end_date"] is not None:
        flat["end_date"] = _coerce_config_date(flat["end_date"])
    if "peak_clock_hours" in flat:
        flat["peak_clock_hours"] = tuple(int(h) for h in flat["peak_clock_hours"])
    if "time_offsets" in flat:
        flat["time_offsets"] = tuple(int(x) for x in flat["time_offsets"])
    if "time_offset_probabilities" in flat:
        flat["time_offset_probabilities"] = tuple(float(x) for x in flat["time_offset_probabilities"])
    if "daily_price_usd_per_kwh" in flat and flat["daily_price_usd_per_kwh"] is not None:
        flat["daily_price_usd_per_kwh"] = tuple(float(x) for x in flat["daily_price_usd_per_kwh"])
    if "allow_emergency_peak_charging" in flat and flat["allow_emergency_peak_charging"] is not None:
        flat["allow_emergency_peak_charging"] = bool(flat["allow_emergency_peak_charging"])

    # Coerce numeric scalars (YAML may leave scientific notation as strings).
    for key in (
        "charger_power_kw",
        "level1_charger_power_kw",
        "level2_charger_power_kw",
        "soc_min_fraction",
        "soc_safety_buffer_fraction",
        "shed_load_penalty_usd_per_kwh",
        "flat_price_usd_per_kwh",
        "initial_soc_kwh",
        "capacity_buffer_fraction",
        "miles_noise_std_fraction",
        "nhts_daily_miles_percentile_low",
        "nhts_daily_miles_percentile_high",
    ):
        if key in flat and flat[key] is not None:
            flat[key] = float(flat[key])
    for key in (
        "min_trip_away_hours",
        "max_departure_hour",
        "max_arrival_hour",
        "random_state",
        "batch_size",
        "max_vehicles",
        "max_workers",
    ):
        if key in flat and flat[key] is not None:
            flat[key] = int(flat[key])

    known = {f.name for f in fields(EVDemandConfig)}
    # match_on_vehicles is derived from ev_assignment; reject legacy YAML keys with a clear error.
    if "match_on_vehicles" in flat:
        raise ValueError(
            f"Unknown EV demand config key 'match_on_vehicles' in {config_path}. "
            "NHTS vehicle-count matching is controlled by sampling.ev_assignment: "
            "pums_vehicles matches on vehicles; resstock_adoption does not."
        )
    unknown = set(flat) - known
    if unknown:
        raise ValueError(f"Unknown EV demand config keys in {config_path}: {sorted(unknown)}")

    required = {"state", "release", "start_date", "end_date"}
    missing = required - set(flat)
    if missing:
        raise ValueError(f"EV demand config {config_path} missing required keys: {sorted(missing)}")

    return EVDemandConfig(**flat)


class EVDemandCalculator:
    """
    Orchestrator for the EV demand pipeline.

    Constructs ``VehicleOwnershipModel``, ``EVAdoptionSampler``, ``EVBatteryAssigner``,
    ``EVChargerAssigner``, ``NHTSProfileSampler``, ``TripScheduleGenerator``, and
    ``ChargingSimulator``.

    Public API:
    - ``match_and_generate_trip_schedules()`` — EV assignment → NHTS profiles →
      daily trip schedules → hourly duty miles → ResStock battery + charger attrs
    - ``generate_soc_schedules()`` — hourly SOC / charge / discharge from trips
      (reuses duty miles when passed)
    """

    def __init__(
        self,
        metadata_df: pl.DataFrame,
        nhts_df: pl.DataFrame,
        ev_battery_df: pl.DataFrame,
        ev_autonomie_df: pl.DataFrame,
        start_date: datetime,
        end_date: datetime,
        ev_ownership_df: pl.DataFrame | None = None,
        pums_df: pl.DataFrame | None = None,
        ev_charger_df: pl.DataFrame | None = None,
        *,
        ev_assignment: EvAssignmentMode = "resstock_adoption",
        max_vehicles: int | None = None,
        vehicle_ownership: VehicleOwnershipModel | None = None,
        random_state: int = 42,
        max_workers: int | None = None,
        min_trip_away_hours: int = DEFAULT_MIN_TRIP_AWAY_HOURS,
        max_departure_hour: int = DEFAULT_MAX_DEPARTURE_HOUR,
        max_arrival_hour: int = DEFAULT_MAX_ARRIVAL_HOUR,
        time_offsets: tuple[int, ...] = DEFAULT_TIME_OFFSETS,
        time_offset_probabilities: tuple[float, ...] = DEFAULT_TIME_OFFSET_PROBABILITIES,
        miles_noise_std_fraction: float = DEFAULT_MILES_NOISE_STD_FRACTION,
        capacity_buffer_fraction: float = DEFAULT_CAPACITY_BUFFER_FRACTION,
        charger_assignment: ChargerAssignmentMode = "fixed",
        charger_power_kw: float | None = None,
        level1_charger_power_kw: float | None = None,
        level2_charger_power_kw: float | None = None,
        temperature_adjustment: TemperatureAdjustmentMode = "none",
        weather_dir: str | Path | None = None,
        weather_map: pl.DataFrame | None = None,
        station_temps: dict[str, pl.DataFrame] | None = None,
        state: str | None = None,
        release: str | None = None,
    ):
        """
        Initialize the EV demand calculator and its pipeline components.

        Args:
            metadata_df: ResStock metadata DataFrame
            nhts_df: NHTS trip data DataFrame
            ev_battery_df: ResStock EV battery option shares (from load_ev_battery_lookup)
            ev_autonomie_df: Autonomie capacity / efficiency params (from load_ev_autonomie_params)
            start_date: Start date for trip generation
            end_date: End date for trip generation
            ev_ownership_df: NREL EV ownership lookup (required for ``resstock_adoption``)
            pums_df: PUMS data DataFrame (required when ``ev_assignment=pums_vehicles``
                unless a fitted ``vehicle_ownership`` is passed)
            ev_charger_df: ResStock charger L1/L2 lookup (required for ``charger_assignment=resstock``)
            ev_assignment: How to assign EV slots — ``pums_vehicles`` or ``resstock_adoption``
            max_vehicles: Cap for the PUMS vehicle-count model (required for ``pums_vehicles``)
            vehicle_ownership: Optional pre-fitted PUMS model (avoids refitting each batch)
            random_state: Random seed for reproducible results
            max_workers: Maximum number of worker threads for parallel execution (None = use all cores)
            charger_assignment: ``fixed`` (global ``charger_power_kw``) or ``resstock`` (L1/L2 TSV)
            charger_power_kw: Global charger kW when ``charger_assignment=fixed``; unused for
                ``resstock`` (defaults to 7.2 when fixed and omitted)
            level1_charger_power_kw: Level 1 kW when ``charger_assignment=resstock``
                (default ResStock 1.6); unused for ``fixed``
            level2_charger_power_kw: Level 2 kW when ``charger_assignment=resstock``
                (default ResStock 5.69); unused for ``fixed``
            temperature_adjustment: When ``resstock``, battery sizing uses temp-scaled duty miles
            weather_dir: ResStock weather CSV directory (required for ``resstock`` temp adj)
            weather_map: Building → weather station map from ``load_all_input_data``
            station_temps: Optional mutable station temp cache shared across batches
            state: State code for weather lookup (required for ``resstock`` temp adj)
            release: ResStock release key for weather lookup (required for ``resstock`` temp adj)
        """
        if ev_assignment not in EV_ASSIGNMENT_MODES:
            raise ValueError(
                f"ev_assignment must be one of {sorted(EV_ASSIGNMENT_MODES)}; got {ev_assignment!r}"
            )
        if charger_assignment not in CHARGER_ASSIGNMENT_MODES:
            raise ValueError(
                f"charger_assignment must be one of {sorted(CHARGER_ASSIGNMENT_MODES)}; "
                f"got {charger_assignment!r}"
            )
        if ev_assignment == "pums_vehicles":
            if max_vehicles is None:
                raise ValueError("max_vehicles is required when ev_assignment=pums_vehicles")
            if vehicle_ownership is None and pums_df is None:
                raise ValueError(
                    "pums_df or a fitted vehicle_ownership model is required when "
                    "ev_assignment=pums_vehicles"
                )
        elif ev_ownership_df is None:
            raise ValueError(
                "ev_ownership_df is required when ev_assignment=resstock_adoption"
            )
        if charger_assignment == "resstock" and ev_charger_df is None:
            raise ValueError(
                "ev_charger_df is required when charger_assignment=resstock"
            )
        # Resolve fixed-rate power once; resstock leaves this None (per-vehicle attrs).
        if charger_assignment == "fixed":
            if charger_power_kw is None:
                charger_power_kw = DEFAULT_LEVEL2_CHARGER_KW
            if charger_power_kw < 0:
                raise ValueError(f"charger_power_kw must be non-negative, got {charger_power_kw}")
            if level1_charger_power_kw is not None or level2_charger_power_kw is not None:
                logging.warning(
                    "level1_charger_power_kw / level2_charger_power_kw are ignored when "
                    "charger_assignment=fixed"
                )
                level1_charger_power_kw = None
                level2_charger_power_kw = None
        else:
            if charger_power_kw is not None:
                logging.warning(
                    "charger_power_kw=%s is ignored when charger_assignment=resstock",
                    charger_power_kw,
                )
                charger_power_kw = None
            if level1_charger_power_kw is None:
                level1_charger_power_kw = RESSTOCK_LEVEL1_CHARGER_KW
            if level2_charger_power_kw is None:
                level2_charger_power_kw = RESSTOCK_LEVEL2_CHARGER_KW
            if level1_charger_power_kw < 0:
                raise ValueError(
                    f"level1_charger_power_kw must be non-negative, got {level1_charger_power_kw}"
                )
            if level2_charger_power_kw < 0:
                raise ValueError(
                    f"level2_charger_power_kw must be non-negative, got {level2_charger_power_kw}"
                )
        match_on_vehicles = ev_assignment == "pums_vehicles"
        # Cap used only for NHTS household-fleet bucketing (tier-1 keys). Adoption mode
        # does not match on vehicles, so the default of 2 is unused for matching but
        # keeps the cache consistent with historical multi-vehicle NHTS data.
        nhts_max_vehicles = max_vehicles if max_vehicles is not None else 2

        np.random.seed(random_state)

        self.metadata_df = metadata_df
        self.nhts_df = nhts_df
        self.pums_df = pums_df
        self.ev_ownership_df = ev_ownership_df
        self.ev_battery_df = ev_battery_df
        self.ev_autonomie_df = ev_autonomie_df
        self.ev_charger_df = ev_charger_df
        self.start_date = start_date
        self.end_date = end_date
        self.ev_assignment = ev_assignment
        self.max_vehicles = max_vehicles
        self.match_on_vehicles = match_on_vehicles
        self.random_state = random_state
        self.max_workers = max_workers
        self.charger_assignment = charger_assignment
        self.charger_power_kw = charger_power_kw
        self.level1_charger_power_kw = level1_charger_power_kw
        self.level2_charger_power_kw = level2_charger_power_kw

        # Pipeline components.
        if vehicle_ownership is not None:
            self.vehicle_ownership = vehicle_ownership
        else:
            self.vehicle_ownership = VehicleOwnershipModel(
                max_vehicles=nhts_max_vehicles,
                random_state=random_state,
            )
            if ev_assignment == "pums_vehicles":
                assert pums_df is not None  # validated above
                logging.info(
                    "Fitting PUMS vehicle-ownership model (max_vehicles=%s)",
                    max_vehicles,
                )
                self.vehicle_ownership.fit(pums_df)

        self.ev_adoption_sampler: EVAdoptionSampler | None = None
        if ev_assignment == "resstock_adoption":
            assert ev_ownership_df is not None  # validated above
            self.ev_adoption_sampler = EVAdoptionSampler(
                ev_ownership_df=ev_ownership_df,
                random_state=random_state,
            )
        self.battery_assigner = EVBatteryAssigner(
            option_probabilities=ev_battery_df,
            autonomie_params=ev_autonomie_df,
            random_state=random_state,
        )
        # Optional ResStock L1/L2 sampler; None when charger_assignment=fixed.
        self.charger_assigner: EVChargerAssigner | None = None
        if charger_assignment == "resstock":
            assert ev_charger_df is not None  # validated above
            assert level1_charger_power_kw is not None  # resolved above
            assert level2_charger_power_kw is not None
            self.charger_assigner = EVChargerAssigner(
                charger_lookup=ev_charger_df,
                random_state=random_state,
                level1_power_kw=level1_charger_power_kw,
                level2_power_kw=level2_charger_power_kw,
            )
        self.nhts_sampler = NHTSProfileSampler(
            nhts_df=nhts_df,
            max_vehicles=nhts_max_vehicles,
            match_on_vehicles=match_on_vehicles,
            random_state=random_state,
        )
        self.trip_schedule_generator = TripScheduleGenerator(
            start_date=start_date,
            end_date=end_date,
            random_state=random_state,
            max_workers=max_workers,
            min_trip_away_hours=min_trip_away_hours,
            max_departure_hour=max_departure_hour,
            max_arrival_hour=max_arrival_hour,
            time_offsets=time_offsets,
            time_offset_probabilities=time_offset_probabilities,
            miles_noise_std_fraction=miles_noise_std_fraction,
        )
        self.charging_simulator = ChargingSimulator(
            start_date=start_date,
            end_date=end_date,
        )
        self.capacity_buffer_fraction = capacity_buffer_fraction
        # Weather / temp-adj fields used when sizing packs against temp-scaled discharge.
        self.temperature_adjustment = temperature_adjustment
        self.weather_dir = weather_dir
        self.weather_map = weather_map
        self.station_temps = station_temps
        self.state = state
        self.release = release

    @classmethod
    def from_config(
        cls,
        config: EVDemandConfig,
        *,
        metadata_df: pl.DataFrame,
        nhts_df: pl.DataFrame,
        ev_battery_df: pl.DataFrame,
        ev_autonomie_df: pl.DataFrame,
        ev_ownership_df: pl.DataFrame | None = None,
        pums_df: pl.DataFrame | None = None,
        ev_charger_df: pl.DataFrame | None = None,
        vehicle_ownership: VehicleOwnershipModel | None = None,
        weather_map: pl.DataFrame | None = None,
        station_temps: dict[str, pl.DataFrame] | None = None,
    ) -> "EVDemandCalculator":
        """Build a calculator from an ``EVDemandConfig`` plus loaded input tables."""
        if config.start_date is None or config.end_date is None:
            raise ValueError("EVDemandConfig.start_date and end_date are required")
        return cls(
            metadata_df=metadata_df,
            nhts_df=nhts_df,
            ev_battery_df=ev_battery_df,
            ev_autonomie_df=ev_autonomie_df,
            start_date=config.start_date,
            end_date=config.end_date,
            ev_ownership_df=ev_ownership_df,
            pums_df=pums_df,
            ev_charger_df=ev_charger_df,
            ev_assignment=config.ev_assignment,
            max_vehicles=config.max_vehicles,
            vehicle_ownership=vehicle_ownership,
            random_state=config.random_state,
            max_workers=config.max_workers,
            min_trip_away_hours=config.min_trip_away_hours,
            max_departure_hour=config.max_departure_hour,
            max_arrival_hour=config.max_arrival_hour,
            time_offsets=config.time_offsets,
            time_offset_probabilities=config.time_offset_probabilities,
            miles_noise_std_fraction=config.miles_noise_std_fraction,
            capacity_buffer_fraction=config.capacity_buffer_fraction,
            charger_assignment=config.charger_assignment,
            charger_power_kw=config.charger_power_kw,
            level1_charger_power_kw=config.level1_charger_power_kw,
            level2_charger_power_kw=config.level2_charger_power_kw,
            temperature_adjustment=config.temperature_adjustment,
            weather_dir=config.weather_dir,
            weather_map=weather_map,
            station_temps=station_temps,
            state=config.state,
            release=config.release,
        )

    @staticmethod
    def _vehicle_slots_from_building_evs(bldg_veh_df: pl.DataFrame) -> pl.DataFrame:
        """Expand buildings with ``vehicles`` > 0 into one row per ``(bldg_id, vehicle_id)``.

        ``vehicle_id`` is 1-based within each building, matching NHTS / trip schedule slots.
        Under ``resstock_adoption``, ``vehicles`` is usually 0 or 1.

        Args:
            bldg_veh_df (pl.DataFrame): The building vehicle DataFrame

        Returns:
            pl.DataFrame: The vehicle slots DataFrame expanded from the building vehicle DataFrame
        """
        if "bldg_id" not in bldg_veh_df.columns or "vehicles" not in bldg_veh_df.columns:
            raise ValueError("bldg_veh_df must include bldg_id and vehicles columns")

        occupied = bldg_veh_df.filter(pl.col("vehicles") > 0)
        if occupied.is_empty():
            return pl.DataFrame(
                schema={
                    "bldg_id": bldg_veh_df.schema.get("bldg_id", pl.Int64),
                    "vehicle_id": pl.Int64,
                }
            )

        return (
            occupied.select("bldg_id", "vehicles")
            .with_columns(
                pl.int_ranges(1, pl.col("vehicles") + 1).alias("vehicle_id"),
            )
            .explode("vehicle_id")
            .select("bldg_id", pl.col("vehicle_id").cast(pl.Int64))
        )

    def match_and_generate_trip_schedules(
        self,
        *,
        hours_base: pl.DataFrame | None = None,
        hourly_temp_f_by_bldg: pl.DataFrame | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Generate trip schedules and assign batteries from a single duty-miles build.

        Pipeline:
        1. Assign EV slots → sample NHTS profiles → generate trip schedules
        2. Build hourly temp-scaled miles **once** (option-independent duty array)
        3. Size packs from peak daily duty miles × (1 + buffer)
        4. Return the hourly duty frame so SOC can do ``discharge = miles * kwh_per_mile``
           without recomputing temperature scaling

        Args:
            hours_base: Optional shared hourly calendar; built from start/end if omitted
            hourly_temp_f_by_bldg: Optional preloaded outdoor temps. When omitted and
                ``temperature_adjustment=resstock``, temps are loaded for EV buildings.

        Returns:
            ``(trip_schedules, ev_attributes, hourly_temp_scaled_miles)``
        """
        bldg_veh_df = self._assign_ev_slots()

        logging.info("Assigning vehicle profiles")
        vehicle_profiles = cast(
            dict[tuple[str, int], VehicleProfile],
            self.nhts_sampler.sample(bldg_veh_df),
        )

        logging.info("Generating trip schedules")
        trip_schedules = self.trip_schedule_generator.generate(vehicle_profiles)

        logging.info("Assigning ResStock EV battery attributes (stock-conditional)")
        vehicle_slots = self._vehicle_slots_from_building_evs(bldg_veh_df)

        # Always materialize a shared hours calendar so duty miles and SOC align.
        resolved_hours = hours_base
        if resolved_hours is None:
            resolved_hours = build_hours_base(self.start_date, self.end_date)

        # Resolve outdoor temps when ResStock temp adjustment is on.
        resolved_temps = hourly_temp_f_by_bldg
        if self.temperature_adjustment == "resstock" and resolved_temps is None:
            if vehicle_slots.height > 0:
                if self.weather_dir is None or self.state is None or self.release is None:
                    raise ValueError(
                        "temperature_adjustment=resstock requires weather_dir, state, and "
                        "release on EVDemandCalculator (pass via from_config / constructor)"
                    )
                logging.info(
                    "Loading outdoor temps for battery sizing (%s EV building(s))",
                    vehicle_slots["bldg_id"].n_unique(),
                )
                resolved_temps = ev_utils.load_hourly_temp_f_for_buildings(
                    vehicle_slots["bldg_id"],
                    hours_base=resolved_hours,
                    state=self.state,
                    release=self.release,
                    weather_dir=self.weather_dir,
                    weather_map=self.weather_map,
                    station_temps=self.station_temps,
                )

        # One expand + optional temp scale → reused for sizing and later discharge.
        hourly_temp_scaled_miles = ChargingSimulator.build_hourly_temp_scaled_miles(
            trip_schedules,
            resolved_hours,
            hourly_temp_f_by_bldg=resolved_temps,
        )
        max_miles = ChargingSimulator.max_daily_miles_from_hourly_temp_scaled(
            hourly_temp_scaled_miles
        )

        vehicle_duty = (
            vehicle_slots.join(max_miles, on=["bldg_id", "vehicle_id"], how="left")
            .with_columns(pl.col("max_daily_miles").fill_null(0.0))
        )
        # Feasible packs: capacity >= duty_miles * kwh_per_mile * (1 + buffer).
        ev_attributes = self.battery_assigner.assign(
            vehicle_duty,
            buffer_fraction=self.capacity_buffer_fraction,
        )
        logging.info(
            "Assigned battery attributes for %s EV vehicle slot(s) "
            "(duty miles %s)",
            ev_attributes.height,
            "temperature-scaled" if resolved_temps is not None else "unscaled",
        )

        # Attach per-vehicle charger level / kW onto the same attribute frame.
        ev_attributes = self._assign_chargers(ev_attributes)

        return trip_schedules, ev_attributes, hourly_temp_scaled_miles

    def _assign_chargers(self, ev_attributes: pl.DataFrame) -> pl.DataFrame:
        """Attach ``charger_level`` and ``charger_power_kw`` to each EV attribute row.

        - ``fixed``: broadcast ``self.charger_power_kw`` (legacy single-rate path).
        - ``resstock``: join metadata FPL / building type / tenure and sample L1 vs L2
          from ``Electric_Vehicle_Charger.tsv``; powers from ``level1/level2_charger_power_kw``.

        Args:
            ev_attributes: DataFrame with ``bldg_id`` and ``vehicle_id`` columns.

        Returns:
            DataFrame with ``charger_level`` and ``charger_power_kw`` columns.
        """
        if ev_attributes.is_empty():
            return ev_attributes.with_columns(
                pl.lit(None, dtype=pl.Utf8).alias("charger_level"),
                pl.lit(None, dtype=pl.Float64).alias("charger_power_kw"),
            )

        if self.charger_assignment == "fixed":
            # Scenario override: every EV uses the config scalar (e.g. 7.2 kW).
            assert self.charger_power_kw is not None  # validated in __init__ / config
            logging.info(
                "Assigning fixed charger power %.3f kW to %s EV(s)",
                self.charger_power_kw,
                ev_attributes.height,
            )
            return ev_attributes.with_columns(
                pl.lit("fixed").alias("charger_level"),
                pl.lit(self.charger_power_kw).alias("charger_power_kw"),
            )

        # ResStock path: need demographic join keys from building metadata.
        assert self.charger_assigner is not None
        meta_keys = {"bldg_id", "fpl", "building_type", "tenure"}
        missing_meta = meta_keys - set(self.metadata_df.columns)
        if missing_meta:
            raise ValueError(
                "metadata_df missing columns required for charger assignment: "
                + ", ".join(sorted(missing_meta))
            )

        vehicles_with_meta = ev_attributes.select("bldg_id", "vehicle_id").join(
            self.metadata_df.select("bldg_id", "fpl", "building_type", "tenure"),
            on="bldg_id",
            how="left",
        )
        # Fail if any EV building is missing demographics (should not happen post load_metadata).
        null_meta = vehicles_with_meta.filter(
            pl.col("fpl").is_null()
            | pl.col("building_type").is_null()
            | pl.col("tenure").is_null()
        )
        if null_meta.height > 0:
            sample_ids = null_meta.get_column("bldg_id").head(5).to_list()
            raise ValueError(
                f"Missing fpl/building_type/tenure for {null_meta.height} EV building(s) "
                f"(e.g. bldg_id={sample_ids})"
            )

        charger_attrs = self.charger_assigner.assign(vehicles_with_meta)
        logging.info(
            "Assigned ResStock L1/L2 chargers for %s EV(s): Level 1=%s, Level 2=%s",
            charger_attrs.height,
            (charger_attrs["charger_level"] == "Level 1").sum(),
            (charger_attrs["charger_level"] == "Level 2").sum(),
        )
        return ev_attributes.join(charger_attrs, on=["bldg_id", "vehicle_id"], how="left")

    def _assign_ev_slots(self) -> pl.DataFrame:
        """Assign a ``vehicles`` column (EV slot count) according to ``ev_assignment``."""
        if self.ev_assignment == "pums_vehicles":
            logging.info(
                "Predicting household vehicle counts from PUMS model "
                "(treating all vehicles as EVs; match_on_vehicles=True)"
            )
            return self.vehicle_ownership.predict(self.metadata_df)

        logging.info(
            "Sampling EV ownership from ResStock adoption rates "
            "(max 1 EV per household; match_on_vehicles=False)"
        )
        assert self.ev_adoption_sampler is not None  # validated in __init__
        bldg_ev_df = self.ev_adoption_sampler.sample(self.metadata_df)
        # NHTS sampler expects a ``vehicles`` column (count of EV slots to fill).
        return bldg_ev_df.with_columns(pl.col("evs").alias("vehicles"))

    def generate_soc_schedules(
        self,
        trip_schedules: pl.DataFrame,
        ev_attributes: pl.DataFrame,
        *,
        vehicle_keys: Iterable[tuple[str | int, int]] | None = None,
        hours_base: pl.DataFrame | None = None,
        presence_by_vehicle: dict[tuple[str | int, int], pl.DataFrame] | None = None,
        charger_power_kw: float | None = DEFAULT_LEVEL2_CHARGER_KW,
        initial_soc_kwh: float | None = None,
        charging_strategy: ChargingStrategy = "immediate",
        hourly_price_usd_per_kwh: np.ndarray | None = None,
        shed_load_penalty_usd_per_kwh: float | np.ndarray | None = None,
        peak_clock_hours: Iterable[int] = DEFAULT_PEAK_CLOCK_HOURS,
        soc_min_fraction: float = DEFAULT_SOC_MIN_FRACTION,
        soc_safety_buffer_fraction: float = DEFAULT_SOC_SAFETY_BUFFER_FRACTION,
        allow_emergency_peak_charging: bool = False,
        hourly_temp_f_by_bldg: pl.DataFrame | None = None,
        hourly_temp_scaled_miles: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Generate hourly SOC / charge / discharge schedules from trip schedules.

        Pass ``hourly_temp_scaled_miles`` from ``match_and_generate_trip_schedules`` to
        avoid recomputing the expand + temperature scale for discharge.

        ``charger_power_kw`` is a fallback when ``ev_attributes`` lacks a per-vehicle
        ``charger_power_kw`` column. Pipeline runs with ``charger_assignment=resstock``
        attach that column and may omit this argument (or pass ``None``).
        """
        if ev_attributes.is_empty():
            raise ValueError("ev_attributes must contain at least one vehicle row")
        # ev_attributes is the authoritative EV slot list; trip_schedules omits vehicles
        # matched to empty NHTS templates. Default to the slot list so a never-driven EV
        # still gets a full-year at-home, zero-discharge schedule instead of vanishing.
        if vehicle_keys is None:
            vehicle_keys = list(
                ev_attributes.select("bldg_id", "vehicle_id").unique().iter_rows()
            )
        return self.charging_simulator.generate_soc(
            trip_schedules,
            vehicle_keys=vehicle_keys,
            hours_base=hours_base,
            presence_by_vehicle=presence_by_vehicle,
            ev_attributes=ev_attributes,
            # None when attrs already carry per-vehicle charger_power_kw (resstock).
            charger_power_kw=charger_power_kw,
            initial_soc_kwh=initial_soc_kwh,
            charging_strategy=charging_strategy,
            hourly_price_usd_per_kwh=hourly_price_usd_per_kwh,
            shed_load_penalty_usd_per_kwh=shed_load_penalty_usd_per_kwh,
            peak_clock_hours=peak_clock_hours,
            soc_min_fraction=soc_min_fraction,
            soc_safety_buffer_fraction=soc_safety_buffer_fraction,
            allow_emergency_peak_charging=allow_emergency_peak_charging,
            hourly_temp_f_by_bldg=hourly_temp_f_by_bldg,
            hourly_temp_scaled_miles=hourly_temp_scaled_miles,
        )


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate EV demand trip schedules from ResStock metadata",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help=f"Path to YAML scenario config (see {CONFIGS_DIR}/)",
    )
    parser.add_argument(
        "--upload-s3",
        action="store_true",
        default=None,
        help="Upload results to S3 (overrides YAML pipeline.upload_s3)",
    )
    return parser.parse_args()


def upload_batch_to_s3(batch_trip_schedules, config, file_name, batch_number):
    """Upload a single batch of trip schedules to S3 with partitioning."""
    import io

    if len(batch_trip_schedules) == 0:
        return True

    # Add metadata columns
    batch_with_metadata = batch_trip_schedules.with_columns([
        pl.lit(config.release).alias("release"),
        pl.lit(config.state).alias("state"),
    ])

    # Partition the batch
    partitions = batch_with_metadata.partition_by(["release", "state"])

    upload_success = True
    for partition in partitions:
        # Get partition values for file naming
        release_val = partition["release"][0] if len(partition["release"]) > 0 else "unknown"
        state_val = partition["state"][0] if len(partition["state"]) > 0 else "unknown"

        # Create partitioned file name with batch number
        partition_file_name = f"{file_name}/release={release_val}/state={state_val}/batch_{batch_number:03d}.parquet"

        # Write partition to memory buffer
        buffer = io.BytesIO()
        partition.write_parquet(buffer)
        file_content = buffer.getvalue()

        # Upload partition to S3
        partition_upload_success = ev_utils.upload_object_to_s3(file_content, partition_file_name)
        if not partition_upload_success:
            print(
                f"Error: S3 upload failed for partition release={release_val}, state={state_val}, batch={batch_number}"
            )
            upload_success = False
            break

    return upload_success


def main():
    """Main function to run EV demand calculation from a YAML config."""
    args = parse_arguments()
    config = load_ev_demand_config(args.config)
    if args.upload_s3 is not None:
        config.upload_s3 = True

    if config.start_date is None or config.end_date is None:
        print("Error: start_date and end_date are required in the config")
        return 1
    if config.start_date >= config.end_date:
        print("Error: Start date must be before end date")
        return 1

    inputs = ev_utils.load_all_input_data(config)
    metadata_df = inputs.metadata_df
    nhts_df = inputs.nhts_df
    pums_df = inputs.pums_df
    ev_ownership_df = inputs.ev_ownership_df
    ev_battery_df = inputs.ev_battery_df
    ev_autonomie_df = inputs.ev_autonomie_df
    ev_charger_df = inputs.ev_charger_df
    weather_map = inputs.weather_map
    station_temps = inputs.station_temps

    print(f"Loaded metadata: {len(metadata_df)} rows")
    print(f"Loaded NHTS data: {len(nhts_df)} rows")
    if pums_df is not None:
        print(f"Loaded PUMS data: {len(pums_df)} rows")
    if ev_ownership_df is not None:
        print(f"Loaded EV ownership lookup: {ev_ownership_df.height:,} rows")
    print(f"Loaded EV battery options: {ev_battery_df.height:,} rows")
    print(f"Loaded Autonomie vehicle params: {ev_autonomie_df.height:,} rows")
    if ev_charger_df is not None:
        print(f"Loaded EV charger lookup (Yes-ownership rows): {ev_charger_df.height:,} rows")
    print(f"EV assignment mode: {config.ev_assignment}")
    print(f"Charger assignment mode: {config.charger_assignment}")
    if config.charger_assignment == "fixed":
        print(f"Fixed charger power: {config.charger_power_kw} kW")
    else:
        print(
            f"ResStock chargers: Level 1 = {config.level1_charger_power_kw} kW, "
            f"Level 2 = {config.level2_charger_power_kw} kW"
        )
    if config.ev_assignment == "pums_vehicles":
        print(f"PUMS max_vehicles: {config.max_vehicles} (NHTS match_on_vehicles=True)")
    else:
        print("ResStock adoption: max 1 EV/household (NHTS match_on_vehicles=False)")
        assert ev_ownership_df is not None
        state_ev_rate = ev_utils.state_ev_ownership_rate(ev_ownership_df, config.state)
        print(
            f"PUMS-weighted mean P(EV) over occupied lookup segments ({config.state}): {state_ev_rate:.4f}"
        )
    print(f"Temperature adjustment: {config.temperature_adjustment}")
    if config.temperature_adjustment == "resstock":
        print(f"Weather dir: {config.weather_dir}")
        if station_temps is not None:
            print(f"Preloaded weather stations: {len(station_temps)}")
    print(
        f"NHTS daily-miles percentile band: "
        f"[{config.nhts_daily_miles_percentile_low}, {config.nhts_daily_miles_percentile_high}]"
    )

    fitted_vehicle_ownership: VehicleOwnershipModel | None = None
    if config.ev_assignment == "pums_vehicles":
        assert config.max_vehicles is not None  # validated in EVDemandConfig
        assert pums_df is not None
        fitted_vehicle_ownership = VehicleOwnershipModel(
            max_vehicles=config.max_vehicles,
            random_state=config.random_state,
        )
        logging.info(
            "Fitting PUMS vehicle-ownership model once (max_vehicles=%s)",
            config.max_vehicles,
        )
        fitted_vehicle_ownership.fit(pums_df)

    hourly_prices = resolve_hourly_prices(config)
    if config.charging_strategy == "cost_minimizing":
        if hourly_prices is None:
            print("Error: cost_minimizing requires prices in the config")
            return 1
        print(f"Loaded hourly prices: {len(hourly_prices):,} hours")
    elif hourly_prices is not None:
        print(
            f"Note: prices loaded ({len(hourly_prices):,} hours) but ignored for "
            f"charging_strategy={config.charging_strategy}"
        )

    hours_base = build_hours_base(config.start_date, config.end_date)

    batch_size = config.batch_size
    total_rows = len(metadata_df)
    all_trip_schedules = []
    all_soc_schedules = []
    all_ev_attributes = []
    trip_file_name = "trip_schedules"
    soc_file_name = "vehicle_soc_schedules"
    attrs_file_name = "ev_attributes"

    for i in range(0, total_rows, batch_size):
        batch_end = min(i + batch_size, total_rows)
        batch_metadata = metadata_df[i:batch_end]
        batch_number = i // batch_size + 1

        print(f"Processing batch {batch_number}: rows {i + 1} to {batch_end} ({len(batch_metadata)} rows)")

        calculator = EVDemandCalculator.from_config(
            config,
            metadata_df=batch_metadata,
            nhts_df=nhts_df,
            ev_ownership_df=ev_ownership_df,
            ev_battery_df=ev_battery_df,
            ev_autonomie_df=ev_autonomie_df,
            ev_charger_df=ev_charger_df,
            vehicle_ownership=fitted_vehicle_ownership,
            weather_map=weather_map,
            station_temps=station_temps,
        )

        # Battery sizing builds hourly temp-scaled miles once; reuse for SOC discharge.
        batch_trip_schedules, batch_ev_attributes, batch_hourly_duty_miles = (
            calculator.match_and_generate_trip_schedules(hours_base=hours_base)
        )
        soc_kwargs: dict[str, Any] = {
            "charging_strategy": config.charging_strategy,
            "initial_soc_kwh": config.initial_soc_kwh,
            "hours_base": hours_base,
            "hourly_temp_scaled_miles": batch_hourly_duty_miles,
        }
        # Fixed mode: pass global kW. Resstock mode: omit — ev_attributes already has
        # per-vehicle charger_power_kw from EVChargerAssigner.
        if config.charger_power_kw is not None:
            soc_kwargs["charger_power_kw"] = config.charger_power_kw
        # Temps only needed if duty miles were not precomputed (legacy / ad-hoc callers).
        # With hourly_temp_scaled_miles set, generate_soc skips the second temp pass.
        if config.temperature_adjustment == "resstock" and batch_hourly_duty_miles.is_empty():
            if config.weather_dir is None:
                raise ValueError("weather_dir is required when temperature_adjustment=resstock")
            if batch_ev_attributes.height > 0:
                soc_kwargs["hourly_temp_f_by_bldg"] = ev_utils.load_hourly_temp_f_for_buildings(
                    batch_ev_attributes["bldg_id"],
                    hours_base=hours_base,
                    state=config.state,
                    release=config.release,
                    weather_dir=config.weather_dir,
                    weather_map=weather_map,
                    station_temps=station_temps,
                )
        if config.charging_strategy == "off_peak":
            soc_kwargs["peak_clock_hours"] = config.peak_clock_hours
            soc_kwargs["soc_min_fraction"] = config.soc_min_fraction
            soc_kwargs["soc_safety_buffer_fraction"] = config.soc_safety_buffer_fraction
        elif config.charging_strategy == "off_peak_immediate":
            soc_kwargs["peak_clock_hours"] = config.peak_clock_hours
            soc_kwargs["allow_emergency_peak_charging"] = config.allow_emergency_peak_charging
        elif config.charging_strategy == "cost_minimizing":
            soc_kwargs["hourly_price_usd_per_kwh"] = hourly_prices
            soc_kwargs["shed_load_penalty_usd_per_kwh"] = config.shed_load_penalty_usd_per_kwh
        batch_soc_schedules = calculator.generate_soc_schedules(
            batch_trip_schedules,
            batch_ev_attributes,
            **soc_kwargs,
        )

        print(f"Completed batch {batch_number}: generated {len(batch_trip_schedules)} trip schedules")
        print(
            f"Completed batch {batch_number}: assigned "
            f"{len(batch_ev_attributes)} EV battery attributes"
        )
        print(f"Completed batch {batch_number}: generated {len(batch_soc_schedules)} hourly SOC rows")

        if config.upload_s3:
            print(f"Uploading batch {batch_number} to S3...")
            trip_upload_success = upload_batch_to_s3(batch_trip_schedules, config, trip_file_name, batch_number)
            soc_upload_success = upload_batch_to_s3(batch_soc_schedules, config, soc_file_name, batch_number)
            attrs_upload_success = True
            if batch_ev_attributes.height > 0:
                attrs_upload_success = upload_batch_to_s3(
                    batch_ev_attributes, config, attrs_file_name, batch_number
                )

            if not trip_upload_success or not soc_upload_success or not attrs_upload_success:
                print(f"Error: S3 upload failed for batch {batch_number}")
                return 1

            print(f"Successfully uploaded batch {batch_number} to S3")
            del batch_trip_schedules
            del batch_soc_schedules
            del batch_ev_attributes
        else:
            all_trip_schedules.append(batch_trip_schedules)
            all_soc_schedules.append(batch_soc_schedules)
            if batch_ev_attributes.height > 0:
                all_ev_attributes.append(batch_ev_attributes)

    if config.upload_s3:
        print(
            f"All batches uploaded to S3 with partitioning: "
            f"{trip_file_name}/, {soc_file_name}/, and {attrs_file_name}/"
        )
        logging.info(
            f"Uploaded all batches to S3 with partitioning: "
            f"{trip_file_name}/, {soc_file_name}/, and {attrs_file_name}/"
        )
    else:
        print("Combining all batches...")
        if config.output_dir is None:
            raise ValueError("config.output_dir")
        os.makedirs(config.output_dir, exist_ok=True)

        if all_trip_schedules:
            combined_trip_schedules = pl.concat(all_trip_schedules)
            logging.info(f"Combined all batches: {len(combined_trip_schedules)} total trip schedules")

            final_trip_schedules = combined_trip_schedules.with_columns([
                pl.lit(config.release).alias("release"),
                pl.lit(config.state).alias("state"),
            ]).sort(["bldg_id", "vehicle_id", "travel_date"])

            local_trip_path = f"{config.output_dir}/{trip_file_name}"
            final_trip_schedules.write_parquet(local_trip_path, partition_by=["release", "state"])

            print(f"Trip schedules written to: {local_trip_path}")
            logging.info(f"Written trip schedules to {local_trip_path}")
        else:
            logging.warning("No trip schedules generated")

        if all_ev_attributes:
            combined_ev_attributes = pl.concat(all_ev_attributes)
            logging.info(f"Combined all batches: {len(combined_ev_attributes)} total EV attribute rows")

            final_ev_attributes = combined_ev_attributes.with_columns([
                pl.lit(config.release).alias("release"),
                pl.lit(config.state).alias("state"),
            ]).sort(["bldg_id", "vehicle_id"])

            local_attrs_path = f"{config.output_dir}/{attrs_file_name}"
            final_ev_attributes.write_parquet(local_attrs_path, partition_by=["release", "state"])

            print(f"EV attributes written to: {local_attrs_path}")
            logging.info(f"Written EV attributes to {local_attrs_path}")
        else:
            logging.warning("No EV attributes generated")
        if all_soc_schedules:
            combined_soc_schedules = pl.concat(all_soc_schedules)
            logging.info(f"Combined all batches: {len(combined_soc_schedules)} total hourly SOC rows")

            final_soc_schedules = combined_soc_schedules.with_columns([
                pl.lit(config.release).alias("release"),
                pl.lit(config.state).alias("state"),
            ]).sort(["bldg_id", "vehicle_id", "hour_index"])

            local_soc_path = f"{config.output_dir}/{soc_file_name}"
            final_soc_schedules.write_parquet(local_soc_path, partition_by=["release", "state"])

            print(f"Vehicle SOC schedules written to: {local_soc_path}")
            logging.info(f"Written vehicle SOC schedules to {local_soc_path}")
        else:
            logging.warning("No vehicle SOC schedules generated")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
