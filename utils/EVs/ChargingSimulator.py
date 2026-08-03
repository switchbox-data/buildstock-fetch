from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

from utils.EVs.charging import (
    ChargingStrategy,
    DEFAULT_PEAK_CLOCK_HOURS,
    DEFAULT_SOC_MIN_FRACTION,
    DEFAULT_SOC_SAFETY_BUFFER_FRACTION,
    build_hours_base,
    build_is_off_peak,
    build_off_peak_charging_params,
    compute_hourly_soc,
    expand_trip_away_hours,
    expand_trips_to_away_hour_rows,
    schedule_cost_minimizing_charging,
    schedule_immediate_charging,
    schedule_off_peak_charging,
    schedule_off_peak_immediate_charging,
    tours_from_trip_schedules,
)
from utils.EVs.ev_utils import resstock_temp_power_mult

DEFAULT_LEVEL2_CHARGER_KW = 7.2  # typical 32 A @ 240 V residential Level 2 charger


@dataclass
class ChargingSimulator:
    """Simulate vehicle presence and SOC/charging over an hourly calendar."""

    start_date: datetime
    end_date: datetime

    def _resolve_hours_base(self, hours_base: pl.DataFrame | None) -> pl.DataFrame:
        if hours_base is None:
            return build_hours_base(self.start_date, self.end_date)
        return hours_base

    def generate_presence(
        self,
        trip_schedules: pl.DataFrame,
        *,
        hours_base: pl.DataFrame | None = None,
        vehicle_keys: Iterable[tuple[str | int, int]] | None = None,
    ) -> dict[tuple[str | int, int], pl.DataFrame]:
        """
        Map each vehicle's trip schedule to an hourly schedule of home/away status for the instance date range.

        Presence uses **tour** away windows (leave-home → return-home), including mid-tour
        parking. Drive legs alone are not enough: a vehicle at work is away even when not
        discharging. Tour intervals are derived via ``tours_from_trip_schedules``.

        Args:
            trip_schedules (pl.DataFrame): DataFrame of trip schedules (with optional tour columns)
            hours_base (pl.DataFrame): hourly calendar for the instance date range
            vehicle_keys (Iterable[tuple[str | int, int]]): Iterable of vehicle keys to generate presence schedules for

        Returns:
            dict[tuple[str | int, int], pl.DataFrame]: Dict of vehicle keys to hourly presence schedules
        """
        hours_base = self._resolve_hours_base(hours_base)  # shared calendar with join keys for matching trip rows to hours

        if vehicle_keys is None:
            if trip_schedules.is_empty():
                return {}  # no vehicles requested and no trips to infer vehicle ids from
            vehicle_keys_df = trip_schedules.select("bldg_id", "vehicle_id").unique()  # infer vehicles from trips
        else:
            vehicle_keys_df = pl.DataFrame({
                "bldg_id": [key[0] for key in vehicle_keys],  # building id for each requested vehicle slot
                "vehicle_id": [key[1] for key in vehicle_keys],  # 1-based vehicle index within the building
            }).unique()  # caller may pass vehicle_profiles.keys(); dedupe just in case

        if trip_schedules.is_empty():
            # no trips means the vehicle never leaves home
            hourly_presence = (
                vehicle_keys_df.join(hours_base, how="cross")  # one hourly schedule per requested vehicle
                .with_columns(
                    pl.lit(True).alias("at_home"),  # default presence state without trip evidence
                    pl.lit(False).alias("away_from_home"),  # explicit complement of at_home
                )
                .select("bldg_id", "vehicle_id", "hour_index", "timestamp", "at_home", "away_from_home")
            )
        else:
            # Tours define away-from-home; mid-tour dwells stay away (no home charging).
            tour_schedules = tours_from_trip_schedules(trip_schedules)
            away_hours = expand_trip_away_hours(tour_schedules, prefix="tour")

            hourly_presence = (
                vehicle_keys_df.join(hours_base, how="cross")  # hourly rows x number of vehicles
                .join(
                    away_hours.with_columns(pl.lit(False).alias("at_home")),  # away rows carry at_home=False
                    on=["bldg_id", "vehicle_id", "date", "hour"],  # match a specific vehicle-hour to a trip hour
                    how="left",  # keep all hours; hours without trips remain null until filled below
                )
                .with_columns(
                    pl.col("at_home").fill_null(True).alias("at_home"),
                )
                .with_columns(
                    (~pl.col("at_home")).alias("away_from_home"),
                )
                .select("bldg_id", "vehicle_id", "hour_index", "timestamp", "at_home", "away_from_home")
            )

        presence_by_vehicle: dict[tuple[str | int, int], pl.DataFrame] = {}
        for vehicle_frame in hourly_presence.partition_by(["bldg_id", "vehicle_id"], as_dict=False):
            bldg_id = vehicle_frame["bldg_id"][0]
            vehicle_id = int(vehicle_frame["vehicle_id"][0])
            presence_by_vehicle[(bldg_id, vehicle_id)] = vehicle_frame.drop("bldg_id", "vehicle_id").sort(
                "hour_index"
            )

        return presence_by_vehicle

    @staticmethod
    def build_hourly_temp_scaled_miles(
        trip_schedules: pl.DataFrame,
        hours_base: pl.DataFrame,
        *,
        hourly_temp_f_by_bldg: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Spread trip miles onto drive hours, optionally × ResStock ``power_mult(T)``.

        This is the shared, option-independent duty array used for both battery
        sizing and SOC discharge:

            discharge_kwh[h] = temp_scaled_miles[h] * kwh_per_mile

        so we build temp-scaled miles **once**, assign packs from peak daily
        totals, then multiply by the drawn ``kwh_per_mile`` (no second temp pass).

        When ``hourly_temp_f_by_bldg`` is ``None``, ``power_mult ≡ 1`` (raw miles).

        Args:
            trip_schedules: Trip legs with miles and trip interval columns
            hours_base: Calendar from ``build_hours_base`` (``date``, ``hour``, ``hour_index``)
            hourly_temp_f_by_bldg: Optional ``bldg_id``, ``hour_index``, ``temp_f`` (°F)

        Returns:
            ``bldg_id``, ``vehicle_id``, ``hour_index``, ``travel_date``,
            ``temp_scaled_miles`` (one row per vehicle-hour that has driving)
        """
        # No trips → empty typed frame for downstream joins / peak-duty aggregation.
        if trip_schedules.is_empty():
            return pl.DataFrame(
                schema={
                    "bldg_id": trip_schedules.schema.get("bldg_id", pl.Int64),
                    "vehicle_id": pl.Int64,
                    "hour_index": pl.UInt32,
                    "travel_date": pl.Datetime(time_unit="us"),
                    "temp_scaled_miles": pl.Float64,
                }
            )

        required_trip = {
            "bldg_id",
            "vehicle_id",
            "travel_date",
            "trip_miles_driven",
            "trip_departure_date",
            "trip_departure_hour",
            "trip_arrival_date",
            "trip_arrival_hour",
        }
        missing_trip = required_trip - set(trip_schedules.columns)
        if missing_trip:
            raise ValueError(f"trip_schedules missing columns: {sorted(missing_trip)}")

        hours_required = {"date", "hour", "hour_index"}
        hours_missing = hours_required - set(hours_base.columns)
        if hours_missing:
            raise ValueError(f"hours_base missing columns: {sorted(hours_missing)}")

        # Tag each leg so mile shares stay tied to one trip_miles total after explode.
        trips = trip_schedules.with_row_index("_trip_idx")

        # Expand drive intervals → hour rows; split miles evenly across those hours.
        away_hour_miles = (
            expand_trips_to_away_hour_rows(trips, prefix="trip")
            .with_columns(
                (pl.col("trip_miles_driven") / pl.col("hour").count().over("_trip_idx")).alias(
                    "miles_share"
                )
            )
            .join(
                hours_base.select("date", "hour", "hour_index"),
                on=["date", "hour"],
                how="inner",
            )
        )

        # Optional outdoor-temp scale (same curve as SOC discharge).
        if hourly_temp_f_by_bldg is not None:
            temp_required = {"bldg_id", "hour_index", "temp_f"}
            temp_missing = temp_required - set(hourly_temp_f_by_bldg.columns)
            if temp_missing:
                raise ValueError(f"hourly_temp_f_by_bldg missing columns: {sorted(temp_missing)}")
            away_hour_miles = away_hour_miles.join(
                hourly_temp_f_by_bldg.select("bldg_id", "hour_index", "temp_f"),
                on=["bldg_id", "hour_index"],
                how="left",
            )
            if away_hour_miles["temp_f"].null_count() > 0:
                missing_temp = (
                    away_hour_miles.filter(pl.col("temp_f").is_null())
                    .select("bldg_id", "hour_index")
                    .unique()
                    .head(5)
                    .to_dicts()
                )
                raise ValueError(
                    "hourly_temp_f_by_bldg missing outdoor temp for trip hour(s); "
                    f"examples: {missing_temp}"
                )
            temps = away_hour_miles["temp_f"].to_numpy()
            power_mult = np.asarray(resstock_temp_power_mult(temps), dtype=np.float64)
            away_hour_miles = away_hour_miles.with_columns(
                pl.Series("power_mult", power_mult),
            ).with_columns(
                (pl.col("miles_share") * pl.col("power_mult")).alias("temp_scaled_miles_share")
            )
        else:
            # No weather → duty miles equal raw mile shares.
            away_hour_miles = away_hour_miles.with_columns(
                pl.col("miles_share").alias("temp_scaled_miles_share")
            )

        # Collapse overlapping legs on the same vehicle-hour and travel day.
        return (
            away_hour_miles.group_by("bldg_id", "vehicle_id", "hour_index", "travel_date")
            .agg(pl.col("temp_scaled_miles_share").sum().alias("temp_scaled_miles"))
        )

    @staticmethod
    def max_daily_miles_from_hourly_temp_scaled(
        hourly_temp_scaled_miles: pl.DataFrame,
    ) -> pl.DataFrame:
        """Peak daily temp-scaled miles from a precomputed hourly duty frame.

        Args:
            hourly_temp_scaled_miles: Output of ``build_hourly_temp_scaled_miles``

        Returns:
            One row per vehicle with ``bldg_id``, ``vehicle_id``, ``max_daily_miles``
        """
        if hourly_temp_scaled_miles.is_empty():
            return pl.DataFrame(
                schema={
                    "bldg_id": hourly_temp_scaled_miles.schema.get("bldg_id", pl.Int64),
                    "vehicle_id": pl.Int64,
                    "max_daily_miles": pl.Float64,
                }
            )

        required = {"bldg_id", "vehicle_id", "travel_date", "temp_scaled_miles"}
        missing = required - set(hourly_temp_scaled_miles.columns)
        if missing:
            raise ValueError(f"hourly_temp_scaled_miles missing columns: {sorted(missing)}")

        # Sum within each NHTS travel day, then take the peak day per vehicle.
        return (
            hourly_temp_scaled_miles.group_by("bldg_id", "vehicle_id", "travel_date")
            .agg(pl.col("temp_scaled_miles").sum().alias("daily_temp_scaled_miles"))
            .group_by("bldg_id", "vehicle_id")
            .agg(pl.col("daily_temp_scaled_miles").max().alias("max_daily_miles"))
        )

    @staticmethod
    def max_daily_temp_scaled_miles_from_trip_schedules(
        trip_schedules: pl.DataFrame,
        *,
        hours_base: pl.DataFrame,
        hourly_temp_f_by_bldg: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Convenience: build hourly duty miles, then take peak daily totals.

        Prefer calling ``build_hourly_temp_scaled_miles`` once and reusing that
        frame for both sizing and discharge when running the full pipeline.
        """
        hourly = ChargingSimulator.build_hourly_temp_scaled_miles(
            trip_schedules,
            hours_base,
            hourly_temp_f_by_bldg=hourly_temp_f_by_bldg,
        )
        return ChargingSimulator.max_daily_miles_from_hourly_temp_scaled(hourly)

    @staticmethod
    def _discharge_kwh_from_temp_scaled_miles(
        hourly_temp_scaled_miles: pl.DataFrame,
        kwh_per_mile_by_vehicle: pl.DataFrame,
    ) -> pl.DataFrame:
        """``discharge_kwh = temp_scaled_miles * kwh_per_mile`` (no temp recomputation).

        Args:
            hourly_temp_scaled_miles: From ``build_hourly_temp_scaled_miles``
            kwh_per_mile_by_vehicle: ``bldg_id``, ``vehicle_id``, ``kwh_per_mile``

        Returns:
            ``bldg_id``, ``vehicle_id``, ``hour_index``, ``discharge_kwh``
        """
        if hourly_temp_scaled_miles.is_empty():
            return pl.DataFrame(
                schema={
                    "bldg_id": pl.Int64,
                    "vehicle_id": pl.Int64,
                    "hour_index": pl.UInt32,
                    "discharge_kwh": pl.Float64,
                }
            )

        required = {"bldg_id", "vehicle_id", "kwh_per_mile"}
        missing = required - set(kwh_per_mile_by_vehicle.columns)
        if missing:
            raise ValueError(f"kwh_per_mile_by_vehicle missing columns: {sorted(missing)}")

        # Collapse travel_date if still present (same clock hour from one travel day only).
        hourly = hourly_temp_scaled_miles
        if "travel_date" in hourly.columns:
            hourly = (
                hourly.group_by("bldg_id", "vehicle_id", "hour_index")
                .agg(pl.col("temp_scaled_miles").sum())
            )

        joined = hourly.join(
            kwh_per_mile_by_vehicle.select("bldg_id", "vehicle_id", "kwh_per_mile"),
            on=["bldg_id", "vehicle_id"],
            how="left",
        )
        if joined["kwh_per_mile"].null_count() > 0:
            missing_keys = (
                joined.filter(pl.col("kwh_per_mile").is_null())
                .select("bldg_id", "vehicle_id")
                .unique()
                .to_dicts()
            )
            raise ValueError(
                "ev_attributes missing kwh_per_mile for trip vehicle(s): "
                + ", ".join(repr((row["bldg_id"], row["vehicle_id"])) for row in missing_keys)
            )

        return joined.select(
            "bldg_id",
            "vehicle_id",
            "hour_index",
            (pl.col("temp_scaled_miles") * pl.col("kwh_per_mile")).alias("discharge_kwh"),
        )

    @staticmethod
    def _build_hourly_discharge_kwh(
        trip_schedules: pl.DataFrame,
        hours_base: pl.DataFrame,
        *,
        kwh_per_mile_by_vehicle: pl.DataFrame,
        hourly_temp_f_by_bldg: pl.DataFrame | None = None,
        hourly_temp_scaled_miles: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Map trips to hourly discharge kWh for the instance date range.

        Prefers a precomputed ``hourly_temp_scaled_miles`` frame (from battery
        sizing) so outdoor-temp scaling is not repeated. Otherwise builds that
        duty array from ``trip_schedules`` then multiplies by ``kwh_per_mile``.

        Args:
            trip_schedules: Trip schedules (ignored when ``hourly_temp_scaled_miles`` set)
            hours_base: Hourly calendar
            kwh_per_mile_by_vehicle: Autonomie efficiency per vehicle
            hourly_temp_f_by_bldg: Outdoor temps when building duty miles from scratch
            hourly_temp_scaled_miles: Optional reusable output of ``build_hourly_temp_scaled_miles``

        Returns:
            ``bldg_id``, ``vehicle_id``, ``hour_index``, ``discharge_kwh``
        """
        # Reuse duty miles from sizing when available; otherwise build them here.
        if hourly_temp_scaled_miles is None:
            hourly_temp_scaled_miles = ChargingSimulator.build_hourly_temp_scaled_miles(
                trip_schedules,
                hours_base,
                hourly_temp_f_by_bldg=hourly_temp_f_by_bldg,
            )
        return ChargingSimulator._discharge_kwh_from_temp_scaled_miles(
            hourly_temp_scaled_miles,
            kwh_per_mile_by_vehicle,
        )

    @staticmethod
    def _battery_attrs_lookup(
        ev_attributes: pl.DataFrame,
        *,
        default_charger_power_kw: float | None,
    ) -> dict[tuple[str | int, int], tuple[float, float, float]]:
        """Map ``(bldg_id, vehicle_id)`` -> ``(battery_capacity_kwh, kwh_per_mile, charger_kw)``.

        Built once per ``generate_soc`` call so the per-vehicle loop can O(1) look up
        ResStock/Autonomie pack size, efficiency, and charger power.

        When ``ev_attributes`` includes ``charger_power_kw`` (from ``EVChargerAssigner``
        or a fixed attach), that per-vehicle value is used. Otherwise every vehicle
        falls back to ``default_charger_power_kw`` (legacy single-rate behavior).

        Args:
            ev_attributes: DataFrame of vehicle attributes
            default_charger_power_kw: Fallback charger power when the column is absent;
                required (non-None) in that case

        Returns:
            Dictionary of vehicle keys to ``(capacity_kwh, kwh_per_mile, charger_power_kw)``
        """
        required = {"bldg_id", "vehicle_id", "battery_capacity_kwh", "kwh_per_mile"}
        missing = required - set(ev_attributes.columns)
        if missing:
            raise ValueError(f"ev_attributes missing columns: {sorted(missing)}")
        if ev_attributes.is_empty():
            raise ValueError("ev_attributes must contain at least one vehicle row")

        # Prefer per-vehicle charger power when the assigner / pipeline attached it.
        has_charger_col = "charger_power_kw" in ev_attributes.columns
        if not has_charger_col and default_charger_power_kw is None:
            raise ValueError(
                "ev_attributes has no charger_power_kw column and no default "
                "charger_power_kw was provided; use charger_assignment=resstock "
                "(per-vehicle attrs) or pass charger_power_kw for the fixed path"
            )
        select_cols = ["bldg_id", "vehicle_id", "battery_capacity_kwh", "kwh_per_mile"]
        if has_charger_col:
            select_cols.append("charger_power_kw")

        lookup: dict[tuple[str | int, int], tuple[float, float, float]] = {}
        for row in ev_attributes.select(select_cols).iter_rows(named=True):
            capacity = float(row["battery_capacity_kwh"])
            efficiency = float(row["kwh_per_mile"])
            if has_charger_col:
                charger_kw = float(row["charger_power_kw"])
            else:
                assert default_charger_power_kw is not None  # checked above
                charger_kw = float(default_charger_power_kw)
            if capacity <= 0:
                raise ValueError(
                    f"battery_capacity_kwh must be positive for "
                    f"({row['bldg_id']}, {row['vehicle_id']}), got {capacity}"
                )
            if efficiency < 0:
                raise ValueError(
                    f"kwh_per_mile must be non-negative for "
                    f"({row['bldg_id']}, {row['vehicle_id']}), got {efficiency}"
                )
            if charger_kw < 0:
                raise ValueError(
                    f"charger_power_kw must be non-negative for "
                    f"({row['bldg_id']}, {row['vehicle_id']}), got {charger_kw}"
                )
            lookup[(row["bldg_id"], int(row["vehicle_id"]))] = (
                capacity,
                efficiency,
                charger_kw,
            )
        return lookup

    def generate_soc(
        self,
        trip_schedules: pl.DataFrame,
        *,
        ev_attributes: pl.DataFrame,
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
        """
        Map each vehicle to an hourly SOC, charging, and discharge schedule for the instance date range.

        Pipeline per vehicle:
        1. Spread trip miles into hourly ``discharge_kwh`` on **drive** hours (temp-scaled)
        2. Build ``charge_kwh`` via ``charging_strategy`` while ``at_home`` from **tours**
        3. Derive ``soc_kwh`` and ``soc_underflow`` from discharge + charge

        ``soc_kwh`` is the battery level at the beginning of each hour (aligned with ``timestamp``).

        Charging strategies:
        - ``immediate``: charge at full power whenever home and not full (default).
        - ``off_peak``: TOU-adapted charging per the value-learning EV doc — charge only during
          off-peak hours in the overnight/pre-departure window until daily ``SOC_req`` is met;
          no peak charging and no emergency override.
        - ``off_peak_immediate``: TOU Immediate — charge at full power whenever home and
          off-peak until the pack is full. Optional ``allow_emergency_peak_charging`` permits
          on-peak home charging when foresight shows an energy shortfall before the next trip.
        - ``cost_minimizing``: perfect-foresight LP that shifts charging to the cheapest
          home hours while meeting trip energy needs. Requires ``hourly_price_usd_per_kwh``.
          Optional ``shed_load_penalty_usd_per_kwh`` penalizes curtailed trip energy; ``None``
          uses a very large default so shedding occurs only when required for LP feasibility.

        Args:
            trip_schedules: DataFrame of trip schedules
            ev_attributes: Per-vehicle ``battery_capacity_kwh`` and ``kwh_per_mile``
                (typically from ``EVBatteryAssigner``); optional ``charger_power_kw``
                per vehicle (from ``EVChargerAssigner``). Required for every simulated vehicle.
            vehicle_keys: Vehicle keys to include when building presence schedules internally
            presence_by_vehicle: Pre-built hourly presence schedules per vehicle; when provided,
                presence is not recomputed and ``vehicle_keys`` is ignored
            hours_base: Hourly calendar for trip-to-hour joins; built from the instance date range if None
            charger_power_kw: Fallback charger power in kW when ``ev_attributes`` lacks
                ``charger_power_kw``. Optional when every vehicle already has a per-vehicle
                rate on ``ev_attributes`` (resstock / fixed attach). If both are missing,
                raises.
            initial_soc_kwh: Initial SOC in kWh at the start of hour 0; when None, each vehicle
                starts full at its own battery capacity. A fixed absolute ``initial_soc_kwh`` is
                only valid when every vehicle uses the same capacity (or the value fits each pack).
            charging_strategy: ``immediate``, ``off_peak``, ``off_peak_immediate``, or ``cost_minimizing``
            hourly_price_usd_per_kwh: Length-``num_hours`` marginal price array for optimized charging
            shed_load_penalty_usd_per_kwh: Penalty on curtailed trip energy for ``cost_minimizing``;
                ``None`` uses ``DEFAULT_SHED_LOAD_PENALTY_USD_PER_KWH``
            peak_clock_hours: On-peak clock hours (0-23) for ``off_peak`` / ``off_peak_immediate``
            soc_min_fraction: Minimum comfortable SOC fraction for ``off_peak`` strategy
            soc_safety_buffer_fraction: Extra SOC fraction above daily trip energy for ``off_peak``
            allow_emergency_peak_charging: For ``off_peak_immediate`` only; allow on-peak home
                charging when remaining off-peak supply cannot cover the next trip
            hourly_temp_f_by_bldg: Optional ``bldg_id``, ``hour_index``, ``temp_f`` (°F) used to
                scale discharge when ``hourly_temp_scaled_miles`` is not provided
            hourly_temp_scaled_miles: Optional reusable duty frame from
                ``build_hourly_temp_scaled_miles`` (skips a second expand/temp pass)

        Returns:
            Long-form DataFrame with one row per vehicle-hour, including presence and SOC columns.

        Raises:
            ValueError: If ``ev_attributes`` is missing required columns / vehicles
            ValueError: If ``charger_power_kw`` is negative, or missing when attrs lack
                a per-vehicle ``charger_power_kw`` column
            ValueError: If ``initial_soc_kwh`` is not within [0, battery capacity] for a vehicle
            ValueError: If a pre-built presence schedule does not match the hourly calendar length
            ValueError: If ``charging_strategy`` is ``cost_minimizing`` without hourly prices
        """
        if charger_power_kw is not None and charger_power_kw < 0:
            raise ValueError(f"charger_power_kw must be non-negative, got {charger_power_kw}")

        attrs_lookup = self._battery_attrs_lookup(
            ev_attributes,
            default_charger_power_kw=charger_power_kw,
        )

        hours_base = self._resolve_hours_base(hours_base)
        num_hours = hours_base.height
        # Shared off-peak mask for all vehicles (depends only on clock hour, not trips).
        is_off_peak = build_is_off_peak(hours_base, peak_clock_hours=peak_clock_hours)

        if charging_strategy == "cost_minimizing":
            if hourly_price_usd_per_kwh is None:
                raise ValueError("hourly_price_usd_per_kwh is required when charging_strategy='cost_minimizing'")
            if len(hourly_price_usd_per_kwh) != num_hours:
                raise ValueError(
                    f"hourly_price_usd_per_kwh must have length {num_hours}, got {len(hourly_price_usd_per_kwh)}"
                )

        if presence_by_vehicle is None:
            presence_by_vehicle = self.generate_presence(
                trip_schedules,
                hours_base=hours_base,
                vehicle_keys=vehicle_keys,
            )
        else:
            for vehicle_key, presence in presence_by_vehicle.items():
                if presence.height != num_hours:
                    raise ValueError(
                        f"presence schedule for {vehicle_key} has {presence.height} rows, "
                        f"expected {num_hours} for the instance date range"
                    )

        missing_attrs = [key for key in presence_by_vehicle if key not in attrs_lookup]
        if missing_attrs:
            raise ValueError(
                "ev_attributes missing rows for vehicle(s): "
                + ", ".join(repr(key) for key in missing_attrs)
            )

        kwh_per_mile_by_vehicle = pl.DataFrame({
            "bldg_id": [key[0] for key in attrs_lookup],
            "vehicle_id": [key[1] for key in attrs_lookup],
            "kwh_per_mile": [vals[1] for vals in attrs_lookup.values()],
        })

        # Discharge = precomputed temp-scaled miles × kwh_per_mile when sizing already
        # built the duty frame; otherwise expand trips (+ optional temp) here.
        discharge_by_hour = self._build_hourly_discharge_kwh(
            trip_schedules,
            hours_base,
            kwh_per_mile_by_vehicle=kwh_per_mile_by_vehicle,
            hourly_temp_f_by_bldg=hourly_temp_f_by_bldg,
            hourly_temp_scaled_miles=hourly_temp_scaled_miles,
        )

        discharge_lookup: dict[tuple[str | int, int], dict[int, float]] = {}
        for discharge_frame in discharge_by_hour.partition_by(["bldg_id", "vehicle_id"], as_dict=False):
            vehicle_key = (discharge_frame["bldg_id"][0], int(discharge_frame["vehicle_id"][0]))
            discharge_lookup[vehicle_key] = dict(
                zip(
                    discharge_frame["hour_index"].to_list(),
                    discharge_frame["discharge_kwh"].to_list(),
                    strict=True,
                )
            )

        # Off-peak scheduling needs per-vehicle trip bounds (first departure / last arrival).
        vehicle_trips_by_key: dict[tuple[str | int, int], pl.DataFrame] = {}
        if charging_strategy == "off_peak" and not trip_schedules.is_empty():
            for trip_frame in trip_schedules.partition_by(["bldg_id", "vehicle_id"], as_dict=False):
                vehicle_key = (trip_frame["bldg_id"][0], int(trip_frame["vehicle_id"][0]))
                vehicle_trips_by_key[vehicle_key] = trip_frame

        # Generate hourly SOC schedules — capacity and charger kW resolved per vehicle.
        soc_by_vehicle: dict[tuple[str | int, int], pl.DataFrame] = {}
        for vehicle_key, presence in presence_by_vehicle.items():
            vehicle_capacity_kwh, _vehicle_kwh_per_mile, vehicle_charger_kw = attrs_lookup[
                vehicle_key
            ]
            # Default: start full at this vehicle's own pack size.
            start_soc = vehicle_capacity_kwh if initial_soc_kwh is None else initial_soc_kwh
            if not 0.0 <= start_soc <= vehicle_capacity_kwh:
                raise ValueError(
                    f"initial_soc_kwh must be within [0, {vehicle_capacity_kwh}] for "
                    f"{vehicle_key}, got {start_soc}"
                )

            discharge_arr = np.zeros(num_hours, dtype=np.float64)  # default: no driving this hour
            # Look up discharge kWh for each hour of the day
            for hour_index, discharge in discharge_lookup.get(vehicle_key, {}).items():
                discharge_arr[int(hour_index)] = discharge

            at_home = presence["at_home"].to_numpy()
            shed_load_kwh = np.zeros(num_hours, dtype=np.float64)
            # Choose hourly charge schedule (strategy-specific), using this vehicle's
            # capacity and its assigned charger power (L1/L2 or fixed).
            if charging_strategy == "immediate":
                charge_kwh = schedule_immediate_charging(
                    at_home,
                    discharge_arr,
                    battery_capacity_kwh=vehicle_capacity_kwh,
                    charger_power_kw=vehicle_charger_kw,
                    initial_soc_kwh=start_soc,
                )
            elif charging_strategy == "off_peak":
                # Derive Window_off and per-hour SOC_req from trip calendar.
                charge_allowed, soc_target_kwh = build_off_peak_charging_params(
                    at_home,
                    discharge_arr,
                    hours_base,
                    vehicle_trips_by_key.get(vehicle_key, pl.DataFrame()),
                    battery_capacity_kwh=vehicle_capacity_kwh,
                    is_off_peak=is_off_peak,
                    soc_min_fraction=soc_min_fraction,
                    soc_safety_buffer_fraction=soc_safety_buffer_fraction,
                )
                # Forward-simulate charge_kwh under the TOU rule (no peak override).
                charge_kwh = schedule_off_peak_charging(
                    at_home,
                    discharge_arr,
                    charge_allowed=charge_allowed,
                    soc_target_kwh=soc_target_kwh,
                    battery_capacity_kwh=vehicle_capacity_kwh,
                    charger_power_kw=vehicle_charger_kw,
                    initial_soc_kwh=start_soc,
                )
            elif charging_strategy == "off_peak_immediate":
                charge_kwh = schedule_off_peak_immediate_charging(
                    at_home,
                    discharge_arr,
                    is_off_peak=is_off_peak,
                    battery_capacity_kwh=vehicle_capacity_kwh,
                    charger_power_kw=vehicle_charger_kw,
                    initial_soc_kwh=start_soc,
                    allow_emergency_peak_charging=allow_emergency_peak_charging,
                )
            else:
                charge_kwh, shed_load_kwh = schedule_cost_minimizing_charging(
                    at_home,
                    discharge_arr,
                    battery_capacity_kwh=vehicle_capacity_kwh,
                    charger_power_kw=vehicle_charger_kw,
                    initial_soc_kwh=start_soc,
                    hourly_price_usd_per_kwh=np.asarray(hourly_price_usd_per_kwh, dtype=np.float64),
                    shed_load_penalty_usd_per_kwh=shed_load_penalty_usd_per_kwh,
                )

            # Derive beginning-of-hour SOC from discharge + charge (shared logic).
            soc_kwh, soc_underflow = compute_hourly_soc(
                discharge_arr,
                charge_kwh,
                initial_soc_kwh=start_soc,
            )

            output_columns = [
                pl.Series("discharge_kwh", discharge_arr),
                pl.Series("charge_kwh", charge_kwh),
                pl.Series("soc_kwh", soc_kwh),
                pl.Series("soc_underflow", soc_underflow),
            ]
            if charging_strategy == "cost_minimizing":
                output_columns.append(pl.Series("shed_load_kwh", shed_load_kwh))
            soc_by_vehicle[vehicle_key] = presence.with_columns(*output_columns)

        return self.to_dataframe(soc_by_vehicle)

    @staticmethod
    def to_dataframe(
        schedules_by_vehicle: dict[tuple[str | int, int], pl.DataFrame],
    ) -> pl.DataFrame:
        """Flatten per-vehicle hourly schedule dicts into a single long-form DataFrame.

        Stacks one row per vehicle-hour so building-level aggregations (e.g. TOU cost) can
        use ``group_by("bldg_id")`` instead of looping over the input dict.

        Args:
            schedules_by_vehicle: Dict keyed by ``(bldg_id, vehicle_id)`` with hourly frames
                from ``generate_presence`` or ``generate_soc``

        Returns:
            DataFrame with the following columns:
            - bldg_id: Building ID
            - vehicle_id: Vehicle ID
            - hour_index: Hour index
            - timestamp: Timestamp
            - at_home: Whether the vehicle is at home
            - away_from_home: Whether the vehicle is away from home
            - discharge_kwh: Discharge kWh
            - charge_kwh: Charge kWh
            - soc_kwh: SOC kWh
            - soc_underflow: Whether the vehicle has a SOC underflow
        """
        if not schedules_by_vehicle:
            return pl.DataFrame()

        # Tag each per-vehicle frame with its building/vehicle keys, then stack.
        frames = [
            frame.with_columns(
                pl.lit(bldg_id).alias("bldg_id"),
                pl.lit(vehicle_id).alias("vehicle_id"),
            )
            for (bldg_id, vehicle_id), frame in schedules_by_vehicle.items()
        ]
        return pl.concat(frames).select(
            "bldg_id",
            "vehicle_id",
            "hour_index",
            "timestamp",
            "at_home",
            "away_from_home",
            "discharge_kwh",
            "charge_kwh",
            "soc_kwh",
            "soc_underflow",
        )
