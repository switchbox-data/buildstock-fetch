from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

from utils.charging import (
    ChargingStrategy,
    DEFAULT_PEAK_CLOCK_HOURS,
    DEFAULT_SOC_MIN_FRACTION,
    DEFAULT_SOC_SAFETY_BUFFER_FRACTION,
    build_hours_base,
    build_is_off_peak,
    build_off_peak_charging_params,
    compute_hourly_soc,
    schedule_cost_minimizing_charging,
    schedule_immediate_charging,
    schedule_off_peak_charging,
)

DEFAULT_BATTERY_CAPACITY_KWH = 90.0  # uniform fleet battery assumption for SOC modeling
DEFAULT_KWH_PER_MILE = 0.30  # simple-model assumption
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

        Uses the same away-hour model as trip schedule generation: a vehicle is
        away from home for hours ``range(departure_hour, arrival_hour)`` on each trip day,
        where ``arrival_hour`` is the first hour at home. It is at home (and available to
        charge) in all other hours.

        Args:
            trip_schedules (pl.DataFrame): DataFrame of trip schedules
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
            # no trips means the vehicle never leaves home; every hour is chargeable
            hourly_presence = (
                vehicle_keys_df.join(hours_base, how="cross")  # one hourly schedule per requested vehicle
                .with_columns(
                    pl.lit(True).alias("at_home"),  # default presence state without trip evidence
                    pl.lit(False).alias("away_from_home"),  # explicit complement of at_home
                    pl.lit(True).alias("can_charge"),  # home charging assumed available whenever at home
                )
                .select("bldg_id", "vehicle_id", "hour_index", "timestamp", "at_home", "away_from_home", "can_charge")
            )
        else:
            away_hours = (
                trip_schedules.with_columns(pl.col("date").cast(pl.Date).alias("date"))  # normalize to date-only key
                .with_columns(
                    # away from departure_hour (inclusive) through arrival_hour (exclusive)
                    pl.int_ranges(pl.col("departure_hour"), pl.col("arrival_hour")).alias("hour"),
                )
                .explode("hour")  # one row per away hour instead of one row per trip interval
                .select("bldg_id", "vehicle_id", "date", "hour")  # minimal join keys for marking away hours
                .unique()  # overlapping trips on the same day collapse to a single away marker
            )

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
                    pl.col("at_home").alias("can_charge"),
                )
                .select("bldg_id", "vehicle_id", "hour_index", "timestamp", "at_home", "away_from_home", "can_charge")
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
    def _build_hourly_discharge_kwh(
        trip_schedules: pl.DataFrame,
        hours_base: pl.DataFrame,
        *,
        kwh_per_mile: float = DEFAULT_KWH_PER_MILE,
        kwh_per_mile_by_vehicle: pl.DataFrame | None = None,
        ev_adoption_rate: float,
    ) -> pl.DataFrame:
        """Map each vehicle's trip schedule to an hourly schedule of discharge kWh for the instance date range.
        Spread each trip's driving energy uniformly over its away-from-home hours.

        Away hours are ``range(departure_hour, arrival_hour)`` where ``arrival_hour`` is the
        first hour at home (exclusive end of the away interval).

        Args:
            trip_schedules (pl.DataFrame): DataFrame of trip schedules
            hours_base (pl.DataFrame): hourly calendar for the instance date range
            kwh_per_mile (float): default kWh per mile when per-vehicle values are absent
            kwh_per_mile_by_vehicle: optional ``bldg_id``, ``vehicle_id``, ``kwh_per_mile`` frame
            ev_adoption_rate (float): EV adoption rate

        Returns:
            pl.DataFrame: hourly discharge kWh for each trip
        """
        if trip_schedules.is_empty():
            # no trips -> no driving discharge; return typed empty frame for downstream joins
            return pl.DataFrame(
                schema={
                    "bldg_id": pl.Int64,
                    "vehicle_id": pl.Int64,
                    "hour_index": pl.UInt32,
                    "discharge_kwh": pl.Float64,
                }
            )

        trips = trip_schedules
        # When ResStock battery attributes are present, use each vehicle's Autonomie
        # kWh/mile instead of the fleet-wide default (e.g. 0.30).
        if kwh_per_mile_by_vehicle is not None:
            trips = trips.join(
                kwh_per_mile_by_vehicle.select("bldg_id", "vehicle_id", "kwh_per_mile"),
                on=["bldg_id", "vehicle_id"],
                how="left",
            ).with_columns(
                # Vehicles missing from ev_attributes fall back to the scalar default.
                pl.col("kwh_per_mile").fill_null(kwh_per_mile)
            )
            trip_kwh_expr = pl.col("miles_driven") * pl.col("kwh_per_mile") * ev_adoption_rate
        else:
            # Legacy path: one efficiency for the whole fleet.
            trip_kwh_expr = pl.col("miles_driven") * kwh_per_mile * ev_adoption_rate

        away_hour_discharge = (
            trips.with_columns(pl.col("date").cast(pl.Date).alias("date"))  # normalize to date-only key
            .with_columns(
                # away from departure_hour (inclusive) through arrival_hour (exclusive)
                pl.int_ranges(pl.col("departure_hour"), pl.col("arrival_hour")).alias("hour"),
                trip_kwh_expr.alias("trip_kwh"),
            )
            .with_columns(
                # divide total trip energy evenly across all away hours for that trip
                (pl.col("trip_kwh") / pl.col("hour").list.len()).alias("discharge_kwh_per_away_hour")
            )
            .explode("hour")  # one row per away hour instead of one row per trip
            .join(
                hours_base.select("date", "hour", "hour_index"),  # map calendar (date, hour) -> hour_index index
                on=["date", "hour"],
                how="inner",
            )
            .group_by("bldg_id", "vehicle_id", "hour_index")
            # overlapping trips contributing to the same hour add their discharge shares together
            .agg(pl.col("discharge_kwh_per_away_hour").sum().alias("discharge_kwh"))
        )
        return away_hour_discharge

    @staticmethod
    def _battery_attrs_lookup(
        ev_attributes: pl.DataFrame | None,
    ) -> dict[tuple[str | int, int], tuple[float, float]]:
        """Map ``(bldg_id, vehicle_id)`` -> ``(battery_capacity_kwh, kwh_per_mile)``.

        Built once per ``generate_soc`` call so the per-vehicle loop can O(1) look up
        ResStock/Autonomie pack size and efficiency.
        """
        if ev_attributes is None or ev_attributes.is_empty():
            return {}

        required = {"bldg_id", "vehicle_id", "battery_capacity_kwh", "kwh_per_mile"}
        missing = required - set(ev_attributes.columns)
        if missing:
            raise ValueError(f"ev_attributes missing columns: {sorted(missing)}")

        lookup: dict[tuple[str | int, int], tuple[float, float]] = {}
        for row in ev_attributes.select(
            "bldg_id", "vehicle_id", "battery_capacity_kwh", "kwh_per_mile"
        ).iter_rows(named=True):
            capacity = float(row["battery_capacity_kwh"])
            efficiency = float(row["kwh_per_mile"])
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
            lookup[(row["bldg_id"], int(row["vehicle_id"]))] = (capacity, efficiency)
        return lookup

    def generate_soc(
        self,
        trip_schedules: pl.DataFrame,
        *,
        vehicle_keys: Iterable[tuple[str | int, int]] | None = None,
        hours_base: pl.DataFrame | None = None,
        presence_by_vehicle: dict[tuple[str | int, int], pl.DataFrame] | None = None,
        battery_capacity_kwh: float = DEFAULT_BATTERY_CAPACITY_KWH,
        kwh_per_mile: float = DEFAULT_KWH_PER_MILE,
        ev_attributes: pl.DataFrame | None = None,
        charger_power_kw: float = DEFAULT_LEVEL2_CHARGER_KW,
        ev_adoption_rate: float = 1.0,
        initial_soc_kwh: float | None = None,
        charging_strategy: ChargingStrategy = "immediate",
        hourly_price_usd_per_kwh: np.ndarray | None = None,
        shed_load_penalty_usd_per_kwh: float | np.ndarray | None = None,
        peak_clock_hours: Iterable[int] = DEFAULT_PEAK_CLOCK_HOURS,
        soc_min_fraction: float = DEFAULT_SOC_MIN_FRACTION,
        soc_safety_buffer_fraction: float = DEFAULT_SOC_SAFETY_BUFFER_FRACTION,
    ) -> pl.DataFrame:
        """
        Map each vehicle to an hourly SOC, charging, and discharge schedule for the instance date range.

        Pipeline per vehicle:
        1. Spread trip miles into hourly ``discharge_kwh`` (away hours only)
        2. Build ``charge_kwh`` via ``charging_strategy`` (immediate, off-peak, or cost-minimizing LP)
        3. Derive ``soc_kwh`` and ``soc_underflow`` from discharge + charge

        ``soc_kwh`` is the battery level at the beginning of each hour (aligned with ``timestamp``).

        Charging strategies:
        - ``immediate``: charge at full power whenever home and not full (default).
        - ``off_peak``: TOU-adapted charging per the value-learning EV doc — charge only during
          off-peak hours in the overnight/pre-departure window until daily ``SOC_req`` is met;
          no peak charging and no emergency override.
        - ``cost_minimizing``: perfect-foresight LP that shifts charging to the cheapest
          home hours while meeting trip energy needs. Requires ``hourly_price_usd_per_kwh``.
          Optional ``shed_load_penalty_usd_per_kwh`` penalizes curtailed trip energy; ``None``
          uses a very large default so shedding occurs only when required for LP feasibility.

        Args:
            trip_schedules: DataFrame of trip schedules
            vehicle_keys: Vehicle keys to include when building presence schedules internally
            presence_by_vehicle: Pre-built hourly presence schedules per vehicle; when provided,
                presence is not recomputed and ``vehicle_keys`` is ignored
            hours_base: Hourly calendar for trip-to-hour joins; built from the instance date range if None
            battery_capacity_kwh: Default battery capacity in kWh when ``ev_attributes`` omits a vehicle
            kwh_per_mile: Default kWh per mile when ``ev_attributes`` omits a vehicle
            ev_attributes: Optional per-vehicle attributes with ``battery_capacity_kwh`` and
                ``kwh_per_mile`` (typically from ``EVBatteryAssigner``)
            charger_power_kw: Charger power in kW
            ev_adoption_rate: EV adoption rate
            initial_soc_kwh: Initial SOC in kWh at the start of hour 0; when None, each vehicle
                starts full at its own battery capacity. A fixed absolute ``initial_soc_kwh`` is
                only valid when every vehicle uses the same capacity (or the value fits each pack).
            charging_strategy: ``immediate``, ``off_peak``, or ``cost_minimizing``
            hourly_price_usd_per_kwh: Length-``num_hours`` marginal price array for optimized charging
            shed_load_penalty_usd_per_kwh: Penalty on curtailed trip energy for ``cost_minimizing``;
                ``None`` uses ``DEFAULT_SHED_LOAD_PENALTY_USD_PER_KWH``
            peak_clock_hours: On-peak clock hours (0-23) for ``off_peak`` strategy
            soc_min_fraction: Minimum comfortable SOC fraction for ``off_peak`` strategy
            soc_safety_buffer_fraction: Extra SOC fraction above daily trip energy for ``off_peak``

        Returns:
            Long-form DataFrame with one row per vehicle-hour, including presence and SOC columns.

        Raises:
            ValueError: If ``battery_capacity_kwh`` is not positive
            ValueError: If ``charger_power_kw`` is negative
            ValueError: If ``kwh_per_mile`` is negative
            ValueError: If ``initial_soc_kwh`` is not within [0, battery capacity] for a vehicle
            ValueError: If a pre-built presence schedule does not match the hourly calendar length
            ValueError: If ``charging_strategy`` is ``cost_minimizing`` without hourly prices
        """
        if battery_capacity_kwh <= 0:
            raise ValueError(f"battery_capacity_kwh must be positive, got {battery_capacity_kwh}")
        if charger_power_kw < 0:
            raise ValueError(f"charger_power_kw must be non-negative, got {charger_power_kw}")
        if kwh_per_mile < 0:
            raise ValueError(f"kwh_per_mile must be non-negative, got {kwh_per_mile}")

        attrs_lookup = self._battery_attrs_lookup(ev_attributes)

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

        # Flatten attrs_lookup into a joinable frame for per-vehicle discharge energy.
        kwh_per_mile_by_vehicle = None
        if attrs_lookup:
            kwh_per_mile_by_vehicle = pl.DataFrame({
                "bldg_id": [key[0] for key in attrs_lookup],
                "vehicle_id": [key[1] for key in attrs_lookup],
                "kwh_per_mile": [vals[1] for vals in attrs_lookup.values()],
            })

        # Build hourly discharge kWh (uses per-vehicle efficiency when available).
        discharge_by_hour = self._build_hourly_discharge_kwh(
            trip_schedules,
            hours_base,
            kwh_per_mile=kwh_per_mile,
            kwh_per_mile_by_vehicle=kwh_per_mile_by_vehicle,
            ev_adoption_rate=ev_adoption_rate,
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

        # Generate hourly SOC schedules — capacity is resolved per vehicle below.
        soc_by_vehicle: dict[tuple[str | int, int], pl.DataFrame] = {}
        for vehicle_key, presence in presence_by_vehicle.items():
            # Prefer ResStock/Autonomie capacity; fall back to scalar defaults for tests / legacy.
            vehicle_capacity_kwh, _vehicle_kwh_per_mile = attrs_lookup.get(
                vehicle_key,
                (battery_capacity_kwh, kwh_per_mile),
            )
            # Default: start full at this vehicle's own pack size (not a fleet-wide 90 kWh).
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
            # Choose hourly charge schedule (strategy-specific), using this vehicle's capacity.
            if charging_strategy == "immediate":
                charge_kwh = schedule_immediate_charging(
                    at_home,
                    discharge_arr,
                    battery_capacity_kwh=vehicle_capacity_kwh,
                    charger_power_kw=charger_power_kw,
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
                    charger_power_kw=charger_power_kw,
                    initial_soc_kwh=start_soc,
                )
            else:
                charge_kwh, shed_load_kwh = schedule_cost_minimizing_charging(
                    at_home,
                    discharge_arr,
                    battery_capacity_kwh=vehicle_capacity_kwh,
                    charger_power_kw=charger_power_kw,
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
            - can_charge: Whether the vehicle can charge
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
            "can_charge",
            "discharge_kwh",
            "charge_kwh",
            "soc_kwh",
            "soc_underflow",
        )
