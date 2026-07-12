from collections.abc import Iterable
from datetime import date, datetime, timedelta
from typing import Final, Literal

import cvxpy as cp
import numpy as np
import polars as pl

ChargingStrategy = Literal["immediate", "cost_minimizing", "off_peak"]

DEFAULT_PEAK_CLOCK_HOURS: Final[tuple[int, ...]] = (17, 18, 19, 20, 21)  # 5pm-9pm on-peak window
DEFAULT_SOC_MIN_FRACTION = 0.2  # minimum comfortable SOC (SOC^min in TOU EV doc)
DEFAULT_SOC_SAFETY_BUFFER_FRACTION = 0.2  # extra SOC buffer above daily trip energy need
# Default shed penalty when none is passed: high enough that shedding is avoided unless
# required for LP feasibility (e.g. trip draw exceeds available SOC with no home charging).
DEFAULT_SHED_LOAD_PENALTY_USD_PER_KWH = 1e6
# How home charging is scheduled before SOC is derived from discharge + charge.

def build_hourly_timestamps(start_date: datetime, end_date: datetime) -> pl.DataFrame:
    """Build hourly timestamps for the instance date range (inclusive, aligned to whole hours).

    Returns:
        pl.DataFrame: hourly timestamps from ``start_date`` 00:00 through ``end_date`` 23:00

    Raises:
        ValueError: If ``end_date`` is before ``start_date``
    """
    start_hour = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_hour = end_date.replace(hour=23, minute=0, second=0, microsecond=0)
    if end_hour < start_hour:
        raise ValueError(
            f"end_date {end_date} must be on or after start_date {start_date}"
        )

    timestamps: list[datetime] = []
    current = start_hour
    while current <= end_hour:
        timestamps.append(current)
        current += timedelta(hours=1)

    return pl.DataFrame({"timestamp": timestamps})

def build_hours_base(start_date: datetime, end_date: datetime) -> pl.DataFrame:
    """Build the hourly calendar for the instance date range, used for trip-to-hour joins.

    Returns:
        pl.DataFrame: hourly calendar for the instance date range
    """
    return (
        build_hourly_timestamps(start_date, end_date)
        .with_row_index("hour_index")  # stable 0..num_hours-1 index in chronological order
        .with_columns(
            pl.col("timestamp").dt.date().alias("date"),  # calendar date for joining daily trip rows
            pl.col("timestamp").dt.hour().alias("hour"),  # clock hour (0..23) for joining trip rows
        )
    )

def schedule_immediate_charging(
    at_home: np.ndarray,
    discharge_kwh: np.ndarray,
    *,
    battery_capacity_kwh: float,
    charger_power_kw: float,
    initial_soc_kwh: float,
) -> np.ndarray:
    """
    Given a vehicle's hourly presence and discharge schedule, build an hourly charge schedule 
    by plugging in at max power whenever home and not full.

    This is the naive "charge as soon as you get home" policy. It forward-simulates SOC
    hour-by-hour because each hour's charge limit depends on energy remaining after that
    hour's trip draw. Returns only ``charge_kwh``; pair with ``compute_hourly_soc`` for
    beginning-of-hour SOC and underflow flags.

    Args:
        at_home: Whether the vehicle is home at the start of each hour
        discharge_kwh: Fixed trip draw ``x_t^DB`` each hour (kWh)
        battery_capacity_kwh: Battery capacity ``K^B`` (kWh)
        charger_power_kw: Max charge rate ``C^B`` when home (kW = kWh/hour)
        initial_soc_kwh: Start-of-hour-0 SOC ``s_0`` (kWh)

    Returns:
        Hourly charge energy ``x_t^CB`` (kWh), same length as ``at_home``
    """
    if len(at_home) != len(discharge_kwh):
        raise ValueError(
            f"at_home and discharge_kwh must have the same length, got {len(at_home)} and {len(discharge_kwh)}"
        )

    num_hours = len(at_home)
    charge_kwh = np.zeros(num_hours, dtype=np.float64)
    current_soc = initial_soc_kwh

    for hour_idx in range(num_hours):
        # Discharge first (same order as compute_hourly_soc).
        trip_draw = discharge_kwh[hour_idx]
        if trip_draw > current_soc:
            current_soc = 0.0  # no public charging; battery empty until next charge
        else:
            current_soc -= trip_draw

        # Charge at Level 2 whenever home and below full capacity.
        if at_home[hour_idx] and current_soc < battery_capacity_kwh:
            added = min(charger_power_kw, battery_capacity_kwh - current_soc)
            charge_kwh[hour_idx] = added
            current_soc += added

    return charge_kwh

def schedule_cost_minimizing_charging(
    at_home: np.ndarray,
    discharge_kwh: np.ndarray,
    *,
    battery_capacity_kwh: float,
    charger_power_kw: float,
    initial_soc_kwh: float,
    hourly_price_usd_per_kwh: np.ndarray,
    shed_load_penalty_usd_per_kwh: float | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a vehicle's hourly presence and discharge schedule, build a perfect-foresight hourly 
    charge schedule that minimizes electricity cost (cvxpy LP).

    Returns ``(charge_kwh, shed_load_kwh)``; pair with ``compute_hourly_soc`` using the full
    planned ``discharge_kwh`` so ``soc_underflow`` flags hours where the battery could not
    cover the trip draw (shed load is reported separately, not subtracted from discharge).
    This is a theoretical lower bound on charging cost when the driver knows all future trips
    and can shift charging to the cheapest home hours.

    LP formulation (``T = num_hours`` clock hours indexed ``0..T-1``):

    - Given: ``s_0`` (``initial_soc_kwh``), fixed discharge ``x_t^DB = discharge_kwh[t]``
    - Decide: charge ``x_t^CB``, shed load ``x_t^SL``, and start-of-hour SOC ``s_t`` for ``t = 1..T``
    - Minimize ``sum_{t=0}^{T-1} (p_t x_t^CB + w_t x_t^SL)``
    - Subject to:
        ``s_1 = s_0 + x_0^CB - x_0^DB + x_0^SL``
        ``s_{t+1} = s_t + x_t^CB - x_t^DB + x_t^SL`` for ``t = 1..T-1``
        ``0 <= x_t^CB <= C^B`` (charger limit when home, else 0)
        ``0 <= x_t^SL <= x_t^DB`` (shed load cannot exceed planned trip draw)
        ``0 <= s_t <= K^B`` for ``t = 1..T``

    Shed load ``x_t^SL`` is trip energy not taken from the battery (effective draw is
    ``x_t^DB - x_t^SL``). Shedding is always allowed (``x_t^SL >= 0``); when
    ``shed_load_penalty_usd_per_kwh`` is ``None``, a very large default penalty is
    used so shedding only occurs when the LP would otherwise be infeasible.

    Discharging and charging are mutually exclusive in this model: trip draw is
    assigned to away hours where ``x_t^CB = 0``, so ``s_t >= x_t^DB - x_t^SL`` follows from
    ``s_{t+1} >= 0`` and the transition equalities without an extra constraint.

    Args:
        at_home: Whether the vehicle is home at the start of each hour
        discharge_kwh: Fixed trip draw ``x_t^DB`` each hour (kWh)
        battery_capacity_kwh: Battery capacity ``K^B`` (kWh)
        charger_power_kw: Max charge rate ``C^B`` when home (kW = kWh/hour)
        initial_soc_kwh: Start-of-hour-0 SOC ``s_0`` (kWh)
        hourly_price_usd_per_kwh: Marginal price ``p_t`` each hour ($/kWh)
        shed_load_penalty_usd_per_kwh: Penalty ``w_t`` on shed load ($/kWh); ``None`` uses
            ``DEFAULT_SHED_LOAD_PENALTY_USD_PER_KWH`` (shed only when needed for feasibility)

    Returns:
        Tuple of hourly charge energy ``x_t^CB`` (kWh) and shed load ``x_t^SL`` (kWh),
        each the same length as ``at_home``

    Raises:
        ValueError: If input arrays differ in length or prices/penalties are negative
        RuntimeError: If the LP solver fails to find a feasible schedule
    """
    num_hours = len(at_home)
    if len(discharge_kwh) != num_hours:
        raise ValueError(
            f"at_home and discharge_kwh must have the same length, got {len(at_home)} and {len(discharge_kwh)}"
        )
    if len(hourly_price_usd_per_kwh) != num_hours:
        raise ValueError(
            "hourly_price_usd_per_kwh must match schedule length, "
            f"got {len(hourly_price_usd_per_kwh)} and expected {num_hours}"
        )
    if np.any(hourly_price_usd_per_kwh < 0):
        raise ValueError("hourly_price_usd_per_kwh must be non-negative")

    discharge = np.asarray(discharge_kwh, dtype=np.float64)
    prices = np.asarray(hourly_price_usd_per_kwh, dtype=np.float64)
    max_charge = np.where(at_home, charger_power_kw, 0.0)
    s_0 = float(initial_soc_kwh)

    # set default shed load penalty if none is passed
    if shed_load_penalty_usd_per_kwh is None:
        shed_penalties = np.full(num_hours, DEFAULT_SHED_LOAD_PENALTY_USD_PER_KWH, dtype=np.float64)
    elif isinstance(shed_load_penalty_usd_per_kwh, np.ndarray):
        shed_penalties = np.asarray(shed_load_penalty_usd_per_kwh, dtype=np.float64)
        if len(shed_penalties) != num_hours:
            raise ValueError(
                "shed_load_penalty_usd_per_kwh must match schedule length, "
                f"got {len(shed_penalties)} and expected {num_hours}"
            )
    else:
        shed_penalties = np.full(num_hours, float(shed_load_penalty_usd_per_kwh), dtype=np.float64)
    if np.any(shed_penalties < 0):
        raise ValueError("shed_load_penalty_usd_per_kwh must be non-negative")

    # Decision variables: x_t^CB (charge), x_t^SL (shed), and s_t for t = 1..T.
    charge = cp.Variable(num_hours, name="charge")
    soc = cp.Variable(num_hours, name="soc")  # s_1..s_T
    shed = cp.Variable(num_hours, name="shed_load")

    constraints: list[cp.Constraint] = [
        soc[0] == s_0 + charge[0] - discharge[0] + shed[0], # first-hour SOC constraint
        charge >= 0, # charge cannot be negative
        charge <= max_charge,  # zero when away (discharge and charge never overlap)
        shed >= 0, # shed cannot be negative
        shed <= discharge, # shed cannot exceed planned trip draw
        soc >= 0, # SOC cannot be negative
        soc <= battery_capacity_kwh, # SOC cannot exceed battery capacity
    ]
    if num_hours > 1:
        constraints.append(soc[1:] == soc[:-1] + charge[1:] - discharge[1:] + shed[1:]) # hour-by-hour SOC constraints

    objective = prices @ charge + shed_penalties @ shed

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Cost-minimizing charging LP failed: {problem.status}")

    charge_kwh = np.asarray(charge.value, dtype=np.float64).reshape(-1)
    shed_load_kwh = np.asarray(shed.value, dtype=np.float64).reshape(-1)
    return charge_kwh, shed_load_kwh

def build_is_off_peak(
    hours_base: pl.DataFrame,
    *,
    peak_clock_hours: Iterable[int] = DEFAULT_PEAK_CLOCK_HOURS,
) -> np.ndarray:
    """Return a boolean mask that is True during off-peak clock hours.
    
    Args:
        hours_base: Hourly calendar with ``date`` and ``hour`` columns
        peak_clock_hours: On-peak clock hours (0-23) for ``off_peak`` strategy

    Returns:
        Boolean mask aligned with ``at_home`` that is True during off-peak clock hours
    """
    peak_hours = set(peak_clock_hours)  # doc set H: on-peak clock hours (default 5-9pm)
    clock_hours = hours_base["hour"].to_numpy()
    # True = off-peak (t ∉ H); reused for every vehicle on the same calendar
    return np.array([hour not in peak_hours for hour in clock_hours], dtype=bool)

def build_off_peak_charging_params(
    at_home: np.ndarray,
    discharge_kwh: np.ndarray,
    hours_base: pl.DataFrame,
    vehicle_trips: pl.DataFrame,
    *,
    battery_capacity_kwh: float,
    is_off_peak: np.ndarray,
    soc_min_fraction: float = DEFAULT_SOC_MIN_FRACTION,
    soc_safety_buffer_fraction: float = DEFAULT_SOC_SAFETY_BUFFER_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build per-hour TOU off-peak charging controls from daily trip bounds.

    Implements the primary TOU strategy in docs_tou_ev_schedule_value_learning.qmd (no emergency
    override): charge only in ``Window_off`` until ``SOC_req`` for the upcoming departure day
    is met.

    For each calendar day *d*:
    - ``Window_avail`` spans from ``last_arrival`` on day *d-1* through ``first_departure``
      on day *d* (overnight + pre-departure morning hours).
    - ``Window_off = Window_avail ∩ off-peak hours ∩ at_home``.
    - ``SOC_req^d = max(SOC_min, daily_trip_kWh / K^B + buffer)`` in SOC fraction,
      converted to kWh via ``battery_capacity_kwh``.

    Midday at-home hours (e.g. between same-day trips) are outside ``Window_avail`` and
    cannot charge, even if off-peak.

    Args:
        at_home: Whether the vehicle is home at the start of each hour
        discharge_kwh: Fixed trip draw each hour (kWh)
        hours_base: Hourly calendar with ``date`` and ``hour`` columns
        vehicle_trips: Trip rows for this vehicle (may be empty)
        battery_capacity_kwh: Battery capacity ``K^B`` (kWh)
        is_off_peak: Off-peak mask aligned with ``at_home``
        soc_min_fraction: Minimum comfortable SOC fraction ``SOC^min``
        soc_safety_buffer_fraction: Extra SOC fraction above daily trip energy

    Returns:
        Tuple of ``charge_allowed`` (bool) and ``soc_target_kwh`` (float) arrays
    """
    num_hours = len(at_home)
    if len(discharge_kwh) != num_hours or len(is_off_peak) != num_hours:
        raise ValueError("at_home, discharge_kwh, and is_off_peak must have the same length")
    if hours_base.height != num_hours:
        raise ValueError(
            f"hours_base must have {num_hours} rows to match schedule length, got {hours_base.height}"
        )
    if not 0.0 <= soc_min_fraction <= 1.0:
        raise ValueError(f"soc_min_fraction must be within [0, 1], got {soc_min_fraction}")
    if not 0.0 <= soc_safety_buffer_fraction <= 1.0:
        raise ValueError(
            f"soc_safety_buffer_fraction must be within [0, 1], got {soc_safety_buffer_fraction}"
        )

    dates = hours_base["date"].to_list()
    clock_hours = hours_base["hour"].to_numpy().astype(np.int64)

    # Sum hourly trip draw into calendar-day totals (miles_total × kWh/mile in discharge_kwh).
    daily_discharge_kwh: dict[date, float] = {}
    for day, discharge in zip(dates, discharge_kwh, strict=True):
        daily_discharge_kwh[day] = daily_discharge_kwh.get(day, 0.0) + float(discharge)

    # Per-day trip bounds: t_dep^first and t_arr^last from trip rows.
    daily_bounds: dict[date, tuple[int, int]] = {}
    if not vehicle_trips.is_empty():
        bounds_frame = (
            vehicle_trips.with_columns(pl.col("date").cast(pl.Date).alias("date"))
            .group_by("date")
            .agg(
                pl.col("departure_hour").min().alias("first_departure"),
                pl.col("arrival_hour").max().alias("last_arrival"),
            )
        )
        for row in bounds_frame.iter_rows(named=True):
            daily_bounds[row["date"]] = (int(row["first_departure"]), int(row["last_arrival"]))

    def soc_req_kwh_for_day(day: date) -> float:
        """SOC_req^d = max(SOC_min, daily_trip_kWh / K^B + buffer), returned in kWh."""
        daily_trip_kwh = daily_discharge_kwh.get(day, 0.0)
        soc_req_fraction = max(
            soc_min_fraction,
            daily_trip_kwh / battery_capacity_kwh + soc_safety_buffer_fraction,
        )
        return soc_req_fraction * battery_capacity_kwh

    charge_allowed = np.zeros(num_hours, dtype=bool)
    soc_target_kwh = np.zeros(num_hours, dtype=np.float64)

    for hour_idx, (day, clock_hour) in enumerate(zip(dates, clock_hours, strict=True)):
        # Default (no trips that day): never depart (24), always home (0) → whole day is Window_avail.
        first_departure, last_arrival = daily_bounds.get(day, (24, 0))
        # Morning slice of Window_avail: hours before first departure today.
        in_morning_window = clock_hour < first_departure
        # Evening slice of Window_avail: hours after last arrival today (overnight for tomorrow).
        in_evening_window = clock_hour >= last_arrival

        # Midday away block (first_dep <= hour < last_arr) is outside Window_avail.
        if not (in_morning_window or in_evening_window):
            continue

        # Decide which departure day this hour is charging toward.
        if in_morning_window and in_evening_window:
            # No-trip day: both windows cover all 24 hours; use today's (minimal) SOC_req.
            target_day = day
        elif in_morning_window:
            target_day = day  # pre-departure hours charge toward today's trips
        else:
            target_day = day + timedelta(days=1)  # post-arrival hours charge toward tomorrow

        soc_target_kwh[hour_idx] = soc_req_kwh_for_day(target_day)
        # Window_off = Window_avail ∩ off-peak ∩ at_home (no emergency peak override).
        charge_allowed[hour_idx] = bool(at_home[hour_idx] and is_off_peak[hour_idx])

    return charge_allowed, soc_target_kwh

def schedule_off_peak_charging(
    at_home: np.ndarray,
    discharge_kwh: np.ndarray,
    *,
    charge_allowed: np.ndarray,
    soc_target_kwh: np.ndarray,
    battery_capacity_kwh: float,
    charger_power_kw: float,
    initial_soc_kwh: float,
) -> np.ndarray:
    """
    Given a vehicle's hourly presence and discharge schedule, build an hourly charge
    schedule using TOU off-peak charging (no emergency override).

    Forward-simulates SOC hour-by-hour. Charges at max power only when
    ``charge_allowed`` is True and SOC is below the per-hour ``soc_target_kwh``
    threshold. Never charges during peak hours or away from home.

    Args:
        at_home: Whether the vehicle is home at the start of each hour
        discharge_kwh: Fixed trip draw ``x_t^DB`` each hour (kWh)
        charge_allowed: Whether off-peak charging is permitted this hour
        soc_target_kwh: Target SOC (kWh) to reach before the next departure day
        battery_capacity_kwh: Battery capacity ``K^B`` (kWh)
        charger_power_kw: Max charge rate ``C^B`` when allowed (kW = kWh/hour)
        initial_soc_kwh: Start-of-hour-0 SOC ``s_0`` (kWh)

    Returns:
        Hourly charge energy ``x_t^CB`` (kWh), same length as ``at_home``
    """
    if len(at_home) != len(discharge_kwh):
        raise ValueError(
            f"at_home and discharge_kwh must have the same length, got {len(at_home)} and {len(discharge_kwh)}"
        )
    if len(charge_allowed) != len(at_home) or len(soc_target_kwh) != len(at_home):
        raise ValueError("charge_allowed and soc_target_kwh must match at_home length")

    num_hours = len(at_home)
    charge_kwh = np.zeros(num_hours, dtype=np.float64)
    current_soc = initial_soc_kwh

    for hour_idx in range(num_hours):
        # Same hour ordering as compute_hourly_soc: discharge first, then charge.
        trip_draw = discharge_kwh[hour_idx]
        if trip_draw > current_soc:
            current_soc = 0.0  # no public charging; battery empty until next allowed hour
        else:
            current_soc -= trip_draw

        # Primary TOU rule: Pow_max when t ∈ Window_off and SOC < SOC_req (no peak override).
        if (
            charge_allowed[hour_idx]
            and at_home[hour_idx]
            and current_soc + 1e-9 < soc_target_kwh[hour_idx]
        ):
            # Stop at SOC_req, not full battery (unlike immediate charging).
            headroom = min(
                soc_target_kwh[hour_idx] - current_soc,
                battery_capacity_kwh - current_soc,
            )
            added = min(charger_power_kw, headroom)
            if added > 0.0:
                charge_kwh[hour_idx] = added
                current_soc += added

    return charge_kwh

def compute_hourly_soc(
    discharge_kwh: np.ndarray,
    charge_kwh: np.ndarray,
    *,
    initial_soc_kwh: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a vehicle's hourly discharge and charge schedule, derive the beginning-of-hour SOC 
    and underflow flags.

    Shared by both charging strategies: once ``charge_kwh`` is chosen, SOC is computed with
    the same hour-by-hour rules. ``soc_kwh[t]`` is the battery level at the **start** of
    hour *t*; within the hour, discharge is applied first, then charge.

    Args:
        discharge_kwh: Fixed trip draw each hour (kWh)
        charge_kwh: Scheduled charge each hour (kWh), from immediate or cost-minimizing policy
        initial_soc_kwh: Start-of-hour-0 SOC (kWh)

    Returns:
        Tuple of beginning-of-hour SOC (kWh) and per-hour underflow flag (True when trip
        draw exceeds available SOC at the start of that hour; SOC is clamped to zero)
    """
    if len(discharge_kwh) != len(charge_kwh):
        raise ValueError(
            f"discharge_kwh and charge_kwh must have the same length, got {len(discharge_kwh)} and {len(charge_kwh)}"
        )

    num_hours = len(discharge_kwh)
    soc_kwh = np.empty(num_hours, dtype=np.float64)
    soc_underflow = np.zeros(num_hours, dtype=bool)

    current_soc = initial_soc_kwh
    for hour_idx in range(num_hours):
        soc_kwh[hour_idx] = current_soc  # record beginning-of-hour SOC

        trip_draw = discharge_kwh[hour_idx]
        if trip_draw > current_soc + 1e-9:
            soc_underflow[hour_idx] = True
            current_soc = 0.0
        else:
            current_soc -= trip_draw

        current_soc += charge_kwh[hour_idx]  # charge after discharge within the hour

    return soc_kwh, soc_underflow

