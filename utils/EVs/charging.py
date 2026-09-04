"""Hourly EV charging policies and calendar helpers.

Pipeline role (inputs come from ``ChargingSimulator`` / ``TripScheduleGenerator``):

1. **Calendar** — ``build_hours_base`` builds the simulation hour grid.
2. **Presence** — expand ``tour_*`` intervals → away hours → ``at_home``.
3. **Discharge** — expand ``trip_*`` drive intervals → ``discharge_kwh``.
4. **Charge** — one of the ``schedule_*`` policies chooses ``charge_kwh``.
5. **SOC** — ``compute_hourly_soc`` applies discharge then charge each hour.

Strategies: ``immediate``, ``off_peak``, ``off_peak_immediate``, ``cost_minimizing``.
"""

from collections.abc import Iterable
from datetime import date, datetime, timedelta
from typing import Final, Literal

import cvxpy as cp
import numpy as np
import polars as pl
from cvxpy.error import SolverError

ChargingStrategy = Literal["immediate", "cost_minimizing", "off_peak", "off_peak_immediate"]

DEFAULT_PEAK_CLOCK_HOURS: Final[tuple[int, ...]] = (17, 18, 19, 20)  # 5pm–9pm; hour 21 (9–10pm) is off-peak
# BGE Schedule EV (P.S.C. Md. E-6): summer Jun–Sep weekdays 10am–8pm; non-summer
# weekdays 7–11am and 5–9pm. Weekends/holidays are off-peak. DST shift not modeled.
DEFAULT_PEAK_CLOCK_HOURS_SUMMER: Final[tuple[int, ...]] = tuple(range(10, 20))  # 10..19
DEFAULT_PEAK_CLOCK_HOURS_WINTER: Final[tuple[int, ...]] = (7, 8, 9, 10, 17, 18, 19, 20)
DEFAULT_TOU_SUMMER_MONTHS: Final[tuple[int, ...]] = (6, 7, 8, 9)  # Jun 1 – Sep 30
DEFAULT_SOC_MIN_FRACTION = 0.2  # minimum comfortable SOC (SOC^min in TOU EV doc)
DEFAULT_SOC_SAFETY_BUFFER_FRACTION = 0.2  # extra SOC buffer above daily trip energy need
# Default shed penalty when none is passed: high enough that shedding is avoided unless
# required for LP feasibility (e.g. trip draw exceeds available SOC with no home charging).
DEFAULT_SHED_LOAD_PENALTY_USD_PER_KWH = 1e6
# Solvers tried in order for the cost-minimizing LP. CLARABEL (interior point) is cvxpy's
# default and fastest here, but struggles on near-degenerate schedules (few or no drive
# hours pin the shed variables to a zero-width box); HIGHS/SCS pick up those cases.
COST_MIN_LP_SOLVERS: Final[tuple[str, ...]] = ("CLARABEL", "HIGHS", "SCS")
# Hours of off-peak-only foresight used by emergency peak charging. 48h covers
# cases where the shortfall only becomes binding more than one day ahead
# (e.g. Level-1 vehicles that need several peak home hours of advance charging).
EMERGENCY_PEAK_LOOKAHEAD_HOURS: Final[int] = 48
# How home charging is scheduled before SOC is derived from discharge + charge.

def build_hourly_timestamps(start_date: datetime, end_date: datetime) -> pl.DataFrame:
    """Build hourly timestamps for the simulation window (inclusive, aligned to whole hours).

    Uses the clock hour of ``start_date`` / ``end_date`` (minutes/seconds cleared).
    Typical NHTS-aligned year: ``2018-01-01 04:00`` through ``2019-01-01 03:00``
    (last hour slot covering 03:00–03:59; 2018 matches AMY weather years).

    Args:
        start_date: Start of the range (hour included)
        end_date: End of the range (hour included)

    Returns:
        pl.DataFrame: hourly timestamps from ``start_date`` through ``end_date``

    Raises:
        ValueError: If ``end_date`` is before ``start_date``
    """
    start_hour = start_date.replace(minute=0, second=0, microsecond=0)
    end_hour = end_date.replace(minute=0, second=0, microsecond=0)
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
    """Build the hourly calendar for the simulation window, used for trip-to-hour joins.

    Args:
        start_date: Start of the range (hour included)
        end_date: End of the range (hour included)

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


def expand_trips_to_away_hour_rows(
    trip_schedules: pl.DataFrame,
    *,
    prefix: str = "trip",
) -> pl.DataFrame:
    """Expand each trip/tour interval to one row per hour in ``[dep, arr)``.

    Adds calendar ``date`` / ``hour`` columns for every hour in the interval.
    Overnight spans emit hours on both calendar days.

    Column names are ``{prefix}_departure_date``, ``{prefix}_departure_hour``,
    ``{prefix}_arrival_date``, ``{prefix}_arrival_hour`` (``prefix`` is ``trip``
    for drive legs or ``tour`` for home-away windows).

    Args:
        trip_schedules (pl.DataFrame): Schedules with prefixed interval columns
        prefix (str): ``trip`` (discharge) or ``tour`` (presence)

    Returns:
        pl.DataFrame: Expanded rows with calendar ``date`` / ``hour``
    """
    # Nothing to expand.
    if trip_schedules.is_empty():
        return trip_schedules

    dep_date = f"{prefix}_departure_date"
    dep_hour = f"{prefix}_departure_hour"
    arr_date = f"{prefix}_arrival_date"
    arr_hour = f"{prefix}_arrival_hour"
    required = {dep_date, dep_hour, arr_date, arr_hour}
    missing = required - set(trip_schedules.columns)
    if missing:
        raise ValueError(f"trip_schedules missing columns: {sorted(missing)}")

    # Normalize join keys to date-only (drop any time component).
    trips = trip_schedules.with_columns(
        pl.col(dep_date).cast(pl.Date).alias(dep_date),
        pl.col(arr_date).cast(pl.Date).alias(arr_date),
    )

    # Same-calendar-day intervals: hours range(dep, arr) on that date.
    same_day = trips.filter(pl.col(arr_date) == pl.col(dep_date)).with_columns(
        pl.int_ranges(pl.col(dep_hour), pl.col(arr_hour)).alias("hour"),
        pl.col(dep_date).alias("date"),
    )

    # Overnight intervals: dep_hour..23 on departure date, then 0..arr_hour on arrival date.
    overnight = trips.filter(pl.col(arr_date) > pl.col(dep_date))

    frames: list[pl.DataFrame] = []
    if same_day.height > 0:
        frames.append(same_day.explode("hour"))
    if overnight.height > 0:
        # Hours after departure on the leave day (up to midnight).
        dep_part = overnight.with_columns(
            pl.int_ranges(pl.col(dep_hour), pl.lit(24)).alias("hour"),
            pl.col(dep_date).alias("date"),
        ).explode("hour")
        # Hours before arrival on the return day (from midnight).
        arr_part = overnight.with_columns(
            pl.int_ranges(pl.lit(0), pl.col(arr_hour)).alias("hour"),
            pl.col(arr_date).alias("date"),
        ).explode("hour")
        frames.append(dep_part)
        frames.append(arr_part)

    # Degenerate case: all intervals filtered out (e.g. arr == dep with no overnight).
    if not frames:
        return trips.clear().with_columns(
            pl.lit(None).cast(pl.Date).alias("date"),
            pl.lit(None).cast(pl.Int64).alias("hour"),
        )

    return pl.concat(frames, how="diagonal_relaxed")


def expand_trip_away_hours(
    trip_schedules: pl.DataFrame,
    *,
    prefix: str = "tour",
) -> pl.DataFrame:
    """Expand interval rows into unique per-hour away markers on the calendar timeline.

    Defaults to ``prefix='tour'`` (presence). Pass ``prefix='trip'`` only if expanding
    drive intervals for diagnostics.

    Args:
        trip_schedules (pl.DataFrame): Intervals with ``{prefix}_departure/arrival_*``
        prefix (str): Column-name prefix for the interval bounds

    Returns:
        pl.DataFrame: DataFrame with ``bldg_id``, ``vehicle_id``, ``date``, ``hour``
    """
    if trip_schedules.is_empty():
        return pl.DataFrame(
            schema={
                "bldg_id": trip_schedules.schema.get("bldg_id", pl.Utf8),
                "vehicle_id": trip_schedules.schema.get("vehicle_id", pl.Int64),
                "date": pl.Date,
                "hour": pl.Int64,
            }
        )

    return (
        expand_trips_to_away_hour_rows(trip_schedules, prefix=prefix)
        .select("bldg_id", "vehicle_id", "date", "hour")
        .unique()  # overlapping intervals (e.g. multi-leg same hour) → one away marker
    )


def tours_from_trip_schedules(trip_schedules: pl.DataFrame) -> pl.DataFrame:
    """Derive unique home-away tour intervals from trip schedule rows.

    When tour columns are present (``tour_departure_*`` / ``tour_arrival_*``),
    each distinct tour becomes one away window (columns kept as ``tour_*``).

    When tour columns are absent, each trip row's ``trip_*`` bounds are copied to
    ``tour_*`` so fixtures without explicit tours still work.

    Args:
        trip_schedules (pl.DataFrame): Trip schedules with ``trip_*`` and ``tour_*`` columns

    Returns:
        pl.DataFrame: Tour schedules with ``tour_*`` columns
    """
    if trip_schedules.is_empty():
        return trip_schedules

    tour_cols = {
        "tour_departure_date",
        "tour_departure_hour",
        "tour_arrival_date",
        "tour_arrival_hour",
    }
    if tour_cols.issubset(trip_schedules.columns):
        # Prefer travel_date as the tour grouping key; fall back to legacy "date".
        key_cols = ["bldg_id", "vehicle_id", "travel_date"]
        if "travel_date" not in trip_schedules.columns and "date" in trip_schedules.columns:
            key_cols = ["bldg_id", "vehicle_id", "date"]
        if "tour_id" in trip_schedules.columns:
            key_cols.append("tour_id")
        keep = [c for c in key_cols if c in trip_schedules.columns]
        # One row per distinct tour window (legs that share a tour collapse here).
        return trip_schedules.select(
            *keep,
            "tour_departure_date",
            "tour_departure_hour",
            "tour_arrival_date",
            "tour_arrival_hour",
        ).unique()

    # Legacy / fixture: treat drive interval as the away window.
    trip_cols = {
        "trip_departure_date",
        "trip_departure_hour",
        "trip_arrival_date",
        "trip_arrival_hour",
    }
    if not trip_cols.issubset(trip_schedules.columns):
        raise ValueError(
            "trip_schedules need tour_* columns or trip_* interval columns; "
            f"got {sorted(trip_schedules.columns)}"
        )
    return trip_schedules.rename({
        "trip_departure_date": "tour_departure_date",
        "trip_departure_hour": "tour_departure_hour",
        "trip_arrival_date": "tour_arrival_date",
        "trip_arrival_hour": "tour_arrival_hour",
    })


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

    # Greedy forward sim: each hour, apply trip draw then maybe charge to full.
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


def _emergency_peak_needed(
    *,
    hour_idx: int,
    current_soc: float,
    at_home: np.ndarray,
    discharge_kwh: np.ndarray,
    is_off_peak: np.ndarray,
    charger_power_kw: float,
    battery_capacity_kwh: float,
    lookahead_hours: int = EMERGENCY_PEAK_LOOKAHEAD_HOURS,
) -> bool:
    """Return whether skipping this peak hour causes an SOC shortfall within the horizon.

    Simulate the normal off-peak-immediate policy over the next ``lookahead_hours``:
    discharge first, then charge at available off-peak home hours. Future on-peak
    emergency charging is intentionally excluded from this forecast, so the current
    hour begins charging when any cumulative trip demand in the next ~2 days would
    otherwise underflow.
    """
    # Start the what-if SOC from the value after this hour's discharge (already applied
    # by the caller). We ask: "if we do *not* charge now, does the next horizon break?"
    forecast_soc = current_soc
    # Exclusive end of the forecast window; clamp at the end of the simulation.
    horizon_end = min(len(discharge_kwh), hour_idx + 1 + lookahead_hours)

    # Replay future hours under the *strict* off-peak policy (no nested emergency).
    for future_idx in range(hour_idx + 1, horizon_end):
        # Same discharge-first rule as compute_hourly_soc / schedule_* loops.
        trip_draw = float(discharge_kwh[future_idx])
        # Any trip that exceeds forecast SOC → this peak hour should start charging now.
        if trip_draw > forecast_soc + 1e-9:
            return True
        forecast_soc -= trip_draw

        # Only credit future *off-peak* home charging. Counting future emergency peak
        # hours here would circularly assume help that only arrives if we already fire.
        if at_home[future_idx] and is_off_peak[future_idx]:
            forecast_soc += min(
                charger_power_kw,
                battery_capacity_kwh - forecast_soc,
            )

    # Horizon survived without underflow → no need to break the peak window this hour.
    return False


def schedule_off_peak_immediate_charging(
    at_home: np.ndarray,
    discharge_kwh: np.ndarray,
    *,
    is_off_peak: np.ndarray,
    battery_capacity_kwh: float,
    charger_power_kw: float,
    initial_soc_kwh: float,
    allow_emergency_peak_charging: bool = False,
) -> np.ndarray:
    """
    TOU Immediate: charge at max power whenever home and off-peak until the pack is full.

    Matches Jones et al. (Energy Reports 2022) TOU Immediate: avoid on-peak charging and
    begin max-power charging as soon as off-peak hours coincide with dwelling. Unlike
    ``schedule_off_peak_charging``, this fills toward full capacity (not ``SOC_req``).

    When ``allow_emergency_peak_charging`` is True, on-peak home hours may charge if a
    48-hour forecast using only future off-peak home charging would underflow on any trip.

    Args:
        at_home: Whether the vehicle is home at the start of each hour
        discharge_kwh: Fixed trip draw ``x_t^DB`` each hour (kWh)
        is_off_peak: Off-peak mask aligned with ``at_home``
        battery_capacity_kwh: Battery capacity ``K^B`` (kWh)
        charger_power_kw: Max charge rate ``C^B`` when home (kW = kWh/hour)
        initial_soc_kwh: Start-of-hour-0 SOC ``s_0`` (kWh)
        allow_emergency_peak_charging: If True, allow on-peak charging when a 48-hour
            forecast shows an energy shortfall on any upcoming trip

    Returns:
        Hourly charge energy ``x_t^CB`` (kWh), same length as ``at_home``
    """
    if len(at_home) != len(discharge_kwh):
        raise ValueError(
            f"at_home and discharge_kwh must have the same length, got {len(at_home)} and {len(discharge_kwh)}"
        )
    if len(is_off_peak) != len(at_home):
        raise ValueError(
            f"is_off_peak and at_home must have the same length, got {len(is_off_peak)} and {len(at_home)}"
        )

    num_hours = len(at_home)
    charge_kwh = np.zeros(num_hours, dtype=np.float64)
    current_soc = initial_soc_kwh

    # Same greedy loop as immediate, but charge only off-peak (or emergency peak).
    for hour_idx in range(num_hours):
        # Apply this hour's trip draw before deciding whether to charge.
        trip_draw = discharge_kwh[hour_idx]
        if trip_draw > current_soc:
            # Underflow: clamp to empty (public charging assumed for the shortfall).
            current_soc = 0.0
        else:
            current_soc -= trip_draw

        # Away or already full → no home charging this hour.
        if not at_home[hour_idx] or current_soc >= battery_capacity_kwh:
            continue

        # Default TOU Immediate rule: charge only in off-peak hours.
        may_charge = bool(is_off_peak[hour_idx])
        # Emergency override: if this is an on-peak home hour, look ahead
        # EMERGENCY_PEAK_LOOKAHEAD_HOURS under a strict off-peak-only policy. If any
        # trip in that window would underflow, break the peak window and charge now.
        if (
            not may_charge
            and allow_emergency_peak_charging
            and _emergency_peak_needed(
                hour_idx=hour_idx,
                current_soc=current_soc,
                at_home=at_home,
                discharge_kwh=discharge_kwh,
                is_off_peak=is_off_peak,
                charger_power_kw=charger_power_kw,
                battery_capacity_kwh=battery_capacity_kwh,
            )
        ):
            may_charge = True

        # Charge at max power toward a full pack (not toward an SOC_req floor).
        if may_charge:
            added = min(charger_power_kw, battery_capacity_kwh - current_soc)
            if added > 0.0:
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

    Vehicles with no trip draw at all skip the LP and return all-zero schedules. Because
    the LP is always feasible, a solver error signals numerical trouble rather than an
    infeasible model, so solvers in ``COST_MIN_LP_SOLVERS`` are tried in order.

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

    # Fixed inputs for the LP.
    discharge = np.asarray(discharge_kwh, dtype=np.float64)
    prices = np.asarray(hourly_price_usd_per_kwh, dtype=np.float64)
    max_charge = np.where(at_home, charger_power_kw, 0.0)  # cannot charge while away
    s_0 = float(initial_soc_kwh)

    # A vehicle that never draws energy (idle inventory vehicle matched to empty NHTS
    # templates, or a template whose legs all report 0 miles) needs no charge: prices are
    # non-negative and there is no terminal SOC requirement, so charge = shed = 0 is
    # optimal. Short-circuit instead of handing the solver an all-degenerate LP where
    # every shed variable is pinned to a zero-width box.
    if not np.any(discharge > 0.0):
        return np.zeros(num_hours, dtype=np.float64), np.zeros(num_hours, dtype=np.float64)

    # Shed penalty: huge default ⇒ shed only when the LP would otherwise be infeasible.
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

    # Decision variables: charge x^CB, end-of-hour SOC s, shed load x^SL.
    charge = cp.Variable(num_hours, name="charge")
    soc = cp.Variable(num_hours, name="soc")  # s after each hour's charge/discharge
    shed = cp.Variable(num_hours, name="shed_load")

    # SOC balance, charger limits, battery box constraints.
    constraints: list[cp.Constraint] = [
        soc[0] == s_0 + charge[0] - discharge[0] + shed[0],
        charge >= 0,
        charge <= max_charge,
        shed >= 0,
        shed <= discharge,
        soc >= 0,
        soc <= battery_capacity_kwh,
    ]
    if num_hours > 1:
        constraints.append(soc[1:] == soc[:-1] + charge[1:] - discharge[1:] + shed[1:])

    # Minimize energy cost + shed penalties (perfect foresight).
    objective = prices @ charge + shed_penalties @ shed
    problem = cp.Problem(cp.Minimize(objective), constraints)

    # The LP is always feasible (shedding covers any trip draw), so a solver error means
    # numerical trouble rather than an infeasible model — retry with a different solver.
    failures: list[str] = []
    for solver in COST_MIN_LP_SOLVERS:
        if solver not in cp.installed_solvers():
            continue
        try:
            problem.solve(solver=solver)
        except SolverError as exc:
            failures.append(f"{solver}: {exc}")
            continue
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            break
        failures.append(f"{solver}: {problem.status}")
    else:
        raise RuntimeError(
            "Cost-minimizing charging LP failed on all solvers: " + "; ".join(failures)
        )

    # Clip solver round-off (e.g. -1e-12) so downstream sums stay physical.
    charge_kwh = np.clip(np.asarray(charge.value, dtype=np.float64).reshape(-1), 0.0, None)
    shed_load_kwh = np.clip(np.asarray(shed.value, dtype=np.float64).reshape(-1), 0.0, None)
    return charge_kwh, shed_load_kwh


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the n-th weekday in month (weekday: Mon=0 … Sun=6; n is 1-based)."""
    d = date(year, month, 1)  # start at the 1st of the month
    # Move forward 0–6 days until we land on the requested weekday.
    d += timedelta(days=(weekday - d.weekday()) % 7)
    # Jump (n-1) full weeks forward (n=1 → first such weekday, n=3 → third, …).
    return d + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Return the last weekday in month (weekday: Mon=0 … Sun=6)."""
    # Start at the last calendar day of the month.
    if month == 12:
        d = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    # Walk backward 0–6 days until we hit the requested weekday.
    d -= timedelta(days=(d.weekday() - weekday) % 7)
    return d


def _easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm for Western Easter Sunday."""
    # Meeus/Jones/Butcher coefficients → month/day of Easter Sunday.
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month = (h + el - 7 * m + 114) // 31
    day = ((h + el - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _observed_fixed_holiday(year: int, month: int, day: int) -> date:
    """Fixed-date holiday; if it falls on Sunday, observe the following Monday (BGE)."""
    d = date(year, month, day)
    if d.weekday() == 6:  # Sunday → BGE observes the Monday after
        return d + timedelta(days=1)
    return d


def bge_schedule_ev_holidays(year: int) -> set[date]:
    """BGE Schedule EV off-peak holidays for a calendar year (no DST adjustments).

    New Year's Day, Presidents' Day, Good Friday, Memorial Day, Independence Day,
    Labor Day, Thanksgiving, Christmas; Monday following when a listed holiday
    falls on Sunday (per Schedule EV).
    """
    holidays = {
        _observed_fixed_holiday(year, 1, 1),  # New Year's Day (Jan 1, or Mon if Sun)
        _nth_weekday(year, 2, 0, 3),  # Presidents' Day — 3rd Monday in February
        _easter_sunday(year) - timedelta(days=2),  # Good Friday = Easter − 2 days
        _last_weekday(year, 5, 0),  # Memorial Day — last Monday in May
        _observed_fixed_holiday(year, 7, 4),  # Independence Day (Jul 4, or Mon if Sun)
        _nth_weekday(year, 9, 0, 1),  # Labor Day — 1st Monday in September
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving — 4th Thursday in November
        _observed_fixed_holiday(year, 12, 25),  # Christmas (Dec 25, or Mon if Sun)
    }
    return holidays


def build_is_off_peak(
    hours_base: pl.DataFrame,
    *,
    peak_clock_hours: Iterable[int] | None = None,
    peak_clock_hours_summer: Iterable[int] | None = None,
    peak_clock_hours_winter: Iterable[int] | None = None,
    summer_months: Iterable[int] = DEFAULT_TOU_SUMMER_MONTHS,
    weekends_off_peak: bool = False,
    holidays_off_peak: bool = False,
) -> np.ndarray:
    """Return a boolean mask that is True during off-peak hours.

    Year-round mode (legacy): pass ``peak_clock_hours`` only — every day uses that
    on-peak set (weekends included unless ``weekends_off_peak``).

    Seasonal mode (e.g. BGE Schedule EV): pass ``peak_clock_hours_summer`` and
    ``peak_clock_hours_winter``. Dates whose month is in ``summer_months`` use the
    summer set; all other months use winter. When both seasonal sets and
    ``peak_clock_hours`` are set, seasonal wins.

    Args:
        hours_base: Hourly calendar with ``date`` and ``hour`` columns
        peak_clock_hours: Year-round on-peak clock hours (0-23)
        peak_clock_hours_summer: Summer on-peak clock hours
        peak_clock_hours_winter: Non-summer on-peak clock hours
        summer_months: Months (1-12) treated as summer (default Jun–Sep)
        weekends_off_peak: If True, Saturday/Sunday are entirely off-peak
        holidays_off_peak: If True, BGE Schedule EV holidays are entirely off-peak

    Returns:
        Boolean mask length ``hours_base.height`` (True = off-peak)
    """
    # ---- choose seasonal vs year-round peak sets ----
    use_seasonal = (
        peak_clock_hours_summer is not None or peak_clock_hours_winter is not None
    )
    if use_seasonal:
        # Both seasons required; refuse a half-configured seasonal TOU.
        if peak_clock_hours_summer is None or peak_clock_hours_winter is None:
            raise ValueError(
                "peak_clock_hours_summer and peak_clock_hours_winter must both be set "
                "for seasonal TOU windows"
            )
        summer_peak = set(int(h) for h in peak_clock_hours_summer)
        winter_peak = set(int(h) for h in peak_clock_hours_winter)
        for label, peak in (("summer", summer_peak), ("winter", winter_peak)):
            if not peak:
                raise ValueError(f"peak_clock_hours_{label} must be non-empty")
            if any(h < 0 or h > 23 for h in peak):
                raise ValueError(
                    f"peak_clock_hours_{label} must be clock hours in 0–23; got {sorted(peak)}"
                )
    else:
        # Legacy path: one on-peak set for every calendar day.
        if peak_clock_hours is None:
            peak_clock_hours = DEFAULT_PEAK_CLOCK_HOURS
        year_round_peak = set(int(h) for h in peak_clock_hours)
        if not year_round_peak:
            raise ValueError("peak_clock_hours must be a non-empty list")
        if any(h < 0 or h > 23 for h in year_round_peak):
            raise ValueError(
                f"peak_clock_hours must be clock hours in 0–23; got {sorted(year_round_peak)}"
            )

    summer_month_set = set(int(m) for m in summer_months)
    if any(m < 1 or m > 12 for m in summer_month_set):
        raise ValueError(f"summer_months must be in 1–12; got {sorted(summer_month_set)}")

    # Pull per-row calendar date and clock hour from the simulation grid.
    dates = hours_base["date"].to_list()
    clock_hours = hours_base["hour"].to_numpy()

    # Precompute holiday dates for every year touched by the window (±1 for edges).
    holiday_years = {d.year for d in dates} | {d.year + 1 for d in dates} | {
        d.year - 1 for d in dates
    }
    holiday_set: set[date] = set()
    if holidays_off_peak:
        for year in holiday_years:
            holiday_set |= bge_schedule_ev_holidays(year)

    # ---- classify each simulation hour ----
    is_off = np.empty(len(clock_hours), dtype=bool)
    for i, (d, hour) in enumerate(zip(dates, clock_hours, strict=True)):
        # Sat/Sun (weekday 5/6) → entire day off-peak when enabled.
        if weekends_off_peak and d.weekday() >= 5:
            is_off[i] = True
            continue
        # BGE Schedule EV holiday → entire day off-peak when enabled.
        if holidays_off_peak and d in holiday_set:
            is_off[i] = True
            continue
        # Pick the on-peak set for this date's season (or year-round legacy set).
        if use_seasonal:
            peak = summer_peak if d.month in summer_month_set else winter_peak
        else:
            peak = year_round_peak
        # Off-peak iff this clock hour is not in the on-peak set.
        is_off[i] = int(hour) not in peak
    return is_off

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

    # --- Daily energy need (from discharge_kwh already spread onto drive hours) ---
    daily_discharge_kwh: dict[date, float] = {}
    for day, discharge in zip(dates, discharge_kwh, strict=True):
        daily_discharge_kwh[day] = daily_discharge_kwh.get(day, 0.0) + float(discharge)

    # --- Per calendar day: (first_departure, last_arrival) from tour windows ---
    # Overnight tours: last_arrival=24 on the leave day; arrival hour lands on next day.
    daily_bounds: dict[date, tuple[int, int]] = {}
    if not vehicle_trips.is_empty():
        required = {
            "tour_departure_date",
            "tour_departure_hour",
            "tour_arrival_date",
            "tour_arrival_hour",
        }
        bound_trips = tours_from_trip_schedules(vehicle_trips)
        missing = required - set(bound_trips.columns)
        if missing:
            raise ValueError(f"vehicle_trips missing columns: {sorted(missing)}")
        trips = bound_trips.with_columns(
            pl.col("tour_departure_date").cast(pl.Date).alias("tour_departure_date"),
            pl.col("tour_arrival_date").cast(pl.Date).alias("tour_arrival_date"),
        )

        # Earliest leave-home hour each calendar day.
        first_dep = (
            trips.group_by("tour_departure_date")
            .agg(pl.col("tour_departure_hour").min().alias("first_departure"))
            .rename({"tour_departure_date": "date"})
        )
        # Same-day returns: last home-arrival hour that day.
        same_day_arr = (
            trips.filter(pl.col("tour_arrival_date") == pl.col("tour_departure_date"))
            .group_by("tour_arrival_date")
            .agg(pl.col("tour_arrival_hour").max().alias("last_arrival"))
            .rename({"tour_arrival_date": "date"})
        )
        # Overnight leave day: still away at midnight → last_arrival sentinel 24.
        overnight_dep = (
            trips.filter(pl.col("tour_arrival_date") > pl.col("tour_departure_date"))
            .select(pl.col("tour_departure_date").alias("date"))
            .unique()
            .with_columns(pl.lit(24).cast(pl.Int64).alias("last_arrival"))
        )
        # Overnight return day: arrival hour is a last_arrival candidate.
        next_day_arr = (
            trips.filter(pl.col("tour_arrival_date") > pl.col("tour_departure_date"))
            .group_by("tour_arrival_date")
            .agg(pl.col("tour_arrival_hour").max().alias("last_arrival"))
            .rename({"tour_arrival_date": "date"})
        )
        last_arr_parts = [df for df in (same_day_arr, overnight_dep, next_day_arr) if df.height > 0]
        if last_arr_parts:
            last_arr = (
                pl.concat(last_arr_parts)
                .group_by("date")
                .agg(pl.col("last_arrival").max().alias("last_arrival"))
            )
        else:
            last_arr = pl.DataFrame({"date": [], "last_arrival": []})

        bounds_frame = first_dep.join(last_arr, on="date", how="full", coalesce=True)
        for row in bounds_frame.iter_rows(named=True):
            first = int(row["first_departure"]) if row["first_departure"] is not None else 24
            last = int(row["last_arrival"]) if row["last_arrival"] is not None else 0
            daily_bounds[row["date"]] = (first, last)

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

    # --- Mark each hour: in Window_avail? toward which day's SOC_req? ---
    for hour_idx, (day, clock_hour) in enumerate(zip(dates, clock_hours, strict=True)):
        # Default (no tours that day): (24, 0) → entire day is Window_avail.
        first_departure, last_arrival = daily_bounds.get(day, (24, 0))
        in_morning_window = clock_hour < first_departure  # before first leave-home
        in_evening_window = clock_hour >= last_arrival  # after last return-home

        # Midday block between first leave and last return is outside Window_avail
        # (includes mid-tour home stops — TOU primary rule does not charge then).
        if not (in_morning_window or in_evening_window):
            continue

        # Which departure day is this hour preparing for?
        if in_morning_window and in_evening_window:
            target_day = day  # no-tour day: both windows cover all 24 hours
        elif in_morning_window:
            target_day = day  # pre-departure → today's trips
        else:
            target_day = day + timedelta(days=1)  # post-arrival → tomorrow's trips

        soc_target_kwh[hour_idx] = soc_req_kwh_for_day(target_day)
        # Window_off = Window_avail ∩ off-peak ∩ at_home (no emergency peak here).
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
        # Record SOC at the start of the hour (before this hour's draw/charge).
        soc_kwh[hour_idx] = current_soc

        trip_draw = discharge_kwh[hour_idx]
        if trip_draw > current_soc + 1e-9:
            soc_underflow[hour_idx] = True
            current_soc = 0.0
        else:
            current_soc -= trip_draw

        # Charge after discharge within the hour (same order as schedule_* policies).
        current_soc += charge_kwh[hour_idx]

    return soc_kwh, soc_underflow


def is_home_charging_soc_feasible(
    at_home: np.ndarray,
    discharge_kwh: np.ndarray,
    *,
    battery_capacity_kwh: float,
    charger_power_kw: float,
    initial_soc_kwh: float | None = None,
    buffer_fraction: float = 0.0,
) -> bool:
    """
    Return whether home charging at ``charger_power_kw`` can cover the trip schedule.

    Uses perfect foresight of the full presence / discharge path. Immediate charging
    maximizes SOC at every hour among home-only policies with the same power cap, so
    underflow under immediate charging is necessary and sufficient for infeasibility
    of *any* home charging schedule at that power.

    ``buffer_fraction`` inflates every hour's discharge by ``(1 + buffer)`` for the
    check (charger headroom analogous to battery ``capacity_buffer_fraction``).

    Args:
        at_home: Hourly home/away mask
        discharge_kwh: Hourly trip draw (kWh)
        battery_capacity_kwh: Pack size ``K^B`` (kWh)
        charger_power_kw: Max home charge rate ``C^B`` (kW)
        initial_soc_kwh: Start SOC; ``None`` → start full at ``battery_capacity_kwh``
        buffer_fraction: Extra fraction of discharge the charger must cover

    Returns:
        True when immediate charging has no SOC underflow under buffered discharge
    """
    if buffer_fraction < 0:
        raise ValueError(f"buffer_fraction must be >= 0, got {buffer_fraction}")
    if charger_power_kw < 0:
        raise ValueError(f"charger_power_kw must be >= 0, got {charger_power_kw}")
    if battery_capacity_kwh < 0:
        raise ValueError(f"battery_capacity_kwh must be >= 0, got {battery_capacity_kwh}")

    start_soc = battery_capacity_kwh if initial_soc_kwh is None else float(initial_soc_kwh)
    if not 0.0 <= start_soc <= battery_capacity_kwh + 1e-9:
        raise ValueError(
            f"initial_soc_kwh must be within [0, {battery_capacity_kwh}], got {start_soc}"
        )

    # Inflate trip energy by (1+buffer) so the charger must cover headroom, not just
    # the raw discharge (same role as capacity_buffer_fraction for packs).
    buffered_discharge = np.asarray(discharge_kwh, dtype=np.float64) * (1.0 + buffer_fraction)

    # Immediate charging is SOC-maximal among home-only policies at this power, so
    # underflow here ⇒ no feasible home schedule exists (perfect-foresight oracle).
    charge_kwh = schedule_immediate_charging(
        np.asarray(at_home, dtype=bool),
        buffered_discharge,
        battery_capacity_kwh=float(battery_capacity_kwh),
        charger_power_kw=float(charger_power_kw),
        initial_soc_kwh=start_soc,
    )
    # Recompute SOC to get per-hour underflow flags (schedule_* returns charge only).
    _, soc_underflow = compute_hourly_soc(
        buffered_discharge,
        charge_kwh,
        initial_soc_kwh=start_soc,
    )
    return not bool(np.any(soc_underflow))

