"""Build home-based tours and driving legs from NHTS vehicle-day trip rows.

NHTS rows are person-trip *legs* (e.g. home→work, work→shop, shop→home). Charging
needs two different timelines derived from the same legs:

- **Tours** — leave-home → return-home windows. The vehicle is away from the
  residence for the whole tour (including parked mid-tour), so home charging is
  off. Presence / ``at_home`` uses tours.
- **Trips (legs)** — actual driving intervals with miles. Discharge kWh (and
  temperature-dependent efficiency) applies only while the wheels are turning.

Resolution order (important):
1. Chain tours at NHTS **minute** resolution using ``STRTTIME`` / ``ENDTIME`` (HHMM)
   and purpose codes ``WHYFROM`` / ``WHYTO``.
2. Only then snap tour and leg bounds to **clock hours** for the hourly SOC grid.

Tours are **not** split by dwell length. Sitting at work for eight hours is still
one tour; ``DWELTIME`` is ignored. A tour ends only when a leg's destination is
home (``WHYTO`` ∈ {1, 2}) or the travel day runs out of legs.

Home purposes follow the 2022 NHTS codebook ``WHYFROM`` / ``WHYTO``:
``01`` regular activities at home, ``02`` work from home (paid).

``TripProfile`` is the single day-template type used from NHTS parse through
schedule generation (weekday/weekend slots on a ``VehicleProfile``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# NHTS WHYFROM / WHYTO codes that mean the vehicle is at the residence.
NHTS_HOME_PURPOSES: frozenset[int] = frozenset({1, 2})


def is_nhts_home_purpose(why: int | None) -> bool:
    """Return True when a WHYFROM/WHYTO code is a home location.
    
    Args:
        why: The why code

    Returns:
        True if the why code is a home location, False otherwise
    """
    if why is None:
        return False
    return int(why) in NHTS_HOME_PURPOSES


def nhts_departure_hour(start_time: int) -> int:
    """Clock hour when a drive leg starts (vehicle is driving during this hour).

    NHTS ``STRTTIME`` is HHMM; e.g. 830 → hour 8.

    Args:
        start_time: The start time in HHMM format

    Returns:
        The departure hour
    """
    return int(start_time) // 100


def nhts_arrival_hour(end_time: int) -> int:
    """First clock hour after a drive leg ends (exclusive end of the drive interval).

    NHTS ``ENDTIME`` is HHMM. Exact hour → done at that hour; mid-hour → next hour.
    Drive hours are ``range(departure_hour, arrival_hour)``.

    Args:
        end_time: The end time in HHMM format

    Returns:
        The arrival hour
    """
    end_time = int(end_time)
    hour, minute = divmod(end_time, 100)
    return hour if minute == 0 else hour + 1


@dataclass
class TripProfile:
    """One NHTS daily template: driving legs plus home-based tours.

    Construct via ``build_tours_from_legs`` (real NHTS) or ``trips_as_singleton_tours``
    (fixtures / one-tour-per-leg). Do not leave tour fields empty.

    **Trips** (parallel ``trip_*`` lists): drive intervals for discharge + temperature.
    **Tours** (parallel ``tour_*`` lists): leave-home → return-home for presence.
    ``tour_ids[i]`` links trip ``i`` to a 1-based index into the tour_* lists.
    """

    # --- Driving legs (discharge + temperature) ---
    trip_departure_hours: list[int] = field(default_factory=list)
    trip_arrival_hours: list[int] = field(default_factory=list)
    trip_miles_driven: list[float] = field(default_factory=list)
    trip_weights: list[float] = field(default_factory=list)
    trip_ids: list[int] = field(default_factory=list)
    tour_ids: list[int] = field(default_factory=list)

    # --- Home-away tours (presence / at_home) ---
    tour_departure_hours: list[int] = field(default_factory=list)
    tour_arrival_hours: list[int] = field(default_factory=list)
    tour_ends_away: list[bool] = field(default_factory=list)

    @property
    def has_trips(self) -> bool:
        return len(self.trip_miles_driven) > 0


def trips_as_singleton_tours(
    *,
    trip_departure_hours: list[int],
    trip_arrival_hours: list[int],
    trip_miles_driven: list[float],
    trip_weights: list[float],
    trip_ids: list[int] | None = None,
) -> TripProfile:
    """Treat each driving leg as its own home-away tour.

    Used for unit-test fixtures and NHTS rows without purpose columns.
    Inputs are already clock hours.
    """
    n = len(trip_departure_hours)
    if not (len(trip_arrival_hours) == len(trip_miles_driven) == len(trip_weights) == n):
        raise ValueError(
            "trip_departure_hours, trip_arrival_hours, trip_miles_driven, "
            "and trip_weights must have equal length"
        )
    ids = trip_ids if trip_ids is not None else list(range(1, n + 1))
    if len(ids) != n:
        raise ValueError("trip_ids must match trip_departure_hours length")

    return TripProfile(
        trip_departure_hours=list(trip_departure_hours),
        trip_arrival_hours=list(trip_arrival_hours),
        trip_miles_driven=list(trip_miles_driven),
        trip_weights=list(trip_weights),
        trip_ids=list(ids),
        tour_ids=list(ids),
        tour_departure_hours=list(trip_departure_hours),
        tour_arrival_hours=list(trip_arrival_hours),
        tour_ends_away=[False] * n,
    )


def build_tours_from_legs(
    *,
    start_times: list[int],
    end_times: list[int],
    trip_miles_driven: list[float],
    trip_weights: list[float],
    why_from: list[int],
    why_to: list[int],
) -> TripProfile:
    """Chain NHTS legs into a ``TripProfile`` (minute tours, then hourly snap).

    Algorithm (vehicle already filtered to HH-vehicle driver trips):

    1. Sort legs by NHTS ``STRTTIME`` / ``ENDTIME`` (HHMM).
    2. Start a tour at the current leg.
    3. Keep appending while ``WHYTO`` is not home — long mid-tour dwells stay
       in the same tour (``DWELTIME`` ignored).
    4. Close on first home destination (or end of day).
    5. Record minute-level bounds, then convert to clock hours.

    Args:
        start_times: List of start times for each leg
        end_times: List of end times for each leg
        trip_miles_driven: List of miles driven for each leg
        trip_weights: List of trip weights for each leg
        why_from: List of why from codes for each leg
        why_to: List of why to codes for each leg

    Returns:
        A ``TripProfile`` object
    """
    n = len(start_times)
    if n == 0:
        return TripProfile()

    lengths = [
        len(end_times),
        len(trip_miles_driven),
        len(trip_weights),
        len(why_from),
        len(why_to),
    ]
    if any(length != n for length in lengths):
        raise ValueError("all leg arrays must have the same length")

    order = sorted(range(n), key=lambda i: (int(start_times[i]), int(end_times[i]), i))

    leg_start_hhmm: list[int] = []
    leg_end_hhmm: list[int] = []
    leg_miles: list[float] = []
    leg_weights: list[float] = []
    leg_tour_ids: list[int] = []
    tour_start_hhmm: list[int] = []
    tour_end_hhmm: list[int] = []
    tour_ends_away: list[bool] = []

    tour_number = 0
    idx = 0
    while idx < n:
        tour_number += 1
        # put together trips on the same tour
        tour_start = idx
        # Non-home destinations (work, shop, …) stay on this tour regardless of dwell.
        while idx < n - 1 and not is_nhts_home_purpose(why_to[order[idx]]):
            idx += 1
        tour_end = idx # inclusive
        ends_away = not is_nhts_home_purpose(why_to[order[tour_end]])

        first = order[tour_start]
        last = order[tour_end]
        tour_start_hhmm.append(int(start_times[first]))
        tour_end_hhmm.append(int(end_times[last]))
        tour_ends_away.append(ends_away)

        for leg in range(tour_start, tour_end + 1):
            j = order[leg]
            leg_start_hhmm.append(int(start_times[j]))
            leg_end_hhmm.append(int(end_times[j]))
            leg_miles.append(float(trip_miles_driven[j]))
            leg_weights.append(float(trip_weights[j]))
            leg_tour_ids.append(tour_number)

        idx = tour_end + 1

    return TripProfile(
        trip_departure_hours=[nhts_departure_hour(t) for t in leg_start_hhmm],
        trip_arrival_hours=[nhts_arrival_hour(t) for t in leg_end_hhmm],
        trip_miles_driven=leg_miles,
        trip_weights=leg_weights,
        trip_ids=list(range(1, len(leg_miles) + 1)),
        tour_ids=leg_tour_ids,
        tour_departure_hours=[nhts_departure_hour(t) for t in tour_start_hhmm],
        tour_arrival_hours=[nhts_arrival_hour(t) for t in tour_end_hhmm],
        tour_ends_away=tour_ends_away,
    )
