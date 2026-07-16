"""Assign ResStock 2025 EV battery class, usable capacity, and efficiency.

ResStock samples each dwelling's EV type from a national BEV stock distribution
(Experian 2023 registrations via TEMPO), then looks up Autonomie parameters for
usable battery capacity (kWh) and combined fuel economy (kWh/mile).

In this pipeline we apply a **stock-conditional** variant: trip schedules are
generated first, then each vehicle draws from the national option shares
restricted to packs that can cover its max daily miles (plus a reserve buffer).
If no option is feasible, assignment raises.

Load reference tables with ``utils.EVs.ev_utils.load_ev_battery_lookup`` and
``load_ev_autonomie_params``, then pass the DataFrames into ``EVBatteryAssigner``.
Callers supply ``max_daily_miles`` (e.g. from
``TripScheduleGenerator.max_daily_miles_from_trip_schedules``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

# Match assign_battery_capacity / off-peak SOC_req convention: size for duty + 20% reserve.
DEFAULT_CAPACITY_BUFFER_FRACTION = 0.2


@dataclass
class EVBatteryAssigner:
    """
    Sample ResStock 2025 EV battery options and attach Autonomie capacity / efficiency.

    Draws from national BEV stock shares, conditioned on each vehicle's max daily
    miles: only options whose usable capacity covers ``miles * kwh_per_mile * (1 + buffer)``
    are eligible. Probabilities are renormalized within that feasible set.
    """

    # Columns: ev_option_name, probability (must sum to ~1)
    option_probabilities: pl.DataFrame
    # Columns: ev_option_name, battery_capacity_kwh, kwh_per_mile, body_class, range_miles
    autonomie_params: pl.DataFrame
    random_state: int = 42
    # Seeded RNG created in __post_init__; not part of construction args.
    _rng: np.random.Generator = field(init=False, repr=False)
    # Joined option tables as arrays for stock-conditional draws.
    _option_names: np.ndarray = field(init=False, repr=False) # ev_option_name
    _option_probs: np.ndarray = field(init=False, repr=False) # probability
    _option_capacity_kwh: np.ndarray = field(init=False, repr=False) # battery_capacity_kwh
    _option_kwh_per_mile: np.ndarray = field(init=False, repr=False) # kwh_per_mile

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

        # Join option probabilities and Autonomie parameters
        joined = self.option_probabilities.join(self.autonomie_params, on="ev_option_name", how="inner")
        self._option_names = joined["ev_option_name"].to_numpy()
        probs = np.asarray(joined["probability"].to_numpy(), dtype=np.float64)
        # ResStock shares can sum to slightly over 1; numpy choice needs exact renormalization.
        self._option_probs = probs / probs.sum()
        self._option_capacity_kwh = np.asarray(joined["battery_capacity_kwh"].to_numpy(), dtype=np.float64)
        self._option_kwh_per_mile = np.asarray(joined["kwh_per_mile"].to_numpy(), dtype=np.float64)

    def _feasible_mask(self, max_daily_miles: float, buffer_fraction: float) -> np.ndarray:
        """Options whose usable pack covers peak-day energy including reserve buffer.
        
        Args:
            max_daily_miles (float): The maximum daily miles driven by the vehicle
            buffer_fraction (float): The buffer fraction to add to the required energy

        Returns:
            np.ndarray: A mask of the options that are feasible
        """
        required_kwh = float(max_daily_miles) * self._option_kwh_per_mile * (1.0 + buffer_fraction)
        return self._option_capacity_kwh >= required_kwh

    def assign(
        self,
        vehicle_duty: pl.DataFrame,
        *,
        buffer_fraction: float = DEFAULT_CAPACITY_BUFFER_FRACTION,
    ) -> pl.DataFrame:
        """
        Draw a stock-conditional EV battery option for each vehicle and join Autonomie params.

        Args:
            vehicle_duty: DataFrame with ``bldg_id``, ``vehicle_id``, and ``max_daily_miles``
            buffer_fraction: Extra fraction of trip energy that usable capacity must cover
                (default 0.2). Option ``i`` is feasible when
                ``battery_capacity_kwh >= max_daily_miles * kwh_per_mile * (1 + buffer_fraction)``.

        Returns:
            One row per vehicle with option name, body class, range, capacity, efficiency,
            and the ``max_daily_miles`` used for the feasibility filter.

        Raises:
            ValueError: If required columns are missing, or no stock option can cover a
                vehicle's peak-day energy need.
        """
        required = {"bldg_id", "vehicle_id", "max_daily_miles"}
        missing = required - set(vehicle_duty.columns)
        if missing:
            raise ValueError(f"vehicle_duty missing columns: {sorted(missing)}")
        if buffer_fraction < 0:
            raise ValueError(f"buffer_fraction must be >= 0, got {buffer_fraction}")

        # Empty input → typed empty output (keeps downstream concat / write happy).
        if vehicle_duty.is_empty():
            return pl.DataFrame(
                schema={
                    "bldg_id": vehicle_duty.schema.get("bldg_id", pl.Int64),
                    "vehicle_id": pl.Int64,
                    "max_daily_miles": pl.Float64,
                    "ev_option_name": pl.Utf8,
                    "body_class": pl.Utf8,
                    "range_miles": pl.Int64,
                    "battery_capacity_kwh": pl.Float64,
                    "kwh_per_mile": pl.Float64,
                }
            )

        miles = vehicle_duty["max_daily_miles"].to_list()
        bldg_ids = vehicle_duty["bldg_id"].to_list()
        vehicle_ids = vehicle_duty["vehicle_id"].to_list()
        drawn: list[str] = []
        # Iterate over the vehicles in the vehicle_duty DataFrame
        for bldg_id, vehicle_id, max_daily_miles in zip(bldg_ids, vehicle_ids, miles, strict=True):
            # identify which vehicle options are feasible
            mask = self._feasible_mask(float(max_daily_miles), buffer_fraction)
            if not np.any(mask):
                # Hard failure for QA: duty cycle exceeds every Autonomie pack under the buffer.
                raise ValueError(
                    f"No ResStock EV battery option can cover bldg_id={bldg_id!r} "
                    f"vehicle_id={vehicle_id} max_daily_miles={float(max_daily_miles):.3f} "
                    f"with buffer_fraction={buffer_fraction}. "
                    "Largest pack usable range "
                    f"(capacity / (kwh_per_mile * (1+buffer))) is "
                    f"{float(np.max(self._option_capacity_kwh / (self._option_kwh_per_mile * (1.0 + buffer_fraction)))):.1f} miles."
                )
            probs = self._option_probs[mask]
            probs = probs / probs.sum()
            # draw a random option from the feasible options
            drawn.append(str(self._rng.choice(self._option_names[mask], p=probs)))
        # create a new DataFrame with the assigned options
        assigned = vehicle_duty.select("bldg_id", "vehicle_id", "max_daily_miles").with_columns(
            pl.Series("ev_option_name", drawn),
        )
        # Attach capacity / efficiency / body_class / range_miles from Autonomie.
        return assigned.join(self.autonomie_params, on="ev_option_name", how="left").select(
            "bldg_id",
            "vehicle_id",
            "max_daily_miles",
            "ev_option_name",
            "body_class",
            "range_miles",
            "battery_capacity_kwh",
            "kwh_per_mile",
        )
