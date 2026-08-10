"""Assign ResStock 2025 EV battery class, usable capacity, and efficiency.

ResStock samples each dwelling's EV type from a national BEV stock distribution
(Experian 2023 registrations via TEMPO), then looks up Autonomie parameters for
usable battery capacity (kWh) and combined fuel economy (kWh/mile).

In this pipeline we apply a **stock-conditional** variant: trip schedules are
generated first, then each vehicle draws from the national option shares
restricted to packs that can cover its peak daily *discharge duty* (plus a
reserve buffer) **and** refill that buffered energy on a Level 2 charger during
the home window on that same peak day (daily-repeat energy balance). Callers should
pass temperature-scaled miles when outdoor-temp adjustment is enabled (see
``ChargingSimulator.build_hourly_temp_scaled_miles``), so capacity feasibility
matches SOC discharge: ``miles_share * kwh_per_mile * power_mult(T)``. Pass
``available_level2_kwh`` (typically ``peak_day_home_hours × Level 2 kW``) so
thirstier packs that fit capacity but cannot recharge are excluded from the draw.
If no option is feasible, assignment raises.

Load reference tables with ``utils.EVs.ev_utils.load_ev_battery_lookup`` and
``load_ev_autonomie_params``, then pass the DataFrames into ``EVBatteryAssigner``.
Callers supply ``max_daily_miles`` (raw or temperature-scaled duty miles).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

# Match assign_battery_capacity / off-peak SOC_req convention: size for duty + 20% reserve.
DEFAULT_CAPACITY_BUFFER_FRACTION = 0.2
# Same headroom the charger assigner / NHTS screen use on recharge energy.
DEFAULT_CHARGER_BUFFER_FRACTION = 0.2


@dataclass
class EVBatteryAssigner:
    """
    Sample ResStock 2025 EV battery options and attach Autonomie capacity / efficiency.

    Draws from national BEV stock shares, conditioned on each vehicle's peak daily
    duty miles: only options whose usable capacity covers
    ``duty_miles * kwh_per_mile * (1 + buffer)`` are eligible. When
    ``available_level2_kwh`` is supplied, options must also satisfy the daily-repeat
    Level 2 recharge gate
    ``duty_miles * kwh_per_mile * (1 + charger_buffer) ≤ available_level2_kwh``.
    When duty miles are temperature-scaled, capacity feasibility is equivalent to
    requiring a 20% buffer above the peak daily temp-adjusted discharge.
    Probabilities are renormalized within the feasible set.
    """

    # Columns: ev_option_name, probability (must sum to ~1)
    option_probabilities: pl.DataFrame
    # Columns: ev_option_name, battery_capacity_kwh, kwh_per_mile, body_class, range_miles
    autonomie_params: pl.DataFrame
    random_state: int = 42
    # Seeded RNG created in __post_init__; not part of construction args.
    _rng: np.random.Generator = field(init=False, repr=False)
    # Joined option tables as arrays for stock-conditional draws.
    _option_names: np.ndarray = field(init=False, repr=False)  # ev_option_name
    _option_probs: np.ndarray = field(init=False, repr=False)  # probability
    _option_capacity_kwh: np.ndarray = field(init=False, repr=False)  # battery_capacity_kwh
    _option_kwh_per_mile: np.ndarray = field(init=False, repr=False)  # kwh_per_mile

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

    def stock_option_parameters(self) -> tuple[tuple[float, float], ...]:
        """Return ``(usable_capacity_kwh, kwh_per_mile)`` for every stock battery option.

        Callers screening duty cycles for feasibility must test all options rather
        than a single reference pack: the largest-capacity option is not the
        longest-range one, and the most efficient option is smaller still.
        """
        return tuple(
            (float(capacity), float(kwh_per_mile))
            for capacity, kwh_per_mile in zip(
                self._option_capacity_kwh, self._option_kwh_per_mile, strict=True
            )
        )

    def _feasible_mask(
        self,
        max_daily_miles: float,
        buffer_fraction: float,
        *,
        available_level2_kwh: float | None = None,
        charger_buffer_fraction: float = DEFAULT_CHARGER_BUFFER_FRACTION,
    ) -> np.ndarray:
        """Options that clear capacity and (optionally) Level 2 daily-repeat recharge.

        ``max_daily_miles`` may be temperature-scaled duty miles. For each Autonomie
        option ``i``, required pack energy is
        ``duty_miles * kwh_per_mile_i * (1 + buffer)``. When ``available_level2_kwh``
        is set, option ``i`` must also satisfy
        ``duty_miles * kwh_per_mile_i * (1 + charger_buffer) ≤ available_level2_kwh``.

        Args:
            max_daily_miles: Peak daily duty miles (raw or temp-scaled)
            buffer_fraction: Extra fraction of trip energy that capacity must cover
            available_level2_kwh: Home charging energy on the peak duty day
                (``peak_day_home_hours × Level 2 kW``). ``None`` disables the
                recharge gate (capacity-only eligibility).
            charger_buffer_fraction: Extra fraction of trip energy Level 2 must cover

        Returns:
            Boolean mask over stock options that are feasible
        """
        # Per-option peak discharge (kWh). Thirstier packs (higher kWh/mi) need more.
        design_kwh = float(max_daily_miles) * self._option_kwh_per_mile
        # Gate 1 — capacity: buffered peak-day energy must fit the usable pack.
        mask = self._option_capacity_kwh >= design_kwh * (1.0 + buffer_fraction)
        if available_level2_kwh is not None:
            # Gate 2 — Level 2 daily-repeat: same peak-day energy (charger buffer)
            # must refill during the home window paired with that peak day.
            # Intersection: only packs that clear *both* stay drawable.
            mask = mask & (
                design_kwh * (1.0 + charger_buffer_fraction) <= float(available_level2_kwh) + 1e-9
            )
        return mask

    def assign(
        self,
        vehicle_duty: pl.DataFrame,
        *,
        buffer_fraction: float = DEFAULT_CAPACITY_BUFFER_FRACTION,
        charger_buffer_fraction: float = DEFAULT_CHARGER_BUFFER_FRACTION,
        level2_power_kw: float | None = None,
    ) -> pl.DataFrame:
        """
        Draw a stock-conditional EV battery option for each vehicle and join Autonomie params.

        Args:
            vehicle_duty: DataFrame with ``bldg_id``, ``vehicle_id``, and ``max_daily_miles``
                (peak daily duty miles; pass temperature-scaled miles when outdoor-temp
                discharge adjustment is enabled so sizing matches SOC draw). Optional
                ``available_level2_kwh`` (or ``peak_day_home_hours`` when
                ``level2_power_kw`` is passed) enables the Level 2 daily-repeat gate.
            buffer_fraction: Extra fraction of peak daily discharge that usable capacity
                must cover (default 0.2).
            charger_buffer_fraction: Extra fraction of peak daily discharge that Level 2
                must cover when the recharge gate is enabled (default 0.2).
            level2_power_kw: When set with a ``peak_day_home_hours`` column, computes
                ``available_level2_kwh = peak_day_home_hours × level2_power_kw``.
                Ignored when ``available_level2_kwh`` is already present.

        Returns:
            One row per vehicle with option name, body class, range, capacity, efficiency,
            and the ``max_daily_miles`` used for the feasibility filter.

        Raises:
            ValueError: If required columns are missing, or no stock option can cover a
                vehicle's peak-day energy need (and Level 2 recharge, when enabled).
        """
        required = {"bldg_id", "vehicle_id", "max_daily_miles"}
        missing = required - set(vehicle_duty.columns)
        if missing:
            raise ValueError(f"vehicle_duty missing columns: {sorted(missing)}")
        if buffer_fraction < 0:
            raise ValueError(f"buffer_fraction must be >= 0, got {buffer_fraction}")
        if charger_buffer_fraction < 0:
            raise ValueError(
                f"charger_buffer_fraction must be >= 0, got {charger_buffer_fraction}"
            )
        if level2_power_kw is not None and level2_power_kw <= 0:
            raise ValueError(f"level2_power_kw must be > 0, got {level2_power_kw}")

        # --- Optional Level 2 energy budget per vehicle ---
        # Callers may pass available_level2_kwh directly, or peak_day_home_hours +
        # level2_power_kw (budget = home_hours × kW). No column → capacity-only draw
        # (legacy / unit tests).
        duty = vehicle_duty
        if "available_level2_kwh" not in duty.columns:
            if "peak_day_home_hours" in duty.columns:
                if level2_power_kw is None:
                    raise ValueError(
                        "level2_power_kw is required when vehicle_duty has "
                        "peak_day_home_hours but not available_level2_kwh"
                    )
                duty = duty.with_columns(
                    (pl.col("peak_day_home_hours") * float(level2_power_kw)).alias(
                        "available_level2_kwh"
                    )
                )
        use_l2_gate = "available_level2_kwh" in duty.columns

        # Empty input → typed empty output (keeps downstream concat / write happy).
        if duty.is_empty():
            return pl.DataFrame(
                schema={
                    "bldg_id": duty.schema.get("bldg_id", pl.Int64),
                    "vehicle_id": pl.Int64,
                    "max_daily_miles": pl.Float64,
                    "ev_option_name": pl.Utf8,
                    "body_class": pl.Utf8,
                    "range_miles": pl.Int64,
                    "battery_capacity_kwh": pl.Float64,
                    "kwh_per_mile": pl.Float64,
                }
            )

        miles = duty["max_daily_miles"].to_list()
        bldg_ids = duty["bldg_id"].to_list()
        vehicle_ids = duty["vehicle_id"].to_list()
        l2_budgets = (
            duty["available_level2_kwh"].to_list() if use_l2_gate else [None] * len(miles)
        )
        drawn: list[str] = []
        # Iterate over the vehicles in the vehicle_duty DataFrame
        for bldg_id, vehicle_id, max_daily_miles, available_l2 in zip(
            bldg_ids, vehicle_ids, miles, l2_budgets, strict=True
        ):
            # Keep packs that cover peak daily discharge + buffer, and (when enabled)
            # that can refill that buffered energy on Level 2 on the peak duty day.
            mask = self._feasible_mask(
                float(max_daily_miles),
                buffer_fraction,
                available_level2_kwh=(
                    None if available_l2 is None else float(available_l2)
                ),
                charger_buffer_fraction=charger_buffer_fraction,
            )
            if not np.any(mask):
                # Hard failure for QA: duty cycle exceeds every Autonomie pack under
                # the capacity / Level 2 gates.
                largest_range = float(
                    np.max(
                        self._option_capacity_kwh
                        / (self._option_kwh_per_mile * (1.0 + buffer_fraction))
                    )
                )
                l2_note = ""
                if available_l2 is not None:
                    l2_note = (
                        f", available_level2_kwh={float(available_l2):.3f} "
                        f"(charger_buffer_fraction={charger_buffer_fraction})"
                    )
                raise ValueError(
                    f"No ResStock EV battery option can cover bldg_id={bldg_id!r} "
                    f"vehicle_id={vehicle_id} max_daily_miles={float(max_daily_miles):.3f} "
                    f"with buffer_fraction={buffer_fraction}{l2_note}. "
                    "Largest pack usable range "
                    f"(capacity / (kwh_per_mile * (1+buffer))) is "
                    f"{largest_range:.1f} miles."
                )
            probs = self._option_probs[mask]
            probs = probs / probs.sum()
            # draw a random option from the feasible options
            drawn.append(str(self._rng.choice(self._option_names[mask], p=probs)))
        # create a new DataFrame with the assigned options
        assigned = duty.select("bldg_id", "vehicle_id", "max_daily_miles").with_columns(
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
