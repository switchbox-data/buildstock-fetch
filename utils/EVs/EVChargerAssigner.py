"""Assign ResStock 2025 home EV charger level (L1 vs L2) and rated power.

ResStock samples charger type from EIA 2020 RECS ``EVCHRGTYPE``, conditioned on
EV ownership, federal poverty level, building type, and tenure
(NREL/TP-5500-93766 §3.2). Dwelling units with an EV always get at least Level 1;
units without an EV get None.

In this pipeline every simulated slot already owns an EV, so we join only the
``Electric Vehicle Ownership = Yes`` rows and draw Level 1 vs Level 2. Draws are
**duty-conditional**: only levels that can cover the vehicle's trip discharge
under perfect SOC foresight (immediate home charging, which is SOC-maximal) are
eligible, with discharge inflated by ``charger_buffer_fraction``. When both
levels are feasible we use the ResStock L1/L2 probabilities; when only one is,
we assign that level. If neither is feasible, assignment raises.

Default rated powers follow the ResStock 2025 TRG / options_lookup mapping
(Level 1 = 1.6 kW, Level 2 = 5.69 kW — the latter is ResStock's *average*
observed 240 V draw, not a typical dedicated EVSE nameplate). Override via
``level1_power_kw`` / ``level2_power_kw`` (or the matching YAML keys).

Load the lookup with ``utils.EVs.ev_utils.load_ev_charger_lookup``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from utils.EVs.charging import is_home_charging_soc_feasible

# ResStock TRG / measure docs: ev_charger_power in watts → kW for the charging simulator.
# L2 = 5.69 kW is ResStock's average 240 V outlet draw, not a 32 A EVSE nameplate (~7.2 kW).
RESSTOCK_LEVEL1_CHARGER_KW = 1.6
RESSTOCK_LEVEL2_CHARGER_KW = 5.69

# Headroom on trip discharge when testing whether a charger level is SOC-feasible.
# Mirrors battery ``capacity_buffer_fraction`` (pack must cover peak discharge × (1+buffer)).
DEFAULT_CHARGER_BUFFER_FRACTION = 0.2


@dataclass
class EVChargerAssigner:
    """Sample L1 vs L2 from ResStock probs, restricted to SOC-feasible levels."""

    # Yes-ownership rows from the thin loader, including ResStock's p_void marker.
    charger_lookup: pl.DataFrame
    random_state: int = 42
    # Rated powers applied after the L1/L2 draw (defaults = ResStock TRG values).
    level1_power_kw: float = RESSTOCK_LEVEL1_CHARGER_KW
    level2_power_kw: float = RESSTOCK_LEVEL2_CHARGER_KW
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Validate the loader/assigner contract without rejecting retained Void rows.
        required = {"fpl", "building_type", "tenure", "p_level1", "p_level2", "p_void"}
        missing = required - set(self.charger_lookup.columns)
        if missing:
            raise ValueError(f"charger_lookup missing columns: {sorted(missing)}")
        if self.level1_power_kw < 0:
            raise ValueError(f"level1_power_kw must be >= 0; got {self.level1_power_kw}")
        if self.level2_power_kw < 0:
            raise ValueError(f"level2_power_kw must be >= 0; got {self.level2_power_kw}")

        # Create one deterministic random stream for reproducible L1/L2 draws.
        self._rng = np.random.default_rng(self.random_state)

    def _level_feasible(
        self,
        *,
        power_kw: float,
        at_home: np.ndarray,
        discharge_kwh: np.ndarray,
        battery_capacity_kwh: float,
        buffer_fraction: float,
    ) -> bool:
        """True when home charging at ``power_kw`` covers buffered trip discharge."""
        return is_home_charging_soc_feasible(
            at_home,
            discharge_kwh,
            battery_capacity_kwh=battery_capacity_kwh,
            charger_power_kw=power_kw,
            buffer_fraction=buffer_fraction,
        )

    def assign(
        self,
        vehicles: pl.DataFrame,
        *,
        presence_by_vehicle: dict[tuple[str | int, int], pl.DataFrame],
        discharge_kwh_by_vehicle: dict[tuple[str | int, int], np.ndarray],
        buffer_fraction: float = DEFAULT_CHARGER_BUFFER_FRACTION,
    ) -> pl.DataFrame:
        """
        Draw a charger level for each EV from the SOC-feasible L1/L2 subset.

        For each vehicle, Level 1 / Level 2 is eligible only when immediate home
        charging at that power covers the trip schedule with perfect foresight
        (discharge inflated by ``1 + buffer_fraction``). If both levels are
        feasible, draw with ResStock ``p_level2``; if only one is, assign it.

        Args:
            vehicles: One row per EV slot with ``bldg_id``, ``vehicle_id``, metadata
                join keys ``fpl``, ``building_type``, ``tenure``, and
                ``battery_capacity_kwh`` (from ``EVBatteryAssigner``).
            presence_by_vehicle: Hourly presence frames keyed by ``(bldg_id, vehicle_id)``,
                each with an ``at_home`` column aligned to the simulation calendar.
            discharge_kwh_by_vehicle: Hourly trip draw (kWh) arrays keyed the same way,
                same length as the matching presence schedule.
            buffer_fraction: Extra fraction of discharge the charger must cover
                (default 0.2).

        Returns:
            ``bldg_id``, ``vehicle_id``, ``charger_level`` (``Level 1`` / ``Level 2``),
            and ``charger_power_kw``.

        Raises:
            ValueError: If required columns are missing, the lookup join misses a row,
                a matched cell is Void / None for an EV owner, presence/discharge are
                missing or misaligned, or neither charger level is SOC-feasible.
        """
        required = {
            "bldg_id",
            "vehicle_id",
            "fpl",
            "building_type",
            "tenure",
            "battery_capacity_kwh",
        }
        missing = required - set(vehicles.columns)
        if missing:
            raise ValueError(f"vehicles missing columns: {sorted(missing)}")
        if buffer_fraction < 0:
            raise ValueError(f"buffer_fraction must be >= 0, got {buffer_fraction}")

        # Empty EV set → typed empty frame for safe concat / parquet writes.
        if vehicles.is_empty():
            return pl.DataFrame(
                schema={
                    "bldg_id": vehicles.schema.get("bldg_id", pl.Int64),
                    "vehicle_id": pl.Int64,
                    "charger_level": pl.Utf8,
                    "charger_power_kw": pl.Float64,
                }
            )

        # Join each vehicle to its conditional L1/L2 probabilities.
        joined = vehicles.select(
            "bldg_id",
            "vehicle_id",
            "fpl",
            "building_type",
            "tenure",
            "battery_capacity_kwh",
        ).join(
            self.charger_lookup,
            on=["fpl", "building_type", "tenure"],
            how="left",
        )

        # A missing join means metadata keys do not exist in the ResStock lookup.
        unmatched = joined.filter(pl.col("p_level1").is_null())
        if unmatched.height > 0:
            sample_ids = unmatched.get_column("bldg_id").head(5).to_list()
            raise ValueError(
                f"EV charger lookup join missed for {unmatched.height} vehicle(s) "
                f"(e.g. bldg_id={sample_ids}). Check fpl, building_type, and tenure "
                "against Electric_Vehicle_Charger.tsv Yes-ownership rows."
            )

        # Void marks a logically impossible dependency combination (typically an
        # occupied/vacant coding conflict). Such a row must never reach an EV draw.
        void_matches = joined.filter(pl.col("p_void") == 1.0)
        if void_matches.height > 0:
            sample_ids = void_matches.get_column("bldg_id").head(5).to_list()
            raise ValueError(
                f"EV charger lookup matched a Void row for {void_matches.height} vehicle(s) "
                f"(e.g. bldg_id={sample_ids}). Check FPL/tenure vacancy consistency."
            )

        # Every usable EV-owner row must allocate all probability to L1 or L2.
        # This catches malformed rows and any unexpected non-binary Void encoding.
        probability_sum = pl.col("p_level1") + pl.col("p_level2")
        invalid_probabilities = joined.filter((probability_sum - 1.0).abs() > 1e-3)
        if invalid_probabilities.height > 0:
            sample_ids = invalid_probabilities.get_column("bldg_id").head(5).to_list()
            raise ValueError(
                "EV charger matched rows must have p_level1 + p_level2 ≈ 1; "
                f"found {invalid_probabilities.height} invalid vehicle(s) "
                f"(e.g. bldg_id={sample_ids})"
            )

        levels: list[str] = []
        powers: list[float] = []
        # When L2 is at least as fast as L1, L1-feasible ⇒ L2-feasible (skip a second sim).
        l2_dominates_l1 = self.level2_power_kw >= self.level1_power_kw - 1e-12

        # Per vehicle: feasibility filter, then ResStock draw (or force sole survivor).
        for row in joined.iter_rows(named=True):
            key = (row["bldg_id"], int(row["vehicle_id"]))
            if key not in presence_by_vehicle:
                raise ValueError(
                    f"presence_by_vehicle missing schedule for bldg_id={row['bldg_id']!r} "
                    f"vehicle_id={row['vehicle_id']}"
                )
            if key not in discharge_kwh_by_vehicle:
                raise ValueError(
                    f"discharge_kwh_by_vehicle missing array for bldg_id={row['bldg_id']!r} "
                    f"vehicle_id={row['vehicle_id']}"
                )

            # Hourly presence / discharge must share the same calendar length.
            presence = presence_by_vehicle[key]
            if "at_home" not in presence.columns:
                raise ValueError(
                    f"presence schedule for {key} missing at_home column"
                )
            at_home = presence["at_home"].to_numpy()
            discharge_kwh = np.asarray(discharge_kwh_by_vehicle[key], dtype=np.float64)
            if len(at_home) != len(discharge_kwh):
                raise ValueError(
                    f"presence/discharge length mismatch for {key}: "
                    f"{len(at_home)} vs {len(discharge_kwh)}"
                )

            # Step 1 — feasible mask over {L1, L2} via perfect-foresight SOC check.
            capacity = float(row["battery_capacity_kwh"])
            l1_ok = self._level_feasible(
                power_kw=self.level1_power_kw,
                at_home=at_home,
                discharge_kwh=discharge_kwh,
                battery_capacity_kwh=capacity,
                buffer_fraction=buffer_fraction,
            )
            if l1_ok and l2_dominates_l1:
                l2_ok = True
            else:
                l2_ok = self._level_feasible(
                    power_kw=self.level2_power_kw,
                    at_home=at_home,
                    discharge_kwh=discharge_kwh,
                    battery_capacity_kwh=capacity,
                    buffer_fraction=buffer_fraction,
                )

            # Hard fail (same spirit as EVBatteryAssigner when no pack fits).
            if not l1_ok and not l2_ok:
                raise ValueError(
                    f"No ResStock EV charger level can cover bldg_id={row['bldg_id']!r} "
                    f"vehicle_id={row['vehicle_id']} under perfect SOC foresight "
                    f"with buffer_fraction={buffer_fraction} "
                    f"(Level 1={self.level1_power_kw} kW, Level 2={self.level2_power_kw} kW, "
                    f"battery_capacity_kwh={capacity})."
                )

            # Both feasible → ResStock Bernoulli; only one → force that level.
            if l1_ok and l2_ok:
                choose_l2 = bool(self._rng.random() < float(row["p_level2"]))
            else:
                choose_l2 = l2_ok

            if choose_l2:
                levels.append("Level 2")
                powers.append(self.level2_power_kw)
            else:
                levels.append("Level 1")
                powers.append(self.level1_power_kw)

        return joined.select("bldg_id", "vehicle_id").with_columns(
            pl.Series("charger_level", levels),
            pl.Series("charger_power_kw", powers, dtype=pl.Float64),
        )
