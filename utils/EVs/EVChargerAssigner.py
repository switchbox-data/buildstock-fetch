"""Assign ResStock 2025 home EV charger level (L1 vs L2) and rated power.

ResStock samples charger type from EIA 2020 RECS ``EVCHRGTYPE``, conditioned on
EV ownership, federal poverty level, building type, and tenure
(NREL/TP-5500-93766 §3.2). Dwelling units with an EV always get at least Level 1;
units without an EV get None.

In this pipeline every simulated slot already owns an EV, so we join only the
``Electric Vehicle Ownership = Yes`` rows and draw Level 1 vs Level 2. Default
rated powers follow the ResStock 2025 TRG / options_lookup mapping
(Level 1 = 1.6 kW, Level 2 = 5.69 kW — the latter is ResStock's *average*
observed 240 V draw, not a typical dedicated EVSE nameplate). Override via
``level1_power_kw`` / ``level2_power_kw`` (or the matching YAML keys).

Load the lookup with ``utils.EVs.ev_utils.load_ev_charger_lookup``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

# ResStock TRG / measure docs: ev_charger_power in watts → kW for the charging simulator.
# L2 = 5.69 kW is ResStock's average 240 V outlet draw, not a 32 A EVSE nameplate (~7.2 kW).
RESSTOCK_LEVEL1_CHARGER_KW = 1.6
RESSTOCK_LEVEL2_CHARGER_KW = 5.69


@dataclass
class EVChargerAssigner:
    """Bernoulli-sample L1 vs L2 charger for each EV vehicle from ResStock TSV probs."""

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

    def assign(self, vehicles: pl.DataFrame) -> pl.DataFrame:
        """
        Draw a charger level for each EV vehicle and attach rated power (kW).

        Args:
            vehicles: One row per EV slot with ``bldg_id``, ``vehicle_id``, plus metadata
                join keys ``fpl``, ``building_type``, and ``tenure``.

        Returns:
            ``bldg_id``, ``vehicle_id``, ``charger_level`` (``Level 1`` / ``Level 2``),
            and ``charger_power_kw``.

        Raises:
            ValueError: If required columns are missing, the lookup join misses a row,
                or a matched cell is Void / None for an EV owner.
        """
        required = {"bldg_id", "vehicle_id", "fpl", "building_type", "tenure"}
        missing = required - set(vehicles.columns)
        if missing:
            raise ValueError(f"vehicles missing columns: {sorted(missing)}")

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
            "bldg_id", "vehicle_id", "fpl", "building_type", "tenure"
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

        # Independent Bernoulli: draw U ~ Unif(0,1); Level 2 if U < p_level2 else Level 1.
        # Equivalent to multinomial([p_level1, p_level2]) since they sum to 1.
        draws = self._rng.random(joined.height)
        p_level2 = joined["p_level2"].to_numpy()
        levels = np.where(draws < p_level2, "Level 2", "Level 1")
        powers = np.where(
            levels == "Level 2",
            self.level2_power_kw,
            self.level1_power_kw,
        )

        return joined.select("bldg_id", "vehicle_id").with_columns(
            pl.Series("charger_level", levels),
            pl.Series("charger_power_kw", powers, dtype=pl.Float64),
        )
