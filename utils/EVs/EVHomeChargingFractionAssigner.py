"""Assign ResStock 2025 home-charging energy fraction per EV slot.

ResStock samples ``Electric Vehicle Charge At Home`` from EIA 2020 RECS, conditioned
on federal poverty level and geometry building type (NREL/TP-5500-93766 §3.5). Each
bin maps to a midpoint scalar ``ev_fraction_charged_home`` via ``options_lookup``:

    0-19% → 0.10, 20-39% → 0.30, …, 100% → 1.00

That scalar multiplies home-attributed discharge (residential meter load) only —
battery sizing still uses full trip duty. Away charging is excluded by shrinking
home discharge rather than modeling workplace/public charge events.

Load the lookup with ``utils.EVs.ev_utils.load_ev_charge_at_home_lookup``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

# ResStock options_lookup: Electric Vehicle Charge At Home → ev_fraction_charged_home.
# Midpoints of the RECS bins (0–19% uses 0.10, not 0.095).
CHARGE_AT_HOME_BIN_TO_FRACTION: dict[str, float] = {
    "0-19%": 0.10,
    "20-39%": 0.30,
    "40-59%": 0.50,
    "60-79%": 0.70,
    "80-99%": 0.90,
    "100%": 1.00,
}

# Ordered bin labels matching probability columns on the thin lookup.
CHARGE_AT_HOME_BINS: tuple[str, ...] = tuple(CHARGE_AT_HOME_BIN_TO_FRACTION.keys())

# Lookup probability columns parallel to CHARGE_AT_HOME_BINS.
_PROB_COLUMNS: tuple[str, ...] = (
    "p_0_19",
    "p_20_39",
    "p_40_59",
    "p_60_79",
    "p_80_99",
    "p_100",
)


@dataclass
class EVHomeChargingFractionAssigner:
    """Sample a home-charging fraction bin from ResStock RECS probabilities."""

    # Thin lookup from ``load_ev_charge_at_home_lookup`` (FPL × building type).
    charge_at_home_lookup: pl.DataFrame
    random_state: int = 42
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Require join keys plus one probability column per RECS bin.
        required = {"fpl", "building_type", *_PROB_COLUMNS}
        missing = required - set(self.charge_at_home_lookup.columns)
        if missing:
            raise ValueError(f"charge_at_home_lookup missing columns: {sorted(missing)}")

        # One deterministic stream for reproducible multinomial draws.
        self._rng = np.random.default_rng(self.random_state)

    def assign(self, vehicles: pl.DataFrame) -> pl.DataFrame:
        """
        Draw a home-charging fraction for each EV from the ResStock TSV.

        Args:
            vehicles: One row per EV slot with ``bldg_id``, ``vehicle_id``, and
                metadata join keys ``fpl``, ``building_type``.

        Returns:
            ``bldg_id``, ``vehicle_id``, ``charge_at_home_bin`` (RECS option label),
            and ``fraction_charged_home`` (midpoint scalar in (0, 1]).

        Raises:
            ValueError: If required columns are missing, the lookup join misses a
                row, or matched probabilities do not sum to ~1.
        """
        required = {"bldg_id", "vehicle_id", "fpl", "building_type"}
        missing = required - set(vehicles.columns)
        if missing:
            raise ValueError(f"vehicles missing columns: {sorted(missing)}")

        # Empty EV set → typed empty frame for safe concat / parquet writes.
        if vehicles.is_empty():
            return pl.DataFrame(
                schema={
                    "bldg_id": vehicles.schema.get("bldg_id", pl.Int64),
                    "vehicle_id": pl.Int64,
                    "charge_at_home_bin": pl.Utf8,
                    "fraction_charged_home": pl.Float64,
                }
            )

        # Join each vehicle to its conditional bin probabilities.
        joined = vehicles.select(
            "bldg_id",
            "vehicle_id",
            "fpl",
            "building_type",
        ).join(
            self.charge_at_home_lookup,
            on=["fpl", "building_type"],
            how="left",
        )

        # A missing join means metadata keys do not exist in the ResStock lookup.
        unmatched = joined.filter(pl.col(_PROB_COLUMNS[0]).is_null())
        if unmatched.height > 0:
            sample_ids = unmatched.get_column("bldg_id").head(5).to_list()
            raise ValueError(
                f"EV charge-at-home lookup join missed for {unmatched.height} vehicle(s) "
                f"(e.g. bldg_id={sample_ids}). Check fpl and building_type against "
                "Electric_Vehicle_Charge_At_Home.tsv."
            )

        # Every row must be a valid multinomial (probabilities sum to one).
        probability_sum = sum(pl.col(c) for c in _PROB_COLUMNS)
        invalid_probabilities = joined.filter((probability_sum - 1.0).abs() > 1e-3)
        if invalid_probabilities.height > 0:
            sample_ids = invalid_probabilities.get_column("bldg_id").head(5).to_list()
            raise ValueError(
                "EV charge-at-home matched rows must have bin probabilities ≈ 1; "
                f"found {invalid_probabilities.height} invalid vehicle(s) "
                f"(e.g. bldg_id={sample_ids})"
            )

        bins: list[str] = []
        fractions: list[float] = []

        # Per vehicle: multinomial draw over the six RECS bins, then map to midpoint.
        for row in joined.iter_rows(named=True):
            probs = np.asarray([float(row[c]) for c in _PROB_COLUMNS], dtype=np.float64)
            # Guard tiny float drift so choice() always sees a valid distribution.
            probs = probs / probs.sum()
            bin_idx = int(self._rng.choice(len(CHARGE_AT_HOME_BINS), p=probs))
            bin_label = CHARGE_AT_HOME_BINS[bin_idx]
            bins.append(bin_label)
            fractions.append(CHARGE_AT_HOME_BIN_TO_FRACTION[bin_label])

        return joined.select("bldg_id", "vehicle_id").with_columns(
            pl.Series("charge_at_home_bin", bins),
            pl.Series("fraction_charged_home", fractions, dtype=pl.Float64),
        )
