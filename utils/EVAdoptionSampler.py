from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class EVAdoptionSampler:
    """Bernoulli-sample EV ownership (0 or 1) from NREL ResStock lookup probabilities."""

    ev_ownership_df: pl.DataFrame
    random_state: int = 42

    def sample(self, metadata_df: pl.DataFrame) -> pl.DataFrame:
        """
        Predict whether each household has an EV (0 or 1) using NREL's ResStock lookup.

        Joins metadata to NREL's conditional P(EV) by FPL, building type, PUMA, and tenure,
        then Bernoulli-samples ownership (max one EV per household). Vacant units always
        receive 0 EVs.

        Args:
            metadata_df: DataFrame with ResStock metadata including fpl, building_type,
                puma_dependency, tenure, and is_vacant columns.

        Returns:
            DataFrame with added columns: ev_ownership_probability, evs

        Raises:
            ValueError: If the metadata DataFrame is missing required columns or an occupied
                building has no matching EV ownership lookup row
        """
        required_columns = {"fpl", "building_type", "tenure", "puma_dependency", "is_vacant"}
        missing_columns = required_columns - set(metadata_df.columns)
        if missing_columns:
            raise ValueError(
                "Missing EV adoption metadata columns: "
                + ", ".join(sorted(missing_columns))
                + ". Ensure load_metadata() was used or provide these columns."
            )

        # conditional P(EV) lookup by FPL, building type, PUMA, and tenure
        ev_lookup = self.ev_ownership_df.select(
            "fpl",
            "building_type",
            "puma_dependency",
            "tenure",
            "ev_ownership_probability",
        )

        # join metadata to conditional P(EV) lookup
        metadata_with_prob = metadata_df.join(
            ev_lookup,
            on=["fpl", "building_type", "puma_dependency", "tenure"],
            how="left",
        ).with_columns(
            pl.when(pl.col("is_vacant"))
            .then(0.0)  # vacant units: Tenure/FPL = "Not Available", set P(EV) = 0
            .otherwise(pl.col("ev_ownership_probability"))
            .alias("ev_ownership_probability"),
        )

        # Fail fast if any occupied building misses the lookup join.
        unmatched = metadata_with_prob.filter(
            ~pl.col("is_vacant") & pl.col("ev_ownership_probability").is_null()
        )
        if unmatched.height > 0:
            sample_ids = unmatched.get_column("bldg_id").head(5).to_list() if "bldg_id" in unmatched.columns else []
            raise ValueError(
                f"EV ownership lookup join missed for {unmatched.height} occupied building(s)"
                + (f" (e.g. bldg_id={sample_ids})" if sample_ids else "")
                + ". Check fpl, building_type, puma_dependency, and tenure against the lookup table."
            )

        # Bernoulli sample — each household gets 0 or 1 EV (max one per household).
        # Matches ev_adoption.ipynb; differs from ResStock 2025 which uses quota sampling.
        rng = np.random.default_rng(self.random_state)
        return metadata_with_prob.with_columns(
            pl.Series("_draw", rng.random(metadata_with_prob.height)),
        ).with_columns(
            pl.when(pl.col("is_vacant"))
            .then(pl.lit(0))
            .when(pl.col("_draw") < pl.col("ev_ownership_probability"))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.Int8)
            .alias("evs"),  # aliased to "vehicles" downstream for NHTS sampling
        ).drop("_draw")
