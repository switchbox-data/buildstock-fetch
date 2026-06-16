"""
Assign EV ownership to ResStock buildings using NREL's ResStock-TEMPO methodology.

NREL pre-computes conditional P(EV) in ``Electric Vehicle Ownership.tsv`` from
Experian BEV registrations, ACS/PUMS geography, and 2020 RECS housing segments.
See NREL/TP-5500-93766 (https://doi.org/10.2172/2584243).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

DEFAULT_OWNERSHIP_TSV = (
    Path(__file__).resolve().parent
    / "ev_data/inputs/resstock_ev_reference/Electric_Vehicle_Ownership.tsv"
)

# National baseline from ResStock options_saturations.csv (upgrade 13, baseline stock).
NATIONAL_EV_OWNERSHIP_RATE = 0.0144856

LOOKUP_RENAME = {
    "Dependency=Federal Poverty Level": "fpl",
    "Dependency=Geometry Building Type RECS": "building_type",
    "Dependency=PUMA": "puma_dependency",
    "Dependency=Tenure": "tenure",
    "Option=Yes": "ev_ownership_probability",
}


def resstock_puma_dependency(state: str, puma_geoid: str) -> str:
    """Map ResStock GEOID PUMA (e.g. ``G24000901``) to lookup key (e.g. ``MD, 00901``)."""
    puma5 = puma_geoid[-5:].zfill(5)
    return f"{state}, {puma5}"


def load_ev_ownership_lookup(path: str | Path = DEFAULT_OWNERSHIP_TSV) -> pl.DataFrame:
    """Load NREL's pre-computed Electric Vehicle Ownership housing characteristic TSV."""
    lookup_path = Path(path)
    if not lookup_path.exists():
        msg = (
            f"Missing {lookup_path}. Run: just download-resstock-ev-reference "
            "(downloads Electric Vehicle Ownership.tsv from github.com/NREL/resstock)."
        )
        raise FileNotFoundError(msg)

    lookup = pl.read_csv(
        lookup_path,
        separator="\t",
        null_values=["", "NA"],
    ).rename(LOOKUP_RENAME)

    return lookup.select(
        "fpl",
        "building_type",
        "puma_dependency",
        "tenure",
        "ev_ownership_probability",
    )


def prepare_metadata_for_ev_lookup(metadata: pl.DataFrame) -> pl.DataFrame:
    """Add join keys expected by NREL's Electric Vehicle Ownership lookup."""
    required = {
        "in.state",
        "in.puma",
        "in.federal_poverty_level",
        "in.geometry_building_type_recs",
        "in.tenure",
    }
    missing = required - set(metadata.columns)
    if missing:
        msg = f"Metadata missing columns for EV ownership lookup: {sorted(missing)}"
        raise ValueError(msg)

    vacancy_col = "in.vacancy_status" if "in.vacancy_status" in metadata.columns else None

    return metadata.with_columns(
        pl.struct(["in.state", "in.puma"])
        .map_elements(
            lambda row: resstock_puma_dependency(row["in.state"], row["in.puma"]),
            return_dtype=pl.Utf8,
        )
        .alias("puma_dependency"),
        pl.col("in.federal_poverty_level").alias("fpl"),
        pl.col("in.geometry_building_type_recs").alias("building_type"),
        pl.col("in.tenure").alias("tenure"),
        (
            pl.col(vacancy_col) == "Vacant"
            if vacancy_col
            else pl.lit(False)
        ).alias("is_vacant"),
    )


def add_ev_ownership_probability(
    metadata: pl.DataFrame,
    lookup: pl.DataFrame | None = None,
    lookup_path: str | Path = DEFAULT_OWNERSHIP_TSV,
    fallback_rate: float = NATIONAL_EV_OWNERSHIP_RATE,
) -> pl.DataFrame:
    """
    Join ResStock metadata to NREL P(EV | PUMA, building type, tenure, FPL).

    Vacant units receive probability 0 (NREL assumption). Unmatched occupied units
    fall back to the national baseline rate (~1.45%).
    """
    lookup_df = lookup if lookup is not None else load_ev_ownership_lookup(lookup_path)
    prepared = prepare_metadata_for_ev_lookup(metadata)

    with_probs = prepared.join(
        lookup_df,
        on=["fpl", "building_type", "puma_dependency", "tenure"],
        how="left",
    ).with_columns(
        pl.when(pl.col("is_vacant"))
        .then(0.0)
        .when(pl.col("ev_ownership_probability").is_null())
        .then(fallback_rate)
        .otherwise(pl.col("ev_ownership_probability"))
        .alias("ev_ownership_probability"),
        pl.col("ev_ownership_probability").is_null().alias("ev_probability_imputed"),
    )

    return with_probs


def sample_ev_ownership(
    metadata: pl.DataFrame,
    *,
    seed: int = 42,
    lookup: pl.DataFrame | None = None,
    lookup_path: str | Path = DEFAULT_OWNERSHIP_TSV,
    probability_column: str = "ev_ownership_probability",
) -> pl.DataFrame:
    """
    Bernoulli-sample ``has_ev`` per building from ``ev_ownership_probability``.

    Reproducible given ``seed``; matches the stochastic assignment ResStock applies
    when sampling housing characteristics.
    """
    if probability_column not in metadata.columns:
        metadata = add_ev_ownership_probability(metadata, lookup=lookup, lookup_path=lookup_path)

    rng = np.random.default_rng(seed)
    probabilities = metadata.get_column(probability_column).to_numpy()
    draws = rng.random(probabilities.shape[0])
    has_ev = draws < probabilities
    is_vacant = (
        metadata.get_column("is_vacant").to_numpy()
        if "is_vacant" in metadata.columns
        else np.zeros(metadata.height, dtype=bool)
    )
    ev_count = np.where(is_vacant, 0, has_ev.astype(np.int8))

    return metadata.with_columns(
        pl.Series("has_ev", has_ev),
        pl.Series("ev_count", ev_count),
    )


def summarize_ev_adoption(metadata_with_ev: pl.DataFrame) -> pl.DataFrame:
    """Weighted and unweighted EV adoption summary for QA."""
    cols = {
        "weight": "weight" if "weight" in metadata_with_ev.columns else None,
        "has_ev": "has_ev" if "has_ev" in metadata_with_ev.columns else None,
        "probability": "ev_ownership_probability"
        if "ev_ownership_probability" in metadata_with_ev.columns
        else None,
    }
    if cols["has_ev"] is None:
        msg = "Expected column 'has_ev' — run sample_ev_ownership() first."
        raise ValueError(msg)

    occupied = (
        metadata_with_ev.filter(~pl.col("is_vacant"))
        if "is_vacant" in metadata_with_ev.columns
        else metadata_with_ev
    )

    summary: dict[str, float] = {
        "buildings": float(occupied.height),
        "unweighted_adoption_rate": float(occupied["has_ev"].mean()),
        "expected_adoption_rate": float(occupied[cols["probability"]].mean())
        if cols["probability"]
        else float("nan"),
    }
    if cols["weight"]:
        weighted_ev = (occupied["has_ev"].cast(pl.Float64) * occupied["weight"]).sum()
        summary["weighted_adoption_rate"] = float(weighted_ev / occupied["weight"].sum())

    return pl.DataFrame({"metric": list(summary.keys()), "value": list(summary.values())})
