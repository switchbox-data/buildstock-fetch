from dataclasses import dataclass

import numpy as np
import polars as pl


def assign_evs_by_priority_sampling(
    probs: np.ndarray,
    weights: np.ndarray,
    draws: np.ndarray,
    target: float,
) -> np.ndarray:
    """Assign EVs by priority sampling on baseline P(EV), then taking a weight prefix.

    Each occupied building gets a priority key ``k_i = u_i / p_i`` (``inf`` when
    ``p_i == 0``). Buildings are visited in ascending ``k_i`` and marked as EVs until
    cumulative ResStock weight reaches ``target * sum(weights)``.

    Why this key: sweeping a threshold ``t`` over ``k_i`` includes building ``i``
    exactly when ``u_i < t * p_i``, so the inclusion probability is ``min(1, t * p_i)``
    — the same linear propensity scaling as ``s * p``, but realized as an exact
    housing-stock count instead of noisy Bernoulli draws. Consequences:

    - adoption **rates** stay proportional to ``p`` across segments (no all-or-nothing
      cascade where the top segment fully saturates before the next one starts)
    - ``τ`` is hit exactly (up to one building's weight)
    - holding ``draws`` fixed nests EV sets as ``τ`` rises
    - ``p == 0`` buildings sort last but still adopt once all ``p > 0`` stock is taken

    Args:
        probs: Baseline P(EV) for occupied buildings (values in [0, 1]).
        weights: Non-negative ResStock sample weights for the same buildings.
        draws: Per-building Uniform(0, 1) draws.
        target: Desired occupied housing-stock EV share in ``[0, 1]``.

    Returns:
        Int8 array of length ``len(probs)`` with 1 = EV, 0 = no EV.

    Raises:
        ValueError: If shapes disagree, ``target`` is outside ``[0, 1]``, or weights
            are invalid.
    """
    if not 0.0 <= target <= 1.0:
        raise ValueError(f"target adoption rate must be in [0, 1]; got {target}")
    probs = np.asarray(probs, dtype=float)
    weights = np.asarray(weights, dtype=float)
    draws = np.asarray(draws, dtype=float)
    if not (probs.shape == weights.shape == draws.shape):
        raise ValueError(
            "probs, weights, and draws must have the same shape; "
            f"got {probs.shape}, {weights.shape}, {draws.shape}"
        )
    if probs.size == 0:
        return np.zeros(0, dtype=np.int8)
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValueError("weights must sum to a positive value")

    evs = np.zeros(probs.shape[0], dtype=np.int8)
    if target == 0.0:
        return evs  # no EVs
    if target >= 1.0:
        evs[:] = 1  # every occupied building in this array
        return evs

    # Priority key k_i = u_i / p_i; p_i == 0 → inf (adopts only after all p > 0).
    # Lower key = adopts earlier. High-p buildings tend to draw low keys, so segment
    # adoption *rates* scale with p rather than saturating one segment at a time.
    safe_probs = np.where(probs > 0.0, probs, 1.0)
    keys = np.where(probs > 0.0, draws / safe_probs, np.inf)

    # np.lexsort: last key is primary. Ties (e.g. all the p==0 rows) fall back to u_i.
    order = np.lexsort((draws, keys))

    # Stop once cumulative housing-stock weight reaches τ * total occupied weight.
    target_weight = target * total_weight
    cumulative = 0.0
    for idx in order:
        # Include the building that crosses τ (may slightly overshoot by one weight).
        # With uniform ResStock weights this is exact to 1/N of occupied stock.
        if cumulative >= target_weight:
            break
        evs[idx] = 1
        cumulative += float(weights[idx])
    return evs


@dataclass
class EVAdoptionSampler:
    """Assign EV ownership (0 or 1) from NREL ResStock lookup probabilities."""

    ev_ownership_df: pl.DataFrame
    random_state: int = 42
    # None = Bernoulli(p_i) at baseline ResStock rates.
    # Else = priority-sample on p_i (key u_i/p_i) and take a ResStock-weight prefix
    # totaling this occupied housing-stock share.
    target_adoption_rate: float | None = None

    def __post_init__(self) -> None:
        if self.target_adoption_rate is not None and not (0.0 <= self.target_adoption_rate <= 1.0):
            raise ValueError(
                f"target_adoption_rate must be in [0, 1]; got {self.target_adoption_rate}"
            )

    def sample(self, metadata_df: pl.DataFrame) -> pl.DataFrame:
        """
        Predict whether each household has an EV (0 or 1) using NREL's ResStock lookup.

        Always:
          1. Join segment P(EV) → baseline ``p_i`` (vacant → 0).
          2. Draw one uniform ``u_i ~ U(0,1)`` **per ResStock building** (not per segment).

        Then either:
          - ``target_adoption_rate is None`` (baseline): Bernoulli — ``evs=1`` iff
            occupied and ``u_i < p_i``.
          - ``target_adoption_rate = τ``: priority-sample occupied buildings on key
            ``u_i / p_i`` (ascending) and assign EVs to a prefix whose ResStock
            ``weight`` share equals ``τ``. Segment adoption *rates* stay proportional
            to ``p_i``; ``p_i == 0`` buildings adopt only after all ``p > 0`` stock is
            taken. Same ``random_state`` + metadata row order ⇒ nested EV sets as
            ``τ`` rises.

        Args:
            metadata_df: DataFrame with ResStock metadata including fpl, building_type,
                puma_dependency, tenure, and is_vacant columns. ``weight`` is required
                when ``target_adoption_rate`` is set.

        Returns:
            DataFrame with added columns: ``ev_ownership_probability`` (baseline) and
            ``evs``.

        Raises:
            ValueError: If the metadata DataFrame is missing required columns or an
                occupied building has no matching EV ownership lookup row.
        """
        required_columns = {"fpl", "building_type", "tenure", "puma_dependency", "is_vacant"}
        # Stock-share targets need ResStock building weights (housing units represented).
        if self.target_adoption_rate is not None:
            required_columns = required_columns | {"weight"}
        missing_columns = required_columns - set(metadata_df.columns)
        if missing_columns:
            raise ValueError(
                "Missing EV adoption metadata columns: "
                + ", ".join(sorted(missing_columns))
                + ". Ensure load_metadata() was used or provide these columns."
            )

        # Segment-level lookup: P(EV | FPL, building type, PUMA, tenure).
        # Many buildings can map to the same segment and therefore the same p.
        ev_lookup = self.ev_ownership_df.select(
            "fpl",
            "building_type",
            "puma_dependency",
            "tenure",
            "ev_ownership_probability",
        )

        # Attach baseline p_i to each ResStock building row (left join preserves
        # one row per building). Vacant stock is forced to p=0.
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

        # Arrays aligned with metadata rows: index i = ResStock building.
        baseline = metadata_with_prob.get_column("ev_ownership_probability").to_numpy()
        occupied_mask = ~metadata_with_prob.get_column("is_vacant").to_numpy()

        # One u_i per building row (building-level, not segment-level).
        # Same random_state + row order ⇒ same u_i across τ for nested sweeps.
        rng = np.random.default_rng(self.random_state)
        draws = rng.random(metadata_with_prob.height)

        if self.target_adoption_rate is None:
            # Baseline snapshot: independent coin flip with Prob = p_i per building.
            # Vacant rows keep evs=0. Differs from ResStock 2025 quota sampling.
            evs = np.where(
                ~occupied_mask,
                0,
                (draws < baseline).astype(np.int8),
            )
        else:
            # Targeted stock share τ (e.g. 0.25 = 25% of occupied housing stock).
            # Priority-sample only occupied buildings; vacants remain 0 in `evs`.
            evs = np.zeros(metadata_with_prob.height, dtype=np.int8)
            if occupied_mask.any():
                occupied_probs = baseline[occupied_mask]
                occupied_weights = metadata_with_prob.get_column("weight").to_numpy()[occupied_mask]
                occupied_draws = draws[occupied_mask]
                occupied_evs = assign_evs_by_priority_sampling(
                    occupied_probs,
                    occupied_weights,
                    occupied_draws,
                    self.target_adoption_rate,
                )
                evs[occupied_mask] = occupied_evs

        return metadata_with_prob.with_columns(
            pl.Series("evs", evs),
        )
