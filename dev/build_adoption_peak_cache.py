"""Build / load τ=100% EV charge matrix cache for ``ev_analysis.ipynb``.

Caches **all four** charging strategies (trips/attrs once per batch; SOC per strategy):

- ``immediate``
- ``off_peak``
- ``off_peak_immediate``
- ``cost_minimizing``

Strategy knobs (``soc_min_fraction``, ``daily_price_usd_per_kwh``, …) are read from
``utils/EVs/configs/md_2024.yaml`` even when the YAML's active ``charging_strategy``
is only one of the four.

Output (``utils/EVs/ev_data/outputs/<state>_<release>/adoption_peak_sweep/``)
----------------------------------------------------------------------------
- ``charge_kwh_by_strategy.npz`` — float32 arrays keyed by strategy name
- ``underflow_hours_by_strategy.npz`` — int32 per-vehicle underflow hour counts
- ``building_meta.parquet`` — ``bldg_id``, ``weight``, ``annual_discharge_kwh``
- ``hours_base.parquet`` — simulation hour calendar
- ``failed_buildings.parquet`` — battery/charger infeasible buildings (if any)

Usage:
  uv run python dev/build_adoption_peak_cache.py
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Imports & path bootstrap
# ---------------------------------------------------------------------------
import logging
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

# Allow `uv run python dev/...` to import `utils.EVs` without installing the package.
BSF_ROOT = Path(__file__).resolve().parents[1]
if str(BSF_ROOT) not in sys.path:
    sys.path.insert(0, str(BSF_ROOT))

from utils.EVs.charging import (
    DEFAULT_SOC_MIN_FRACTION,
    DEFAULT_SOC_SAFETY_BUFFER_FRACTION,
    build_hours_base,
)
from utils.EVs.ev_demand import (
    EVDemandCalculator,
    EVDemandConfig,
    load_ev_demand_config,
    resolve_hourly_prices,
)
from utils.EVs import ev_utils
from utils.EVs.ev_utils import EVDemandInputs

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------------
# Constants: config path, batching, strategy set, and on-disk artifact names
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = BSF_ROOT / "utils" / "EVs" / "configs" / "md_2024.yaml"
DEFAULT_BATCH_SIZE = 50

# Order matches presentation notebook fleet plots.
CHARGING_STRATEGIES: tuple[str, ...] = (
    "immediate",
    "off_peak",
    "off_peak_immediate",
    "cost_minimizing",
)

CHARGE_NPZ_NAME = "charge_kwh_by_strategy.npz"
UNDERFLOW_NPZ_NAME = "underflow_hours_by_strategy.npz"
# Legacy single-strategy file (pre multi-strategy caches).
LEGACY_CHARGE_NPY_NAME = "charge_kwh_f32.npy"
META_PARQUET_NAME = "building_meta.parquet"
HOURS_PARQUET_NAME = "hours_base.parquet"
FAILED_PARQUET_NAME = "failed_buildings.parquet"


# ---------------------------------------------------------------------------
# Cache container returned to notebooks / callers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChargeCache:
    """τ=100% per-building hourly charge for each charging strategy."""

    # strategy → float32 [n_buildings, n_hours], rows aligned with building_meta
    charge_by_strategy: dict[str, np.ndarray]
    # strategy → int32 [n_buildings] SOC underflow hour counts (0 if cache predates this)
    underflow_hours_by_strategy: dict[str, np.ndarray]
    building_meta: pl.DataFrame  # bldg_id, weight [, annual_discharge_kwh]
    hours_base: pl.DataFrame
    failed_df: pl.DataFrame  # bldg_id, error
    cache_dir: Path
    strategies: tuple[str, ...] = CHARGING_STRATEGIES
    underflow_available: bool = True

    @property
    def charge_mat(self) -> np.ndarray:
        """Default strategy matrix (config / off_peak_immediate) for simple callers."""
        preferred = "off_peak_immediate"
        if preferred in self.charge_by_strategy:
            return self.charge_by_strategy[preferred]
        return next(iter(self.charge_by_strategy.values()))


def strategy_energy_underflow_summary(cache: ChargeCache) -> pl.DataFrame:
    """Fleet-level annual charge / discharge / underflow hours (presentation-style)."""
    n = cache.building_meta.height
    if "annual_discharge_kwh" in cache.building_meta.columns:
        discharge_total = float(cache.building_meta["annual_discharge_kwh"].sum())
    else:
        discharge_total = float("nan")

    rows: list[dict[str, Any]] = []
    for strategy in cache.strategies:
        charge = cache.charge_by_strategy[strategy]
        uf = cache.underflow_hours_by_strategy.get(
            strategy, np.zeros(n, dtype=np.int32)
        )
        rows.append(
            {
                "strategy": strategy,
                "annual_charge_kwh": round(float(charge.sum()), 1),
                "annual_discharge_kwh": round(discharge_total, 1)
                if discharge_total == discharge_total
                else None,
                "underflow_hours": int(uf.sum()),
                "n_vehicles_with_underflow": int((uf > 0).sum()),
            }
        )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cache path helpers & existence checks
# ---------------------------------------------------------------------------
def adoption_peak_cache_dir(
    config: EVDemandConfig,
    *,
    root: Path = BSF_ROOT,
) -> Path:
    """Directory for the adoption-peak sweep cache for this state/release."""
    return (
        root
        / "utils"
        / "EVs"
        / "ev_data"
        / "outputs"
        / f"{config.state}_{config.release}"
        / "adoption_peak_sweep"
    )


def _charge_npz_complete(cache_dir: Path) -> bool:
    """True when multi-strategy charge matrices + meta/hours are present."""
    npz = cache_dir / CHARGE_NPZ_NAME
    if not (
        npz.exists()
        and (cache_dir / META_PARQUET_NAME).exists()
        and (cache_dir / HOURS_PARQUET_NAME).exists()
    ):
        return False
    with np.load(npz) as data:
        return all(s in data.files for s in CHARGING_STRATEGIES)


def _underflow_npz_complete(cache_dir: Path) -> bool:
    """True when per-strategy underflow hour counts are present."""
    npz = cache_dir / UNDERFLOW_NPZ_NAME
    if not npz.exists():
        return False
    with np.load(npz) as data:
        return all(s in data.files for s in CHARGING_STRATEGIES)


def cache_exists(cache_dir: Path) -> bool:
    """True when multi-strategy charge artifacts are present (underflow optional)."""
    return _charge_npz_complete(cache_dir)


# ---------------------------------------------------------------------------
# Load a previously written cache (with legacy-file hint on failure)
# ---------------------------------------------------------------------------
def load_charge_cache(cache_dir: Path) -> ChargeCache:
    """Load a previously written multi-strategy charge matrix cache."""
    npz_path = cache_dir / CHARGE_NPZ_NAME
    underflow_path = cache_dir / UNDERFLOW_NPZ_NAME
    meta_path = cache_dir / META_PARQUET_NAME
    hours_path = cache_dir / HOURS_PARQUET_NAME
    failed_path = cache_dir / FAILED_PARQUET_NAME

    # Incomplete / old single-strategy caches must be rebuilt.
    if not cache_exists(cache_dir):
        legacy = cache_dir / LEGACY_CHARGE_NPY_NAME
        hint = (
            f" Incomplete multi-strategy cache under {cache_dir}."
            + (
                f" Found legacy {LEGACY_CHARGE_NPY_NAME}; rebuild with "
                "`uv run python dev/build_adoption_peak_cache.py`."
                if legacy.exists()
                else ""
            )
        )
        raise FileNotFoundError(hint)

    building_meta = pl.read_parquet(meta_path)
    hours_base = pl.read_parquet(hours_path)
    # Failed list is optional; empty schema if the file was never written.
    failed_df = (
        pl.read_parquet(failed_path)
        if failed_path.exists()
        else pl.DataFrame(schema={"bldg_id": pl.Int64, "error": pl.Utf8})
    )

    # Load each strategy array and sanity-check shapes vs meta/hours.
    charge_by_strategy: dict[str, np.ndarray] = {}
    with np.load(npz_path) as data:
        for strategy in CHARGING_STRATEGIES:
            mat = data[strategy]
            if hours_base.height != mat.shape[1]:
                raise ValueError(
                    f"{strategy}: hours_base length {hours_base.height} != "
                    f"charge columns {mat.shape[1]}"
                )
            if building_meta.height != mat.shape[0]:
                raise ValueError(
                    f"{strategy}: building_meta rows {building_meta.height} != "
                    f"charge rows {mat.shape[0]}"
                )
            charge_by_strategy[strategy] = mat

    n_ev = building_meta.height
    underflow_hours_by_strategy: dict[str, np.ndarray] = {}
    underflow_available = _underflow_npz_complete(cache_dir)
    if underflow_available:
        with np.load(underflow_path) as data:
            for strategy in CHARGING_STRATEGIES:
                uf = np.asarray(data[strategy], dtype=np.int32)
                if uf.shape != (n_ev,):
                    raise ValueError(
                        f"{strategy}: underflow length {uf.shape} != "
                        f"building_meta rows {(n_ev,)}"
                    )
                underflow_hours_by_strategy[strategy] = uf
    else:
        underflow_hours_by_strategy = {
            s: np.zeros(n_ev, dtype=np.int32) for s in CHARGING_STRATEGIES
        }

    return ChargeCache(
        charge_by_strategy=charge_by_strategy,
        underflow_hours_by_strategy=underflow_hours_by_strategy,
        building_meta=building_meta,
        hours_base=hours_base,
        failed_df=failed_df,
        cache_dir=cache_dir,
        strategies=CHARGING_STRATEGIES,
        underflow_available=underflow_available,
    )


# ---------------------------------------------------------------------------
# Per-strategy kwargs for generate_soc_schedules
# ---------------------------------------------------------------------------
def _soc_kwargs_for_strategy(
    config: EVDemandConfig,
    strategy: str,
    *,
    hours_base: pl.DataFrame,
    duty: pl.DataFrame,
    hourly_prices: np.ndarray | None,
) -> dict[str, Any]:
    """Build ``generate_soc_schedules`` kwargs for one charging strategy."""
    # Shared by all strategies: identity, calendar, and temperature-scaled miles.
    kwargs: dict[str, Any] = {
        "charging_strategy": strategy,
        "initial_soc_kwh": config.initial_soc_kwh,
        "hours_base": hours_base,
        "hourly_temp_scaled_miles": duty,
    }
    if strategy == "immediate":
        # Charge as soon as parked; no peak / price knobs.
        return kwargs
    if strategy == "off_peak":
        # Defer charging outside peak hours; respect SOC floors / buffers.
        kwargs.update(config.peak_window_kwargs())
        kwargs["soc_min_fraction"] = (
            config.soc_min_fraction
            if config.soc_min_fraction is not None
            else DEFAULT_SOC_MIN_FRACTION
        )
        kwargs["soc_safety_buffer_fraction"] = (
            config.soc_safety_buffer_fraction
            if config.soc_safety_buffer_fraction is not None
            else DEFAULT_SOC_SAFETY_BUFFER_FRACTION
        )
        return kwargs
    if strategy == "off_peak_immediate":
        # Prefer off-peak, optionally allow emergency peak charging.
        kwargs.update(config.peak_window_kwargs())
        kwargs["allow_emergency_peak_charging"] = bool(
            config.allow_emergency_peak_charging
        )
        return kwargs
    if strategy == "cost_minimizing":
        # Needs a full hourly price series and a shed-load penalty from config.
        if hourly_prices is None:
            raise ValueError(
                "cost_minimizing requires prices in the config "
                "(seasonal TOU / daily_price_usd_per_kwh / flat / hourly_price_path)"
            )
        if config.shed_load_penalty_usd_per_kwh is None:
            raise ValueError("cost_minimizing requires shed_load_penalty_usd_per_kwh")
        kwargs["hourly_price_usd_per_kwh"] = hourly_prices
        kwargs["shed_load_penalty_usd_per_kwh"] = config.shed_load_penalty_usd_per_kwh
        return kwargs
    raise ValueError(f"unknown charging strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Long-form SOC → charge matrix + per-vehicle underflow / discharge totals
# ---------------------------------------------------------------------------
def _soc_to_charge_and_underflow(
    soc: pl.DataFrame,
    attrs: pl.DataFrame,
    n_hours: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse long-form SOC to charge matrix + underflow/discharge vectors.

    Returns
    -------
    charge : float32 [n_ev, n_hours] in ``attrs`` row order
    underflow_hours : int32 [n_ev] count of ``soc_underflow`` hours
    annual_discharge_kwh : float64 [n_ev] sum of trip discharge
    """
    n_ev = attrs.height
    if n_ev == 0 or soc.is_empty():
        return (
            np.zeros((0, n_hours), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float64),
        )

    # Per-vehicle lists / totals, then left-join onto attrs for stable row order.
    per_veh = (
        soc.sort("bldg_id", "vehicle_id", "hour_index")
        .group_by("bldg_id", "vehicle_id", maintain_order=True)
        .agg(
            pl.col("charge_kwh"),
            pl.col("soc_underflow").sum().cast(pl.Int32).alias("underflow_hours"),
            pl.col("discharge_kwh").sum().alias("annual_discharge_kwh"),
        )
    )
    aligned = attrs.select("bldg_id", "vehicle_id").join(
        per_veh, on=["bldg_id", "vehicle_id"], how="left"
    )
    charge_rows = [
        np.asarray(c, dtype=np.float32) if c is not None else np.zeros(n_hours, dtype=np.float32)
        for c in aligned["charge_kwh"].to_list()
    ]
    charge = np.vstack(charge_rows) if charge_rows else np.zeros((0, n_hours), dtype=np.float32)
    underflow = (
        aligned["underflow_hours"]
        .fill_null(0)
        .to_numpy()
        .astype(np.int32, copy=False)
    )
    discharge = (
        aligned["annual_discharge_kwh"]
        .fill_null(0.0)
        .to_numpy()
        .astype(np.float64, copy=False)
    )
    return charge, underflow, discharge


# ---------------------------------------------------------------------------
# Core batch pipeline: trips once, then SOC for every strategy
# ---------------------------------------------------------------------------
def _empty_strategy_mats(
    n_hours: int,
    strategies: tuple[str, ...] = CHARGING_STRATEGIES,
) -> dict[str, np.ndarray]:
    """Zero-row placeholder charge matrices when a batch has no successful EVs."""
    return {s: np.zeros((0, n_hours), dtype=np.float32) for s in strategies}


def _empty_underflow(
    strategies: tuple[str, ...] = CHARGING_STRATEGIES,
) -> dict[str, np.ndarray]:
    """Zero-row placeholder underflow vectors."""
    return {s: np.zeros(0, dtype=np.int32) for s in strategies}


def _run_pipeline_charge_all_strategies(
    batch_md: pl.DataFrame,
    *,
    config_full: EVDemandConfig,
    inputs: EVDemandInputs,
    hours_base: pl.DataFrame,
    hourly_prices: np.ndarray | None,
    strategies: tuple[str, ...] = CHARGING_STRATEGIES,
) -> tuple[pl.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Trips/attrs once, then SOC charge + underflow for each strategy.

    Returns
    -------
    attrs_w : attrs with bldg_id/weight
    charge_by_strategy : strategy → float32 [n_ev, n_hours]
    underflow_by_strategy : strategy → int32 [n_ev]
    annual_discharge_kwh : float64 [n_ev] (strategy-independent trip draw)
    """
    # Wire calculator to this batch's metadata + shared lookup tables.
    calc = EVDemandCalculator.from_config(
        config_full,
        metadata_df=batch_md,
        nhts_df=inputs.nhts_df,
        ev_ownership_df=inputs.ev_ownership_df,
        ev_battery_df=inputs.ev_battery_df,
        ev_autonomie_df=inputs.ev_autonomie_df,
        ev_charger_df=inputs.ev_charger_df,
        ev_charge_at_home_df=inputs.ev_charge_at_home_df,
        weather_map=inputs.weather_map,
        station_temps=inputs.station_temps,
    )
    # Expensive step shared across strategies: trip matching + duty cycles.
    trips, attrs, duty = calc.match_and_generate_trip_schedules(hours_base=hours_base)
    n_hours = hours_base.height
    if attrs.is_empty():
        return (
            attrs,
            _empty_strategy_mats(n_hours, strategies),
            _empty_underflow(strategies),
            np.zeros(0, dtype=np.float64),
        )

    # Cheap(er) per-strategy SOC rollouts reuse the same trips/attrs/duty.
    matrices: dict[str, np.ndarray] = {}
    underflow: dict[str, np.ndarray] = {}
    annual_discharge: np.ndarray | None = None
    for strategy in strategies:
        soc = calc.generate_soc_schedules(
            trips,
            attrs,
            **_soc_kwargs_for_strategy(
                config_full,
                strategy,
                hours_base=hours_base,
                duty=duty,
                hourly_prices=hourly_prices,
            ),
        )
        charge, uf_hours, discharge = _soc_to_charge_and_underflow(soc, attrs, n_hours)
        matrices[strategy] = charge
        underflow[strategy] = uf_hours
        # Discharge is trip-driven and identical across strategies — keep first.
        if annual_discharge is None:
            annual_discharge = discharge

    assert annual_discharge is not None
    # Attach ResStock sample weights for later adoption-rate scaling.
    attrs_w = attrs.join(batch_md.select("bldg_id", "weight"), on="bldg_id", how="left")
    return attrs_w, matrices, underflow, annual_discharge


# ---------------------------------------------------------------------------
# Batch runner with binary-split skip of infeasible buildings
# ---------------------------------------------------------------------------
def _vstack_or_take(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Concatenate non-empty arrays along axis 0 (tolerate empty halves)."""
    if left.shape[0] == 0:
        return right
    if right.shape[0] == 0:
        return left
    return np.concatenate([left, right], axis=0)


def _generate_charge_with_skip(
    batch_md: pl.DataFrame,
    *,
    config_full: EVDemandConfig,
    inputs: EVDemandInputs,
    hours_base: pl.DataFrame,
    hourly_prices: np.ndarray | None,
    strategies: tuple[str, ...] = CHARGING_STRATEGIES,
) -> tuple[
    list[dict],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    list[dict],
]:
    """Like ``_run_pipeline_charge_all_strategies``, isolating infeasible buildings.

    On ValueError (e.g. battery/charger can't meet duty), bisect the batch until
    the bad building(s) are identified; return successes + a failed list.
    """
    n_hours = hours_base.height
    if batch_md.height == 0:
        return (
            [],
            _empty_strategy_mats(n_hours, strategies),
            _empty_underflow(strategies),
            np.zeros(0, dtype=np.float64),
            [],
        )
    try:
        attrs_w, matrices, underflow, discharge = _run_pipeline_charge_all_strategies(
            batch_md,
            config_full=config_full,
            inputs=inputs,
            hours_base=hours_base,
            hourly_prices=hourly_prices,
            strategies=strategies,
        )
        # One meta row per successful EV (bldg_id may repeat if multi-vehicle).
        meta_rows = [
            {
                "bldg_id": int(r["bldg_id"]),
                "weight": float(r["weight"]),
                "annual_discharge_kwh": float(discharge[i]),
            }
            for i, r in enumerate(
                attrs_w.select("bldg_id", "weight").iter_rows(named=True)
            )
        ]
        return meta_rows, matrices, underflow, discharge, []
    except ValueError as exc:
        # Leaf of the bisect: record this single building as failed.
        if batch_md.height == 1:
            return (
                [],
                _empty_strategy_mats(n_hours, strategies),
                _empty_underflow(strategies),
                np.zeros(0, dtype=np.float64),
                [
                    {
                        "bldg_id": int(batch_md["bldg_id"][0]),
                        "error": str(exc).split("\n")[0][:300],
                    }
                ],
            )
        # Otherwise split in half and recurse; merge successful matrices.
        mid = batch_md.height // 2
        (
            left_meta,
            left_mats,
            left_uf,
            left_dis,
            left_fail,
        ) = _generate_charge_with_skip(
            batch_md.slice(0, mid),
            config_full=config_full,
            inputs=inputs,
            hours_base=hours_base,
            hourly_prices=hourly_prices,
            strategies=strategies,
        )
        (
            right_meta,
            right_mats,
            right_uf,
            right_dis,
            right_fail,
        ) = _generate_charge_with_skip(
            batch_md.slice(mid, batch_md.height - mid),
            config_full=config_full,
            inputs=inputs,
            hours_base=hours_base,
            hourly_prices=hourly_prices,
            strategies=strategies,
        )
        matrices = {
            strategy: _vstack_or_take(left_mats[strategy], right_mats[strategy])
            for strategy in strategies
        }
        underflow = {
            strategy: _vstack_or_take(left_uf[strategy], right_uf[strategy])
            for strategy in strategies
        }
        discharge = _vstack_or_take(left_dis, right_dis)
        return (
            left_meta + right_meta,
            matrices,
            underflow,
            discharge,
            left_fail + right_fail,
        )


# ---------------------------------------------------------------------------
# Full build: load inputs, batch over buildings, write artifacts
# ---------------------------------------------------------------------------
def build_charge_cache(
    config: EVDemandConfig,
    *,
    inputs: EVDemandInputs | None = None,
    metadata_df: pl.DataFrame | None = None,
    max_buildings: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    cache_dir: Path | None = None,
    root: Path = BSF_ROOT,
    strategies: tuple[str, ...] = CHARGING_STRATEGIES,
) -> ChargeCache:
    """Generate τ=100% charge matrices for all strategies and write cache files."""
    # Resolve inputs / metadata (optional overrides for notebooks / tests).
    if inputs is None:
        inputs = ev_utils.load_all_input_data(config)
    if metadata_df is None:
        metadata_df = inputs.metadata_df
    if max_buildings is not None:
        metadata_df = metadata_df.head(max_buildings)

    # Force full adoption so every eligible home contributes a charge profile.
    config_full = replace(config, target_adoption_rate=1.0)
    hours_base = build_hours_base(config.start_date, config.end_date)
    hourly_prices = resolve_hourly_prices(config_full)
    if cache_dir is None:
        cache_dir = adoption_peak_cache_dir(config, root=root)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Starting τ=100% multi-strategy charge cache for {metadata_df.height:,} buildings "
        f"→ {cache_dir}\n  strategies={list(strategies)}",
        flush=True,
    )
    if "cost_minimizing" in strategies and hourly_prices is None:
        raise ValueError(
            "cost_minimizing is requested but no prices are set in the config "
            "(uncomment daily_price_usd_per_kwh / shed_load_penalty in md_2024.yaml)"
        )

    # Accumulators across batches.
    t0 = time.time()
    meta_acc: list[dict] = []
    mats_acc: dict[str, list[np.ndarray]] = {s: [] for s in strategies}
    uf_acc: dict[str, list[np.ndarray]] = {s: [] for s in strategies}
    failed_acc: list[dict] = []
    n_batches = (metadata_df.height + batch_size - 1) // batch_size

    for batch_i, start in enumerate(range(0, metadata_df.height, batch_size)):
        meta_rows, matrices, underflow, _discharge, failed = _generate_charge_with_skip(
            metadata_df.slice(start, batch_size),
            config_full=config_full,
            inputs=inputs,
            hours_base=hours_base,
            hourly_prices=hourly_prices,
            strategies=strategies,
        )
        meta_acc.extend(meta_rows)
        failed_acc.extend(failed)
        for strategy, matrix in matrices.items():
            if matrix.shape[0]:
                mats_acc[strategy].append(matrix)
                uf_acc[strategy].append(underflow[strategy])
        print(
            f"batch {batch_i + 1}/{n_batches}: ok={len(meta_acc)} fail={len(failed_acc)} "
            f"elapsed={(time.time() - t0) / 60:.1f} min",
            flush=True,
        )

    # Stack batch chunks → final [n_ev, n_hours] / [n_ev] per strategy.
    charge_by_strategy = {
        strategy: (
            np.vstack(chunks)
            if chunks
            else np.zeros((0, hours_base.height), dtype=np.float32)
        )
        for strategy, chunks in mats_acc.items()
    }
    underflow_hours_by_strategy = {
        strategy: (
            np.concatenate(chunks).astype(np.int32, copy=False)
            if chunks
            else np.zeros(0, dtype=np.int32)
        )
        for strategy, chunks in uf_acc.items()
    }
    building_meta = (
        pl.DataFrame(meta_acc)
        if meta_acc
        else pl.DataFrame(
            schema={
                "bldg_id": pl.Int64,
                "weight": pl.Float64,
                "annual_discharge_kwh": pl.Float64,
            }
        )
    )
    failed_df = (
        pl.DataFrame(failed_acc)
        if failed_acc
        else pl.DataFrame(schema={"bldg_id": pl.Int64, "error": pl.Utf8})
    )

    # Persist: npz of matrices + underflow counts + parquet sidecars.
    np.savez_compressed(cache_dir / CHARGE_NPZ_NAME, **charge_by_strategy)
    np.savez_compressed(cache_dir / UNDERFLOW_NPZ_NAME, **underflow_hours_by_strategy)
    building_meta.write_parquet(cache_dir / META_PARQUET_NAME)
    hours_base.write_parquet(cache_dir / HOURS_PARQUET_NAME)
    failed_df.write_parquet(cache_dir / FAILED_PARQUET_NAME)
    shapes = {s: charge_by_strategy[s].shape for s in strategies}
    uf_totals = {s: int(underflow_hours_by_strategy[s].sum()) for s in strategies}
    print(
        f"DONE shapes={shapes} fail={failed_df.height} "
        f"underflow_hours={uf_totals} "
        f"in {(time.time() - t0) / 60:.1f} min",
        flush=True,
    )
    return ChargeCache(
        charge_by_strategy=charge_by_strategy,
        underflow_hours_by_strategy=underflow_hours_by_strategy,
        building_meta=building_meta,
        hours_base=hours_base,
        failed_df=failed_df,
        cache_dir=cache_dir,
        strategies=strategies,
        underflow_available=True,
    )


# ---------------------------------------------------------------------------
# Convenience: load if complete, otherwise build
# ---------------------------------------------------------------------------
def load_or_build_charge_cache(
    config: EVDemandConfig,
    *,
    force_rebuild: bool = False,
    inputs: EVDemandInputs | None = None,
    metadata_df: pl.DataFrame | None = None,
    max_buildings: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    cache_dir: Path | None = None,
    root: Path = BSF_ROOT,
    strategies: tuple[str, ...] = CHARGING_STRATEGIES,
) -> ChargeCache:
    """Load multi-strategy cache if present (and not forced), otherwise build it."""
    if cache_dir is None:
        cache_dir = adoption_peak_cache_dir(config, root=root)
    if cache_exists(cache_dir) and not force_rebuild:
        cache = load_charge_cache(cache_dir)
        first = next(iter(cache.charge_by_strategy.values()))
        print(
            f"Loaded multi-strategy cache: {first.shape[0]:,} buildings × "
            f"{first.shape[1]:,} hours × {len(cache.strategies)} strategies "
            f"({(cache_dir / CHARGE_NPZ_NAME).stat().st_size / 1e6:.1f} MB)",
            flush=True,
        )
        print(f"strategies={list(cache.strategies)}", flush=True)
        print(f"Skipped infeasible buildings: {cache.failed_df.height:,}", flush=True)
        if cache.underflow_available:
            uf_totals = {
                s: int(cache.underflow_hours_by_strategy[s].sum())
                for s in cache.strategies
            }
            print(f"underflow_hours={uf_totals}", flush=True)
        else:
            print(
                "WARNING: underflow_hours_by_strategy.npz missing — rebuild with "
                "`FORCE_REBUILD_CACHE=True` or "
                "`uv run python dev/build_adoption_peak_cache.py` to report "
                "SOC underflow hours.",
                flush=True,
            )
        return cache
    return build_charge_cache(
        config,
        inputs=inputs,
        metadata_df=metadata_df,
        max_buildings=max_buildings,
        batch_size=batch_size,
        cache_dir=cache_dir,
        root=root,
        strategies=strategies,
    )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
def main() -> int:
    """CLI: build full-MD multi-strategy cache for the default MD 2024 config."""
    config = load_ev_demand_config(DEFAULT_CONFIG_PATH)
    build_charge_cache(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
