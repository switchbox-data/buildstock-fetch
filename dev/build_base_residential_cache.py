"""Build weight-weighted base residential hourly load cache for ev_analysis.ipynb."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

BSF_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BSF_ROOT))

from utils.EVs.charging import build_hours_base
from utils.EVs.ev_demand import load_ev_demand_config

CACHE_DIR = (
    BSF_ROOT
    / "utils/EVs/ev_data/outputs/MD_res_2024_tmy3_2/adoption_peak_sweep"
)
PATH_LOADS_DIR = Path(
    "/ebs/data/nrel/resstock/res_2024_amy2018_2_sb/load_curve_hourly/state=MD/upgrade=00"
)
ELEC_COL = "out.electricity.total.energy_consumption"
OUT = CACHE_DIR / "base_residential_weighted_kwh_f32.npy"


def main() -> int:
    meta = pl.read_parquet(CACHE_DIR / "building_meta.parquet")
    bldg_ids = meta["bldg_id"].to_numpy()
    weights = meta["weight"].to_numpy().astype(np.float64)
    cfg = load_ev_demand_config(BSF_ROOT / "utils/EVs/configs/md_2024.yaml")
    hours_base = build_hours_base(cfg.start_date, cfg.end_date)
    n_hours = hours_base.height
    ts_list = hours_base["timestamp"].to_list()
    cal_to_idx = {(ts.month, ts.day, ts.hour): i for i, ts in enumerate(ts_list)}

    sample_ts = pl.read_parquet(
        PATH_LOADS_DIR / f"{int(bldg_ids[0])}-0.parquet", columns=["timestamp"]
    )["timestamp"]
    amy_months = sample_ts.dt.month().to_numpy()
    amy_days = sample_ts.dt.day().to_numpy()
    amy_hours = sample_ts.dt.hour().to_numpy()
    n_amy = sample_ts.len()
    amy_to_hb = np.array(
        [
            cal_to_idx.get((int(amy_months[j]), int(amy_days[j]), int(amy_hours[j])), -1)
            for j in range(n_amy)
        ],
        dtype=np.int32,
    )
    matched = amy_to_hb >= 0
    hb_idx = amy_to_hb[matched]
    amy_idx = np.nonzero(matched)[0]

    print(f"Building base load for {len(bldg_ids):,} buildings → {OUT}", flush=True)
    base_load = np.zeros(n_hours, dtype=np.float64)
    t0 = time.time()
    n_missing = 0
    for i, (bid, w) in enumerate(zip(bldg_ids.tolist(), weights.tolist())):
        path = PATH_LOADS_DIR / f"{int(bid)}-0.parquet"
        if not path.exists():
            n_missing += 1
            continue
        elec = (
            pl.read_parquet(path, columns=[ELEC_COL])[ELEC_COL]
            .to_numpy()
            .astype(np.float64)
        )
        base_load[hb_idx] += w * elec[amy_idx]
        if (i + 1) % 1000 == 0 or i + 1 == len(bldg_ids):
            print(
                f"  {i + 1:,}/{len(bldg_ids):,}  elapsed={(time.time() - t0) / 60:.1f} min",
                flush=True,
            )

    covered = np.zeros(n_hours, dtype=bool)
    covered[hb_idx] = True
    base_load[~covered] = np.nan
    np.save(OUT, base_load.astype(np.float32))
    print(
        f"DONE peak_MW={np.nanmax(base_load) / 1000:.2f}  "
        f"finite={np.isfinite(base_load).sum()}  missing={n_missing}  "
        f"in {(time.time() - t0) / 60:.1f} min",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
