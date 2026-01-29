from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

DATA_ROOT = Path("top12_stage2_data")
PREFIX_RMSD = "rmsd_protein"
PREFIX_RG = "rg_protein"
PREFIX_SASA = "sasa_protein"
PREFIX_HB_L2P_AVG = "hb_lig_prot_avg"
PREFIX_HB_P2L_AVG = "hb_PROT_to_LIG_S2_3rep_avg"

def find_first_dat(lig_dir: Path, prefix: str) -> Path:
    candidates = sorted(lig_dir.glob(f"{prefix}*.dat"))
    if not candidates:
        raise FileNotFoundError(f"{lig_dir}: can't find {prefix}*.dat")
    if len(candidates) > 1:
        print(f"[WARN] {lig_dir.name}: find {prefix}*.dat，use {candidates[0].name}")
    return candidates[0]


def load_xy(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, comments=['#', '@'])

    if arr.ndim == 1:
        y = arr.astype(float)
        x = np.arange(len(y), dtype=float)
    else:
        x = arr[:, 0].astype(float)
        y = arr[:, 1].astype(float)
        if x.max() > 50:
            x = x / 1000.0
    return x, y


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    start = int(len(x) * 0.2)
    x_fit = x[start:]
    y_fit = y[start:]
    A = np.vstack([x_fit, np.ones_like(x_fit)]).T
    m, _b = np.linalg.lstsq(A, y_fit, rcond=None)[0]
    return float(m)


def max_deviation_percent(series: np.ndarray) -> float:
    mean = float(series.mean())
    if mean == 0:
        return np.nan
    dev = np.max(np.abs(series - mean))
    return float(dev / mean * 100.0)


def parse_hbond_avg(path_avg: Path) -> Tuple[float, int, float]:
    if not path_avg.exists():
        return 0.0, 0, 0.0
    fracs: List[float] = []
    with path_avg.open() as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith(('#', '@')):
                continue
            parts = line.split()
            try:
                frac = float(parts[4])
            except ValueError:
                continue
            fracs.append(frac)

    if not fracs:
        return 0.0, 0, 0.0
    fracs_arr = np.asarray(fracs, dtype=float)
    main_occ_pct = float(fracs_arr.max() * 100.0)
    n_frac_ge_0_1 = int((fracs_arr >= 0.1).sum())
    mean_count = float(fracs_arr.sum())
    return main_occ_pct, n_frac_ge_0_1, mean_count

def analyze_one_ligand(lig_dir: Path) -> dict:
    lig_id = int(lig_dir.name)

    # --- RMSD ---
    x_rmsd, y_rmsd = load_xy(find_first_dat(lig_dir, PREFIX_RMSD))
    rmsd_mean = float(y_rmsd.mean())
    rmsd_slope = linear_slope(x_rmsd, y_rmsd)  # Å/ns

    # --- Rg ---
    _x_rg, y_rg = load_xy(find_first_dat(lig_dir, PREFIX_RG))
    rg_mean = float(y_rg.mean())
    rg_dev_pct = max_deviation_percent(y_rg)

    # --- SASA ---
    _x_sasa, y_sasa = load_xy(find_first_dat(lig_dir, PREFIX_SASA))
    sasa_mean = float(y_sasa.mean())
    sasa_dev_pct = max_deviation_percent(y_sasa)

    # --- hbond:L→P 、 P→L ---
    main_occ_l2p, n_ge_0_1_l2p, hb_l2p_mean = parse_hbond_avg(
        find_first_dat(lig_dir, PREFIX_HB_L2P_AVG)
    )
    main_occ_p2l, n_ge_0_1_p2l, hb_p2l_mean = parse_hbond_avg(
        find_first_dat(lig_dir, PREFIX_HB_P2L_AVG)
    )

    # summary
    hb_mean_total = hb_l2p_mean + hb_p2l_mean
    main_occ_total = max(main_occ_l2p, main_occ_p2l)
    n_ge_0_1_total = n_ge_0_1_l2p + n_ge_0_1_p2l

    # thershold---
    cond_rmsd_level = rmsd_mean <= 2.5
    cond_rg = rg_dev_pct <= 3.0
    cond_sasa = sasa_dev_pct <= 10.0  
    cond_hb_mean = hb_mean_total >= 1.0
    cond_main_occ = main_occ_total >= 30.0
    cond_frac_ge_0_1 = n_ge_0_1_total >= 2

    pass_stage2 = all([
        cond_rmsd_level,
        cond_rg,
        cond_sasa,
        cond_hb_mean,
        cond_main_occ,
        cond_frac_ge_0_1,
    ])

    return {
        "id": lig_id,

        "rmsd_mean": rmsd_mean,
        "rmsd_slope_A_per_ns": rmsd_slope,
        "rg_mean": rg_mean,
        "rg_dev_pct": rg_dev_pct,
        "sasa_mean": sasa_mean,
        "sasa_dev_pct": sasa_dev_pct,

        "hb_mean_L2P": hb_l2p_mean,
        "hb_mean_P2L": hb_p2l_mean,
        "hb_mean_total": hb_mean_total,
        "hb_main_occ_pct": main_occ_total,
        "hb_n_frac_ge_0_1_total": n_ge_0_1_total,

        "cond_rmsd_level": cond_rmsd_level,
        "cond_rg": cond_rg,
        "cond_sasa": cond_sasa,
        "cond_hb_mean": cond_hb_mean,
        "cond_main_occ": cond_main_occ,
        "cond_frac_ge_0_1": cond_frac_ge_0_1,

        "pass_stage2": pass_stage2,
    }


def main():
    if not DATA_ROOT.exists():
        raise SystemExit(f"dataset not exist：{DATA_ROOT}")

    lig_dirs = [
        d for d in DATA_ROOT.iterdir()
        if d.is_dir() and d.name.isdigit()
    ]
    lig_dirs = sorted(lig_dirs, key=lambda p: int(p.name))

    results = []
    for d in lig_dirs:
        print(f"[Stage2] analyze ligand {d.name} ...")
        res = analyze_one_ligand(d)
        results.append(res)

    df = pd.DataFrame(results).sort_values("id")
    df.to_csv("stage2_summary.csv", index=False, encoding="utf-8-sig")

# Sort the ligands that have passed the screening
    top = df[df["pass_stage2"]].copy()
    top = top.sort_values(
    ["hb_mean_total", "hb_n_frac_ge_0_1_total", "hb_main_occ_pct", "rmsd_mean"],
    ascending=[False,          False,                    False,             True]
)

# take top5
    top5 = top.head(5)
    top5.to_csv("stage2_top5.csv", index=False, encoding="utf-8-sig")

    print("\nanalyze done:")
    print(df[["id", "pass_stage2"]])
    print(f"\nnumbers of ligands that have passed the screening:{len(top)}")
    print("TOP5:", list(top5["id"].values))
    print("see:stage2_summary.csv, stage2_top5.csv")



if __name__ == "__main__":
    main()
