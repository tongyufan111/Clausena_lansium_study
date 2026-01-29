import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_series(folder: Path, pattern: str) -> pd.DataFrame:
    files = sorted(folder.glob(f"{pattern}*.dat"), key=lambda p: len(p.name))
    if not files:
        raise FileNotFoundError(f"No {pattern}*.dat found in {folder}")
    df = pd.read_csv(
        files[0],
        delim_whitespace=True,
        comment="#",
        header=None
    )
    return df

def compute_metrics_for_folder(folder: Path, dt_ns: float = 0.01) -> dict:
    sim_id = int(folder.name)
    rmsd_df = load_series(folder, "rmsd_protein")
    frames = rmsd_df.iloc[:, 0].to_numpy()
    rmsd = rmsd_df.iloc[:, 1].to_numpy()
    mask = frames >= 201
    frames_sel = frames[mask]
    rmsd_sel = rmsd[mask]
    t_sel = (frames_sel - 1) * dt_ns 
    rmsd_mean = float(rmsd_sel.mean())
    rmsd_std = float(rmsd_sel.std(ddof=1))
    t_center = t_sel - t_sel.mean()
    slope = float((t_center * rmsd_sel).sum() / (t_center ** 2).sum())
    rg_df = load_series(folder, "rg_protein")
    rg_vals = rg_df.iloc[:, 1].to_numpy()
    rg_early = float(rg_vals[:200].mean())
    rg_late = float(rg_vals[-200:].mean())
    rg_drift_pct = float((rg_late - rg_early) / rg_early * 100.0)
    sasa_df = load_series(folder, "sasa_protein")
    sasa_vals = sasa_df.iloc[:, 1].to_numpy()
    sasa_early = float(sasa_vals[:200].mean())
    sasa_late = float(sasa_vals[-200:].mean())
    sasa_drift_pct = float((sasa_late - sasa_early) / sasa_early * 100.0)
    hb_df = load_series(folder, "hb_lig_prot_avg")
    
    if hb_df.shape[1] >= 5:
        fracs = hb_df.iloc[:, 4].astype(float).to_numpy()
        hb_avg_total = float(fracs.sum())
        hb_main_frac = float(fracs.max())
        hb_num_ge_0_1 = int((fracs >= 0.1).sum())
        hb_avg_total = 0.0
        hb_main_frac = 0.0
        hb_num_ge_0_1 = 0

    return {
        "id": sim_id,
        "rmsd_mean": rmsd_mean,
        "rmsd_std": rmsd_std,
        "rmsd_slope_per_ns": slope,
        "rg_early": rg_early,
        "rg_late": rg_late,
        "rg_drift_pct": rg_drift_pct,
        "sasa_early": sasa_early,
        "sasa_late": sasa_late,
        "sasa_drift_pct": sasa_drift_pct,
        "hb_avg_total": hb_avg_total,
        "hb_main_frac": hb_main_frac,
        "hb_num_ge_0.1": hb_num_ge_0_1,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze 20 ns MD (stage 1) for top30 ligands."
    )
    parser.add_argument("root",)
    parser.add_argument(
        "--dt_ns",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--out",
        default="stage1_summary.csv",
    )

    args = parser.parse_args()
    root = Path(args.root)

    rows = []
    for sub in sorted(root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 9999):
        if not sub.is_dir():
            continue
        if not sub.name.isdigit():
            continue
        rows.append(compute_metrics_for_folder(sub, dt_ns=args.dt_ns))

    df = pd.DataFrame(rows).sort_values("id")
    df["pass_stage1"] = (df["hb_avg_total"] >= 0.70) & (df["hb_main_frac"] >= 0.20)

    df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")
    passed = df[df["pass_stage1"]].sort_values("id")
    passed.to_csv("stage1_top12.csv", index=False)
    print(f"Top {len(passed)} ligands saved to stage1_top12.csv")


if __name__ == "__main__":
    main()
