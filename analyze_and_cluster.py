"""
Ligand screening & clustering (CSV outputs + publication-ready figures)
- Canonicalizes SMILES before fingerprinting/clustering
- Computes LBP_avg, Vina_med, Vina_std
- Filters: LBP >= P70, Vina_med <= -7.0 kcal/mol, LibDock >= P60
- Ranks: LBP_avg (desc) -> Vina_med (asc) -> LibDock (desc)
- Clusters: RDKit MorganFP (r=2, 2048) + Tanimoto, Tc=0.7
- Per-cluster cap selection (default 2) to fill TopK (default 30)
- Outputs (to ./results/):
    CSVs (canonical SMILES only), publication-grade plots (PNG+PDF+SVG),
    English figure captions (figure_captions_en.txt),
    Threshold histograms with cut-lines, and thresholds_summary.txt
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ----- Matplotlib: publication style, headless backend -----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'DejaVu Serif',   # replace with 'Times New Roman' if installed
    'font.size': 12,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.facecolor': '#f5f5f5',
    'axes.facecolor': '#f7f7f7'
})

from scipy.stats import pearsonr, spearmanr, linregress
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.MolStandardize import rdMolStandardize


def load_table(infile: Path) -> pd.DataFrame:
    # read csv---
    if infile.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(infile)
    else:
        try:
            df = pd.read_csv(infile, sep=None, engine="python", encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(infile, sep=None, engine="python", encoding="utf-8-sig")
    df = df.dropna(how="all").reset_index(drop=True)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    def _strip_hidden(s):
        return (str(s)
                .replace("\ufeff","").replace("\u3000","").replace("\xa0","")
                .strip())
    df.columns = [_strip_hidden(c).lower() for c in df.columns]
    import re
    def _clean_cell(x):
        if pd.isna(x): return x
        s = str(x)
        s = (s.replace("\ufeff","").replace("\u3000","").replace("\xa0","").replace("\t"," ")
               .replace("−","-").replace("–","-").replace("—","-"))
        s = re.sub(r"\s+", " ", s).strip()
        if re.fullmatch(r"-?\d+,\d+", s):
            s = s.replace(",", ".")
        return s
    df = df.applymap(_clean_cell)
    rename_map = {}
    if "smiles" in df.columns: rename_map["smiles"] = "Smiles"
    if "smi"     in df.columns: rename_map["smi"]     = "Smiles"
    # libdock
    for c in ["libdock_score", "libdockscore"]:
        if c in df.columns: rename_map[c] = "libdock"
    if "libdock" in df.columns: rename_map["libdock"] = "libdock"
    df = df.rename(columns=rename_map)

    return df

# ------------------------ Metrics & thresholds ------------------------
def compute_metrics(df: pd.DataFrame):
    cols = list(df.columns)
    lbp_cols  = [c for c in cols if c.lower().startswith("lbp_")]
    vina_cols = [c for c in cols if c.lower().startswith("vina") and c.lower() != "vina_ave"]
    libdock_col = "libdock" if "libdock" in cols else None

    for col in lbp_cols + vina_cols + ([libdock_col] if libdock_col else []):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["LBP_avg"] = df[lbp_cols].mean(axis=1, skipna=True) if lbp_cols else np.nan

    if "vina_ave" in df.columns:
        df["Vina_mean"] = pd.to_numeric(df["vina_ave"], errors="coerce")
    else:
        df["Vina_mean"] = df[vina_cols].mean(axis=1, skipna=True) if vina_cols else np.nan

    if vina_cols:
        df["Vina_med"] = df[vina_cols].median(axis=1, skipna=True)
        df["Vina_std"] = df[vina_cols].std(axis=1, ddof=0)
    else:
        df["Vina_med"] = df["Vina_mean"]
        df["Vina_std"] = np.nan

    p70_lbp = np.nanpercentile(df["LBP_avg"], 70) if df["LBP_avg"].notna().any() else np.nan
    p60_libdock = np.nanpercentile(df[libdock_col], 60) if libdock_col else np.nan

    meta = {
        "lbp_cols": lbp_cols,
        "vina_cols": vina_cols,
        "libdock_col": libdock_col,
        "p70_lbp": float(p70_lbp) if np.isfinite(p70_lbp) else np.nan,
        "p60_libdock": float(p60_libdock) if np.isfinite(p60_libdock) else np.nan,
        "vina_cut": -7.0
    }
    return df, meta

def screen_and_rank(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    libdock_col = meta["libdock_col"]
    df["pass_LBP"] = df["LBP_avg"] >= meta["p70_lbp"]
    df["pass_Vina"] = df["Vina_med"] <= meta["vina_cut"]
    df["pass_LibDock"] = (df[libdock_col] >= meta["p60_libdock"]) if libdock_col else True
    passed = df[df["pass_LBP"] & df["pass_Vina"] & df["pass_LibDock"]].copy()
    sort_keys, ascending = ["LBP_avg", "Vina_med"], [False, True]
    if libdock_col:
        sort_keys.append(libdock_col)
        ascending.append(False)
    return passed.sort_values(sort_keys, ascending=ascending).reset_index(drop=True)


# ----------------------------- Canonical SMILES -----------------------------
def canonicalize_smiles_col(df: pd.DataFrame,
                            smiles_col: str = "Smiles",
                            keep_raw: bool = False) -> pd.DataFrame:
    raw = df[smiles_col] if smiles_col in df.columns else pd.Series([""] * len(df))
    canon_list, raw_list = [], []

    norm = rdMolStandardize.Normalizer()
    reion = rdMolStandardize.Reionizer()
    chooser = rdMolStandardize.LargestFragmentChooser()
    taut = rdMolStandardize.TautomerEnumerator()

    for s in raw.fillna(""):
        raw_list.append(s)
        try:
            mol = Chem.MolFromSmiles(str(s))
            if mol is None:
                canon_list.append("")
                continue
            mol = chooser.choose(mol)
            mol = norm.normalize(mol)
            mol = reion.reionize(mol)
            mol = taut.Canonicalize(mol)
            Chem.SanitizeMol(mol)
            smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
            canon_list.append(smi)
        except Exception:
            canon_list.append("")

    df = df.copy()
    if keep_raw:
        df["Smiles_raw"] = raw_list
    if smiles_col in df.columns:
        df.drop(columns=[smiles_col], inplace=True)
    df["Smiles"] = canon_list
    return df

# ----------------------------- Clustering -----------------------------
def cluster_assign(passed: pd.DataFrame, tc=0.7) -> pd.DataFrame:
    if "Smiles" not in passed.columns:
        passed["cluster_id"] = -1
        return passed
    smiles = passed["Smiles"].fillna("")
    mols = [Chem.MolFromSmiles(s) if s else None for s in smiles]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None for m in mols]
    cluster_ids, centroids = [-1]*len(fps), []
    for i, fp in enumerate(fps):
        if fp is None:
            continue
        for cid, (cfp, _) in enumerate(centroids):
            if DataStructs.TanimotoSimilarity(fp, cfp) >= tc:
                cluster_ids[i] = cid
                break
        else:
            centroids.append((fp, i))
            cluster_ids[i] = len(centroids)-1

    out = passed.copy()
    out["cluster_id"] = cluster_ids
    return out

# ----------------------- Per-cluster Top-K pick -----------------------
def select_topk_allow_per_cluster(passed, topk, per_cluster_max):
    sort_keys = ["LBP_avg", "Vina_med"] + (["libdock"] if "libdock" in passed.columns else [])
    ascending = [False, True] + ([False] if "libdock" in passed.columns else [])
    ranked = passed.sort_values(sort_keys, ascending=ascending).reset_index(drop=True)

    best = ranked.groupby("cluster_id", as_index=False).first()
    best = best.sort_values(sort_keys, ascending=ascending).reset_index(drop=True)
    cluster_order = best["cluster_id"].tolist()

    selected, counts = [], {}
    while len(selected) < topk:
        added = 0
        for cid in cluster_order:
            grp = ranked[ranked["cluster_id"] == cid]
            k = counts.get(cid, 0)
            if k < per_cluster_max and k < grp.shape[0]:
                selected.append(grp.iloc[k])
                counts[cid] = k + 1
                added += 1
                if len(selected) >= topk:
                    break
        if added == 0:
            break
    return pd.DataFrame(selected).reset_index(drop=True)


# -------------------------- Plotting helpers --------------------------
def safe_corr(x, y):
    x = np.array(pd.to_numeric(x, errors="coerce"), dtype=float)
    y = np.array(pd.to_numeric(y, errors="coerce"), dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan, np.nan, 0
    r = pearsonr(x[mask], y[mask])[0]
    rho = spearmanr(x[mask], y[mask])[0]
    return float(r), float(rho), int(mask.sum())


def save_figure_all_formats(fig: plt.Figure, path_no_ext: Path):
    fig.savefig(path_no_ext.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_no_ext.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_no_ext.with_suffix(".svg"), bbox_inches="tight")


def plot_scatter_with_corr_and_fit(passed, results_dir: Path):
    x = pd.to_numeric(passed["LBP_avg"], errors="coerce")
    y = pd.to_numeric(passed["Vina_med"], errors="coerce")
    mask = x.notna() & y.notna()
    xv, yv = x[mask].values, y[mask].values

    r, rho, n_xy = safe_corr(xv, yv)

    # regression
    if len(xv) >= 2:
        lr = linregress(xv, yv)
        xfit = np.linspace(xv.min(), xv.max(), 200)
        yfit = lr.slope * xfit + lr.intercept
    else:
        lr = None
        xfit = yfit = None

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(xv, yv, marker='x', linewidths=1.5, label="Passed compounds")

    if xfit is not None:
        ax.plot(xfit, yfit, linewidth=2.2, label="Linear fit")
        ax.text(0.02, 0.02,
                f"Fit: y = {lr.slope:.2f}x + {lr.intercept:.2f}\n"
                f"R = {abs(lr.rvalue):.2f}, p = {lr.pvalue:.3g}",
                transform=ax.transAxes, va='bottom', ha='left',
                bbox=dict(boxstyle='round', fc='white', ec='0.5', alpha=0.95))

    ax.set_xlabel("LBP_avg")
    ax.set_ylabel("Vina_med (kcal/mol)")
    ax.set_title("LBP vs Vina (passed)")
    ax.text(0.02, 0.98,
            f"Pearson r = {r:.2f}\nSpearman ρ = {rho:.2f}\nN = {n_xy}",
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', fc='white', ec='0.5', alpha=0.95))
    ax.legend(frameon=True)

    fig.tight_layout()
    save_figure_all_formats(fig, results_dir / "plot_LBP_vs_Vina_pub")
    plt.close(fig)

    slope = lr.slope if lr else np.nan
    intercept = lr.intercept if lr else np.nan
    pval = lr.pvalue if lr else np.nan
    return r, rho, n_xy, slope, intercept, pval


def plot_cluster_sizes_pub(passed, tc: float, results_dir: Path):
    sizes = passed["cluster_id"].value_counts().sort_index()
    mean_size = float(sizes.mean()) if not sizes.empty else float('nan')
    max_size = int(sizes.max()) if not sizes.empty else 0

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(sizes.index, sizes.values, width=0.8, linewidth=0.8, edgecolor='0.4')
    ax.set_xlabel("cluster_id")
    ax.set_ylabel("size")
    ax.set_title(f"Cluster size distribution (Tc = {tc})")
    ax.text(0.98, 0.98, f"mean size = {mean_size:.2f}\nmax size = {max_size}",
            transform=ax.transAxes, va='top', ha='right',
            bbox=dict(boxstyle='round', fc='white', ec='0.5', alpha=0.95))

    fig.tight_layout()
    save_figure_all_formats(fig, results_dir / "plot_cluster_sizes_pub")
    plt.close(fig)
    return mean_size, max_size


# -------- Percentile histograms with threshold lines & summary -------
def plot_threshold_histograms(df_all: pd.DataFrame, meta: dict, results_dir: Path):
    """Plot LBP P70, LibDock P60, Vina_med cut (-7.0) histograms and return counts."""
    counts = {}

    # LBP
    x = pd.to_numeric(df_all["LBP_avg"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(x, bins=30)
    ax.axvline(meta["p70_lbp"], linestyle='-', linewidth=2.0, label=f'P70 = {meta["p70_lbp"]:.3f}')
    pass_lbp = int((df_all["LBP_avg"] >= meta["p70_lbp"]).sum())
    total = int(df_all["LBP_avg"].notna().sum())
    frac = pass_lbp / total if total else 0.0
    ax.text(0.02, 0.98, f"pass = {pass_lbp}/{total} ({frac:.1%})",
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', fc='white', ec='0.5', alpha=0.95))
    ax.set_xlabel("LBP_avg"); ax.set_ylabel("Count"); ax.set_title("LBP distribution with P70")
    ax.legend(frameon=True)
    fig.tight_layout(); save_figure_all_formats(fig, results_dir / "hist_LBP_P70"); plt.close(fig)
    counts["LBP"] = (pass_lbp, total, frac)

    # LibDock
    if meta["libdock_col"]:
        ld = pd.to_numeric(df_all[meta["libdock_col"]], errors="coerce").dropna()
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.hist(ld, bins=30)
        ax.axvline(meta["p60_libdock"], linestyle='-', linewidth=2.0,
                   label=f'P60 = {meta["p60_libdock"]:.3f}')
        pass_ld = int((df_all[meta["libdock_col"]] >= meta["p60_libdock"]).sum())
        total_ld = int(df_all[meta["libdock_col"]].notna().sum())
        frac_ld = pass_ld / total_ld if total_ld else 0.0
        ax.text(0.02, 0.98, f"pass = {pass_ld}/{total_ld} ({frac_ld:.1%})",
                transform=ax.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', fc='white', ec='0.5', alpha=0.95))
        ax.set_xlabel("LibDock"); ax.set_ylabel("Count"); ax.set_title("LibDock distribution with P60")
        ax.legend(frameon=True)
        fig.tight_layout(); save_figure_all_formats(fig, results_dir / "hist_LibDock_P60"); plt.close(fig)
        counts["LibDock"] = (pass_ld, total_ld, frac_ld)

    # Vina_med cut
    v = pd.to_numeric(df_all["Vina_med"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(v, bins=30)
    ax.axvline(meta["vina_cut"], linestyle='-', linewidth=2.0, label=f'cut = {meta["vina_cut"]:.1f}')
    pass_v = int((df_all["Vina_med"] <= meta["vina_cut"]).sum())
    total_v = int(df_all["Vina_med"].notna().sum())
    frac_v = pass_v / total_v if total_v else 0.0
    ax.text(0.02, 0.98, f"pass = {pass_v}/{total_v} ({frac_v:.1%})",
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', fc='white', ec='0.5', alpha=0.95))
    ax.set_xlabel("Vina_med (kcal/mol)"); ax.set_ylabel("Count"); ax.set_title("Vina_med distribution with cut")
    ax.legend(frameon=True)
    fig.tight_layout(); save_figure_all_formats(fig, results_dir / "hist_Vina_cut"); plt.close(fig)
    counts["Vina"] = (pass_v, total_v, frac_v)

    # All three conditions
    cond = (df_all["LBP_avg"] >= meta["p70_lbp"]) & (df_all["Vina_med"] <= meta["vina_cut"])
    if meta["libdock_col"]:
        cond &= (df_all[meta["libdock_col"]] >= meta["p60_libdock"])
    n_all = int(cond.sum()); total_all = int(len(df_all))
    frac_all = n_all / total_all if total_all else 0.0
    counts["All"] = (n_all, total_all, frac_all)

    return counts


def save_thresholds_summary(results_dir: Path, meta: dict, counts: dict):
    lines = []
    lines.append("Thresholds summary (percentile-based intra-library filtering)\n")
    lines.append(f"LBP P70 value: {meta['p70_lbp']:.6f}   pass: {counts['LBP'][0]}/{counts['LBP'][1]} ({counts['LBP'][2]:.1%})")
    if meta["libdock_col"]:
        lines.append(f"LibDock P60 value: {meta['p60_libdock']:.6f}   pass: {counts['LibDock'][0]}/{counts['LibDock'][1]} ({counts['LibDock'][2]:.1%})")
    lines.append(f"Vina_med cut: {meta['vina_cut']:.1f} kcal/mol   pass: {counts['Vina'][0]}/{counts['Vina'][1]} ({counts['Vina'][2]:.1%})")
    lines.append(f"\nAll criteria passed: {counts['All'][0]}/{counts['All'][1]} ({counts['All'][2]:.1%})\n")
    (results_dir / "thresholds_summary.txt").write_text("\n".join(lines), encoding="utf-8")


# ------------------------------ Save CSVs & captions ------------------------------
def save_csvs(df_all, passed, topk_df, results_dir, topk, per_cluster_max):
    df_all.to_csv(results_dir / "all_with_metrics.csv", index=False)
    passed.to_csv(results_dir / "passed_sorted.csv", index=False)
    topk_df.to_csv(results_dir / f"Top{topk}_clustered_allow{per_cluster_max}.csv", index=False)
    passed.head(topk).to_csv(results_dir / f"Top{topk}_passed_unclustered.csv", index=False)


def save_captions(results_dir: Path,
                  r, rho, n_xy, slope, intercept, pval,
                  mean_size, max_size, tc):
    cap1 = (
        "Figure X. Correlation between ML-predicted activity and docking energy.\n"
        "Scatter plot of model-averaged LBP scores versus Vina binding affinities "
        "(kcal/mol) for compounds passing percentile-based internal filtering "
        "(LBP ≥ P70; Vina_med ≤ −7.0; LibDock ≥ P60). "
        f"Pearson r = {r:.2f}, Spearman ρ = {rho:.2f}, N = {n_xy}. "
        "A linear regression (solid line) is overlaid with equation "
        f"y = {slope:.2f}x + {intercept:.2f} (p = {pval:.3g}). "
        "The negative correlation indicates that higher predicted activities are "
        "generally associated with stronger binding (more negative energies)."
    )

    cap2 = (
        "Figure Y. Chemical diversity of selected ligands.\n"
        "Distribution of cluster sizes obtained with Morgan fingerprints (radius = 2, "
        "2048 bits) and Tanimoto similarity (Tc shown). "
        f"Average cluster size = {mean_size:.2f}; maximum = {max_size}. "
        "The small cluster sizes demonstrate minimal redundancy and good chemical "
        "diversity among screened candidates."
    )

    with open(results_dir / "figure_captions_en.txt", "w", encoding="utf-8") as f:
        f.write(cap1 + "\n\n" + cap2 + "\n")


# ---------------------- correlation plots ----------------------
def plot_libdock_vs_vina_pub(passed: pd.DataFrame, results_dir: Path):
    if "libdock" not in passed.columns:
        return np.nan, np.nan, 0, np.nan, np.nan, np.nan

    x = pd.to_numeric(passed["libdock"], errors="coerce")
    y = pd.to_numeric(passed["Vina_med"], errors="coerce")
    mask = x.notna() & y.notna()
    xv, yv = x[mask].values, y[mask].values

    r, rho, n_xy = safe_corr(xv, yv)
    if len(xv) >= 2:
        lr = linregress(xv, yv)
        xfit = np.linspace(xv.min(), xv.max(), 200)
        yfit = lr.slope * xfit + lr.intercept
    else:
        lr = None
        xfit = yfit = None

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(xv, yv, marker='x', linewidths=1.5, label="Passed compounds")
    if xfit is not None:
        ax.plot(xfit, yfit, linewidth=2.2, label="Linear fit")
        ax.text(0.02, 0.02,
                f"Fit: y = {lr.slope:.2f}x + {lr.intercept:.2f}\n"
                f"R = {abs(lr.rvalue):.2f}, p = {lr.pvalue:.3g}",
                transform=ax.transAxes, va='bottom', ha='left',
                bbox=dict(boxstyle='round', fc='white', ec='0.5', alpha=0.95))

    ax.set_xlabel("LibDock score")
    ax.set_ylabel("Vina_med (kcal/mol)")
    ax.set_title("LibDock vs Vina (passed)")
    ax.text(0.02, 0.98,
            f"Pearson r = {r:.2f}\nSpearman ρ = {rho:.2f}\nN = {n_xy}",
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', fc='white', ec='0.5', alpha=0.95))
    ax.legend(frameon=True)

    fig.tight_layout()
    save_figure_all_formats(fig, results_dir / "plot_LibDock_vs_Vina_pub")
    plt.close(fig)

    slope = lr.slope if lr else np.nan
    intercept = lr.intercept if lr else np.nan
    pval = lr.pvalue if lr else np.nan
    return r, rho, n_xy, slope, intercept, pval


def plot_lbp_vs_libdock_pub(passed: pd.DataFrame, results_dir: Path):
    if "libdock" not in passed.columns:
        return np.nan, np.nan, 0, np.nan, np.nan, np.nan
    x = pd.to_numeric(passed["LBP_avg"], errors="coerce")
    y = pd.to_numeric(passed["libdock"], errors="coerce")
    mask = x.notna() & y.notna()
    xv, yv = x[mask].values, y[mask].values
    r, rho, n_xy = safe_corr(xv, yv)
    if len(xv) >= 2:
        lr = linregress(xv, yv)
        xfit = np.linspace(xv.min(), xv.max(), 200)
        yfit = lr.slope * xfit + lr.intercept
    else:
        lr = None
        xfit = yfit = None

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(xv, yv, marker='x', linewidths=1.5, label="Passed compounds")
    if xfit is not None:
        ax.plot(xfit, yfit, linewidth=2.2, label="Linear fit")
        ax.text(0.02, 0.02,
                f"Fit: y = {lr.slope:.2f}x + {lr.intercept:.2f}\n"
                f"R = {abs(lr.rvalue):.2f}, p = {lr.pvalue:.3g}",
                transform=ax.transAxes, va='bottom', ha='left',
                bbox=dict(boxstyle='round', fc='white', ec='0.5', alpha=0.95))

    ax.set_xlabel("LBP_avg")
    ax.set_ylabel("LibDock score")
    ax.set_title("LBP vs LibDock (passed)")
    ax.text(0.02, 0.98,
            f"Pearson r = {r:.2f}\nSpearman ρ = {rho:.2f}\nN = {n_xy}",
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', fc='white', ec='0.5', alpha=0.95))
    ax.legend(frameon=True)

    fig.tight_layout()
    save_figure_all_formats(fig, results_dir / "plot_LBP_vs_LibDock_pub")
    plt.close(fig)

    slope = lr.slope if lr else np.nan
    intercept = lr.intercept if lr else np.nan
    pval = lr.pvalue if lr else np.nan
    return r, rho, n_xy, slope, intercept, pval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--tc", type=float, default=0.7)
    ap.add_argument("--per_cluster_max", type=int, default=2)
    ap.add_argument("--topk", type=int, default=30)
    args = ap.parse_args()

    infile = Path(args.infile)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    df = load_table(infile)
    df, meta = compute_metrics(df)
    df = canonicalize_smiles_col(df, smiles_col="Smiles", keep_raw=False)
    passed = screen_and_rank(df, meta)
    passed = cluster_assign(passed, tc=args.tc)
    topk_df = select_topk_allow_per_cluster(passed, args.topk, args.per_cluster_max)
    r, rho, n_xy, slope, intercept, pval = plot_scatter_with_corr_and_fit(passed, results_dir)
    mean_size, max_size = plot_cluster_sizes_pub(passed, args.tc, results_dir)
    plot_libdock_vs_vina_pub(passed, results_dir)
    plot_lbp_vs_libdock_pub(passed, results_dir)
    counts = plot_threshold_histograms(df, meta, results_dir)
    save_thresholds_summary(results_dir, meta, counts)
    save_csvs(df, passed, topk_df, results_dir, args.topk, args.per_cluster_max)
    save_captions(results_dir, r, rho, n_xy, slope, intercept, pval, mean_size, max_size, args.tc)


if __name__ == "__main__":
    main()

# command：python .\analyze_and_cluster.py --infile "ligands_screen.csv" --tc 0.7 --per_cluster_max 2 --topk 30
