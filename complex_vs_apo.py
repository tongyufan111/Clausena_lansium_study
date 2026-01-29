from pathlib import Path
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
COMPLEX_DIR = BASE_DIR / "Quercetin_3-arabinoside"
APO_DIR     = BASE_DIR / "single_protein"
PREFIX_RMSD = "rmsd_protein"
PREFIX_RG   = "rg_protein"
PREFIX_SASA = "sasa_protein"
PREFIX_RMSF = "rmsf_protein"

TOTAL_NS    = 500.0
ROLLING_NS  = 5.0

RAW_LW        = 0.8
SMOOTH_LW     = 1.2
RAW_ALPHA     = 0.30
SMOOTH_ALPHA  = 0.80

COL_COMPLEX = "#1f77b4"
COL_APO     = "#2ca02c"

LABEL_COMPLEX = "Quercetin 3-arabinoside"
LABEL_APO     = "Apo"



def npj_style():
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 600,
        "font.size": 9,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 9,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def find_first_dat(folder: Path, prefix: str) -> Path:
    patterns = [
        f"{prefix}*.dat",
        f"{prefix}*.DAT",
        f"{prefix}*.xvg",
        f"{prefix}*.XVG",
        f"{prefix}*.txt",
        f"{prefix}*.TXT",
    ]

    candidates = []
    for pat in patterns:
        candidates.extend(sorted(folder.glob(pat)))
    uniq = []
    seen = set()
    for p in candidates:
        if p.resolve() not in seen:
            uniq.append(p)
            seen.add(p.resolve())
    candidates = uniq

    if not candidates:
        existing = sorted([p.name for p in folder.iterdir() if p.is_file()])
        raise FileNotFoundError(
            f"{folder}: can't find {prefix}*.(dat/xvg/txt)\n"
            f"current file：\n  " + "\n  ".join(existing[:50])
        )

    if len(candidates) > 1:
        print(f"[WARN] {folder.name}: find {prefix}*，use {candidates[0].name}")

    return candidates[0]


def load_two_cols(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(("#", "@")):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0]); y = float(parts[1])
            except ValueError:
                continue
            xs.append(x); ys.append(y)
    if not xs:
        raise ValueError(f"{path}: no valid data")
    return np.asarray(xs, float), np.asarray(ys, float)


def make_time_axis(n_points: int, total_ns: float = TOTAL_NS) -> np.ndarray:
    if n_points < 2:
        return np.zeros(n_points)
    return np.linspace(0.0, total_ns, n_points)


def rolling_mean(y: np.ndarray, window_ns: float, total_ns: float = TOTAL_NS) -> np.ndarray:
    if window_ns <= 0 or len(y) < 3:
        return np.full_like(y, np.nan)

    dt_ns = total_ns / (len(y) - 1)
    window_pts = int(round(window_ns / dt_ns))
    if window_pts < 2:
        return np.full_like(y, np.nan)

    kernel = np.ones(window_pts, dtype=float) / window_pts
    valid = np.convolve(y, kernel, mode="valid")
    pad = (len(y) - len(valid)) // 2
    out = np.full_like(y, np.nan)
    out[pad:pad + len(valid)] = valid
    return out


def add_panel_label(ax: plt.Axes, label: str):
    ax.text(
        0.02, 0.96, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10,
        fontweight="bold"
    )


# ===================== 单张时间序列图 =====================

def plot_time_series_single(
    t_c: np.ndarray,
    y_c: np.ndarray,
    t_a: np.ndarray,
    y_a: np.ndarray,
    ylabel: str,
    title: str,
    outname: str,
):
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.plot(t_c, y_c, color=COL_COMPLEX, linewidth=RAW_LW, alpha=RAW_ALPHA,
            label=LABEL_COMPLEX)
    ax.plot(t_a, y_a, color=COL_APO, linewidth=RAW_LW, alpha=RAW_ALPHA,
            linestyle="--", label=LABEL_APO)
    y_c_s = rolling_mean(y_c, ROLLING_NS)
    y_a_s = rolling_mean(y_a, ROLLING_NS)
    ax.plot(t_c, y_c_s, color=COL_COMPLEX,
            linewidth=SMOOTH_LW, alpha=SMOOTH_ALPHA)
    ax.plot(t_a, y_a_s, color=COL_APO,
            linewidth=SMOOTH_LW, alpha=SMOOTH_ALPHA,
            linestyle="--")

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(outname, dpi=600)
    plt.close(fig)
    print(f"[SAVE] {outname}")


# ===================== RMSF=====================
def plot_rmsf_single(
    x_res: np.ndarray,
    y_c: np.ndarray,
    y_a: np.ndarray,
    outname: str,
):
    fig, ax = plt.subplots(figsize=(4.0, 3.0))

    ax.plot(x_res, y_c, color=COL_COMPLEX,
            linewidth=SMOOTH_LW, alpha=SMOOTH_ALPHA,
            label=LABEL_COMPLEX)
    ax.plot(x_res, y_a, color=COL_APO,
            linewidth=SMOOTH_LW, alpha=SMOOTH_ALPHA,
            linestyle="--", label="Apo (protein only)")

    ax.set_xlabel("Residue index")
    ax.set_ylabel("RMSF (Å)")
    ax.set_title("RMSF comparison")
    ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(outname, dpi=600)
    plt.close(fig)
    print(f"[SAVE] {outname}")


# ===================== 2×2  =====================
def plot_panel(
    t_c: np.ndarray,
    t_a: np.ndarray,
    rmsd_c: np.ndarray, rmsd_a: np.ndarray,
    rg_c: np.ndarray,   rg_a: np.ndarray,
    sasa_c: np.ndarray, sasa_a: np.ndarray,
    x_res: np.ndarray,  rmsf_c: np.ndarray, rmsf_a: np.ndarray,
    outname: str,
):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
    ax_rmsd, ax_rg = axes[0]
    ax_sasa, ax_rmsf = axes[1]

    # ---- (a) RMSD ----
    ax_rmsd.plot(t_c, rmsd_c, color=COL_COMPLEX, linewidth=RAW_LW, alpha=RAW_ALPHA)
    ax_rmsd.plot(t_a, rmsd_a, color=COL_APO, linewidth=RAW_LW, alpha=RAW_ALPHA,
                 linestyle="--")

    rmsd_c_s = rolling_mean(rmsd_c, ROLLING_NS)
    rmsd_a_s = rolling_mean(rmsd_a, ROLLING_NS)
    ax_rmsd.plot(t_c, rmsd_c_s, color=COL_COMPLEX,
                 linewidth=SMOOTH_LW, alpha=SMOOTH_ALPHA,
                 label=LABEL_COMPLEX)
    ax_rmsd.plot(t_a, rmsd_a_s, color=COL_APO,
                 linewidth=SMOOTH_LW, alpha=SMOOTH_ALPHA,
                 linestyle="--", label=LABEL_APO)

    ax_rmsd.set_ylabel("Cα-RMSD (Å)")
    ax_rmsd.set_title("Cα-RMSD vs Time")
    add_panel_label(ax_rmsd, "(a)")
    ax_rmsd.legend(loc="upper right", frameon=False)

    # ---- (b) Rg ----
    ax_rg.plot(t_c, rg_c, color=COL_COMPLEX, linewidth=RAW_LW, alpha=RAW_ALPHA)
    ax_rg.plot(t_a, rg_a, color=COL_APO, linewidth=RAW_LW, alpha=RAW_ALPHA,
               linestyle="--")

    rg_c_s = rolling_mean(rg_c, ROLLING_NS)
    rg_a_s = rolling_mean(rg_a, ROLLING_NS)
    ax_rg.plot(t_c, rg_c_s, color=COL_COMPLEX,
               linewidth=SMOOTH_LW, alpha=SMOOTH_ALPHA,
               label=LABEL_COMPLEX)
    ax_rg.plot(t_a, rg_a_s, color=COL_APO,
               linewidth=SMOOTH_LW, alpha=SMOOTH_ALPHA,
               linestyle="--", label=LABEL_APO)

    ax_rg.set_ylabel("Radius of gyration (Å)")
    ax_rg.set_title("Radius of gyration vs Time")
    add_panel_label(ax_rg, "(b)")
    ax_rg.set_xlabel("Time (ns)")

    # ---- (c) SASA ----
    ax_sasa.plot(t_c, sasa_c, color=COL_COMPLEX, linewidth=RAW_LW, alpha=RAW_ALPHA)
    ax_sasa.plot(t_a, sasa_a, color=COL_APO, linewidth=RAW_LW, alpha=RAW_ALPHA,
                 linestyle="--")

    sasa_c_s = rolling_mean(sasa_c, ROLLING_NS)
    sasa_a_s = rolling_mean(sasa_a, ROLLING_NS)
    ax_sasa.plot(t_c, sasa_c_s, color=COL_COMPLEX,
                 linewidth=SMOOTH_LW, alpha=SMOOTH_ALPHA,
                 label=LABEL_COMPLEX)
    ax_sasa.plot(t_a, sasa_a_s, color=COL_APO,
                 linewidth=SMOOTH_LW, alpha=SMOOTH_ALPHA,
                 linestyle="--", label=LABEL_APO)

    ax_sasa.set_xlabel("Time (ns)")
    ax_sasa.set_ylabel("SASA (Å²)")
    ax_sasa.set_title("SASA vs Time")
    add_panel_label(ax_sasa, "(c)")

    # ---- (d) RMSF ----
    ax_rmsf.plot(x_res, rmsf_c, color=COL_COMPLEX,
                 linewidth=SMOOTH_LW, alpha=SMOOTH_ALPHA,
                 label=LABEL_COMPLEX)
    ax_rmsf.plot(x_res, rmsf_a, color=COL_APO,
                 linewidth=SMOOTH_LW, alpha=SMOOTH_ALPHA,
                 linestyle="--", label=LABEL_APO)

    ax_rmsf.set_xlabel("Residue index")
    ax_rmsf.set_ylabel("RMSF (Å)")
    ax_rmsf.set_title("RMSF comparison")
    add_panel_label(ax_rmsf, "(d)")
    ax_rmsf.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(outname, dpi=600)
    tiff_name = outname.replace(".png", ".tiff")
    fig.savefig(tiff_name, dpi=600)
    plt.close(fig)
    print(f"[SAVE] {outname}")


def main():
    npj_style()
    rmsd_c_path = find_first_dat(COMPLEX_DIR, PREFIX_RMSD)
    rg_c_path   = find_first_dat(COMPLEX_DIR, PREFIX_RG)
    sasa_c_path = find_first_dat(COMPLEX_DIR, PREFIX_SASA)
    rmsf_c_path = find_first_dat(COMPLEX_DIR, PREFIX_RMSF)

    _, rmsd_c = load_two_cols(rmsd_c_path)
    _, rg_c   = load_two_cols(rg_c_path)
    _, sasa_c = load_two_cols(sasa_c_path)
    x_res_c, rmsf_c = load_two_cols(rmsf_c_path)

    rmsd_a_path = find_first_dat(APO_DIR, PREFIX_RMSD)
    rg_a_path   = find_first_dat(APO_DIR, PREFIX_RG)
    sasa_a_path = find_first_dat(APO_DIR, PREFIX_SASA)
    rmsf_a_path = find_first_dat(APO_DIR, PREFIX_RMSF)

    _, rmsd_a = load_two_cols(rmsd_a_path)
    _, rg_a   = load_two_cols(rg_a_path)
    _, sasa_a = load_two_cols(sasa_a_path)
    x_res_a, rmsf_a = load_two_cols(rmsf_a_path)

    t_c = make_time_axis(len(rmsd_c), TOTAL_NS)
    t_a = make_time_axis(len(rmsd_a), TOTAL_NS)

    x_res = x_res_c

    plot_time_series_single(
        t_c, rmsd_c,
        t_a, rmsd_a,
        ylabel="Cα-RMSD (Å)",
        title="Cα-RMSD vs Time",
        outname="rmsd_single.png",
    )

    plot_time_series_single(
        t_c, rg_c,
        t_a, rg_a,
        ylabel="Radius of gyration (Å)",
        title="Radius of gyration vs Time",
        outname="rg_single.png",
    )

    plot_time_series_single(
        t_c, sasa_c,
        t_a, sasa_a,
        ylabel="SASA (Å²)",
        title="SASA vs Time",
        outname="sasa_single.png",
    )

    plot_rmsf_single(
        x_res, rmsf_c, rmsf_a,
        outname="rmsf_single.png",
    )

    plot_panel(
        t_c, t_a,
        rmsd_c, rmsd_a,
        rg_c,   rg_a,
        sasa_c, sasa_a,
        x_res,  rmsf_c, rmsf_a,
        outname="md_panel_npj.png",
    )


if __name__ == "__main__":
    main()
