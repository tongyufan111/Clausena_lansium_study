import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

WORKDIR = Path(r"D:\projects\Huangpi\PBSA")

FILES = [
    "clauselansine D.dat",
    "Quercetin 3-arabinoside.dat",
]

TARGET_NAME = "Quercetin 3-arabinoside"
PBSA_WINDOW_TEXT = "450–500 ns (500 frames)"
OUT_PNG = WORKDIR / "Quercetin_3-arabinoside_PBSA_bar.png"

def _norm_key(k: str) -> str:
    k = k.strip()
    k = re.sub(r"\s+", " ", k)
    return k.upper()


def parse_mmpbsa_pbsa(text: str) -> dict:
    anchor = re.search(
        r"Differences\s*\(Complex\s*-\s*Receptor\s*-\s*Ligand\)\s*:",
        text,
        flags=re.I
    )
    if not anchor:
        return {}
    sub = text[anchor.end():]
    line_pat = re.compile(
        r"^\s*([A-Za-z][A-Za-z0-9_\s]+?)\s+([-+]?\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)",
        flags=re.M
    )
    out = {}
    for m in line_pat.finditer(sub):
        key = _norm_key(m.group(1))
        avg = float(m.group(2))
        sd  = float(m.group(3))
        out[key] = (avg, sd)
    return out

def read_one_dat(fp: Path) -> dict:
    return parse_mmpbsa_pbsa(fp.read_text(errors="ignore"))


def make_vertical_table(raw: dict) -> pd.DataFrame:
    def fmt(mean, sd):
        return f"{mean:.2f} ± {sd:.2f}"
    vdw = raw.get("VDWAALS", (None, None))
    eel = raw.get("EEL", (None, None))
    epb = raw.get("EPB", (None, None))
    enp = raw.get("ENPOLAR", (None, None))
    eds = raw.get("EDISPER", (0.0, 0.0))
    tot = raw.get("DELTA TOTAL", (None, None))

    rows = []
    rows.append(("Δ Evdw", fmt(*vdw) if vdw[0] is not None else ""))
    rows.append(("Δ Eele", fmt(*eel) if eel[0] is not None else ""))
    rows.append(("Δ Gpolar  (EPB)", fmt(*epb) if epb[0] is not None else ""))

    if enp[0] is not None:
        nonpolar_mean = enp[0] + (eds[0] if eds[0] is not None else 0.0)
        nonpolar_sd = ((enp[1] or 0.0) ** 2 + (eds[1] or 0.0) ** 2) ** 0.5
        rows.append(("Δ Gnonpolar  (ENPOLAR\n+ EDISPER)", fmt(nonpolar_mean, nonpolar_sd)))
    else:
        rows.append(("Δ Gnonpolar  (ENPOLAR\n+ EDISPER)", ""))

    rows.append(("Δ Gbind", fmt(*tot) if tot[0] is not None else ""))

    return pd.DataFrame(rows, columns=["Energy term", "Mean ± SD (kcal/mol)"])


def write_tables_to_csv(tables: dict, out_dir: Path):
    for ligand, df in tables.items():
        safe = ligand.replace(" ", "_")
        out_csv = out_dir / f"PBSA_{safe}.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] CSV 已输出：{out_csv}")


def plot_pbsa_bar_journal(raw: dict, out_png: Path, window_text: str):
    def get(term, default=(None, None)):
        return raw.get(term, default)
    vdw = get("VDWAALS")
    eel = get("EEL")
    epb = get("EPB")
    enp = get("ENPOLAR")
    eds = get("EDISPER", (0.0, 0.0))
    tot = get("DELTA TOTAL")

    nonpolar_mean, nonpolar_sd = None, None
    if enp[0] is not None:
        nonpolar_mean = enp[0] + (eds[0] if eds[0] is not None else 0.0)
        nonpolar_sd = ((enp[1] or 0.0) ** 2 + (eds[1] or 0.0) ** 2) ** 0.5

    labels = ["ΔEvdW", "ΔEele", "ΔGpolar (EPB)", "ΔGnonpolar", "ΔGbind"]
    means  = [vdw[0], eel[0], epb[0], nonpolar_mean, tot[0]]
    sds    = [vdw[1], eel[1], epb[1], nonpolar_sd,  tot[1]]

    data = [(l, m, s) for l, m, s in zip(labels, means, sds) if m is not None]
    labels = [d[0] for d in data]
    means  = np.array([d[1] for d in data], dtype=float)
    sds    = np.array([0.0 if d[2] is None else float(d[2]) for d in data], dtype=float)

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(5.4, 3.6), dpi=200)
    ax = fig.add_subplot(111)

    x = np.arange(len(means))

    bars = ax.bar(
        x, means,
        width=0.72,
        yerr=sds,
        capsize=3,
        error_kw={"elinewidth": 1.1, "capthick": 1.1},
        zorder=2
    )

    for b, lab in zip(bars, labels):
        b.set_alpha(1.0 if lab == "ΔGbind" else 0.85)

    ax.axhline(0, color="black", linewidth=1.3, zorder=3)
    y_min = np.min(means - sds)
    y_max = np.max(means + sds)
    pad = max(3.0, 0.12 * (y_max - y_min))
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")

    ax.set_ylabel("Energy (kcal/mol)")
    ax.set_title("MM-PBSA free energy decomposition", pad=8)

    ax.text(
        0.99, 0.97,
        window_text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", width=1.2, length=5)

    fig.tight_layout()

    fig.savefig(out_png, dpi=600, bbox_inches="tight")

    out_tiff = out_png.with_suffix(".tiff")
    fig.savefig(
        out_tiff,
        dpi=600,
        format="tiff",
        pil_kwargs={"compression": "tiff_lzw"}
    )

    plt.close(fig)

def main():
    tables = {}
    raws = {}

    for fn in FILES:
        fp = WORKDIR / fn
        if not fp.exists():
            raise FileNotFoundError(f"can't find：{fp}")

        ligand = Path(fn).stem
        raw = read_one_dat(fp)

        if not raw:
            raise ValueError(f"can't find PBSA Differences csv：{fp}")

        raws[ligand] = raw
        tables[ligand] = make_vertical_table(raw)

    write_tables_to_csv(tables, WORKDIR)
    print("Done")

    raw_q = None
    if TARGET_NAME in raws:
        raw_q = raws[TARGET_NAME]
    else:
        raw_q = raws.get(Path(FILES[-1]).stem)

    if raw_q:
        plot_pbsa_bar_journal(raw_q, OUT_PNG, PBSA_WINDOW_TEXT)
        print(f"bar Done：{OUT_PNG}")
    else:
        print("can't find ligand dataset。")


if __name__ == "__main__":
    main()
