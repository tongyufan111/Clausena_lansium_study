import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("RUNNING FILE:", os.path.abspath(__file__))


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = SCRIPT_DIR
LIG_IDS = ["3", "7", "17"]
COM_FILE   = "lig_pocket_com_last50ns.dat"
RES_FILE   = "nc_res_last50ns.dat"
FRAME_FILE = "nc_frame_last50ns.dat"
THRESH_A = 4.5
TIME_START_NS = 450.0
TIME_END_NS   = 500.0
FRAMES_PER_NS = 10.0
DT_NS = 1.0 / FRAMES_PER_NS
PDB_OFFSET = 324
OUT_DIR = os.path.join(SCRIPT_DIR, "figs_last50ns")


os.makedirs(OUT_DIR, exist_ok=True)
print("OUTPUT DIR:", OUT_DIR)


def must_exist(path, what):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {what}: {path}")


def frame_to_ns_by_rate(frame_series, start_ns=450.0, dt_ns=0.1):
    f = np.asarray(frame_series, dtype=float)
    fmin = np.nanmin(f)
    if abs(fmin - 1.0) < 1e-6:
        f0 = f - 1.0
    elif abs(fmin - 0.0) < 1e-6:
        f0 = f
    else:
        f0 = f - fmin
    return start_ns + f0 * dt_ns


def read_com(path):
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
    if df.shape[1] < 2:
        raise ValueError(f"COM file has <2 columns: {path}")
    df = df.iloc[:, :2].copy()
    df.columns = ["frame", "dist"]
    df["time"] = frame_to_ns_by_rate(df["frame"], start_ns=TIME_START_NS, dt_ns=DT_NS)
    return df[["time", "dist", "frame"]]

def com_stats(dist, thresh=4.5):
    dist = np.asarray(dist, dtype=float)
    mean = float(dist.mean())
    sd   = float(dist.std(ddof=1)) if len(dist) > 1 else 0.0
    mx   = float(dist.max())
    frac_over = float((dist > thresh).mean() * 100.0)
    return mean, sd, mx, frac_over


def read_nc_res(path):
    """读取 nc_res_last50ns.dat，列：#Res1 #Res2 TotalFrac Contacts，并 Res2 + offset"""
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
    if df.shape[1] < 4:
        raise ValueError(f"nc_res file has <4 columns: {path}")
    df = df.iloc[:, :4].copy()
    df.columns = ["Res1", "Res2", "TotalFrac", "Contacts"]
    df["Res_PDB"] = df["Res2"].astype(int) + PDB_OFFSET
    return df


def read_nc_frame(path):
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
    if df.shape[1] < 2:
        raise ValueError(f"nc_frame file has <2 columns: {path}")
    df = df.copy()
    df.columns = [f"c{i}" for i in range(df.shape[1])]
    df = df.rename(columns={"c0": "frame", f"c{df.shape[1]-1}": "contacts"})
    return df[["frame", "contacts"]]


def save_png_tiff(fig, out_base, dpi=600):
    png_path  = os.path.join(OUT_DIR, out_base + ".png")
    tiff_path = os.path.join(OUT_DIR, out_base + ".tiff")
    fig.savefig(png_path,  dpi=dpi, bbox_inches="tight")
    fig.savefig(tiff_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


lig_data = {}
for lig in LIG_IDS:
    lig_dir = os.path.join(ROOT_DIR, lig)
    if not os.path.isdir(lig_dir):
        raise FileNotFoundError(f"Missing ligand folder: {lig_dir}")

    com_path   = os.path.join(lig_dir, COM_FILE)
    res_path   = os.path.join(lig_dir, RES_FILE)
    frame_path = os.path.join(lig_dir, FRAME_FILE)

    must_exist(com_path,   f"{lig} COM file")
    must_exist(res_path,   f"{lig} nc_res file")
    com = read_com(com_path)
    res = read_nc_res(res_path)
    frame = read_nc_frame(frame_path) if os.path.exists(frame_path) else None
    lig_data[lig] = {"dir": lig_dir, "com": com, "res": res, "frame": frame}

print("Loaded ligands:", ", ".join(lig_data.keys()))


fig = plt.figure(figsize=(14, 5))
ax = fig.gca()

stats_lines = []
for lig in LIG_IDS:
    df = lig_data[lig]["com"].copy()
    df = df[(df["time"] >= TIME_START_NS) & (df["time"] <= TIME_END_NS)].copy()

    ax.plot(df["time"], df["dist"], linewidth=1.3, label=f"Lig{lig}")
    mean, sd, mx, over = com_stats(df["dist"], THRESH_A)
    stats_lines.append(
        f"Lig{lig}: Mean±SD {mean:.2f}±{sd:.2f} Å, Max {mx:.2f} Å, >{THRESH_A}Å {over:.1f}%"
    )

ax.axhline(THRESH_A, linestyle="--", linewidth=1.5)
ax.text(0.01, 0.98, "\n".join(stats_lines), transform=ax.transAxes, va="top")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Ligand–pocket COM distance (Å)")
ax.set_title("COM distance in 450–500 ns (final 50 ns, 3-ligand comparison)")
ax.set_xlim(TIME_START_NS, TIME_END_NS)
ax.legend()
fig.tight_layout()
save_png_tiff(fig, "FigA_COM_timeseries_3lig")
fig = plt.figure(figsize=(10, 6))
ax = fig.gca()
bins = 30

for lig in LIG_IDS:
    df = lig_data[lig]["com"]
    ax.hist(df["dist"], bins=bins, alpha=0.55, label=f"Lig{lig}")

ax.axvline(THRESH_A, linestyle="--", linewidth=1.5)
ax.set_xlabel("Ligand–pocket COM distance (Å)")
ax.set_ylabel("Counts")
ax.set_title("Distribution of COM distance (450–500 ns, final 50 ns)")
ax.legend()
fig.tight_layout()
save_png_tiff(fig, "FigC_COM_hist_3lig")


fig = plt.figure(figsize=(14, 6))
ax = fig.gca()
any_frame = False

for lig in LIG_IDS:
    fr = lig_data[lig]["frame"]
    if fr is None:
        continue
    any_frame = True

    t_ns = frame_to_ns_by_rate(fr["frame"], start_ns=TIME_START_NS, dt_ns=DT_NS)
    ax.plot(t_ns, fr["contacts"], linewidth=1.2, label=f"Lig{lig}")

if any_frame:
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Contact number")
    ax.set_title("Contact number vs time (450–500 ns, final 50 ns)")
    ax.set_xlim(TIME_START_NS, TIME_END_NS)
    ax.legend()
    fig.tight_layout()
    save_png_tiff(fig, "FigD_Contacts_timeseries_3lig")
else:
    plt.close(fig)
    print("Skip FigD: nc_frame_last50ns.dat not found for all ligands.")

topn = 12
RES_COL = "Res_PDB"
VAL_COL = "TotalFrac"

res_union = set()
for lig in LIG_IDS:
    df = lig_data[lig]["res"]
    top_res = df.sort_values(VAL_COL, ascending=False).head(topn)[RES_COL].astype(int).tolist()
    res_union.update(top_res)
residues = sorted(res_union)

vals = {}
for lig in LIG_IDS:
    df = lig_data[lig]["res"]
    m = dict(zip(df[RES_COL].astype(int), df[VAL_COL].astype(float)))
    vals[lig] = [m.get(r, 0.0) for r in residues]

x = np.arange(len(residues))
k = len(LIG_IDS)
width = 0.78 / k

fig = plt.figure(figsize=(18, 6))
ax = fig.gca()
for i, lig in enumerate(LIG_IDS):
    ax.bar(x - 0.39 + width*(i+0.5), vals[lig], width=width, label=f"Lig{lig}")

ax.set_xticks(x)
ax.set_xticklabels([str(int(r)) for r in residues], rotation=0)
ax.set_xlabel("Pocket residue number (PDB numbering)")
ax.set_ylabel("TotalFrac (contact occupancy)")
ax.set_title(f"Top pocket residues by contact occupancy (450–500 ns, top {topn} union)")
ax.legend()
fig.tight_layout()
save_png_tiff(fig, "FigB_TopResidues_TotalFrac_3lig")

print("DONE. Figures saved to:", OUT_DIR)
