import re
import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = "FINAL_DECOMP_MMPBSA.dat"
TOP_N = 10
OUT_PNG = "GBSA_residue_decomposition_topN_protein.png"

PDB_OFFSET = 324
HIGHLIGHT = "ARG-415"

LIG_RESNAMES = {"UNL", "LIG", "MOL"}


def find_header_line(path, prefix="Residue,Location"):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.strip().startswith(prefix):
                return i
    raise RuntimeError("❌ 未找到 Residue,Location 表头行")


def load_decomp(path):
    start = find_header_line(path)

    df = pd.read_csv(
        path,
        skiprows=start,
        sep=",",
        engine="python"
    )

    df["Residue"] = df["Residue"].astype(str)

    resname, amberidx = [], []
    pat = re.compile(r"^\s*([A-Z]{3})\s+(\d+)\s*$")

    for s in df["Residue"]:
        m = pat.match(s)
        if m:
            resname.append(m.group(1))
            amberidx.append(int(m.group(2)))
        else:
            resname.append(None)
            amberidx.append(None)

    df["ResName"] = resname
    df["AmberIdx"] = amberidx

    df = df.dropna(subset=["ResName", "AmberIdx"]).copy()

    df["TOTAL"] = pd.to_numeric(df["TOTAL"], errors="coerce")
    df = df.dropna(subset=["TOTAL"])

    df["PdbNum"] = df["AmberIdx"].astype(int) + PDB_OFFSET
    df["Residue_Label"] = df["ResName"] + "-" + df["PdbNum"].astype(str)

    return df


df = load_decomp(INPUT_FILE)
df_prot = df[~df["ResName"].isin(LIG_RESNAMES)].copy()

df_agg = (
    df_prot
    .groupby("Residue_Label", as_index=False)["TOTAL"]
    .mean()
)

top = (
    df_agg
    .sort_values("TOTAL", ascending=True)
    .head(TOP_N)
    .sort_values("TOTAL", ascending=True)
)

plt.figure(figsize=(7.2, 4.2))
bars = plt.bar(top["Residue_Label"], top["TOTAL"])

if HIGHLIGHT:
    for bar, lab in zip(bars, top["Residue_Label"]):
        bar.set_alpha(1.0 if lab == HIGHLIGHT else 0.35)

plt.axhline(0, lw=1)
plt.ylabel("Per-residue contribution (kcal/mol)", fontsize=11)
plt.xlabel("Residue", fontsize=11)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()

plt.savefig(OUT_PNG, dpi=600)
plt.savefig(OUT_PNG.replace(".png", ".tiff"), dpi=600)
plt.show()

print("Saved:", OUT_PNG)
print(top.to_string(index=False))
