from pathlib import Path
import numpy as np
import pandas as pd

DATA_ROOT = Path("stage3")
TOTAL_TIME_NS = 500.0
EXPECTED_FRAMES = 5000
PREFIX_RMSD = "rmsd_protein_S4_rep1"
PREFIX_RG   = "rg_protein_S4_rep1"
PREFIX_SASA = "sasa_protein_S4_rep1"
PREFIX_HB_L2P_SERIES = "hb_LIG_to_PROT_S4_rep1_ts"
PREFIX_HB_P2L_SERIES = "hb_PROT_to_LIG_S4_rep1_ts"
PREFIX_HB_L2P_AVG = "hb_LIG_to_PROT_S4_rep1_avg"
PREFIX_HB_P2L_AVG = "hb_PROT_to_LIG_S4_rep1_avg"

# ========== threshold ==========
THRESH_RMSD_MEAN = 2.5
THRESH_RG        = 3.0
THRESH_SASA      = 10.0
THRESH_HB        = 1.0
THRESH_MAIN_OCC  = 30.0
THRESH_FRAC      = 2


def find_file(dir_, prefix):
    files = sorted(dir_.glob(f"{prefix}*.dat"))
    if not files:
        raise FileNotFoundError(f"{dir_} lack {prefix}*.dat")
    return files[0]


def load_series_500ns(path):
    arr = np.loadtxt(path, comments=['#', '@'])
    if arr.ndim == 1:
        y = arr.astype(float)
    else:
        y = arr[:, 1].astype(float)
    n = len(y)
    x = np.linspace(0, TOTAL_TIME_NS, n, endpoint=False)
    return x, y


def dev_percent(arr):
    mean = arr.mean()
    return np.max(np.abs(arr - mean)) / mean * 100.0


def parse_hb_avg(path):
    fracs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if (not line) or line.startswith(('#', '@')):
            continue
        try:
            fracs.append(float(line.split()[4]))
        except Exception:
            pass
    if not fracs:
        return 0.0, 0, 0.0
    fracs = np.array(fracs)
    return fracs.max() * 100.0, (fracs >= 0.1).sum(), fracs.sum()


# ========== anlyze  ==========
def analyze_complex(ligid):
    path = DATA_ROOT / str(ligid)

    # ---- RMSD ----
    xr, yr = load_series_500ns(find_file(path, PREFIX_RMSD))
    rmsd_mean = yr.mean()

    # ---- Rg ----
    xg, yg = load_series_500ns(find_file(path, PREFIX_RG))
    rg_dev = dev_percent(yg)

    # ---- SASA ----
    xs, ys = load_series_500ns(find_file(path, PREFIX_SASA))
    sasa_dev = dev_percent(ys)

    # ---- avgout hbond（frac / frac≥0.1  / total average H-bond）----
    main_l2p, n_l2p, hb_l2p = parse_hb_avg(find_file(path, PREFIX_HB_L2P_AVG))
    main_p2l, n_p2l, hb_p2l = parse_hb_avg(find_file(path, PREFIX_HB_P2L_AVG))
    main_occ = max(main_l2p, main_p2l)
    n_frac = n_l2p + n_p2l
    hb_avg_total = hb_l2p + hb_p2l

    # ---- hbond timeseries（L→P + P→L）----
    xhb1, yhb1 = load_series_500ns(find_file(path, PREFIX_HB_L2P_SERIES))
    xhb2, yhb2 = load_series_500ns(find_file(path, PREFIX_HB_P2L_SERIES))
    hb_total = yhb1 + yhb2

    hb_all = hb_total.mean()

    # ---- stage judgement ----
    pass_stage3 = all([
        rmsd_mean <= THRESH_RMSD_MEAN,
        rg_dev <= THRESH_RG,
        sasa_dev <= THRESH_SASA,
        hb_all >= THRESH_HB,
        main_occ >= THRESH_MAIN_OCC,
        n_frac >= THRESH_FRAC,
    ])

    return {
        "id": ligid,
        "rmsd_mean": rmsd_mean,
        "rg_dev_pct": rg_dev,
        "sasa_dev_pct": sasa_dev,
        "hb_avg_all": hb_all,
        "hb_main_occ": main_occ,
        "hb_frac_ge_0.1": n_frac,
        "hb_avg_total_from_avgout": hb_avg_total,
        "pass_stage3": pass_stage3,
    }


def main():
    ligids = sorted(
        int(p.name) for p in DATA_ROOT.iterdir()
        if p.is_dir() and p.name.isdigit()
    )
    results = []
    for ligid in ligids:
        try:
            res = analyze_complex(ligid)
        except Exception as e:
            print(f"[SKIP] Complex {ligid}: {e}")
            continue
        if res["pass_stage3"]:
            results.append(res)
            print("\n===== Complex", ligid, "(PASS) =====")
            for k, v in res.items():
                print(f"{k}: {v}")
    df = pd.DataFrame(results)
    df.to_csv("stage3_summary.csv", index=False, encoding="utf-8-sig")
    print("\nresults in stage3_summary.csv")


if __name__ == "__main__":
    main()
