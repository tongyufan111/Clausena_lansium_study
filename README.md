
This repository provides a **fully reproducible, multi-stage computational workflow** for screening bioactive small molecules using **machine learning, molecular docking, clustering analyisis, molecular dynamics simulations, free-energy analysis, and quantum chemical calculations**.

The pipeline is designed for **ligand selection**, combining **statistical ML filtering**, **structure-aware clustering**, and **progressively stricter MD-based stability criteria**

## Overview of the workflow

ML + Docking screening + Clustering -> Stage 1 MD (20 ns) rapid elimination -> Stage 2 MD (100 ns) intermediate refinement ->Stage 3 MD (500 ns)long-timescale validation -> COM analysis (last 50 ns) -> MM-PBSA / PCA / FEL / QM


Each stage applies **explicit, quantitative thresholds** to ensure physical interpretability and reproducibility.


## Repository structure
```text
.
├── analyze_and_cluster.py     # ML + docking screening & clustering
├── analyze_stage1.py          # Stage-1 MD screening (20 ns)
├── analyze_stage2.py          # Stage-2 MD screening (100 ns)
├── analyze_stage3.py          # Stage-3 MD screening (500 ns)
├── COM.py                     # COM-based binding stability analysis
├── complex_vs_apo.py          # Complex vs apo MD comparison
├── PCA&FEL.py                 # PCA and free energy landscape
├── pbsa.py                    # MM-PBSA energy analysis
├── gbsa.py                    # Per-residue energy decomposition
├── homo-lumo.py               # HOMO–LUMO visualization
└── README.md
```

## Requirements

* Python ≥ 3.8
* numpy, pandas, matplotlib, scipy
* RDKit (for clustering and SMILES processing)
* AMBER (for MD simulations and MMPBSA.py outputs)

All plotting scripts run in **headless mode** and are suitable for HPC environments.


## Step 1. ML + docking screening and clustering

**Script:** `analyze_and_cluster.py`

### Purpose

* Integrate ML-predicted activity scores (LBP)
* Integrate docking scores (AutoDock Vina, LibDock)
* Apply percentile-based internal thresholds
* Enforce chemical diversity via fingerprint clustering
* Select Top-K ligands for MD simulations

### Run

```bash
python analyze_and_cluster.py \
  --infile ligands_screen.csv \
  --tc 0.7 \
  --per_cluster_max 2 \
  --topk 30
```

### Output (`./results/`)

* `all_with_metrics.csv`
* `passed_sorted.csv`
* `Top30_clustered_allow2.csv`
* Threshold histograms (LBP / Vina / LibDock)
* Correlation plots
* `figure_captions_en.txt`


## Step 2. Stage-1 MD screening (20 ns)

**Script:** `analyze_stage1.py`

### Purpose

Rapid elimination of unstable complexes using short MD trajectories.

### Input structure

```text
stage1/
├── 1/
│   ├── rmsd_protein*.dat
│   ├── rg_protein*.dat
│   ├── sasa_protein*.dat
│   └── hb_lig_prot_avg*.dat
├── 2/
└── ...
```

### Run

```bash
python analyze_stage1.py top30_stage1_data

```

### Screening criteria

* Mean H-bond count ≥ **0.70**
* Main H-bond occupancy ≥ **20%**

### Output

* `stage1_summary.csv`
* `stage1_top12.csv`


## Step 3. Stage-2 MD screening (100 ns)

**Script:** `analyze_stage2.py`

### Purpose

Intermediate screening using longer trajectories and stricter structural criteria.

### Input structure

```text
top12_stage2_data/
├── 3/
│   ├── rmsd_protein*.dat
│   ├── rg_protein*.dat
│   ├── sasa_protein*.dat
│   ├── hb_lig_prot_avg*.dat
│   └── hb_PROT_to_LIG_S2_3rep_avg*.dat
├── 7/
└── ...
```

### Run

```bash
python analyze_stage2.py
```

### Screening criteria

* Mean Cα-RMSD ≤ **2.5 Å**
* Rg deviation ≤ **3%**
* SASA deviation ≤ **10%**
* Mean H-bond count ≥ **1.0**
* Main H-bond occupancy ≥ **30%**
* H-bond frac ≥ 0.1 count ≥ **2**

### Output

* `stage2_summary.csv`
* `stage2_top5.csv`


## Step 4. Stage-3 MD screening (500 ns)

**Script:** `analyze_stage3.py`

### Purpose

Long-timescale validation of binding stability.

### Input structure

```text
stage3/
├── 3/
│   ├── rmsd_protein_S4_rep1*.dat
│   ├── rg_protein_S4_rep1*.dat
│   ├── sasa_protein_S4_rep1*.dat
│   ├── hb_LIG_to_PROT_S4_rep1_avg*.dat
│   ├── hb_PROT_to_LIG_S4_rep1_avg*.dat
│   ├── hb_LIG_to_PROT_S4_rep1_ts*.dat
│   └── hb_PROT_to_LIG_S4_rep1_ts*.dat
├── 7/
└── ...
```

### Run

```bash
python analyze_stage3.py
```

### Screening criteria

Same as Stage-2, applied over **500 ns** trajectories.

### Output

* `stage3_summary.csv`


## Step 5. COM-based binding stability analysis (final 50 ns)

**Script:** `COM.py`

### Purpose

Quantitative evaluation of ligand retention in the binding pocket during the equilibrated window.

### Input structure

```text
.
├── 3/
│   ├── lig_pocket_com_last50ns.dat
│   ├── nc_res_last50ns.dat
│   └── nc_frame_last50ns.dat
├── 7/
├── 17/
```

### Run

```bash
python COM.py
```

### Criterion

* Stable binding: ligand–pocket COM distance ≤ **4.5 Å**

### Output (`figs_last50ns/`)

* COM distance time series
* COM distance distributions
* Contact number vs time
* Per-residue contact occupancy (PDB numbering)

---

## Step 6. MD stability: complex vs apo protein

**Script:** `complex_vs_apo.py`

### Purpose

Compare structural stability between ligand-bound complex and apo protein.

### Run

```bash
python complex_vs_apo.py
```

### Output

* RMSD / Rg / SASA / RMSF plots
* 2×2 NPJ-style composite figure (`.png` / `.tiff`)

---

## Step 7. PCA and free energy landscape (FEL)

**Script:** `PCA&FEL.py`

### Input

`pca_projection.dat` generated by `cpptraj`.

### Run

```bash
python PCA&FEL.py
```

### Output

* PC1–PC2 scatter
* 2D FEL contour map
* 3D FEL surface

## Step 8. MM-PBSA free energy analysis

**Script:** `pbsa.py`

### Purpose

Parse `MMPBSA.py` outputs and generate publication-quality energy decomposition plots.

### Run

```bash
python pbsa.py
```

### Output

* `PBSA_*.csv`
* Binding energy bar plots (`.png` / `.tiff`)


## Step 9. Per-residue energy decomposition

**Script:** `gbsa.py`

### Run

```bash
python gbsa.py
```

### Output

* Top contributing residues
* Highlighted residue bar plots

## Step 10. HOMO–LUMO visualization

**Script:** `homo-lumo.py`

### Input

```text
homo.png
lumo.png
```

### Run

```bash
python homo-lumo.py
```

### Output

* Two-panel HOMO/LUMO figure (NPJ-style)

---


## Notes on reproducibility

* All thresholds are explicitly defined and reported
* Screening is **progressive and irreversible**
* COM analysis precedes MM-PBSA to ensure valid energy windows
* All figures are exported at **600 dpi** in **PNG + TIFF** formats


