# PathwayMAML: Pathway-aware Meta-learning for Cancer-to-Non-Cancer Transfer

This repository provides the source code for **PathwayMAML**, a pathway-aware MAML framework for cancer-to-non-cancer transfer in few-shot multi-omics disease classification.

<p align="center">
  <img src="figure/overview.png" width="100%" alt="PathwayMAML overview">
</p>

---
## Overview

PathwayMAML follows a three-stage pipeline:
1. **Meta-training** on multiple TCGA cancer tasks to learn a transferable initialization.
2. **Meta-testing / adaptation** to unseen GEO disease cohorts under *k*-shot supervision.
3. **Biomarker identification** by computing pathway-level attributions and performing cohort-level statistical testing.

The model takes **2,534 joint mRNA+miRNA features** as input and projects them into a **Reactome pathway layer (383 neurons)**
via a sparse gene→pathway mapping, followed by a **hidden layer (95 neurons)** and a binary output layer.
This design enables pathway-level interpretability by attributing predictions to pathway neurons (e.g., via Layer Integrated Gradients).

## Environment Setup (Using anaconda)
```bash
conda create -n PathwayMAML python=3.9
conda activate PathwayMAML

pip install -r requirements.txt
```
---
## Datasets

1. Download the datasets from:  
   https://drive.google.com/drive/folders/1nXHURlv2M6AoGz43JBTj7l3qofa1DaSh?usp=sharing
2. Place the downloaded files under `data/`:

```bash
PathwayMAML/
└── data/
    └── <downloaded_tcga_files_here>
```
---
## Tutorial

This repository provides the core 3-step pipeline used in the paper:

### Step 1 — Meta-learning evaluation + raw attributions
Run PathwayMAML to obtain **few-shot performance** and generate **raw LIG attributions** on GEO target cohorts.

- Script: `model/pathway_MAML.py`
- Outputs: evaluation metrics (PR-AUC/ROC-AUC) and per-sample raw LIG/IG attribution files under `results/`.

#### Key switch: `--init_mode`
`--init_mode` controls whether evaluation uses a meta-trained initialization (PathwayMAML) or the ablation without meta-training (Pathway MLP).

- `--init_mode pathway_maml` (default): evaluates using the **meta-trained checkpoint** (`--ckpt_path`)
- `--init_mode pathway_mlp`: runs the **Pathway MLP ablation (no meta-training)** by evaluating from a **random initialization** (the script saves and loads `data/weights/random_init.pth`)

---

### Step 2 — Biomarker identification (MWU + BH-FDR + effect size)
Identify cohort-level biomarker pathways by performing statistical testing on the raw attributions.

- Script: `preprocessing/lig_mwu_biomarker.py`
- Inputs: `results/LIG_raw/<disease>/query_LIG_task*.csv` (and optionally `results/IG_raw/<disease>/query_GENE_IG_task*.csv`)
- Outputs: labeled biomarker tables under `results/biomarker/` (and `results/gene_biomarker/`)

#### Key switch: `--random_init`
`--random_init` selects whether to process the **PathwayMAML outputs** or the **random-init ablation outputs**.

- (default) **PathwayMAML** mode:
  - Input: `results/LIG_raw/` and `results/IG_raw/`
  - Output: `results/biomarker/` and `results/gene_biomarker/`
- `--random_init` **Pathway MLP (random init) ablation** mode:
  - Input: `results/LIG_raw_random_init/` and `results/IG_raw_random_init/`
  - Output: `results/biomarker_random_init/` and `results/gene_biomarker_random_init/`

---

### Step 3 — Method comparison (PathwayMAML vs Pathway MLP)
Compare biomarker evidence between **PathwayMAML** and the ablation **Pathway MLP** using summary visualizations.

- Script: `results/histogram.py`
- Inputs (from Step 2):
  - `results/biomarker/` vs `results/biomarker_random_init/`
  - `results/gene_biomarker/` vs `results/gene_biomarker_random_init/`
- Outputs:
  - Figures and summary CSVs saved under `results/histogram/`





