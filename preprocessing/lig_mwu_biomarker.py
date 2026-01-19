#!/usr/bin/env python3
"""
lig_mwu_biomarker.py

Run the full pipeline in ONE pass:
1) From per-disease LIG CSVs (query_LIG_task*.csv), build per-sample median LIG table.
2) Run Mann–Whitney U per pathway with BH-FDR and Cliff's delta, save MWU_BH_results.csv to lig_root/<disease>/.
3) Filter q<=alpha, label effect-size magnitude + direction, and save significant_pathways_labeled.csv to out_root/<disease>/.
4) Write a summary_significant_counts.csv under out_root/ summarizing counts per disease.

Usage (example):
    python lig_mwu_biomarker.py --lig_root /path/to/LIG_raw --out_root /path/to/biomarker --alpha 0.05 --save_median_table

Notes:
- If you already have MWU_BH_results.csv and just want the biomarker tables, use --reuse_mwu.
- If labels are inconsistent across tasks for the same sample_idx, the default is to raise.
  Use --non_strict to auto-resolve by the mode label.
"""
import os
import glob
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# ==========================================
# Dynamic Path Configuration
# ==========================================

# 1. Determine the script's directory
# Location: .../Pathway_MAML/preprocessing/lig_mwu_biomarker.py
CURRENT_DIR = Path(__file__).resolve().parent

# 2. Define Project Root
# Move up 1 level: preprocessing/ -> Pathway_MAML/
PROJECT_ROOT = CURRENT_DIR.parent

# 3. Define Base Results Directory
# Path: .../Pathway_MAML/results/
RESULTS_DIR = PROJECT_ROOT / "results"

# 4. Define Default Paths (Converted to String for argparse compatibility)
DEFAULT_LIG_ROOT = str(RESULTS_DIR / "LIG_raw")
DEFAULT_OUT_ROOT = None
DEFAULT_IG_ROOT = str(RESULTS_DIR / "IG_raw")
DEFAULT_GENE_OUT_ROOT = None

# Cliff's delta magnitude thresholds
TH_NEGL = 0.147
TH_SMALL = 0.330
TH_MED  = 0.474

# -------------------------- Core utilities --------------------------

def discover_diseases_from_lig(lig_root: str):
    """Find diseases that have query_LIG_task*.csv under lig_root/<disease>/"""
    diseases = []
    for name in os.listdir(lig_root):
        dpath = os.path.join(lig_root, name)
        if os.path.isdir(dpath) and glob.glob(os.path.join(dpath, "query_LIG_task*.csv")):
            diseases.append(name)
    diseases.sort()
    return diseases

def read_and_concat(csv_dir):
    files = sorted(glob.glob(os.path.join(csv_dir, "query_LIG_task*.csv")))
    if not files:
        raise FileNotFoundError(f"No query_LIG_task*.csv found in {csv_dir}")
    dfs = []
    for p in files:
        df = pd.read_csv(p)
        # minimal validation
        if not {'sample_idx', 'label'}.issubset(df.columns):
            raise ValueError(f"CSV {p} missing required columns 'sample_idx' and 'label'")
        dfs.append(df)
    big = pd.concat(dfs, ignore_index=True)
    return big

def label_consistency_check(df, strict=True):
    """
    Ensures each sample_idx has a unique label.
    If strict=True, raises if inconsistency is found.
    Otherwise, keeps the *mode* label per sample and warns.
    """
    nunique = df.groupby('sample_idx')['label'].nunique()
    bad = nunique[nunique > 1]
    if len(bad) > 0:
        msg = f"Found {len(bad)} sample_idx with inconsistent labels. "
        if strict:
            raise ValueError(msg + "Set --non_strict to auto-resolve by mode label.")
        else:
            warnings.warn(msg + "Resolving by mode label per sample_idx.")
            # mode per sample_idx
            label_mode = (df.groupby('sample_idx')['label']
                            .agg(lambda s: s.mode().iloc[0]))
            # apply: first map mode labels
            mode_map = label_mode.to_dict()
            df['label'] = df['sample_idx'].map(mode_map)
    return df

def make_sample_median_table(df):
    """
    From tall df (rows = sample occurrences across tasks), produce:
    - median_df: per-sample median LIG per pathway + label column
    """
    path_cols = [c for c in df.columns if c not in ('sample_idx', 'label')]
    # build label map (assumes consistency)
    label_map = df.groupby('sample_idx')['label'].agg(lambda s: s.iloc[0]).to_dict()
    # per-sample median across tasks
    med = df.groupby('sample_idx')[path_cols].median().reset_index()
    med['label'] = med['sample_idx'].map(label_map)
    # reorder columns
    med = med[['sample_idx', 'label'] + path_cols]
    return med, path_cols

def run_mwu_for_pathways(median_df, path_cols, alpha=0.05, min_group_n=1):
    """
    Run two-sided Mann-Whitney U per pathway with BH-FDR.
    Also compute Cliff's delta.
    Returns a result DataFrame sorted by q_value.
    """
    res = {
        'Pathway': [],
        'n_label0': [],
        'n_label1': [],
        'median_label0': [],
        'median_label1': [],
        'delta_median(label1 - label0)': [],
        'U_stat': [],
        'p_value': [],
        'cliffs_delta': [],
    }

    for p in path_cols:
        x = median_df.loc[median_df['label'] == 0, p].astype(float).dropna().values
        y = median_df.loc[median_df['label'] == 1, p].astype(float).dropna().values
        n0, n1 = len(x), len(y)
        if n0 < min_group_n or n1 < min_group_n:
            U = np.nan
            pval = np.nan
            cd = np.nan
            m0 = np.nan
            m1 = np.nan
        else:
            # SciPy >=1.7 supports method='auto'; fallback if older
            try:
                U, pval = mannwhitneyu(x, y, alternative='two-sided', method='auto')
            except TypeError:
                U, pval = mannwhitneyu(x, y, alternative='two-sided')

            try:
                Uy, _ = mannwhitneyu(y, x, alternative='greater', method='auto')
            except TypeError:
                Uy, _ = mannwhitneyu(y, x, alternative='greater')
                
            m0 = float(np.median(x)) if n0 > 0 else np.nan
            m1 = float(np.median(y)) if n1 > 0 else np.nan
            cd = 2.0 * (Uy / (n0 * n1)) - 1.0 # δ > 0 => y > x

        res['Pathway'].append(p)
        res['n_label0'].append(n0)
        res['n_label1'].append(n1)
        res['median_label0'].append(m0)
        res['median_label1'].append(m1)
        res['delta_median(label1 - label0)'].append(
            (m1 - m0) if (not np.isnan(m1) and not np.isnan(m0)) else np.nan
        )
        res['U_stat'].append(U)
        res['p_value'].append(pval)
        res['cliffs_delta'].append(cd)

    out = pd.DataFrame(res)

    # BH-FDR
    # handle all-NaN p-values gracefully
    pvals = out['p_value'].values
    mask_valid = ~np.isnan(pvals)
    qvals = np.full_like(pvals, fill_value=np.nan, dtype=float)
    rejs = np.zeros_like(pvals, dtype=bool)
    if mask_valid.sum() > 0:
        rej, q, _, _ = multipletests(pvals[mask_valid], alpha=alpha, method='fdr_bh')
        qvals[mask_valid] = q
        rejs[mask_valid] = rej
    out['q_value'] = qvals
    out['significant(q<=%.3f)' % alpha] = rejs

    # sort by q_value (NaNs at bottom)
    out = out.sort_values(by=['q_value', 'p_value'], na_position='last').reset_index(drop=True)
    return out

# -------------------------- IG (gene) helpers --------------------------
def discover_diseases_from_ig(ig_root: str):
    """Find diseases that have query_GENE_IG_task*.csv under ig_root/<disease>/"""
    import glob, os
    diseases = []
    for name in os.listdir(ig_root):
        dpath = os.path.join(ig_root, name)
        if os.path.isdir(dpath) and glob.glob(os.path.join(dpath, "query_GENE_IG_task*.csv")):
            diseases.append(name)
    diseases.sort()
    return diseases

def read_and_concat_geneIG(csv_dir):
    """Concatenate query_GENE_IG_task*.csv under csv_dir"""
    import glob, os, pandas as pd
    files = sorted(glob.glob(os.path.join(csv_dir, "query_GENE_IG_task*.csv")))
    if not files:
        raise FileNotFoundError(f"No query_GENE_IG_task*.csv found in {csv_dir}")
    dfs = []
    for p in files:
        df = pd.read_csv(p)
        if not {'sample_idx', 'label'}.issubset(df.columns):
            raise ValueError(f"CSV {p} missing required columns 'sample_idx' and 'label'")
        dfs.append(df)
    big = pd.concat(dfs, ignore_index=True)
    return big

def run_mwu_for_genes(median_df, gene_cols, alpha=0.05, min_group_n=1):
    """
    Reuse run_mwu_for_pathways, then rename 'Pathway' -> 'Gene' in the output.
    """
    out = run_mwu_for_pathways(median_df, gene_cols, alpha=alpha, min_group_n=min_group_n)
    if 'Pathway' in out.columns:
        out = out.rename(columns={'Pathway': 'Gene'})
    return out

def biomarker_from_mwu_df_genes(out_df: pd.DataFrame, out_root: str, disease: str, alpha: float):
    """
    Same as biomarker_from_mwu_df but for genes.
    Writes <out_root>/<disease>/significant_genes_labeled.csv
    """
    # Ensure expected columns exist (after rename)
    required = {"Gene", "q_value", "p_value", "cliffs_delta"}
    missing = [c for c in required if c not in out_df.columns]
    if missing:
        print(f"[WARN] Skip {disease}: missing columns {missing}")
        return None

    # Filter q<=alpha and label effect size
    sig = out_df[out_df["q_value"].astype(float) <= float(alpha)].copy()
    if sig.empty:
        print(f"[INFO] {disease}: no genes with q_value <= {alpha}")
        return None

    sig["effect_size_magnitude"] = sig["cliffs_delta"].apply(label_magnitude_from_delta)

    def direction_text(delta):
        if pd.isna(delta):
            return "NA"
        return "higher_in_label1(disease)" if delta > 0 else ("higher_in_label0(control)" if delta < 0 else "no_diff")
    sig["direction"] = sig["cliffs_delta"].apply(direction_text)

    sig["q_value"] = pd.to_numeric(sig["q_value"], errors="coerce")
    sig["cliffs_delta"] = pd.to_numeric(sig["cliffs_delta"], errors="coerce")
    sig["abs_delta"] = sig["cliffs_delta"].abs()
    sig = sig.sort_values(by=["q_value", "abs_delta"],
                          ascending=[True, False],
                          na_position="last").drop(columns=["abs_delta"])

    out_dir = os.path.join(out_root, disease)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "significant_genes_labeled.csv")

    cols = [c for c in ["Gene",
                        "q_value", "p_value",
                        "cliffs_delta",
                        "delta_median(label1 - label0)",
                        "median_label0", "median_label1",
                        "n_label0", "n_label1",
                        "effect_size_magnitude", "direction"]
            if c in sig.columns]
    sig[cols].to_csv(out_csv, index=False)
    print(f"[INFO] Saved: {out_csv}  (n={len(sig)})")
    return out_csv

# --------------------- Biomarker labeling utilities ---------------------

def label_magnitude_from_delta(delta: float) -> str:
    """Label magnitude by |Cliff's δ| thresholds."""
    if pd.isna(delta):
        return "NA"
    a = abs(delta)
    if a < TH_NEGL:
        return "negligible"
    elif a < TH_SMALL:
        return "small"
    elif a < TH_MED:
        return "medium"
    else:
        return "large"

def biomarker_from_mwu_df(out_df: pd.DataFrame, out_root: str, disease: str, alpha: float):
    """Filter q<=alpha, add effect-size magnitude + direction, save CSV to out_root/<disease>/"""
    # Basic checks
    required = {"Pathway", "q_value", "p_value", "cliffs_delta"}
    missing = [c for c in required if c not in out_df.columns]
    if missing:
        print(f"[WARN] Skip {disease}: missing columns {missing}")
        return None

    sig = out_df[out_df["q_value"].astype(float) <= float(alpha)].copy()
    if sig.empty:
        print(f"[INFO] {disease}: no pathways with q_value <= {alpha}")
        return None

    # Label effect-size magnitude and direction
    sig["effect_size_magnitude"] = sig["cliffs_delta"].apply(label_magnitude_from_delta)

    def direction_text(delta):
        if pd.isna(delta):
            return "NA"
        return "higher_in_label1(disease)" if delta > 0 else ("higher_in_label0(control)" if delta < 0 else "no_diff")
    sig["direction"] = sig["cliffs_delta"].apply(direction_text)

    sig["q_value"] = pd.to_numeric(sig["q_value"], errors="coerce")
    sig["cliffs_delta"] = pd.to_numeric(sig["cliffs_delta"], errors="coerce")
    sig["abs_delta"] = sig["cliffs_delta"].abs()

    sig = sig.sort_values(by=["q_value", "abs_delta"],
                          ascending=[True, False],
                          na_position="last")
    sig = sig.drop(columns=["abs_delta"])

    out_dir = os.path.join(out_root, disease)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "significant_pathways_labeled.csv")

    cols = [c for c in ["Pathway",
                        "q_value", "p_value",
                        "cliffs_delta",
                        "delta_median(label1 - label0)",
                        "median_label0", "median_label1",
                        "n_label0", "n_label1",
                        "effect_size_magnitude", "direction"]
            if c in sig.columns]
    sig[cols].to_csv(out_csv, index=False)
    print(f"[INFO] Saved: {out_csv}  (n={len(sig)})")
    return out_csv

# ------------------------------- Main -------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="End-to-end: per-sample median LIG -> MWU+BH-FDR -> biomarker labeling (q<=alpha)."
    )
    ap.add_argument("--lig_root", type=str, default=DEFAULT_LIG_ROOT,
                    help=f"Root dir with per-disease LIG CSVs (default: {DEFAULT_LIG_ROOT})")
    ap.add_argument("--out_root", type=str, default=DEFAULT_OUT_ROOT,
                    help=f"Output root for labeled biomarker tables (default: {DEFAULT_OUT_ROOT})")
    ap.add_argument("--alpha", type=float, default=0.05, help="FDR/BH q-value threshold and reporting threshold.")
    ap.add_argument("--diseases", type=str, nargs="*", default=None,
                    help="Specific disease folder names to process. If omitted, auto-discover all under lig_root.")
    ap.add_argument("--min_group_n", type=int, default=1,
                    help="Minimum samples per label group to run a test.")
    ap.add_argument("--non_strict", action="store_true",
                    help="Resolve label conflicts by mode instead of raising an error.")
    ap.add_argument("--save_median_table", action="store_true",
                    help="Save per-sample median LIG table for each disease.")
    ap.add_argument("--reuse_mwu", action="store_true",
                    help="If MWU_BH_results.csv already exists, reuse it instead of recomputing.")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--process_gene_ig", dest="process_gene_ig", action="store_true",
                    help="Also run the same pipeline for gene IG CSVs (query_GENE_IG_task*.csv).")
    group.add_argument("--no_process_gene_ig", dest="process_gene_ig", action="store_false",
                    help="Skip the gene IG pipeline.")
    ap.set_defaults(process_gene_ig=True) 
    ap.add_argument("--ig_root", type=str, default=DEFAULT_IG_ROOT,
                    help=f"Root dir with per-disease gene IG CSVs (default: {DEFAULT_IG_ROOT})")
    ap.add_argument("--out_root_gene", type=str, default=DEFAULT_GENE_OUT_ROOT,
                    help=f"Output root for gene-level biomarker tables (default: {DEFAULT_GENE_OUT_ROOT})")
    ap.add_argument("--save_median_table_gene", action="store_true",
                    help="Save per-sample median Gene IG table for each disease.")
    ap.add_argument("--reuse_mwu_gene", action="store_true",
                    help="If MWU_BH_results_gene.csv already exists, reuse it instead of recomputing.")
    ap.add_argument("--random_init", action="store_true",
                help="Use *_raw_random_init as inputs and write to *_biomarker_random_init automatically.")

    args = ap.parse_args()

    if args.random_init:
        args.lig_root = str(RESULTS_DIR / "LIG_raw_random_init")
        args.ig_root  = str(RESULTS_DIR / "IG_raw_random_init")

    lig_root = args.lig_root
    ig_root  = args.ig_root
    
    # 1) Pathway biomarker out_root auto
    if args.out_root is None:
        if "random_init" in Path(lig_root).name or "random_init" in str(lig_root):
            args.out_root = str(RESULTS_DIR / "biomarker_random_init")
        else:
            args.out_root = str(RESULTS_DIR / "biomarker")

    # 2) Gene biomarker out_root_gene auto
    if args.out_root_gene is None:
        if "random_init" in Path(ig_root).name or "random_init" in str(ig_root):
            args.out_root_gene = str(RESULTS_DIR / "gene_biomarker_random_init")
        else:
            args.out_root_gene = str(RESULTS_DIR / "gene_biomarker")
    
    out_root = args.out_root
    out_root_gene = args.out_root_gene  
    alpha = args.alpha

    if not os.path.isdir(lig_root):
        raise NotADirectoryError(f"{lig_root} is not a directory")

    # Determine diseases
    diseases = args.diseases if args.diseases else discover_diseases_from_lig(lig_root)
    if not diseases:
        raise RuntimeError("No diseases found (no query_LIG_task*.csv under lig_root).")

    print(f"[INFO] Diseases to process: {diseases}")
    os.makedirs(out_root, exist_ok=True)

    made = []
    for disease in diseases:
        print(f"\n[INFO] Processing disease: {disease}")
        ddir = os.path.join(lig_root, disease)

        mwu_csv_path = os.path.join(ddir, "MWU_BH_results.csv")
        out_df = None

        if args.reuse_mwu and os.path.isfile(mwu_csv_path):
            print(f"[INFO] Reusing existing MWU results: {mwu_csv_path}")
            out_df = pd.read_csv(mwu_csv_path)
        else:
            # 1) load & concat
            df = read_and_concat(ddir)
            # 2) label consistency
            df = label_consistency_check(df, strict=not args.non_strict)
            # 3) build per-sample median table
            median_df, path_cols = make_sample_median_table(df)
            if args.save_median_table:
                med_out = os.path.join(ddir, "per_sample_median_LIG.csv")
                median_df.to_csv(med_out, index=False)
                print(f"[INFO] Saved per-sample median LIG table: {med_out}")
            # 4) run MWU + BH-FDR
            out_df = run_mwu_for_pathways(median_df, path_cols, alpha=args.alpha, min_group_n=args.min_group_n)
            # 5) save MWU results
            out_df.to_csv(mwu_csv_path, index=False)
            print(f"[INFO] Saved MWU+BH results: {mwu_csv_path}")

        # 6) biomarker labeling from MWU results
        label_csv = biomarker_from_mwu_df(out_df, out_root, disease, alpha)
        if label_csv:
            made.append((disease, label_csv))

    # 7) summary
    if made:
        summary = []
        for d, p in made:
            n = sum(1 for _ in open(p)) - 1
            summary.append({"disease": d, f"n_significant_q<={alpha:.3f}": n})
        summary_df = pd.DataFrame(summary).sort_values("disease")
        summary_csv = os.path.join(out_root, "summary_significant_counts.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"[INFO] Saved summary: {summary_csv}")

    if args.process_gene_ig:
        ig_root = args.ig_root
        out_root_gene = args.out_root_gene
        alpha = args.alpha

        if not os.path.isdir(ig_root):
            print(f"[WARN] IG root '{ig_root}' not found; skipping gene IG.")
        else:
            gene_diseases = args.diseases if args.diseases else discover_diseases_from_ig(ig_root)

            if not gene_diseases:
                print("[WARN] No gene IG CSVs found under ig_root; skipping gene IG.")
            else:
                print(f"\n[INFO] (Gene IG) Diseases to process: {gene_diseases}")
                os.makedirs(out_root_gene, exist_ok=True)

            made_gene = []
            for disease in gene_diseases:
                print(f"\n[INFO] (Gene IG) Processing disease: {disease}")
                ddir = os.path.join(ig_root, disease)

                mwu_csv_path = os.path.join(ddir, "MWU_BH_results_gene.csv")
                out_df = None

                if args.reuse_mwu_gene and os.path.isfile(mwu_csv_path):
                    print(f"[INFO] (Gene IG) Reusing existing MWU results: {mwu_csv_path}")
                    out_df = pd.read_csv(mwu_csv_path)
                else:
                    # 1) load & concat gene IG
                    df = read_and_concat_geneIG(ddir)
                    # 2) label consistency
                    df = label_consistency_check(df, strict=not args.non_strict)
                    # 3) build per-sample median table
                    median_df, gene_cols = make_sample_median_table(df)
                    if args.save_median_table_gene:
                        med_out = os.path.join(ddir, "per_sample_median_GENE_IG.csv")
                        median_df.to_csv(med_out, index=False)
                        print(f"[INFO] (Gene IG) Saved per-sample median IG table: {med_out}")
                    # 4) run MWU + BH-FDR
                    out_df = run_mwu_for_genes(median_df, gene_cols, alpha=args.alpha, min_group_n=args.min_group_n)
                    # 5) save MWU results
                    out_df.to_csv(mwu_csv_path, index=False)
                    print(f"[INFO] (Gene IG) Saved MWU+BH results: {mwu_csv_path}")

                # 6) biomarker labeling from MWU results
                label_csv = biomarker_from_mwu_df_genes(out_df, out_root_gene, disease, alpha)
                if label_csv:
                    made_gene.append((disease, label_csv))

            # 7) summary for gene IG
            if made_gene:
                summary = []
                for d, p in made_gene:
                    n = sum(1 for _ in open(p)) - 1
                    summary.append({"disease": d, f"n_significant_q<={alpha:.3f}": n})
                summary_df = pd.DataFrame(summary).sort_values("disease")
                summary_csv = os.path.join(out_root_gene, "summary_significant_gene_counts.csv")
                summary_df.to_csv(summary_csv, index=False)
                print(f"[INFO] (Gene IG) Saved summary: {summary_csv}")

    else:
        print("[INFO] Skipped gene IG processing.")

    print("\n[DONE] All diseases processed.")

if __name__ == "__main__":
    main()
