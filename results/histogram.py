#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare two biomarker experiments and quantify how many *significant genes* are involved
in non-overlapping (unique) significant pathways per disease.

- Experiment 1 (pathways): ROOT_PATHWAY1
- Experiment 2 (pathways): ROOT_PATHWAY2
- Experiment 1 (genes):    ROOT_GENE1
- Experiment 2 (genes):    ROOT_GENE2

For each common disease:

  1) Load significant pathways from each experiment
       ROOT_PATHWAY*/<disease>/significant_pathways_labeled.csv

  2) Load significant genes from each experiment
       ROOT_GENE*/<disease>/significant_genes_labeled.csv

  3) Build a pathway -> gene set mapping for this disease using:
       build_pathway_to_genes_for_disease(gene_pathway_file, target_file)

  4) For pathways that are unique to each experiment:
       - For Exp1: unique1 = P1 - P2
         For each p in unique1:
           gene_count = # of genes that are BOTH
                        (significant genes in Exp1)
                        AND (mapped to pathway p by Reactome)

       - Likewise for Exp2.

  5) Summarize and draw:
       - Per-disease histograms of "significant gene count per unique pathway"
       - One global histogram across all diseases

This allows us to see which experiment tends to utilize more significant genes
to discover non-overlapping significant pathways.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ====================== PATH SETUP (RELATIVE) ======================

# 1. Determine Project Root
# This script is located in: .../Pathway_MAML/results/histogram.py
# So Project Root is: .../Pathway_MAML/
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# 2. Define Base Directories
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# ====================== CONFIG ======================

# Pathway biomarker results (significant_pathways_labeled.csv)
ROOT_PATHWAY1 = RESULTS_DIR / "biomarker"
ROOT_PATHWAY2 = RESULTS_DIR / "biomarker_random_init"

LABEL1 = "Pathway_MAML"
LABEL2 = "Pathway_MLP"

# Gene biomarker results (significant_genes_labeled.csv)
ROOT_GENE1 = RESULTS_DIR / "gene_biomarker"
ROOT_GENE2 = RESULTS_DIR / "gene_biomarker_random_init"

# Reactome gene-to-pathway mapping file (CSV)
GENE_PATHWAY_FILE = (
    DATA_DIR / "Reactome" / "final" / "percentile90_min10_matched_combined_pathways.csv"
)

# Disease-specific target files
diseases = [
    "idiopathic_pulmonary_fibrosis",
    "HBV-HCC",
    "cirrhosis",
    "ipf_ssc",
    "IgA_nephropathy",
]

target_files = {
    disease: DATA_DIR / "NCBI" / disease / "second_filtered_combined_counts_transposed.tsv"
    for disease in diseases
}

# Output directory for CSV and figures  # === NEW ===
HIST_OUT_DIR = RESULTS_DIR / "histogram"
HIST_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional filters by effect size magnitude.
# Possible values: None, "negligible", "small", "medium", "large"
MIN_MAGNITUDE_PATHWAY: Optional[str] = "large"  # Pathway-level filter
MIN_MAGNITUDE_GENE: Optional[str] = "large"     # Gene-level filter


# ====================== GENE–PATHWAY MAPPING ======================

def build_pathway_to_genes_for_disease(gene_pathway_file: str, target_file: str):
    """
    Build a mapping: pathway_name -> set(gene_id) for a given disease.

    - Uses a Reactome-style CSV with columns ['ID', 'Pathway'].
    - Keeps only genes that appear in the disease-specific target_file
      (all columns except the last one).
    """
    # 1) Valid gene IDs from the target file
    target_df = pd.read_csv(target_file, sep="\t")
    valid_ids = set(target_df.columns[:-1])

    # 2) Load gene–pathway mapping and filter by valid IDs
    mapping_df = pd.read_csv(gene_pathway_file)
    mapping_df = mapping_df[mapping_df["ID"].isin(valid_ids)]
    mapping_df = mapping_df.dropna(subset=["ID", "Pathway"])

    # 3) Group by pathway and collect genes into a set
    pathway_to_genes = (
        mapping_df.groupby("Pathway")["ID"]
                  .apply(lambda s: set(s.astype(str)))
                  .to_dict()
    )

    return pathway_to_genes


# ====================== LOADERS ======================

def filter_by_magnitude(df: pd.DataFrame, col: str, min_magnitude: Optional[str]) -> pd.DataFrame:
    """
    Filter a DataFrame by effect size magnitude.

    Args:
        df: Input DataFrame.
        col: Column name containing magnitude labels (e.g., "effect_size_magnitude").
        min_magnitude: Minimum magnitude to keep; one of
                       None, "negligible", "small", "medium", "large".

    Returns:
        Filtered DataFrame.
    """
    if min_magnitude is None:
        return df

    if col not in df.columns:
        return df

    order = ["negligible", "small", "medium", "large"]
    if min_magnitude not in order:
        raise ValueError(f"Unknown min_magnitude={min_magnitude}, must be in {order}.")

    min_idx = order.index(min_magnitude)
    allowed = set(order[min_idx:])
    return df[df[col].isin(allowed)]


def load_significant_pathways(root: str, min_magnitude: Optional[str] = None):
    """
    Load significant pathways for each disease from:
        root/<disease>/significant_pathways_labeled.csv

    Returns:
        dict: disease_name -> DataFrame of significant pathways.
    """
    disease_to_df = {}

    if not os.path.isdir(root):
        raise NotADirectoryError(f"{root} is not a directory")

    for name in sorted(os.listdir(root)):
        ddir = os.path.join(root, name)
        if not os.path.isdir(ddir):
            continue

        csv_path = os.path.join(ddir, "significant_pathways_labeled.csv")
        if not os.path.isfile(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if "Pathway" not in df.columns:
            print(f"[WARN] {csv_path} has no 'Pathway' column; skip.")
            continue

        df = filter_by_magnitude(df, "effect_size_magnitude", min_magnitude)
        if df.empty:
            print(f"[INFO] {csv_path} has no significant pathways after filtering.")
            continue

        disease_to_df[name] = df

    return disease_to_df


def load_significant_genes(root: str, min_magnitude: Optional[str] = None):
    """
    Load significant genes for each disease from:
        root/<disease>/significant_genes_labeled.csv

    Returns:
        dict: disease_name -> set of gene IDs.
    """
    disease_to_genes = {}

    if not os.path.isdir(root):
        raise NotADirectoryError(f"{root} is not a directory")

    for name in sorted(os.listdir(root)):
        ddir = os.path.join(root, name)
        if not os.path.isdir(ddir):
            continue

        csv_path = os.path.join(ddir, "significant_genes_labeled.csv")
        if not os.path.isfile(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if "Gene" not in df.columns:
            print(f"[WARN] {csv_path} has no 'Gene' column; skip.")
            continue

        df = filter_by_magnitude(df, "effect_size_magnitude", min_magnitude)
        if df.empty:
            print(f"[INFO] {csv_path} has no significant genes after filtering.")
            continue

        genes = set(df["Gene"].dropna())
        disease_to_genes[name] = genes

    return disease_to_genes


# ====================== MAIN LOGIC ======================

def main():
    # 1) Load pathway-level results
    print(f"[INFO] Loading significant pathways from: {ROOT_PATHWAY1}")
    disease_to_df_p1 = load_significant_pathways(ROOT_PATHWAY1, MIN_MAGNITUDE_PATHWAY)
    print(f"[INFO] {len(disease_to_df_p1)} diseases in {LABEL1}")

    print(f"[INFO] Loading significant pathways from: {ROOT_PATHWAY2}")
    disease_to_df_p2 = load_significant_pathways(ROOT_PATHWAY2, MIN_MAGNITUDE_PATHWAY)
    print(f"[INFO] {len(disease_to_df_p2)} diseases in {LABEL2}")

    # 2) Load gene-level results (significant genes)
    print(f"[INFO] Loading significant genes from: {ROOT_GENE1}")
    disease_to_genes_1 = load_significant_genes(ROOT_GENE1, MIN_MAGNITUDE_GENE)
    print(f"[INFO] {len(disease_to_genes_1)} diseases with significant genes in {LABEL1}")

    print(f"[INFO] Loading significant genes from: {ROOT_GENE2}")
    disease_to_genes_2 = load_significant_genes(ROOT_GENE2, MIN_MAGNITUDE_GENE)
    print(f"[INFO] {len(disease_to_genes_2)} diseases with significant genes in {LABEL2}")

    # 3) Determine common diseases across all four sets
    common_diseases = sorted(
        set(disease_to_df_p1.keys())
        & set(disease_to_df_p2.keys())
        & set(disease_to_genes_1.keys())
        & set(disease_to_genes_2.keys())
        & set(target_files.keys())  # ensure target_file exists  # === NEW ===
    )

    if not common_diseases:
        raise RuntimeError(
            "No common diseases with both pathway and gene results in both experiments."
        )

    print(f"[INFO] Common diseases: {common_diseases}")

    # Summary table rows
    summary_rows = []

    # For global histogram: significant gene counts per unique pathway (across all diseases)
    all_gene_counts_1 = []
    all_gene_counts_2 = []
    # global (across diseases) unions
    global_unique_pathways_1 = set()  # union of MAML-only pathways across diseases
    global_unique_pathways_2 = set()  # union of MLP-only pathways across diseases
    global_unique_valid_genes_1 = set()  # union of MAML-only valid genes across diseases
    global_unique_valid_genes_2 = set()  # union of MLP-only valid genes across diseases

    for disease in common_diseases:
        print(f"\n[INFO] Disease: {disease}")

        if disease not in target_files:
            raise KeyError(
                f"target_files for disease='{disease}' is not set. "
                f"Please add the correct target_file path for this disease."
            )
        target_file = target_files[disease]

        # Build Reactome pathway->genes mapping for this disease
        print(f"  [INFO] Building pathway->genes mapping using target_file: {target_file}")
        pathway_to_genes = build_pathway_to_genes_for_disease(GENE_PATHWAY_FILE, target_file)

        df_p1 = disease_to_df_p1[disease]
        df_p2 = disease_to_df_p2[disease]
        sig_genes_1 = disease_to_genes_1[disease]
        sig_genes_2 = disease_to_genes_2[disease]

        set_p1 = set(df_p1["Pathway"].dropna())
        set_p2 = set(df_p2["Pathway"].dropna())

        unique1 = set_p1 - set_p2
        unique2 = set_p2 - set_p1

        print(f"  {LABEL1}: total {len(set_p1)} pathways, {len(unique1)} non-overlapping")
        print(f"  {LABEL2}: total {len(set_p2)} pathways, {len(unique2)} non-overlapping")

        gene_counts_1 = []
        gene_counts_2 = []

        missing_p1 = []
        missing_p2 = []

        # Experiment 1
        for pw in unique1:
            genes_in_mapping = pathway_to_genes.get(pw)
            if genes_in_mapping is None:
                missing_p1.append(pw)
                continue

            valid_genes = genes_in_mapping & sig_genes_1
            gene_counts_1.append(len(valid_genes))

        # Experiment 2
        for pw in unique2:
            genes_in_mapping = pathway_to_genes.get(pw)
            if genes_in_mapping is None:
                missing_p2.append(pw)
                continue

            valid_genes = genes_in_mapping & sig_genes_2
            gene_counts_2.append(len(valid_genes))

        if missing_p1:
            print(
                f"  [WARN] {LABEL1}: {len(missing_p1)} unique pathways not found in Reactome mapping."
            )
        if missing_p2:
            print(
                f"  [WARN] {LABEL2}: {len(missing_p2)} unique pathways not found in Reactome mapping."
            )

        # === NEW: per-disease histogram =====================================
        combined_disease = gene_counts_1 + gene_counts_2
        if combined_disease and max(combined_disease) > 0:
            max_count_d = max(combined_disease)
            bins_d = np.arange(-0.5, max_count_d + 1.5, 1)

            plt.figure(figsize=(8, 6))
            plt.hist(
                gene_counts_1,
                bins=bins_d,
                alpha=0.5,
                label=LABEL1,
            )
            plt.hist(
                gene_counts_2,
                bins=bins_d,
                alpha=0.5,
                label=LABEL2,
            )
            plt.xlabel("Number of significant genes per non-overlapping significant pathway")
            plt.ylabel("Number of pathways")
            plt.title(
                "Significant gene counts per non-overlapping significant pathway\n"
                f"Disease: {disease}"
            )
            plt.legend()
            plt.tight_layout()

            out_png_d = os.path.join(
                HIST_OUT_DIR,
                f"valid_gene_coverage_unique_pathways_histogram_{disease}.png",
            )
            plt.savefig(out_png_d, dpi=300)
            plt.close()
            print(f"  [INFO] Saved per-disease histogram: {out_png_d}")
        else:
            print(f"  [INFO] No valid genes mapped to non-overlapping pathways in {disease}; "
                  "skipping per-disease histogram.")
        # ====================================================================

        # Append to global lists for global histogram
        all_gene_counts_1.extend(gene_counts_1)
        all_gene_counts_2.extend(gene_counts_2)

        # Per-disease summary: sum of counts (allowing duplicates across pathways)
        total_genes_sum_1 = int(np.sum(gene_counts_1)) if gene_counts_1 else 0
        total_genes_sum_2 = int(np.sum(gene_counts_2)) if gene_counts_2 else 0

        # Per-disease summary: union of genes across all unique pathways
        unique_valid_genes_1 = set()
        for pw in unique1:
            genes_in_mapping = pathway_to_genes.get(pw)
            if genes_in_mapping is not None:
                unique_valid_genes_1.update(genes_in_mapping & sig_genes_1)

        unique_valid_genes_2 = set()
        for pw in unique2:
            genes_in_mapping = pathway_to_genes.get(pw)
            if genes_in_mapping is not None:
                unique_valid_genes_2.update(genes_in_mapping & sig_genes_2)

        total_genes_union_1 = len(unique_valid_genes_1)
        total_genes_union_2 = len(unique_valid_genes_2)
        global_unique_pathways_1.update(unique1)
        global_unique_pathways_2.update(unique2)
        global_unique_valid_genes_1.update(unique_valid_genes_1)
        global_unique_valid_genes_2.update(unique_valid_genes_2)
        
        summary_rows.append({
            "disease": disease,
            f"{LABEL1}_n_unique_pathways": len(unique1),
            f"{LABEL2}_n_unique_pathways": len(unique2),
            f"{LABEL1}_sum_valid_genes_in_unique_pathways": total_genes_sum_1,
            f"{LABEL2}_sum_valid_genes_in_unique_pathways": total_genes_sum_2,
            f"{LABEL1}_n_unique_valid_genes_union": total_genes_union_1,
            f"{LABEL2}_n_unique_valid_genes_union": total_genes_union_2,
        })

    # 4) Save summary CSV
    summary_df = pd.DataFrame(summary_rows).sort_values("disease")
    out_summary_csv = os.path.join(
        HIST_OUT_DIR, "unique_pathways_valid_gene_coverage_summary.csv"
    )
    summary_df.to_csv(out_summary_csv, index=False)
    print(f"\n[INFO] Saved summary CSV: {out_summary_csv}")
    print(summary_df)

    # 5) Global histogram: distribution of "significant gene count per non-overlapping pathway"
    if all_gene_counts_1 or all_gene_counts_2:
        combined = all_gene_counts_1 + all_gene_counts_2
        if not combined or max(combined) == 0:
            print("[WARN] All valid gene counts are zero; skip global histogram plotting.")
        else:
            max_count = max(combined)

            # Use integer-aligned bins: each bar corresponds to an integer gene count
            bins = np.arange(-0.5, max_count + 1.5, 1)

            plt.figure(figsize=(8, 6))

            plt.hist(
                all_gene_counts_1,
                bins=bins,
                alpha=0.5,
                label=LABEL1,
            )
            plt.hist(
                all_gene_counts_2,
                bins=bins,
                alpha=0.5,
                label=LABEL2,
            )

            plt.xlabel("Number of significant genes per non-overlapping significant pathway")
            plt.ylabel("Number of pathways")
            plt.title(
                "Significant gene counts per non-overlapping significant pathway\n"
                f"{LABEL1} vs {LABEL2} (all diseases)"
            )
            plt.legend()
            plt.tight_layout()

            out_png = os.path.join(
                HIST_OUT_DIR, "valid_gene_coverage_unique_pathways_histogram_all_diseases.png"
            )
            plt.savefig(out_png, dpi=300)
            plt.close()
            print(f"[INFO] Saved global histogram: {out_png}")
    else:
        print("[INFO] No valid genes mapped to non-overlapping pathways; nothing to plot.")


if __name__ == "__main__":
    main()
