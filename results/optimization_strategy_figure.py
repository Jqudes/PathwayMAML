import sys
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

# ====================== PATH SETUP (RELATIVE) ======================

# 1. Determine Project Root based on file location
# This ensures the script works regardless of where you run the command from.
FILE_PATH = Path(__file__).resolve()

# Assuming this script is in: .../Pathway_MAML/results/figures/plot_figure1.py
# Move up until we find the project root (e.g., look for 'data' folder)
PROJECT_ROOT = FILE_PATH.parent

# Loop to find the root directory containing 'data'
while not (PROJECT_ROOT / 'data').exists():
    if PROJECT_ROOT == PROJECT_ROOT.parent: # Reached system root
        raise FileNotFoundError("Could not find project root containing 'data' folder.")
    PROJECT_ROOT = PROJECT_ROOT.parent

# 2. Define Directories
RESULTS_DIR = PROJECT_ROOT / 'results'
LR_SWEEP_DIR = RESULTS_DIR / 'lr_sweep'
FIGURE_OUTPUT_DIR = RESULTS_DIR / 'figures'

# Create output directory if not exists
FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[INFO] Project Root: {PROJECT_ROOT}")
print(f"[INFO] Loading data from: {LR_SWEEP_DIR}")
print(f"[INFO] Saving figures to: {FIGURE_OUTPUT_DIR}")

# ---------------------------------------------------------
# 1. Data Loading & Preprocessing
# ---------------------------------------------------------
file_5k = LR_SWEEP_DIR / 'lr_sweep_results_iter5000_vast.tsv'
file_15k = LR_SWEEP_DIR / 'lr_sweep_results_iter15000_vast.tsv'

if not file_5k.exists() or not file_15k.exists():
    raise FileNotFoundError(f"Data files not found in {LR_SWEEP_DIR}. Check paths.")

df_5k_vast = pd.read_csv(file_5k, sep='\t')
df_15k_vast = pd.read_csv(file_15k, sep='\t')

# Merge datasets to analyze the full spectrum of iterations and parameters
df_vast_all = pd.concat([df_15k_vast, df_5k_vast], ignore_index=True)

# Prepare Data for Panel A (Inner LR Sensitivity)
# Objective: Find the maximum capability (ROC-AUC) at each Inner LR.
# We aggregate by taking the 'max' to filter out suboptimal meta-learning rates or iterations.
df_inner_curve = df_vast_all.groupby(['K', 'inner_lr']).agg({
    'overall_avg_auc': 'max',
    'overall_avg_pr_auc': 'max'
}).reset_index()
# ---------------------------------------------------------
# 2. Figure Setup
# ---------------------------------------------------------
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13})

fig = plt.figure(figsize=(18, 6)) # (18,8)

gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

# Define a consistent color palette: K=1 (Red), K=3 (Blue), K=5 (Orange)
palette = {1: '#D62728', 3: '#1F77B4', 5: '#FF7F0E'}

# ---------------------------------------------------------
# 3. Panel A: Adaptation Strategy (Inner LR vs. ROC-AUC)
# ---------------------------------------------------------
ax1 = fig.add_subplot(gs[0])

for k in [1, 3, 5]:
    subset = df_inner_curve[df_inner_curve['K'] == k]
    
    # Plot Line with Markers
    sns.lineplot(ax=ax1, data=subset, x='inner_lr', y='overall_avg_auc', 
                 color=palette[k], linewidth=2.5, marker='o', markersize=9, label=f'k={k}')
    
    # Annotate the Peak (Sweet Spot)
    if not subset.empty:
        peak_idx = subset['overall_avg_auc'].idxmax()
        peak_row = subset.loc[peak_idx]
        ax1.annotate(f"{peak_row['inner_lr']}", 
                     (peak_row['inner_lr'], peak_row['overall_avg_auc']),
                     xytext=(0, -18), textcoords='offset points', ha='center',
                     fontsize=10, fontweight='bold', color=palette[k])

# Styling Panel A
ax1.set_xscale('log')
ax1.set_title('(A) Shot-Dependent Shift in Optimal Inner Learning Rates', fontweight='bold', loc='left')
ax1.set_xlabel('Inner Loop Learning Rate')
ax1.set_ylabel('ROC-AUC')
ax1.grid(True, which='both', linestyle='--', alpha=0.5)
ax1.legend(title='Support Set Size (k)', loc='center right')

# ---------------------------------------------------------
# 4. Panel B: Generalization Strategy (Iteration vs. PR-AUC)
# ---------------------------------------------------------
ax2 = fig.add_subplot(gs[1])

# Scatter plot
sns.scatterplot(ax=ax2, data=df_vast_all, x='meta_train_stop_iter', y='overall_avg_pr_auc', 
                hue='K', palette=palette, alpha=0.4, s=50, legend=False)

# Regression lines
for k in [1, 3, 5]:
    subset = df_vast_all[df_vast_all['K'] == k]
    if not subset.empty and len(subset) > 1:
        corr = subset['meta_train_stop_iter'].corr(subset['overall_avg_pr_auc'])
        label_text = f'k={k} Trend (ρ={corr:.2f})'
        
        sns.regplot(ax=ax2, data=subset, x='meta_train_stop_iter', y='overall_avg_pr_auc', 
                    scatter=False, color=palette[k], ci=None, 
                    line_kws={'linewidth': 3, 'linestyle': '-' if k==1 else '--'}, 
                    label=label_text)

# Styling Panel B
ax2.set_title('(B) Opposing Impact of Training Duration', fontweight='bold', loc='left')
ax2.set_xlabel('Meta-Training Iteration')
ax2.set_ylabel('PR-AUC')
ax2.legend(loc='center right', frameon=True)
ax2.grid(True, linestyle='--', alpha=0.5)

# ---------------------------------------------------------
# 6. Save and Show
# ---------------------------------------------------------
plt.tight_layout(pad=2.0)

output_path = FIGURE_OUTPUT_DIR / 'figure1_dual_optimization_strategy.png'
plt.savefig(output_path, dpi=600)
plt.show()

print("Figure 1 (Resized) generated successfully.")