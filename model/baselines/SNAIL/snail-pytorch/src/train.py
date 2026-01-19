# coding=utf-8
import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from snail import SnailFewShot
from gene_expression_dataset import GeneExpressionDataset, Column_match


def get_project_root() -> Path:
    """
    This file is at:
      Pathway_MAML/model/baselines/SNAIL/snail-pytorch/src/train.py

    parents:
      [0]=src
      [1]=snail-pytorch
      [2]=SNAIL
      [3]=baselines
      [4]=model
      [5]=Pathway_MAML (project root)
    """
    return Path(__file__).resolve().parents[5]


def parse_args():
    root = get_project_root()
    p = argparse.ArgumentParser()

    # Base dirs (project-relative defaults; override via CLI)
    p.add_argument("--data_dir", type=Path, default=root / "data")
    p.add_argument("--results_dir", type=Path, default=root / "results")

    # Data subdirs under data_dir
    p.add_argument("--tcga_subdir", type=Path, default=Path("TCGA/5_TCGA_NCBI"))
    p.add_argument("--ncbi_subdir", type=Path, default=Path("NCBI"))

    # Diseases (comma-separated)
    p.add_argument(
        "--diseases",
        type=str,
        default="idiopathic_pulmonary_fibrosis,HBV-HCC,cirrhosis,ipf_ssc,IgA_nephropathy"
    )

    # Output directory for checkpoints (inside results by default)
    p.add_argument("--exp_dir", type=Path, default=(root / "results" / "snail"))

    # Training params
    p.add_argument("--tasks_per_meta_batch", type=int, default=4)
    p.add_argument("--K", type=int, default=1)                 # K-shot per class (binary -> 2K support total)
    p.add_argument("--epochs", type=int, default=15000)         # Number of epochs/iterations (same as your original)
    p.add_argument("--lr", type=float, default=1.5e-5)
    p.add_argument("--weight_decay", type=float, default=3e-4)
    p.add_argument("--metatest_tasks", type=int, default=50)

    # Device / seed
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cuda", action="store_true")

    return p.parse_args()


def init_model(opt):
    opt.num_classes = 2
    feature_dim = len(opt.gene_to_index)
    model = SnailFewShot(N=opt.num_classes, K=opt.K, feature_dim=feature_dim, use_cuda=opt.cuda)
    model = model.cuda() if opt.cuda else model
    return model.to(opt.device)


def build_snail_sequence(opt, x_sup, y_sup, x_q, y_q):

    x_sup = x_sup.to(opt.device)                                # (2K, F)
    y_sup = y_sup.squeeze(-1).long().to(opt.device)             # (2K,)
    x_q = x_q.to(opt.device)                                    # (1, F)
    y_q = y_q.squeeze(-1).long().to(opt.device)                 # (1,)

    x_seq = torch.cat([x_sup, x_q], dim=0)

    y_all = torch.cat([y_sup, y_q], dim=0)  # (2K+1,)
    y_oh = F.one_hot(y_all, num_classes=opt.num_classes).float().to(opt.device)

    return x_seq, y_oh, y_q


def train_rna_seq(opt, dataset, model, optimizer):
    loss_fn = nn.CrossEntropyLoss()
    best_loss = float("inf")
    patience = 100
    patience_counter = 0
    best_state = None

    best_path = opt.exp / "best_model.pth"
    last_path = opt.exp / "last_model_rna.pth"

    for epoch in range(opt.epochs):
        model.train()
        epoch_loss = 0.0

        # Sample meta-train tasks every epoch (your original behavior)
        tr_tasks = dataset.create_train_tasks(opt.tasks_per_meta_batch, opt.K)
        if len(tr_tasks) == 0:
            continue

        for ((x_sup, y_sup), (x_q_all, y_q_all)) in tqdm(tr_tasks, desc=f"Epoch {epoch+1}/{opt.epochs}"):
            x_q_all = x_q_all.to(opt.device)                     # (Q, F)
            y_q_all = y_q_all.squeeze(-1).long().to(opt.device)  # (Q,)

            for q_idx in range(x_q_all.size(0)):
                x_q = x_q_all[q_idx:q_idx+1]                    # (1, F)
                y_q = y_q_all[q_idx:q_idx+1].unsqueeze(-1)      # (1, 1) to match build function expectations

            x_seq, y_oh, y_q_true = build_snail_sequence(opt, x_sup, y_sup, x_q, y_q)

            # Forward: SNAIL outputs a sequence; we use the last step logits
            out_seq = model(x_seq, y_oh)                    # expected shape: (1, 2K+1, N)
            last_logits = out_seq[:, -1, :]                 # (1, N)

            loss = loss_fn(last_logits, y_q_true)           # y_q_true is (1,)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(tr_tasks))
        print(f"[Epoch {epoch+1}/{opt.epochs}] Avg Loss: {avg_loss:.6f}")

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), str(best_path))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (best_loss={best_loss:.6f})")
                break

    # Save final model
    torch.save(model.state_dict(), str(last_path))
    return best_state, best_loss


@torch.no_grad()
def evaluate_rna_seq(opt, target_files, dataset, model, metatest_tasks=50):
    model.eval()
    print("\n=== SNAIL ROC-AUC / PR-AUC Evaluation ===")

    results = {}
    pr_all, auc_all = [], []

    for tgt in target_files:
        disease = os.path.basename(os.path.dirname(tgt))
        pr_list, auc_list = [], []

        for _ in range(metatest_tasks):
            try:
                (x_sup, y_sup), (x_q_all, y_q_all) = dataset.create_test_task(opt.K, tgt)
            except ValueError:
                continue

            # Collect predictions for ALL queries in this episode
            y_true_ep, y_prob_ep = [], []

            for q_idx in range(x_q_all.size(0)):
                x_q = x_q_all[q_idx:q_idx+1]
                y_q = y_q_all[q_idx:q_idx+1]

                x_seq, y_oh, y_q_true = build_snail_sequence(opt, x_sup, y_sup, x_q, y_q)

                logits = model(x_seq, y_oh)[:, -1, :].squeeze(0)      # (N,)
                prob_pos = logits.softmax(0)[1].item()                # P(class=1)

                y_true_ep.append(int(y_q_true.item()))
                y_prob_ep.append(prob_pos)

            # AUC requires both classes present
            if len(set(y_true_ep)) < 2:
                continue

            roc = roc_auc_score(y_true_ep, y_prob_ep)
            prec, rec, _ = precision_recall_curve(y_true_ep, y_prob_ep)
            pr = auc(rec, prec)

            auc_list.append(roc)
            pr_list.append(pr)

        if not pr_list:
            continue

        m_pr, s_pr = float(np.mean(pr_list)), float(np.std(pr_list))
        m_auc, s_auc = float(np.mean(auc_list)), float(np.std(auc_list))
        results[disease] = (m_pr, s_pr, m_auc, s_auc)

        pr_all.extend(pr_list)
        auc_all.extend(auc_list)

        print(f"{disease:20s} PR-AUC {m_pr:.4f}±{s_pr:.4f} | ROC-AUC {m_auc:.4f}±{s_auc:.4f}")

    if pr_all:
        print("-" * 55)
        print(f"Overall            PR-AUC {np.mean(pr_all):.4f}±{np.std(pr_all):.4f} | "
              f"ROC-AUC {np.mean(auc_all):.4f}±{np.std(auc_all):.4f}")


def main():
    opt = parse_args()

    # Device
    opt.device = torch.device("cuda") if opt.cuda and torch.cuda.is_available() else torch.device("cpu")

    # Seed
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # Directories
    opt.exp = opt.exp_dir
    opt.exp.mkdir(parents=True, exist_ok=True)

    tcga_dir = opt.data_dir / opt.tcga_subdir
    ncbi_dir = opt.data_dir / opt.ncbi_subdir

    # Build disease list and target files
    diseases = [d.strip() for d in opt.diseases.split(",") if d.strip()]
    target_files = [ncbi_dir / d / "second_filtered_combined_counts_transposed.tsv" for d in diseases]

    # Validate target files exist
    for tf in target_files:
        if not tf.exists():
            raise FileNotFoundError(f"Target file not found: {tf}")

    # Training files (TCGA CSV)
    training_files = sorted([p.name for p in tcga_dir.iterdir() if p.is_file() and p.suffix == ".csv"])
    if len(training_files) == 0:
        raise RuntimeError(f"No training CSV files found in: {tcga_dir}")

    # Build gene_to_index using first train file and first target file
    first_train = str(tcga_dir / training_files[0])
    first_test = str(target_files[0])
    gene_list = Column_match.Column_match(first_train, first_test)
    gene_to_index = {gene: idx for idx, gene in enumerate(gene_list)}
    opt.gene_to_index = gene_to_index

    # Dataset
    dataset = GeneExpressionDataset(
        data_folder=str(tcga_dir),
        training_files=training_files,
        target_files=[str(p) for p in target_files],
        gene_to_index=opt.gene_to_index
    )

    # Model + optimizer
    model = init_model(opt)
    optimizer = AdamW(params=model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    # Train
    train_rna_seq(opt, dataset, model, optimizer)

    # Load best model and evaluate
    best_path = opt.exp / "best_model.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(str(best_path), map_location=opt.device))
    evaluate_rna_seq(opt, [str(p) for p in target_files], dataset, model, metatest_tasks=opt.metatest_tasks)


if __name__ == "__main__":
    main()
