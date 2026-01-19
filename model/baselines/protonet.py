import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch.optim as optim
from torch.autograd import Variable
from collections import defaultdict
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
import argparse

device = None


def get_project_root() -> Path:
    # This file is located at:
    # Pathway_MAML/model/baselines/<this_file>.py
    # parents[0]=baselines, parents[1]=model, parents[2]=Pathway_MAML (project root)
    return Path(__file__).resolve().parents[2]


def parse_args():
    root = get_project_root()
    p = argparse.ArgumentParser()

    # Use project-relative paths by default; allow override via CLI args
    p.add_argument("--data_dir", type=Path, default=root / "data")
    p.add_argument("--results_dir", type=Path, default=root / "results")

    # Data subdirectories (relative to data_dir)
    p.add_argument("--tcga_subdir", type=Path, default=Path("TCGA/5_TCGA_NCBI"))
    p.add_argument("--ncbi_subdir", type=Path, default=Path("NCBI"))

    # Diseases (comma-separated)
    p.add_argument(
        "--diseases",
        type=str,
        default="idiopathic_pulmonary_fibrosis,HBV-HCC,cirrhosis,ipf_ssc,IgA_nephropathy"
    )

    # Model / training params
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pathway_dim", type=int, default=383)
    p.add_argument("--hidden_dim", type=int, default=None)  # If None, hidden_dim = pathway_dim // 4
    p.add_argument("--K", type=int, default=1)
    p.add_argument("--tasks_per_meta_batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--weight_decay", type=float, default=3e-4)
    p.add_argument("--num_iters", type=int, default=15000)
    p.add_argument("--metatest_tasks", type=int, default=50)

    # Checkpoint path (best ProtoNet weights)
    p.add_argument(
        "--ckpt_path",
        type=Path,
        default=root / "results" / "weights" / "protonet_best.pth"
    )

    return p.parse_args()

class Column_match:
    @staticmethod
    def Column_match(training_file, target_file):

        target_df = pd.read_csv(target_file, sep='\t') 
        target_ID = set(target_df.columns[:-1])

        training_df = pd.read_csv(training_file)
        matching_columns = set(training_df.columns).intersection(target_ID)
        matching_columns = sorted(list(matching_columns))
        training_df = training_df[matching_columns]

        gene_to_index = training_df.columns.to_list()

        return gene_to_index
    
def linear_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr=1e-4, warmup_lr_init=0.0):
    """
    Linear warm-up followed by cosine annealing.
    
    Args:
        optimizer: Optimizer to which the scheduler is applied.
        warmup_steps (int): Number of steps for linear warm-up.
        total_steps (int): Total number of steps (warm-up + decay).
        min_lr (float): Minimum learning rate for cosine annealing.
        warmup_lr_init (float): Initial learning rate during warm-up.

    Returns:
        LambdaLR: Scheduler object.
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warm-up phase
            return (warmup_lr_init +
                    (1.0 - warmup_lr_init) * (current_step / max(1, warmup_steps)))
        else:
            # Cosine annealing phase
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)

class GeneExpressionDataset(Dataset):
    def __init__(self, data_folder, training_files, target_files, gene_to_index):
        """
        Initialize the GeneExpressionDataset:
          1. Load training files for meta-training.
          2. Load target files for meta-testing.
          3. Use gene_to_index for mapping gene IDs to indices.
        """
        self.data_folder = data_folder
        self.training_files = training_files  # Each file is treated as an individual task.
        self.meta_train_dataset = {}
        self.meta_test_dataset = {}
        self.gene_to_index = gene_to_index
        self.gene_order = sorted(gene_to_index.keys(), key=lambda g: gene_to_index[g])

        # Preload data for meta-training
        for train_file in training_files:
            file_path = os.path.join(data_folder, train_file)
            data = pd.read_csv(file_path)
            
            # Filter columns based on gene_to_index
            data = data[self.gene_order + ['Label']]

            train_features = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float)
            train_labels = torch.tensor(data.iloc[:, -1].values, dtype=torch.float)
            
            print(f"[Meta-train] {train_file} → features: {train_features.shape}, labels: {train_labels.shape}")

            # Store in the dictionary: key=file path, value=features and labels
            self.meta_train_dataset[train_file] = {
                'all_features': train_features,
                'all_labels': train_labels
            }

        # Preload and process data for meta-testing
        for target_file in target_files:
            # Load and process target files
            meta_test_data = pd.read_csv(target_file, sep='\t') 
            meta_test_data = meta_test_data[self.gene_order + ['Label']]

            test_features = torch.tensor(meta_test_data.iloc[:, :-1].values, dtype=torch.float)
            test_labels = torch.tensor(meta_test_data.iloc[:, -1].values, dtype=torch.float)
            
            print(f"[Meta-test ] {os.path.basename(target_file)} → features: {test_features.shape}, labels: {test_labels.shape}")

            # Store in the dictionary: key=file path, value=features and labels
            self.meta_test_dataset[target_file] = {
                'all_features': test_features,
                'all_labels': test_labels
            }
            
    def create_train_tasks(self, tasks_per_meta_batch, K):
        """
        Create multiple meta-training tasks:
        1. Select up to 'tasks_per_meta_batch' files from training_files.
        2. For each selected file:
            a. Randomly sample K examples per label (0 and 1) for the support set.
            b. Randomly sample an equal number of examples from the remaining data to form the query set, maintaining a 1:1 label ratio.
        3. Return a list of tasks.
        """
        # Filter out any undesired files (e.g., containing 'breast')
        selected_files = [f for f in self.training_files]
        
        # Limit the number of tasks to create
        max_tasks = min(tasks_per_meta_batch, len(selected_files))
        selected_files = selected_files[:max_tasks]

        tasks = []
        for file_name in selected_files:
            # Skip if data is not found (safety check)
            if file_name not in self.meta_train_dataset:
                continue
            
            # Retrieve pre-loaded features/labels (now assume they're Torch tensors)
            all_features = self.meta_train_dataset[file_name]['all_features']  # shape: (N, num_features)
            all_labels   = self.meta_train_dataset[file_name]['all_labels']    # shape: (N,)
            
            # Split indices by label
            label_0_idx = (all_labels == 0).nonzero(as_tuple=True)[0]
            label_1_idx = (all_labels == 1).nonzero(as_tuple=True)[0]
            
            # Check if we have enough examples for K-shot
            if label_0_idx.size(0) < K or label_1_idx.size(0) < K:
                continue
            
            # Randomly select K examples for support set (label 0 and label 1)
            support_idx_0 = label_0_idx[torch.randperm(len(label_0_idx))[:K]]
            support_idx_1 = label_1_idx[torch.randperm(len(label_1_idx))[:K]]
            
            support_indices = torch.cat([support_idx_0, support_idx_1], dim=0) # concatenate
            support_indices = support_indices[torch.randperm(len(support_indices))] # shuffle
            
            x_support = all_features[support_indices]
            y_support = all_labels[support_indices].unsqueeze(-1)
            
            # Query set: remaining indices
            remain_idx_0 = label_0_idx[~label_0_idx.unsqueeze(1).eq(support_idx_0).any(1)]
            remain_idx_1 = label_1_idx[~label_1_idx.unsqueeze(1).eq(support_idx_1).any(1)]
            
            # Determine how many query samples to take (at most the smaller label leftover)
            num_query_samples = min(len(remain_idx_0), len(remain_idx_1))
            if num_query_samples == 0:
                continue
            
            # Randomly sample query indices
            query_idx_0 = remain_idx_0[torch.randperm(len(remain_idx_0))[:num_query_samples]]
            query_idx_1 = remain_idx_1[torch.randperm(len(remain_idx_1))[:num_query_samples]]
            
            query_indices = torch.cat([query_idx_0, query_idx_1], dim=0)
            query_indices = query_indices[torch.randperm(len(query_indices))]
            
            x_query = all_features[query_indices]
            y_query = all_labels[query_indices].unsqueeze(-1)

            # Move to GPU (if not already). If already on device, this is harmless.
            x_support = x_support.to(device)
            y_support = y_support.to(device)
            x_query   = x_query.to(device)
            y_query   = y_query.to(device)
            
            tasks.append(((x_support, y_support), (x_query, y_query)))
        
        return tasks


    def create_test_task(self, K, target_file):
        """
        Create a single meta-testing task using a specified target file.
        1. Randomly select K examples per label (0 and 1) to form the support set.
        2. Select up to 30 remaining examples for the query set, balancing the number of samples per label as much as possible.
        3. Return the task in PyTorch tensor format, moved to the appropriate device.
        """
        # Retrieve pre-loaded features and labels for the target file
        if target_file not in self.meta_test_dataset:
            raise ValueError(f"Target file {target_file} not found in meta_test_dataset.")

        all_features = self.meta_test_dataset[target_file]['all_features']
        all_labels = self.meta_test_dataset[target_file]['all_labels']

        # Split indices by label
        label_0_idx = (all_labels == 0).nonzero(as_tuple=True)[0]
        label_1_idx = (all_labels == 1).nonzero(as_tuple=True)[0]

        # Ensure there are enough samples for K-shot
        if label_0_idx.size(0) < K or label_1_idx.size(0) < K:
            raise ValueError("Not enough samples for K-shot testing.")

        # Randomly select K examples for support set (label 0 and label 1)
        support_idx_0 = label_0_idx[torch.randperm(len(label_0_idx))[:K]]
        support_idx_1 = label_1_idx[torch.randperm(len(label_1_idx))[:K]]

        support_indices = torch.cat([support_idx_0, support_idx_1], dim=0)
        support_indices = support_indices[torch.randperm(len(support_indices))]  # Shuffle

        x_support = all_features[support_indices]
        y_support = all_labels[support_indices].unsqueeze(-1)

        # Query set: remaining indices
        remain_idx_0 = label_0_idx[~label_0_idx.unsqueeze(1).eq(support_idx_0).any(1)]
        remain_idx_1 = label_1_idx[~label_1_idx.unsqueeze(1).eq(support_idx_1).any(1)]

        # Dynamically determine query sample sizes based on availability
        available_query_0 = remain_idx_0.size(0)
        available_query_1 = remain_idx_1.size(0)

        if (available_query_0 + available_query_1) >= 30:  # Balanced query set if total available samples >= 30
            num_query_samples_0 = min(15, available_query_0)
            num_query_samples_1 = min(30 - num_query_samples_0, available_query_1)
        else:  # Use all remaining samples if total < 30
            num_query_samples_0 = available_query_0
            num_query_samples_1 = available_query_1

        # Randomly sample query indices
        query_idx_0 = remain_idx_0[torch.randperm(len(remain_idx_0))[:num_query_samples_0]]
        query_idx_1 = remain_idx_1[torch.randperm(len(remain_idx_1))[:num_query_samples_1]]

        query_indices = torch.cat([query_idx_0, query_idx_1], dim=0)
        query_indices = query_indices[torch.randperm(len(query_indices))]  # Shuffle

        x_query = all_features[query_indices]
        y_query = all_labels[query_indices].unsqueeze(-1)

        # Move tensors to device
        x_support = x_support.to(device)
        y_support = y_support.to(device)
        x_query = x_query.to(device)
        y_query = y_query.to(device)

        return (x_support, y_support), (x_query, y_query)
    
# squared euclidean distance
def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

# MLPEncoder
class MLPEncoder(nn.Module):
    def __init__(self, input_dim, pathway_dim, hidden_dim):
        super().__init__()
        # input → pathway
        self.fc        = nn.Linear(input_dim, pathway_dim)
        self.ln_pathway = nn.LayerNorm(pathway_dim)
        # pathway → hidden
        self.fc2       = nn.Linear(pathway_dim, hidden_dim)
        self.ln_hidden = nn.LayerNorm(hidden_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        # 1) input → pathway
        x = self.fc(x)                    
        x = self.ln_pathway(x)
        x = self.activation(x)

        # 2) pathway → hidden
        x = self.fc2(x)
        x = self.ln_hidden(x)
        x = self.activation(x)

        return x

# Prototypical Network
class ProtoNet(nn.Module):
    """
    Few-shot episode 단위 forward / loss 계산
    """
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    # 내부 유틸 
    @staticmethod
    def _get_prototypes(z_s, y_s):
        """
        클래스별 support 임베딩 평균.
        z_s:[S,d], y_s:[S] → proto:[N_way,d]
        """
        classes = torch.unique(y_s)
        return torch.stack([z_s[y_s == c].mean(0) for c in classes])

    # forward 
    def forward(self, xs, ys, xq):
        z_s = self.encoder(xs)                # [S,d]
        z_q = self.encoder(xq)                # [Q,d]
        proto = self._get_prototypes(z_s, ys) # [N,d]
        dists = euclidean_dist(z_q, proto)    # [Q,N]
        return -dists                         # logits  (–distance)

    # loss 
    def loss(self, xs, ys, xq, yq):
        logits   = self.forward(xs, ys, xq)          # = -dists
        loss   = F.cross_entropy(logits, yq)         # cross_entropy = log_softmax+NLL
        return loss

class ProtoNetTrainer:
    def __init__(self, input_dim, pathway_dim, hidden_dim, dataset, K=1, tasks_per_meta_batch=4,
                 lr=1e-3, weight_decay=3e-4, device='cuda', ckpt_path: Path = None):
        self.dataset  = dataset
        self.K        = K
        self.tasks_per_meta_batch = tasks_per_meta_batch
        self.device   = device

        encoder = MLPEncoder(input_dim, pathway_dim, hidden_dim).to(device)
        self.model = ProtoNet(encoder).to(device)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Early‑Stopping
        self.best_loss        = float('inf')
        self.patience         = 100        
        self.patience_counter = 0

        self.ckpt_path = ckpt_path
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Warm-up + Cosine Annealing Scheduler
        warmup_steps = 10  # Adjust based on your setup
        total_steps = 15000  # Total number of iterations
        
        self.scheduler = linear_warmup_cosine_scheduler(
            self.opt,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=1e-4,          
            warmup_lr_init=1e-4
        )

    # Meta-training 
    def train(self, num_iters=15000, log_every=200):
        self.model.train()
        for it in range(1, num_iters + 1):
            tasks = self.dataset.create_train_tasks(self.tasks_per_meta_batch, self.K)         # list

            if len(tasks) == 0:
                continue
            
            loss_batch = 0.
            for (xs, ys), (xq, yq) in tasks:
                xs = xs.to(self.device)                               # [S, D]
                xq = xq.to(self.device)                               # [Q, D]
                ys = ys.squeeze(-1).long().to(self.device)            # [S]
                yq = yq.squeeze(-1).long().to(self.device)            # [Q]

                loss = self.model.loss(xs, ys, xq, yq)                # cross‑entropy
                loss_batch += loss

            meta_loss = loss_batch / len(tasks)
            
            self.opt.zero_grad()
            meta_loss.backward()
            self.opt.step()
            self.scheduler.step() # linear_warmup_cosine_scheduler

            # Early‑Stopping 
            if meta_loss < self.best_loss:
                self.best_loss = meta_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.ckpt_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early-stopped at iter {it:05d} | best loss {self.best_loss:.4e}")
                    print(f"len(tasks)= ", len(tasks))
                    break
            
            if it % log_every == 0:
                print(f"[{it:05d}] loss = {meta_loss.item():.4e}")

    # Meta-testing 
    @torch.no_grad()
    def evaluate(self, target_files, metatest_tasks=50): 
        self.model.eval()
        results = {}
        pr_all, auc_all = [], []

        for tgt in target_files:
            disease = os.path.basename(os.path.dirname(tgt))
            pr_list, auc_list = [], []

            for _ in range(metatest_tasks):
                try:
                    (xs, ys), (xq, yq) = self.dataset.create_test_task(self.K, tgt)
                except ValueError:
                    continue

                xs = xs.to(self.device)
                xq = xq.to(self.device)
                ys = ys.squeeze(-1).long().to(self.device)
                yq = yq.squeeze(-1).long().to(self.device)

                logits = self.model(xs, ys, xq)          # [Q, N_way]

                classes = torch.unique(ys)
                if 1 not in classes:                      # no positive class -> continue
                    continue
                
                # Find the positive class index
                pos_idx = (torch.unique(ys) == 1).nonzero(as_tuple=True)[0].item()
                probs  = logits.softmax(1)[:, pos_idx].cpu()
                y_true = yq.cpu()

                roc_val = roc_auc_score(y_true, probs)
                prec, rec, _ = precision_recall_curve(y_true, probs)
                pr_val  = auc(rec, prec)

                pr_list.append(pr_val);  auc_list.append(roc_val)

            if pr_list:
                results[disease] = (np.mean(pr_list), np.std(pr_list),
                                    np.mean(auc_list), np.std(auc_list))
                pr_all.extend(pr_list);  auc_all.extend(auc_list)

        # output
        for d, (mpr, spr, mau, sau) in results.items():
            print(f"{d:20s}  PR-AUC {mpr:.4f}±{spr:.4f} | AUC {mau:.4f}±{sau:.4f}")
        print("-"*55)
        print(f"Overall        PR-AUC {np.mean(pr_all):.4f}±{np.std(pr_all):.4f} | "
              f"AUC {np.mean(auc_all):.4f}±{np.std(auc_all):.4f}")
        
def main():
    args = parse_args()

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    diseases = [d.strip() for d in args.diseases.split(",") if d.strip()]

    data_dir = args.data_dir
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build paths from data_dir
    data_folder = data_dir / args.tcga_subdir
    ncbi_dir = data_dir / args.ncbi_subdir

    target_files = [
        str(ncbi_dir / disease / "second_filtered_combined_counts_transposed.tsv")
        for disease in diseases
    ]

    # Collect training CSVs
    training_files = sorted([p.name for p in data_folder.iterdir() if p.is_file() and p.suffix == ".csv"])
    if len(training_files) == 0:
        raise RuntimeError(f"No training CSV files found in: {data_folder}")

    gene_list = Column_match.Column_match(
        str(data_folder / training_files[0]),
        target_files[0]
    )
    gene_to_index = {gene: idx for idx, gene in enumerate(gene_list)}

    dataset = GeneExpressionDataset(
        data_folder=str(data_folder),
        training_files=training_files,
        target_files=target_files,
        gene_to_index=gene_to_index
    )

    input_dim = len(gene_to_index)
    pathway_dim = args.pathway_dim
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else pathway_dim // 4

    print("input_dim =", input_dim)
    print("pathway_dim =", pathway_dim)
    print("hidden_dim =", hidden_dim)

    # Checkpoint path under results by default
    ckpt_path = args.ckpt_path

    proto_trainer = ProtoNetTrainer(
        input_dim=input_dim,
        pathway_dim=pathway_dim,
        hidden_dim=hidden_dim,
        dataset=dataset,
        K=args.K,
        tasks_per_meta_batch=args.tasks_per_meta_batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        ckpt_path=ckpt_path
    )

    proto_trainer.train(num_iters=args.num_iters)

    # Load best model before evaluation
    proto_trainer.model.load_state_dict(torch.load(str(proto_trainer.ckpt_path), map_location=device))
    proto_trainer.evaluate(target_files, metatest_tasks=args.metatest_tasks)

    print(input_dim)
    print(pathway_dim)
    print(hidden_dim)


if __name__ == "__main__":
    main()