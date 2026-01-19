import os
import csv
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch.optim.lr_scheduler import LambdaLR, StepLR
from collections import defaultdict
from tqdm import tqdm
import torch.nn.functional as F
from captum.attr import LayerIntegratedGradients, IntegratedGradients
import itertools
import shutil
from datetime import datetime
from pathlib import Path

# ==========================================
# PATH SETUP (Relative to Project Root)
# ==========================================

FILE_PATH = Path(__file__).resolve()

PROJECT_ROOT = FILE_PATH.parent.parent

# Define Base Directories
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'

# Define Specific Subdirectories (Constants)
WEIGHTS_DIR = DATA_DIR / 'weights'
NCBI_DIR = DATA_DIR / 'NCBI'
TCGA_DIR = DATA_DIR / 'TCGA'
REACTOME_DIR = DATA_DIR / 'Reactome'

# Results Subdirectories
LIG_RAW_DIR = RESULTS_DIR / 'LIG_raw'
IG_RAW_DIR = RESULTS_DIR / 'IG_raw'
BIOMARKER_DIR = RESULTS_DIR / 'biomarker'
LR_SWEEP_DIR = RESULTS_DIR / 'lr_sweep'
PATHWAY_SCORES_DIR = RESULTS_DIR / 'pathway_scores_per_disease_diff'

# Ensure critical directories exist
for d in [WEIGHTS_DIR, LIG_RAW_DIR, IG_RAW_DIR, BIOMARKER_DIR, LR_SWEEP_DIR, PATHWAY_SCORES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"[INFO] Project Root: {PROJECT_ROOT}")
print(f"[INFO] Data Directory: {DATA_DIR}")
print(f"[INFO] Results Directory: {RESULTS_DIR}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")  

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

def LIG(model, sample_input, fast_weights, pathway_to_index, n_steps=100, topk=30, return_full=False, use_logit=True):
    """
    Computes the contribution of each pathway neuron in the gene_to_pathway module using Layer Integrated Gradients.
    
    For each target neuron (by index), this function calculates integrated gradients, sums the attributions over
    the input(pathway layer in this case) feature dimension to obtain a scalar contribution, and then returns the top 5 pathways.
    
    Args:
        model (HierarchicalMAMLModel): The model instance.
        sample_input (Tensor): A sample input tensor of shape (1, input_dim).
        fast_weights (list[Tensor]): Adapted parameters from the inner-loop updates.
        pathway_to_index (dict): Mapping from pathway name to its index.
        n_steps (int): Number of steps for integration.
        
    Returns:
        top_pathways (list of tuple): List of (pathway_name, attribution_value) tuples for the top 5 pathways.
        delta: (None, not used)
    """

    pathway_block = nn.Sequential(
        model.gene_to_pathway,
        model.ln_pathway,
        model.activation
    )

    def forward_func(x):
        orig_weight = model.gene_to_pathway.weight.data.clone()
        model.gene_to_pathway.weight.data = fast_weights[0] * model.gene_to_pathway_mask
        out = pathway_block(x)                       # (batch, pathway_dim)

        # manually apply the rest of the model's layers
        h = F.linear(out, fast_weights[1], fast_weights[2])
        h = model.ln_hidden(h)
        h = model.activation(h)
        logits = F.linear(h, fast_weights[3], fast_weights[4]).squeeze(-1)
        prob = torch.sigmoid(logits)
        
        model.gene_to_pathway.weight.data = orig_weight  # restore the original weight
        return logits if use_logit else prob

    # BASELINE
    # create a baseline tensor of zeros with the same shape as the sample_input.
    baseline = torch.zeros_like(sample_input)
    lig = LayerIntegratedGradients(forward_func, pathway_block)
    

    attr = lig.attribute(inputs=sample_input,
                         baselines=baseline,
                         n_steps=n_steps)                 # shape: (batch, pathway_dim)

    if return_full:
        # return per-sample attributions (no batch-mean)
        return attr.detach().cpu().numpy()  # (batch, pathway_dim)
    else:
        neuron_attr = attr.mean(dim=0).cpu().numpy()  # (pathway_dim,)
        top_idx = np.argsort(-neuron_attr)[:topk]
        idx2path = {idx: name for name, idx in pathway_to_index.items()}
        top_pathways = [(idx2path[i], neuron_attr[i]) for i in top_idx]
        return top_pathways, None

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

class GenePathwayMapping:
    @staticmethod
    def load_gene_pathway_mapping(gene_pathway_file, target_file):
        """
        Load and process gene-to-pathway mapping and filter data based on target IDs.
        1. Filters rows in the gene-pathway mapping file based on valid IDs from the target file.
        2. Maps gene and pathway names/IDs to unique indices.
        3. Constructs a mapping from gene indices to a list of associated pathway indices.

        Args:
            gene_pathway_file (str): Path to the CSV file containing gene-to-pathway mapping.
            target_file (str): Path to the target file (e.g., disease-specific data). The valid
                              IDs are determined from the columns excluding the last one.

        Returns:
            tuple: A tuple containing the following:
                - gene_pathway_mapping (defaultdict): Mapping of gene indices to pathway indices.
                - gene_to_index (dict): Mapping of gene names/IDs to unique indices.
                - pathway_to_index (dict): Mapping of pathway names/IDs to unique indices.
                - target_ID (set): Set of valid IDs extracted from the target file.
        """
        target_df = pd.read_csv(target_file, sep='\t') 
        target_ID = set(target_df.columns[:-1])

        mapping_df = pd.read_csv(gene_pathway_file)
        mapping_df = mapping_df[mapping_df['ID'].isin(target_ID)]
    
        # Map unique gene and pathway IDs to indices
        gene_to_index = {gene: idx for idx, gene in enumerate(mapping_df['ID'].unique())} 
        pathway_to_index = {pathway: idx for idx, pathway in enumerate(mapping_df['Pathway'].unique())}

        # Iterate through each row in the mapping file to populate the mapping
        # Example row format: {'ID': 'GeneA', 'Pathway': 'Pathway1'}
        gene_pathway_mapping = defaultdict(list)
        for _, row in mapping_df.iterrows():
            gene_id = gene_to_index[row['ID']]
            pathway_id = pathway_to_index[row['Pathway']]
            gene_pathway_mapping[gene_id].append(pathway_id) # Map gene index to pathway indices

        return gene_pathway_mapping, gene_to_index, pathway_to_index, target_ID

class GeneExpressionDataset(Dataset):
    def __init__(self, data_folder, training_files, target_files, gene_to_index):
        """
        Initialize the GeneExpressionDataset:
          1. Load training files for meta-training.
          2. Load target files for meta-testing.
          3. Use gene_to_index for mapping gene IDs to indices.

        Args:
            data_folder (str): Path to the folder containing training files.
            training_files (list[str]): List of files for meta-training (outer loop).
            target_files (list[str]): List of files for meta-testing.
            gene_to_index (dict): Mapping of gene names/IDs to unique indices.
        """
        self.data_folder = Path(data_folder)
        self.training_files = training_files  # Each file is treated as an individual task.
        self.meta_train_dataset = {}
        self.meta_test_dataset = {}
        self.gene_to_index = gene_to_index

        # Build gene order ONCE to match gene_to_index indices (mask order)
        self.gene_order = sorted(self.gene_to_index.keys(), key=lambda g: self.gene_to_index[g])

        # Preload data for meta-training
        for train_file in training_files:
            file_path = self.data_folder / train_file
            data = pd.read_csv(file_path)

            if "Label" not in data.columns:
                raise ValueError(f"'Label' column not found in {file_path}")

            # Fail fast if any required gene is missing (recommended: do not silently misalign)
            missing = [g for g in self.gene_order if g not in data.columns]
            if missing:
                raise ValueError(
                    f"[META-TRAIN] Missing {len(missing)} genes in {file_path}. First few: {missing[:5]}"
                )

            # Reorder columns to align with gene_to_index (mask) order
            data = data[self.gene_order + ["Label"]]

            # Optional sanity check (ensures gene_order is index-sorted 0..G-1)
            assert all(self.gene_to_index[self.gene_order[i]] == i for i in range(len(self.gene_order)))

            train_features = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float)
            train_labels = torch.tensor(data.iloc[:, -1].values, dtype=torch.float)

            # Store in the dictionary: key=file path, value=features and labels
            self.meta_train_dataset[train_file] = {
                'all_features': train_features,
                'all_labels': train_labels
            }

        # Preload and process data for meta-testing
        for target_file in target_files:
            # Load and process target files
            meta_test_data = pd.read_csv(target_file, sep='\t') 

            # (Optional) Fail fast if any required gene is missing
            missing = [g for g in self.gene_order if g not in meta_test_data.columns]
            if missing:
                raise ValueError(
                    f"[{os.path.basename(os.path.dirname(target_file))}] Missing {len(missing)} genes in {target_file}. "
                    f"First few: {missing[:5]}"
                )

            # Reorder columns to align feature columns with gene_to_index order
            meta_test_data = meta_test_data[self.gene_order + ['Label']]
            assert all(self.gene_to_index[self.gene_order[i]] == i for i in range(len(self.gene_order)))

            test_features = torch.tensor(meta_test_data.iloc[:, :-1].values, dtype=torch.float)
            test_labels = torch.tensor(meta_test_data.iloc[:, -1].values, dtype=torch.float)

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
        
        Args:
            tasks_per_meta_batch (int): Number of tasks to create in one meta-batch.
            K (int): K-shot (number of samples per label for support set).
            
        Returns:
            tasks (list): A list of tasks, each is ((x_support, y_support), (x_query, y_query)).
        """
        
        selected_files = self.training_files.copy()
        
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

        Args:
            K (int): Number of samples per label for the support set (K-shot).
            target_file (str): Path to the target file for meta-testing.

        Returns:
            ((x_support, y_support), (x_query, y_query)): Tuple containing support and query sets.

        Raises:
            ValueError: If there are insufficient samples for K-shot or no query samples.
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
            raise ValueError() #(f"Not enough samples in {target_file} for K-shot testing.")

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

        return (x_support, y_support, support_indices.cpu()), (x_query, y_query, query_indices.cpu())

def save_query_lig_to_csv(disease_name, task_idx, query_indices, y_query_cpu, attr_matrix, pathway_to_index, out_root=LIG_RAW_DIR):
    """
    Save per-sample pathway LIG for the query set of one task.
    - attr_matrix: (num_query, pathway_dim) numpy array
    - query_indices: (num_query,) tensor/cpu numpy indices into target_file rows
    - y_query_cpu: (num_query, 1) tensor on CPU with labels 0/1
    """
    out_dir = os.path.join(out_root, disease_name)
    os.makedirs(out_dir, exist_ok=True)

    # Save pathway index->name once per disease (idempotent)
    idx2path = {idx: name for name, idx in pathway_to_index.items()}
    map_path = os.path.join(out_dir, 'pathway_index_to_name.csv')
    if not os.path.exists(map_path):
        with open(map_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Index', 'Pathway'])
            for idx in range(len(idx2path)):
                w.writerow([idx, idx2path[idx]])

    # Prepare header: sample_idx, label, then pathway columns in index order
    header = ['sample_idx', 'label'] + [idx2path[i] for i in range(len(idx2path))]
    save_path = os.path.join(out_dir, f'query_LIG_task{task_idx:03d}.csv')

    # Ensure proper shapes/dtypes
    sample_idx = np.asarray(query_indices, dtype=np.int64).reshape(-1)
    labels = np.asarray(y_query_cpu.squeeze(), dtype=np.int64).reshape(-1)
    assert attr_matrix.shape[0] == sample_idx.shape[0] == labels.shape[0], "Row size mismatch"

    with open(save_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(attr_matrix.shape[0]):
            w.writerow([int(sample_idx[i]), int(labels[i]), *attr_matrix[i].tolist()])


def compute_gene_ig_total(model, sample_input, fast_weights, n_steps=100, use_logit=True):
    """
    Compute Integrated Gradients from input genes to the final output.
    By default, the logit (pre-sigmoid output) is used as the target to mitigate saturation (use_logit=True).
    Returns: tensor of shape (batch_size, input_dim)
    """
    # Wrapper for performing the forward pass with fast_weights
    # (injecting gene_to_pathway weights in the same way as in LIG)
    pathway_block = nn.Sequential(model.gene_to_pathway, model.ln_pathway, model.activation)

    def forward_func(x):
        # Mask fast_weights[0] * mask into gene_to_pathway
        orig_weight = model.gene_to_pathway.weight.data.clone()
        model.gene_to_pathway.weight.data = fast_weights[0] * model.gene_to_pathway_mask

        # pathway block
        out = pathway_block(x)  # (B, pathway_dim)
        # hidden & output (using fast_weights)
        h = F.linear(out, fast_weights[1], fast_weights[2])
        h = model.ln_hidden(h)
        h = model.activation(h)
        logits = F.linear(h, fast_weights[3], fast_weights[4]).squeeze(-1)

        # Restore original weights
        model.gene_to_pathway.weight.data = orig_weight

        return logits if use_logit else torch.sigmoid(logits)

    baseline = torch.zeros_like(sample_input)  # Can be replaced with a per-gene median baseline if needed
    ig = IntegratedGradients(forward_func)
    gene_attr = ig.attribute(inputs=sample_input, baselines=baseline, n_steps=n_steps)  # (B, G)
    return gene_attr


def save_query_gene_ig_to_csv(
    disease_name, task_idx, query_indices, y_query_cpu, gene_attr_matrix, gene_to_index,
    out_root=IG_RAW_DIR
):
    """
    Save per-sample gene-level IG for the query set to a CSV file.
    - gene_attr_matrix: (num_query, input_dim) numpy array
    - gene_to_index: {gene_name -> idx}
    """
    out_dir = Path(out_root) / disease_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gene index -> name mapping file (once per disease, idempotent)
    idx2gene = {idx: name for name, idx in gene_to_index.items()}
    map_path = out_dir / 'pathway_index_to_name.csv'
    if not map_path.exists():
        with open(map_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Index', 'Gene'])
            for idx in range(len(idx2gene)):
                w.writerow([idx, idx2gene[idx]])

    # Construct header
    header = ['sample_idx', 'label'] + [idx2gene[i] for i in range(len(idx2gene))]
    save_path = out_dir / f'query_LIG_task{task_idx:03d}.csv'

    sample_idx = np.asarray(query_indices, dtype=np.int64).reshape(-1)
    labels = np.asarray(y_query_cpu.squeeze(), dtype=np.int64).reshape(-1)
    assert gene_attr_matrix.shape[0] == sample_idx.shape[0] == labels.shape[0], "Row size mismatch"

    with open(save_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(gene_attr_matrix.shape[0]):
            w.writerow([int(sample_idx[i]), int(labels[i]), *gene_attr_matrix[i].tolist()])
    
class HierarchicalMAMLModel(nn.Module):
    def __init__(self, input_dim, pathway_dim, hidden_dim, output_dim, gene_pathway_mapping):
        """
        HierarchicalMAMLModel:
        A hierarchical model with the following structure:
        1. Input features (genes) are aggregated into pathway-level representations.
        2. Pathway representations are processed and transformed into a hidden layer.
        3. The hidden layer is used to produce the final output.
        4. A mask is applied to enforce connections between genes and pathways as defined by the gene_pathway_mapping.

        Args:
            input_dim (int): Number of input features (genes).
            pathway_dim (int): Number of unique pathways.
            hidden_dim (int): Dimension of the hidden layer.
            output_dim (int): Dimension of the final output.
            gene_pathway_mapping (dict): Mapping of genes to pathways as {gene_idx: [pathway_idx1, pathway_idx2, ...]}.
        """
        super(HierarchicalMAMLModel, self).__init__()
        self.input_dim = input_dim
        self.pathway_dim = pathway_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Define layers
        # gene_to_pathway: Aggregates input features into pathway-level representations
        self.gene_to_pathway = nn.Linear(input_dim, pathway_dim, bias=False)
        # pathway_to_hidden: Processes pathway representations to produce hidden layer features
        self.pathway_to_hidden = nn.Linear(pathway_dim, hidden_dim)
        # hidden_to_output: Produces the final output from hidden layer features
        self.hidden_to_output = nn.Linear(hidden_dim, output_dim)
        
        # Add Layer Normalization
        self.ln_pathway = nn.LayerNorm(pathway_dim)  # Pathway Layer Normalization
        self.ln_hidden = nn.LayerNorm(hidden_dim)    # Hidden Layer Normalization
    
        # Create masking matrix
        # Generate a binary mask to enforce connections between genes and pathways based on gene_pathway_mapping
        mask = self.create_gene_to_pathway_mask(input_dim, pathway_dim, gene_pathway_mapping)
        self.register_buffer('gene_to_pathway_mask', mask)  # Register mask as a buffer (not trainable but moves with the model to the correct device)
        
        # Apply the mask to the initial weights of gene_to_pathway
        # Weights corresponding to disconnected gene-pathway pairs are set to zero
        self.gene_to_pathway.weight.data *= self.gene_to_pathway_mask

        # Activation functions
        self.activation = nn.LeakyReLU()
        self.output_activation = nn.Sigmoid()

    # Gene-to-Pathway mask creation function
    def create_gene_to_pathway_mask(self, input_dim, pathway_dim, gene_pathway_mapping):
        """
        Create a mask for the gene-to-pathway layer.
        The mask ensures only connected gene-pathway pairs are considered in computations.

        Args:
            input_dim (int): Number of genes (columns).
            pathway_dim (int): Number of pathways (rows).
            gene_pathway_mapping (dict): Mapping of genes to pathways as {gene_idx: [pathway_idx1, ...]}.

        Returns:
            mask (Tensor): Binary mask of shape (pathway_dim, input_dim), where connected pairs are 1, others are 0.
        """
        mask = torch.zeros(pathway_dim, input_dim) # Initialize mask as a zero tensor of size (pathway_dim x input_dim)
        for gene_idx, pathways in gene_pathway_mapping.items(): # Iterate over genes and their connected pathways
            for pathway_idx in pathways: # Set mask value to 1 for each valid gene-pathway connection
                mask[pathway_idx, gene_idx] = 1
        return mask

    # Parameterized forward function for weight injection
    def parameterized(self, x, weights): 
        """
        Parameterized forward function:
        Performs forward propagation using externally injected weights, useful for MAML-style meta-learning.

        Args:
            x (Tensor): Input tensor of shape (batch_size x input_dim).
            weights (list[Tensor]): List of weight tensors, in the following order:
                - gene_to_pathway_weight: shape (pathway_dim, input_dim)
                - pathway_to_hidden_weight: shape (hidden_dim, pathway_dim)
                - pathway_to_hidden_bias: shape (hidden_dim,)
                - hidden_to_output_weight: shape (output_dim, hidden_dim)
                - hidden_to_output_bias: shape (output_dim,)

        Returns:
            x (Tensor): Final output tensor of shape (batch_size x output_dim).
        """
        # Apply the mask to the gene-to-pathway weights to ensure only valid connections are used
        gene_to_pathway_weight = weights[0] * self.gene_to_pathway_mask
        pathway_to_hidden_weight = weights[1]
        pathway_to_hidden_bias = weights[2]
        hidden_to_output_weight = weights[3]
        hidden_to_output_bias = weights[4]
       
        # Gene -> Pathway: Apply linear transformation with masked weights
        x = F.linear(x, gene_to_pathway_weight, None)
        x = self.ln_pathway(x)  
        x = self.activation(x) 

        # Pathway -> Hidden: Apply linear transformation and normalization
        x = F.linear(x, pathway_to_hidden_weight, pathway_to_hidden_bias)
        x = self.ln_hidden(x) 
        x = self.activation(x) 

        # Hidden -> Output: Apply linear transformation and sigmoid activation
        x = F.linear(x, hidden_to_output_weight, hidden_to_output_bias)
        x = self.output_activation(x) 
        
        return x

def save_mean_pathway_scores(mean_attr, disease_name, label, pathway_to_index):
    """Save averaged pathway scores **per label group** (0=control, 1=disease).

    Args:
        mean_attr (np.ndarray): 1-D array of averaged pathway attributions.
        disease_name (str): Name of the disease (directory name).
        label (int): 0 for negative / control, 1 for positive / case.
        pathway_to_index (dict): Mapping pathway→index.
    """
    ps_dir = PATHWAY_SCORES_DIR
    ps_dir.mkdir(parents=True, exist_ok=True)

    label_tag = 'label1_pos' if label == 1 else 'label0_neg'
    csv_path = ps_dir / f'{disease_name}_{label_tag}_pathway_scores.csv'

    idx2path = {idx: name for name, idx in pathway_to_index.items()}
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Pathway', 'Score'])
        for idx, score in enumerate(mean_attr):
            writer.writerow([idx2path[idx], score])
    print(f'Saved averaged pathway scores: {csv_path}')

class MAML:
    def __init__(self, model, dataset, inner_lr, meta_lr, K=1, inner_steps=1, tasks_per_meta_batch=3, L2_lambda=3e-4, use_second_order=True,
                ckpt_path=WEIGHTS_DIR / 'meta_trained_model_fc.pth',
                eval_csv_path=RESULTS_DIR / 'evaluation_results_by_diseases.csv',
                save_attributions=False):
        """
        Initialize the MAML (Model-Agnostic Meta-Learning) object.

        Args:
            model (nn.Module): The model to be meta-trained (e.g., HierarchicalMAMLModel).
            dataset: A dataset capable of generating tasks (support set and query set).
            inner_lr (float): Learning rate for the inner loop (task-specific updates).
            meta_lr (float): Learning rate for the outer loop (meta-optimization).
            K (int): Number of support samples per class (K-shot learning).
            inner_steps (int): Number of parameter updates in the inner loop.
            tasks_per_meta_batch (int): Number of tasks per meta-batch during meta-training.
            L2_lambda (float): Weight decay coefficient (L2 regularization) for the AdamW optimizer.
            use_second_order (bool): Whether to use second-order gradients (True for ordinary MAML, False for FOMAML).
        """
        self.dataset = dataset
        self.model = model
        self.weights = list(model.parameters())
        self.criterion = nn.BCELoss() 
        self.meta_optimizer = optim.AdamW(self.weights, meta_lr, weight_decay=L2_lambda)
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.K = K
        self.inner_steps = inner_steps
        self.tasks_per_meta_batch = tasks_per_meta_batch
        self.use_second_order = use_second_order  # Flag for selecting first-order or second-order MAML
        self.ckpt_path = ckpt_path
        self.eval_csv_path = eval_csv_path
        self.save_attributions = save_attributions

        # Warm-up + Cosine Annealing Scheduler
        warmup_steps = 10  # Adjust based on your setup
        total_steps = 5000  # Total number of iterations
        self.scheduler = linear_warmup_cosine_scheduler(
            self.meta_optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=1e-4,
            warmup_lr_init=1e-4 # Starting LR during warm-up
        )
        
        #self.scheduler = StepLR(self.meta_optimizer, step_size=30, gamma=0.1)
        
        # Early stopping parameters
        self.best_loss = float('inf')
        self.patience = 100
        self.patience_counter = 0
        
        self.meta_train_stop_iter = None
        
    def inner_loop(self, task):
        """
        Perform an inner-loop update for a single meta-learning task.
        1. Unpack support/query sets.
        2. Clone the current meta-parameters (self.weights) for task-specific adaptation.
        3. Update the cloned parameters (fast_weights) for a fixed number of steps (inner_steps) using the support set.
        4. Compute and return the loss on the query set using the updated parameters.
        
        Args:
            task (tuple): ((x_support, y_support), (x_query, y_query)) where:
                - x_support, y_support: Support set features and labels.
                - x_query, y_query: Query set features and labels.

        Returns:
            query_loss (Tensor): Loss computed on the query set with the updated parameters.
        """
        # Unpack support and query sets
        (x_support, y_support), (x_query, y_query) = task
        x_support, y_support = x_support.to(device), y_support.to(device) 
        x_query, y_query = x_query.to(device), y_query.to(device)
        
        # Clone the meta-parameters for task-specific updates
        fast_weights = [w.clone() for w in self.weights]
        
        # Update 'fast_weights' for 'inner_steps' times on the support set
        for _ in range(self.inner_steps):
            # Compute predictions and loss on the support set
            support_preds = self.model.parameterized(x_support, fast_weights)
            support_loss = self.criterion(support_preds, y_support)
            
            # Compute gradients of the loss with respect to the parameters
            grad = torch.autograd.grad(support_loss, fast_weights, create_graph=self.use_second_order, allow_unused=True)
            
            # Update fast_weights using gradient descent
            fast_weights = [w - self.inner_lr * g if g is not None else w for w, g in zip(fast_weights, grad)] 

        # Compute the final loss on the query set using the updated parameters
        query_preds = self.model.parameterized(x_query, fast_weights)
        query_loss = self.criterion(query_preds, y_query)
            
        return query_loss

    def main_loop(self, num_iterations):
        """
        Perform the outer loop (meta-training) for a num_iteartions.

        1. Sample tasks_per_meta_batch tasks per iteration.
        2. Perform inner_loop updates for each task and compute the query set loss.
        3. Aggregate and average the losses from all tasks to compute the meta-loss.
        4. Update meta-parameters (self.weights) using the meta-loss and the meta-optimizer.
        5. Apply learning rate scheduler and early stopping if conditions are met.

        Args:
            num_iterations (int): Number of outer loop iterations to perform.
        """
        self.model.train()
        
        self.meta_train_stop_iter = num_iterations
        
        with tqdm(total=num_iterations, desc="Meta-Training Progress") as pbar:
            for iteration in range(1, num_iterations + 1):
                # Sample tasks for the current iteration
                tasks = self.dataset.create_train_tasks(self.tasks_per_meta_batch, self.K)
                total_loss = torch.tensor(0., device=device) # Initialize total loss for the meta-batch

                # Perform inner loop for each task
                for task in tasks: 
                    task_loss = self.inner_loop(task) # Compute task-specific loss via inner loop
                    total_loss += task_loss

                # Compute the average loss for the meta-batch
                meta_loss = total_loss / len(tasks)

                # Compute gradients of meta-loss w.r.t. meta-parameters
                grads = torch.autograd.grad(meta_loss, self.weights, create_graph=False)
                
                # Update meta-parameters using the computed gradients
                for w, g in zip(self.weights, grads):
                    if w.grad is None :
                        w.grad = torch.zeros_like(w)
                    w.grad.copy_(g)  # Copy the computed gradient to w.grad
                        
                self.meta_optimizer.step() # Update meta-parameters using the optimizer
                self.meta_optimizer.zero_grad() # Clear gradients for the next iteration


                # Update progress bar with current meta-loss
                pbar.set_postfix({'Loss': meta_loss.item()})
                pbar.update(1)                    

                # Early Stopping
                if meta_loss < self.best_loss:
                    self.best_loss = meta_loss
                    self.patience_counter = 0
                    torch.save(self.model.state_dict(), self.ckpt_path)
                    
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        self.meta_train_stop_iter = iteration
                        print(f"Early stopping at iteration {iteration}")
                        break
                    
                # Step the learning rate scheduler
                self.scheduler.step()
                
        return self.meta_train_stop_iter
        
    def evaluate(self, K, target_files, metatest_tasks, metatest_steps, metatest_lr):
        """
        Perform meta-testing using the trained model on the given target files.

        For each target file:
        1. Generate a specified number of meta-test tasks, each consisting of a support set and a query set.
        2. Perform a fixed number of inner-loop gradient updates (metatest_steps) on the support set.
        3. Evaluate the updated parameters (fast_weights) on the query set to compute PR_AUC and AUC.
        4. Aggregate PR_AUC and AUC across tasks, calculating their mean and standard deviation for each target file.
        5. Compute and display overall statistics (PR_AUC and AUC) across all target files.

        Additionally:
        - Results for each target file are printed.
        - Predictions and actual labels are saved to a CSV file.

        Args:
            K (int): Number of samples in the support set (K-shot).
            target_files (list[str]): List of file paths corresponding to the diseases to evaluate.
            metatest_tasks (int): Number of tasks to generate for each target file (default: 10).
            metatest_steps (int): Number of inner-loop updates to perform on the support set (default: 10).
            metatest_lr (float): Learning rate for the inner loop during meta-testing.

        Returns: 
            None
            (Prints results and saves them to a CSV file.)
        """
        # Load the trained meta-parameters and set the model to evaluation mode
        self.model.load_state_dict(torch.load(self.ckpt_path, map_location=device))
        self.model.eval()
                
        overall_pr_auc_list, overall_auc_list = [], []  # Collect PR_AUC and AUC across all tasks and target files
        all_predictions, all_targets = [], []  # Collect all predictions and true labels for saving
        results_by_disease = {}  # Store individual results for each disease    
        
        base_dir = BIOMARKER_DIR 
        
        # Evaluate on each target file
        for target_file in target_files:
            target_path = Path(target_file)
            disease_name = target_path.parent.name
            
            pr_auc_list, auc_list = [], []
            
            # make disease-specific directory once
            out_dir = base_dir / disease_name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            for task_idx in range(metatest_tasks):
                try:
                    test_task = self.dataset.create_test_task(K, target_file)
                except ValueError as e:
                    print(f"Skipping task {task_idx} for {disease_name}: {e}")
                    continue

                # Unpack with indices
                (x_support, y_support, support_indices), (x_query, y_query, query_indices) = test_task
                x_support, y_support = x_support.to(device), y_support.to(device)
                x_query,   y_query   = x_query.to(device),   y_query.to(device)
            
                # Clone the meta-parameters for task-specific adaptation
                fast_weights = [w.clone() for w in self.weights]

                # Inner-loop updates on the support set
                for _ in range(metatest_steps):
                    # Compute the loss on the support set using the current fast_weights
                    support_loss = self.criterion(self.model.parameterized(x_support, fast_weights).squeeze(), y_support.squeeze())
                    # Calculate gradients of the support loss with respect to fast_weights
                    grad = torch.autograd.grad(support_loss, fast_weights, create_graph=False, allow_unused=True)
                    # Update fast_weights using gradient descent with the specified meta-test learning rate
                    fast_weights = [w - metatest_lr * g if g is not None else w for w, g in zip(fast_weights, grad)]

                if self.save_attributions:
                    attr_mat = LIG(
                        model=self.model,
                        sample_input=x_query,               # (num_query, input_dim)
                        fast_weights=fast_weights,
                        pathway_to_index=pathway_to_index,
                        n_steps=100,
                        return_full=True,                   # return (num_query, pathway_dim)
                        use_logit=True                      # True = logit, False = sigmoid
                    )
                    save_query_lig_to_csv(
                        disease_name=disease_name,
                        task_idx=task_idx,
                        query_indices=query_indices.numpy(),       # (num_query,)
                        y_query_cpu=y_query.detach().cpu(),        # (num_query,1)
                        attr_matrix=attr_mat,                      # (num_query, pathway_dim)
                        pathway_to_index=pathway_to_index,
                        out_root=LIG_RAW_DIR
                    )

                    gene_ig_t = compute_gene_ig_total(
                        model=self.model,
                        sample_input=x_query,         # (num_query, input_dim)
                        fast_weights=fast_weights,
                        n_steps=100,
                        use_logit=True                
                    )
                    save_query_gene_ig_to_csv(
                        disease_name=disease_name,
                        task_idx=task_idx,
                        query_indices=query_indices.numpy(),
                        y_query_cpu=y_query.detach().cpu(),
                        gene_attr_matrix=gene_ig_t.detach().cpu().numpy(),  # (num_query, input_dim)
                        gene_to_index=self.dataset.gene_to_index,           
                        out_root=IG_RAW_DIR
                    )
                
                # Evaluate performance on the query set
                with torch.no_grad():
                    predictions = self.model.parameterized(x_query, fast_weights).squeeze()
                    # Compute AUC
                    roc_auc_value = roc_auc_score(y_query.squeeze().cpu().numpy(), predictions.cpu().numpy())
                    # Compute PR-AUC
                    precision, recall, _ = precision_recall_curve(y_query.squeeze().cpu().numpy(), predictions.cpu().numpy())
                    PR_AUC = auc(recall, precision)
                    # Save predictions and targets
                    all_predictions.extend(predictions.cpu().tolist())
                    all_targets.extend(y_query.squeeze().cpu().tolist())

                # Store results for this task
                pr_auc_list.append(PR_AUC)
                auc_list.append(roc_auc_value)

            # Skip if no valid tasks were processed
            if len(pr_auc_list) == 0 or len(auc_list) == 0:
                print(f"Skipping {disease_name}: No valid tasks generated.")
                continue

            # Calculate statistics for the current disease
            avg_pr_auc = np.mean(pr_auc_list)
            std_pr_auc = np.std(pr_auc_list)
            avg_auc = np.mean(auc_list)
            std_auc = np.std(auc_list)

            results_by_disease[disease_name] = {
                "Average PR_AUC": avg_pr_auc,
                "Std PR_AUC": std_pr_auc,
                "Average AUC": avg_auc,
                "Std AUC": std_auc
            }

            print(f"{disease_name} - Average PR_AUC: {avg_pr_auc:.4f} ± {std_pr_auc:.4f}, Average AUC: {avg_auc:.4f} ± {std_auc:.4f}")
            overall_pr_auc_list.extend(pr_auc_list)
            overall_auc_list.extend(auc_list)
            
        # Calculate overall metrics
        overall_avg_pr_auc = np.mean(overall_pr_auc_list)
        overall_std_pr_auc = np.std(overall_pr_auc_list)
        overall_avg_auc = np.mean(overall_auc_list)
        overall_std_auc = np.std(overall_auc_list)

        print(f"Overall - Average PR_AUC: {overall_avg_pr_auc:.4f} ± {overall_std_pr_auc:.4f}")
        print(f"Overall - Average AUC: {overall_avg_auc:.4f} ± {overall_std_auc:.4f}")

        # Save results to a CSV file
        self.save_results_to_csv(all_predictions, all_targets, overall_avg_pr_auc, overall_std_pr_auc, overall_avg_auc, overall_std_auc, results_by_disease)
        
        return results_by_disease, (overall_avg_pr_auc, overall_std_pr_auc, overall_avg_auc, overall_std_auc)
    
    def save_results_to_csv(self, predictions, targets, overall_avg_pr_auc, overall_std_pr_auc, overall_avg_auc, overall_std_auc, results_by_disease):
        """
        Save predictions, targets, overall metrics, and disease-specific metrics to a CSV file.
        """
        filename = self.eval_csv_path
        filename.parent.mkdir(parents=True, exist_ok=True)
        
        # Open the file in write mode
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write overall metrics
            writer.writerow(['Metrics', 'Value'])
            writer.writerow(['Overall Average PR_AUC', overall_avg_pr_auc])
            writer.writerow(['Overall Std PR_AUC', overall_std_pr_auc])
            writer.writerow(['Overall Average AUC', overall_avg_auc])
            writer.writerow(['Overall Std AUC', overall_std_auc])
            writer.writerow(['Meta-Training Stop Iteration', getattr(self, "meta_train_stop_iter", "")])
            
            # Write per-disease metrics
            writer.writerow(['Disease', 'Average PR_AUC', 'Std PR_AUC', 'Average AUC', 'Std AUC'])
            for disease, metrics in results_by_disease.items():
                writer.writerow([disease, metrics["Average PR_AUC"], metrics["Std PR_AUC"], metrics["Average AUC"], metrics["Std AUC"]])

            # Write predictions and targets
            writer.writerow([])
            writer.writerow(['Predicted', 'Actual'])
            for pred, target in zip(predictions, targets):
                writer.writerow([pred, target])

        print(f"Results and metrics saved to {filename}")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _read_eval_csv(eval_csv_path: str):
    """
    Parse MAML.save_results_to_csv() output file:
      - "Metrics,Value" section contains Overall metrics
      - blank line
      - "Disease,Average PR_AUC,Std PR_AUC,Average AUC,Std AUC" section
    Returns:
      overall: dict
      per_disease: dict[disease] -> dict
    """
    overall = {}
    per_disease = {}

    if not os.path.exists(eval_csv_path):
        raise FileNotFoundError(f"Eval CSV not found: {eval_csv_path}")

    rows = []
    with open(eval_csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for r in reader:
            rows.append(r)

    # 1) overall metrics (look for 'Overall Average PR_AUC' etc.)
    for r in rows:
        if len(r) >= 2:
            k, v = r[0].strip(), r[1].strip()
            if k in ["Overall Average PR_AUC", "Overall Std PR_AUC", "Overall Average AUC", "Overall Std AUC"]:
                try:
                    overall[k] = float(v)
                except:
                    overall[k] = v

    # 2) find per-disease header row index
    header_idx = None
    for i, r in enumerate(rows):
        if len(r) >= 5 and r[0].strip() == "Disease" and r[1].strip() == "Average PR_AUC":
            header_idx = i
            break

    if header_idx is not None:
        for r in rows[header_idx + 1:]:
            if not r or len(r) < 5:
                # stop at first empty row or short row (pred/actual section begins later)
                break
            disease = r[0].strip()
            try:
                per_disease[disease] = {
                    "Average PR_AUC": float(r[1]),
                    "Std PR_AUC": float(r[2]),
                    "Average AUC": float(r[3]),
                    "Std AUC": float(r[4]),
                }
            except:
                per_disease[disease] = {
                    "Average PR_AUC": r[1],
                    "Std PR_AUC": r[2],
                    "Average AUC": r[3],
                    "Std AUC": r[4],
                }

    return overall, per_disease

def _append_sweep_tsv(
    tsv_path: str,
    run_id: str,
    K: int,
    inner_lr: float,
    meta_lr: float,
    overall: dict,
    per_disease: dict,
    diseases_in_order: list,
    meta_train_stop_iter: int
):
    """
    Append one sweep result into a TSV file.
    One row = one (inner_lr, meta_lr) run.
    """
    header = [
        "run_id", "timestamp", "K", "inner_lr", "meta_lr", "meta_train_stop_iter",
        "overall_avg_pr_auc", "overall_std_pr_auc", "overall_avg_auc", "overall_std_auc",
    ]
    # Add per-disease columns (stable order)
    for d in diseases_in_order:
        header += [
            f"{d}.avg_pr_auc", f"{d}.std_pr_auc", f"{d}.avg_auc", f"{d}.std_auc"
        ]

    row = [
        run_id,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        K,
        f"{inner_lr:.10g}",
        f"{meta_lr:.10g}",
        meta_train_stop_iter,
        overall.get("Overall Average PR_AUC", ""),
        overall.get("Overall Std PR_AUC", ""),
        overall.get("Overall Average AUC", ""),
        overall.get("Overall Std AUC", ""),
    ]

    for d in diseases_in_order:
        m = per_disease.get(d, {})
        row += [
            m.get("Average PR_AUC", ""),
            m.get("Std PR_AUC", ""),
            m.get("Average AUC", ""),
            m.get("Std AUC", ""),
        ]

    path = Path(tsv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists(tsv_path)
    with open(tsv_path, "a", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if not file_exists:
            w.writerow(header)
        w.writerow(row)

def run_lr_sweep(
    *,
    inner_lrs: list,
    meta_lrs: list,
    build_model_fn,
    dataset,
    K: int,
    tasks_per_meta_batch: int,
    inner_steps: int,
    L2_lambda: float,
    use_second_order: bool,
    num_iterations: int,
    target_files: list,
    metatest_tasks: int,
    metatest_steps: int,
    metatest_lr: float,
    sweep_tsv_path: str,
    sweep_artifact_dir: str,
    diseases_in_order: list,
    eval_csv_fixed_path: str = RESULTS_DIR / 'evaluation_results_by_diseases.csv',
    weight_fixed_path: str = WEIGHTS_DIR / 'meta_trained_model_fc.pth',
    sweep_seed: int = 42,
):
    """
    Sweep inner_lr x meta_lr and append results to TSV.
    Keeps original evaluate/save_results_to_csv behavior (fixed paths) and copies artifacts out per run.
    """
    _ensure_dir(sweep_artifact_dir)

    for inner_lr, meta_lr in itertools.product(inner_lrs, meta_lrs):
        # reset seed for comparability (optional but recommended)
        torch.manual_seed(sweep_seed)
        np.random.seed(sweep_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sweep_seed)

        run_id = f"inner{inner_lr:.2e}_meta{meta_lr:.2e}"
        print(f"\n\n===== [LR SWEEP] {run_id} =====")

        model = build_model_fn().to(device)

        maml = MAML(
            model=model,
            dataset=dataset,
            inner_lr=inner_lr,
            meta_lr=meta_lr,
            K=K,
            tasks_per_meta_batch=tasks_per_meta_batch,
            inner_steps=inner_steps,
            L2_lambda=L2_lambda,
            use_second_order=use_second_order,
        )

        # meta-train
        stop_iter = maml.main_loop(num_iterations=num_iterations)

        # meta-test (writes eval csv to fixed path)
        maml.evaluate(K, target_files, metatest_tasks, metatest_steps, metatest_lr)

        # parse the fixed eval csv and append TSV
        overall, per_disease = _read_eval_csv(eval_csv_fixed_path)
        _append_sweep_tsv(
            tsv_path=sweep_tsv_path,
            run_id=run_id,
            K = K,
            inner_lr=inner_lr,
            meta_lr=meta_lr,
            overall=overall,
            per_disease=per_disease,
            diseases_in_order=diseases_in_order,
            meta_train_stop_iter=stop_iter,
        )
        print(f"[LR SWEEP] appended -> {sweep_tsv_path}")

        # copy artifacts (eval csv + weights) to avoid overwrite in next run
        run_dir = os.path.join(sweep_artifact_dir, run_id)
        _ensure_dir(run_dir)

        if os.path.exists(eval_csv_fixed_path):
            shutil.copy2(eval_csv_fixed_path, os.path.join(run_dir, "evaluation_results_by_diseases.csv"))

        if os.path.exists(weight_fixed_path):
            shutil.copy2(weight_fixed_path, os.path.join(run_dir, "meta_trained_model_fc.pth"))

# Define meta-testing diseases      
diseases = ['idiopathic_pulmonary_fibrosis', 'HBV-HCC', 'cirrhosis', 'ipf_ssc','IgA_nephropathy']  

target_files = [
NCBI_DIR / disease / 'second_filtered_combined_counts_transposed.tsv'
for disease in diseases
]

# Define the pre-training data path
data_folder = TCGA_DIR / '5_TCGA_NCBI'

# Mapping file path
mapping_file_path = REACTOME_DIR / 'final' / 'percentile90_min10_matched_combined_pathways.csv'

# Get list of files in the data folder for meta-training
training_files = sorted(os.listdir(data_folder))

# Generate gene_pathway_mapping
gene_pathway_mapping, gene_to_index, pathway_to_index, target_ID = GenePathwayMapping.load_gene_pathway_mapping(mapping_file_path, target_files[0])

# Initialize the dataset with specified training files and target file
dataset = GeneExpressionDataset(data_folder, training_files, target_files, gene_to_index)

# Set input and output dimensions based on the mapping and the model's architecture
input_dim = len(gene_to_index)  # Number of genes
pathway_dim = len(pathway_to_index)  # Number of pathways
hidden_dim = pathway_dim // 4  # Hidden dimension size; PASNet uses a fixed size of 100
output_dim = 1  # Binary classification

# Define MAML-specific parameters
K= 1 # Number of samples per task
tasks_per_meta_batch = 4 # Number of tasks in each meta-batch
inner_steps = 1 # Number of inner-loop optimization steps during meta-training

# Define meta-testing parameters
metatest_steps = 10 # Number of optimization steps during meta-testing
metatest_lr = 0.005 # Learning rate for meta-testing # 0.005

print('input_dim = ', input_dim)
print('pathway_dim = ', pathway_dim)
print('hidden_dim = ', hidden_dim)

# Initialize the model with the hidden layer
model = HierarchicalMAMLModel(
    input_dim=input_dim,
    pathway_dim=pathway_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    gene_pathway_mapping=gene_pathway_mapping
).to(device)

torch.save(model.state_dict(), WEIGHTS_DIR / 'INIT_RANDOM.pth')

maml = MAML(
    model=model,
    dataset=dataset, 
    inner_lr=3e-7,  #3e-8, #0.001, #3e-7, #3e-8, #8e-8
    meta_lr=3e-6,  #3e-7, #0.001 #3e-7, #3e-9, #8e-7
    K=K,
    tasks_per_meta_batch=tasks_per_meta_batch,
    inner_steps=inner_steps,
    L2_lambda=3e-4,
    use_second_order=True # Flag for selecting first-order or second-order MAML
)

# Define Inner Loop Learning Rates
# We use a strategic grid covering the range [1e-4, 2e-2] to capture the 
# shift in adaptation dynamics from low-shot to high-shot scenarios.
INNER_LRS = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 7e-3, 1e-2, 2e-2]
META_LRS = [1e-05]

COMBINATIONS = list(itertools.product(INNER_LRS, META_LRS))

SWEEP_TSV_PATH = RESULTS_DIR / 'lr_sweep' / 'lr_sweep_results_iter5000_vast.tsv'
SWEEP_ARTIFACT_DIR = RESULTS_DIR / 'lr_sweep' / 'artifacts'

DO_LR_SWEEP = True

def build_model_fn():
    m = HierarchicalMAMLModel(
        input_dim=input_dim,
        pathway_dim=pathway_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        gene_pathway_mapping=gene_pathway_mapping,
    )
    return m

metatest_tasks = 50

K_LIST = [1, 3, 5]

if DO_LR_SWEEP:
    
    _ensure_dir(SWEEP_ARTIFACT_DIR)

    for K in K_LIST:
        for inner_lr, meta_lr in COMBINATIONS:
            
            torch.manual_seed(42)
            np.random.seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)

            run_id = f"K{K}_inner{inner_lr:.2e}_meta{meta_lr:.2e}"
            print(f"\n\n===== [LR SWEEP] {run_id} =====")

            model = build_model_fn().to(device)

            maml = MAML(
                model=model,
                dataset=dataset,
                inner_lr=inner_lr,
                meta_lr=meta_lr,
                K=K,
                tasks_per_meta_batch=tasks_per_meta_batch,
                inner_steps=inner_steps,
                L2_lambda=3e-4,
                use_second_order=True,
            )

            # 1. Meta-Training
            stop_iter = maml.main_loop(num_iterations=5000)

            # 2. Meta-Testing
            maml.evaluate(K, target_files, metatest_tasks, metatest_steps, metatest_lr)

            # 3. Parse the fixed evaluation CSV and append results to the TSV log
            eval_csv_path = RESULTS_DIR / 'evaluation_results_by_diseases.csv'
            overall, per_disease = _read_eval_csv(eval_csv_path)
            
            _append_sweep_tsv(
                tsv_path=SWEEP_TSV_PATH,
                run_id=run_id,
                K=K,
                inner_lr=inner_lr,
                meta_lr=meta_lr,
                overall=overall,
                per_disease=per_disease,
                diseases_in_order=diseases,
                meta_train_stop_iter=stop_iter,
            )
            print(f"[LR SWEEP] appended -> {SWEEP_TSV_PATH}")

            # 4. Backup artifacts (result files, model weights) to a separate directory
            run_dir = os.path.join(SWEEP_ARTIFACT_DIR, run_id)
            _ensure_dir(run_dir)

            if os.path.exists(eval_csv_path):
                shutil.copy2(eval_csv_path, os.path.join(run_dir, "evaluation_results_by_diseases.csv"))

            weight_path = WEIGHTS_DIR / 'meta_trained_model_fc.pth'
            if os.path.exists(weight_path):
                shutil.copy2(weight_path, os.path.join(run_dir, "meta_trained_model_fc.pth"))

else:
    # ===== Single Run Execution (when DO_LR_SWEEP = False) =====
    model = build_model_fn().to(device)
    torch.save(model.state_dict(), WEIGHTS_DIR / 'INIT_RANDOM.pth')

    maml = MAML(
        model=model,
        dataset=dataset,
        inner_lr=8e-7,
        meta_lr=8e-6,
        K=K,
        tasks_per_meta_batch=tasks_per_meta_batch,
        inner_steps=inner_steps,
        L2_lambda=3e-4,
        use_second_order=True
    )

    maml.main_loop(num_iterations=5000)
    maml.evaluate(K, target_files, metatest_tasks, metatest_steps, metatest_lr)