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
from pathlib import Path
import argparse

# Global device used across dataset/model code (minimal-change approach)
device = None


def get_project_root() -> Path:
    # This file is located at:
    # Pathway_MAML/model/baselines/naive_maml.py
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

    # Checkpoint path (meta-trained weights)
    p.add_argument(
        "--ckpt_path",
        type=Path,
        default=root / "data" / "weights" / "naive_maml_meta_trained.pth"
    )

    # Experiment params
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--K", type=int, default=1)
    p.add_argument("--tasks_per_meta_batch", type=int, default=4)
    p.add_argument("--inner_steps", type=int, default=1)
    p.add_argument("--inner_lr", type=float, default=3e-4)
    p.add_argument("--meta_lr", type=float, default=1.5e-5)
    p.add_argument("--num_iterations", type=int, default=15000)

    # Meta-test params
    p.add_argument("--metatest_tasks", type=int, default=50)
    p.add_argument("--metatest_steps", type=int, default=10)
    p.add_argument("--metatest_lr", type=float, default=0.005)

    # Model dims (keep defaults consistent with your current script)
    p.add_argument("--pathway_dim", type=int, default=383)

    return p.parse_args()

def linear_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr=1e-4, warmup_lr_init=0.0):
    # Linear warm-up followed by cosine annealing.
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

class GeneExpressionDataset(Dataset):
    def __init__(self, data_folder, training_files, target_files, gene_to_index):

        self.data_folder = data_folder
        self.training_files = training_files  # Each file is treated as an individual task.
        self.meta_train_dataset = {}
        self.meta_test_dataset = {}
        self.gene_to_index = gene_to_index

        # Preload data for meta-training
        for train_file in training_files:
            file_path = os.path.join(data_folder, train_file)
            data = pd.read_csv(file_path)
            
            # Filter columns based on gene_to_index
            data = data[self.gene_to_index + ['Label']]

            train_features = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float)
            train_labels = torch.tensor(data.iloc[:, -1].values, dtype=torch.float)

            self.meta_train_dataset[train_file] = {
                'all_features': train_features,
                'all_labels': train_labels
            }

        # Preload and process data for meta-testing
        for target_file in target_files:
            meta_test_data = pd.read_csv(target_file, sep='\t') 
            meta_test_data = meta_test_data[self.gene_to_index + ['Label']]

            test_features = torch.tensor(meta_test_data.iloc[:, :-1].values, dtype=torch.float)
            test_labels = torch.tensor(meta_test_data.iloc[:, -1].values, dtype=torch.float)

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
        selected_files = self.training_files.copy()
        
        # Limit the number of tasks to create
        max_tasks = min(tasks_per_meta_batch, len(selected_files))
        selected_files = selected_files[:max_tasks]

        tasks = []
        for file_name in selected_files:
            if file_name not in self.meta_train_dataset:
                continue
            
            all_features = self.meta_train_dataset[file_name]['all_features']  # shape: (N, num_features)
            all_labels   = self.meta_train_dataset[file_name]['all_labels']    # shape: (N,)
            
            label_0_idx = (all_labels == 0).nonzero(as_tuple=True)[0]
            label_1_idx = (all_labels == 1).nonzero(as_tuple=True)[0]
            
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
        if target_file not in self.meta_test_dataset:
            raise ValueError(f"Target file {target_file} not found in meta_test_dataset.")

        all_features = self.meta_test_dataset[target_file]['all_features']
        all_labels = self.meta_test_dataset[target_file]['all_labels']

        label_0_idx = (all_labels == 0).nonzero(as_tuple=True)[0]
        label_1_idx = (all_labels == 1).nonzero(as_tuple=True)[0]

        if label_0_idx.size(0) < K or label_1_idx.size(0) < K:
            raise ValueError("Not enough samples to create support set.")

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

class HierarchicalMAMLModel(nn.Module):
    def __init__(self, input_dim, pathway_dim, hidden_dim, output_dim):

        super(HierarchicalMAMLModel, self).__init__()
        self.input_dim = input_dim
        self.pathway_dim = pathway_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Define layers
        # gene_to_pathway: Aggregates input features into pathway-level representations
        self.gene_to_pathway = nn.Linear(input_dim, pathway_dim)
        # pathway_to_hidden: Processes pathway representations to produce hidden layer features
        self.pathway_to_hidden = nn.Linear(pathway_dim, hidden_dim)
        # hidden_to_output: Produces the final output from hidden layer features
        self.hidden_to_output = nn.Linear(hidden_dim, output_dim)
        
        # Add Layer Normalization
        self.ln_pathway = nn.LayerNorm(pathway_dim)  # Pathway Layer Normalization
        self.ln_hidden = nn.LayerNorm(hidden_dim)    # Hidden Layer Normalization

        # Activation functions
        self.activation = nn.LeakyReLU()
        self.output_activation = nn.Sigmoid()

    # Parameterized forward function for weight injection
    def parameterised(self, x, weights): 
        
        # Apply the mask to the gene-to-pathway weights to ensure only valid connections are used
        gene_to_pathway_weight = weights[0]
        gene_to_pathway_bias = weights[1]
        pathway_to_hidden_weight = weights[2]
        pathway_to_hidden_bias = weights[3]
        hidden_to_output_weight = weights[4]
        hidden_to_output_bias = weights[5]
       
        # Gene -> Pathway: Apply linear transformation
        x = F.linear(x, gene_to_pathway_weight, gene_to_pathway_bias)
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
    
class MAML:
    def __init__(self, model, dataset, inner_lr, meta_lr, ckpt_path: Path, results_dir: Path, 
                 K=1, inner_steps=1, tasks_per_meta_batch=3, L2_lambda=3e-4, use_second_order=False):

        self.dataset = dataset
        self.data_folder = dataset.data_folder
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
        self.results_dir = results_dir

        # Warm-up + Cosine Annealing Scheduler
        warmup_steps = 10  # Adjust based on your setup
        total_steps = 15000  # Total number of iterations
        self.scheduler = linear_warmup_cosine_scheduler(
            self.meta_optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=1e-4,
            warmup_lr_init=1e-4 # Starting LR during warm-up
        )
        
        # Early stopping parameters
        self.best_loss = float('inf')
        self.patience = 100
        self.patience_counter = 0

    def inner_loop(self, task):
        """
        Perform an inner-loop update for a single meta-learning task.
        1. Unpack support/query sets.
        2. Clone the current meta-parameters (self.weights) for task-specific adaptation.
        3. Update the cloned parameters (fast_weights) for a fixed number of steps (inner_steps) using the support set.
        4. Compute and return the loss on the query set using the updated parameters.
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
            support_preds = self.model.parameterised(x_support, fast_weights)
            support_loss = self.criterion(support_preds, y_support)
            # Compute gradients of the loss with respect to the parameters
            grad = torch.autograd.grad(support_loss, fast_weights, create_graph=self.use_second_order, allow_unused=True)
            # Update fast_weights using gradient descent
            fast_weights = [w - self.inner_lr * g if g is not None else w for w, g in zip(fast_weights, grad)] 

        # Compute the final loss on the query set using the updated parameters
        query_preds = self.model.parameterised(x_query, fast_weights)
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
        """
        self.model.train()
        
        with tqdm(total=num_iterations, desc="Meta-Training Progress", mininterval=10) as pbar:
            for iteration in range(1, num_iterations + 1):
                # Sample tasks for the current iteration
                tasks = self.dataset.create_train_tasks(self.tasks_per_meta_batch, self.K)
                
                if len(tasks) == 0:
                    continue
                
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
                    # Save the meta-trained model
                    self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), str(self.ckpt_path))
                
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print(f"Early stopping at iteration {iteration}")
                        break
                    
                # Step the learning rate scheduler
                self.scheduler.step()

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
        """
        # Load the trained meta-parameters and set the model to evaluation mode
        self.model.load_state_dict(torch.load(str(self.ckpt_path), map_location=device))
        self.model.eval()
        
        overall_pr_auc_list, overall_auc_list = [], []  # Collect PR_AUC and AUC across all tasks and target files
        all_predictions, all_targets = [], []  # Collect all predictions and true labels for saving
        results_by_disease = {}  # Store individual results for each disease     
        
        # Evaluate on each target file
        for target_file in target_files:
            disease_name = os.path.basename(os.path.dirname(target_file))  # Extract disease name from directory
            pr_auc_list, auc_list = [], [] # Track PR_AUC and AUC for this disease
            
            # Generate and evaluate meta-test tasks
            for _ in range(metatest_tasks):
                try:
                    test_task = self.dataset.create_test_task(K, target_file)
                except ValueError as e:
                    #print(f"Skipping task generation for {disease_name}: {e}")
                    continue

                (x_support, y_support), (x_query, y_query) = test_task
                x_support, y_support = x_support.to(device), y_support.to(device)
                x_query, y_query = x_query.to(device), y_query.to(device)
            
                # Clone the meta-parameters for task-specific adaptation
                fast_weights = [w.clone() for w in self.weights]

                # Inner-loop updates on the support set
                for _ in range(metatest_steps):
                    # Compute the loss on the support set using the current fast_weights
                    support_loss = self.criterion(self.model.parameterised(x_support, fast_weights).squeeze(), y_support.squeeze())
                    # Calculate gradients of the support loss with respect to fast_weights
                    grad = torch.autograd.grad(support_loss, fast_weights, create_graph=True, allow_unused=True)
                    # Update fast_weights using gradient descent with the specified meta-test learning rate
                    fast_weights = [w - metatest_lr * g if g is not None else w for w, g in zip(fast_weights, grad)]

                # Evaluate performance on the query set
                with torch.no_grad():
                    predictions = self.model.parameterised(x_query, fast_weights).squeeze()
                    
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

    def save_results_to_csv(self, predictions, targets, overall_avg_pr_auc, overall_std_pr_auc, overall_avg_auc, overall_std_auc, results_by_disease):
        """
        Save predictions, targets, overall metrics, and disease-specific metrics to a CSV file.
        """
        # Define the directory and filename for saving the results
        self.results_dir.mkdir(parents=True, exist_ok=True)
        filename = self.results_dir / "evaluation_results_by_diseases.csv"
        
        # Open the file in write mode
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write overall metrics
            writer.writerow(['Metrics', 'Value'])
            writer.writerow(['Overall Average PR_AUC', overall_avg_pr_auc])
            writer.writerow(['Overall Std PR_AUC', overall_std_pr_auc])
            writer.writerow(['Overall Average AUC', overall_avg_auc])
            writer.writerow(['Overall Std AUC', overall_std_auc])
            writer.writerow([]) 
            
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
    training_files = [p.name for p in data_folder.iterdir() if p.is_file() and p.suffix == ".csv"]
    training_files = sorted(training_files)

    if len(training_files) == 0:
        raise RuntimeError(f"No training CSV files found in: {data_folder}")

    gene_to_index = Column_match.Column_match(
        str(data_folder / training_files[0]),
        target_files[0]
    )

    dataset = GeneExpressionDataset(
        data_folder=str(data_folder),
        training_files=training_files,
        target_files=target_files,
        gene_to_index=gene_to_index
    )

    input_dim = len(gene_to_index)
    pathway_dim = args.pathway_dim
    hidden_dim = pathway_dim // 4
    output_dim = 1

    print("input_dim =", input_dim)
    print("pathway_dim =", pathway_dim)
    print("hidden_dim =", hidden_dim)

    model = HierarchicalMAMLModel(
        input_dim=input_dim,
        pathway_dim=pathway_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    ).to(device)

    maml = MAML(
        model=model,
        dataset=dataset,
        inner_lr=args.inner_lr,
        meta_lr=args.meta_lr,
        ckpt_path=args.ckpt_path,
        results_dir=results_dir,
        K=args.K,
        tasks_per_meta_batch=args.tasks_per_meta_batch,
        inner_steps=args.inner_steps,
        L2_lambda=3e-4,
        use_second_order=True
    )

    maml.main_loop(num_iterations=args.num_iterations)

    maml.evaluate(
        K=args.K,
        target_files=target_files,
        metatest_tasks=args.metatest_tasks,
        metatest_steps=args.metatest_steps,
        metatest_lr=args.metatest_lr
    )


if __name__ == "__main__":
    main()