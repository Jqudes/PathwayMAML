import os
import pandas as pd
import torch
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")  

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
        
        Args:
            tasks_per_meta_batch (int): Number of tasks to create in one meta-batch.
            K (int): K-shot (number of samples per label for support set).
            
        Returns:
            tasks (list): A list of tasks, each is ((x_support, y_support), (x_query, y_query)).
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

        return (x_support, y_support), (x_query, y_query)
    
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