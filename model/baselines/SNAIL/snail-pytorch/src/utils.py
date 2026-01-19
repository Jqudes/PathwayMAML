# coding=utf-8
import torch
import os
import torch
from torch.utils.data import DataLoader, Dataset
from gene_expression_dataset import GeneExpressionDataset


def init_dataset(opt):

    # 1) generate GeneExpressionDataset
    dataset = GeneExpressionDataset(
        data_folder=opt.data_folder,
        training_files=opt.train_files,
        target_files=opt.test_files,
        gene_to_index=opt.gene_to_index
    )
    # 2) list of meta-training tasks
    tr_tasks = dataset.create_train_tasks(
        tasks_per_meta_batch=opt.tasks_per_meta_batch,
        K=opt.K
    )
    # 3) list of meta-testing tasks
    test_tasks = [
        dataset.create_test_task(K=opt.K, target_file=target_file)
        for target_file in opt.test_files
    ]

    return tr_tasks, test_tasks
