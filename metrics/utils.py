"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import collections
import logging
import math

import numpy as np
import torch
from tqdm import tqdm, trange

from utils import math
import metrics
from utils.helpers import get_cpu_core_num


METRICS = [
    'mig',
    'sap_d',
    'dci',
    'modularity_d',
    'reconstruction_error'
]

def select_metric(name, **kwargs):
    if name not in METRICS:
        err = f"{name} not a valid metric. Select from {METRICS}."
        raise ValueError(err)

    if name == 'mig':
        return metrics.MIG(**kwargs)
    if name == 'sap_d':
        return metrics.SAPd(**kwargs)
    if name == 'dci':
        return metrics.DCI(**kwargs)
    if name == 'modularity_d':
        return metrics.Modularityd(**kwargs)
    if name == 'reconstruction_error':
        return metrics.ReconstructionError(**kwargs)

class MetricAggregator: # TODO Add capability to compute metrics like reconstruction error that do not require latent representations
    """
    This class aggregates multiple disentanglement metrics
    """
    def __init__(self, metrics):
        """Initializes the metric computation utility.

        Args:
            metrics (list): A list that contains dict that specify the metric name and its argument
            num_workers (int, optional): The num of workers that is set for computing the metric. Defaults to 8.
        """
        assert isinstance(metrics, list), "Metrics must be provided as a list."
        for metric_config in metrics:
            assert isinstance(metric_config, dict), "Each metric configuration must be a dictionary."
            assert 'name' in metric_config, "Each metric dictionary must contain a 'name' key."
            assert 'args' in metric_config, "Each metric dictionary must contain an 'args' key."
        self.metrics = metrics
    
    def _get_representation_dataloader(self, model, data_loader, device='cpu'):
        """Get the latent representation and ground truth factors from the model and data loader.

        Args:
            model (torch.nn.Module): The model to evaluate.
            data_loader (torch.utils.data.DataLoader): The data loader containing the dataset.
            device (torch.device): The device to perform computations on.

        Returns:
            tuple: A tuple containing the latent representations and ground truth factors in cpu
        """
        model.to(device)
        model.eval()
        latent_reps = []
        gt_factors = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Computing representations"):
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                z = model.get_representations(inputs, is_deterministic=True)
                latent_reps.append(z.cpu())
                gt_factors.append(labels.cpu())

        return torch.cat(latent_reps), torch.cat(gt_factors)

    def _get_representation_dataset(self, model, dataset, seed, sample_num=None, device='cpu'):
        """Get the latent representation and ground truth factors from the model and dataset.

        Args:
            model (torch.nn.Module): The model to evaluate.
            dataset (torch.utils.data.Dataset): The dataset containing the data.
            device (torch.device): The device to perform computations on.

        Returns:
            tuple: A tuple containing the latent representations and ground truth factors in cpu
        """

        if sample_num is not None:
            if sample_num > len(dataset):
                raise ValueError(f"Sample number {sample_num} exceeds dataset size {len(dataset)}.")

            # Create a torch generator for reproducible sampling of dataset indices
            g = torch.Generator()
            g.manual_seed(seed)
            
            # Generate random indices using torch.randperm and the seeded generator
            # Then take the first 'sample_num' indices
            indices = torch.randperm(len(dataset), generator=g).tolist()[:sample_num]
            
            sample_dataset = torch.utils.data.Subset(dataset, indices)
        else:
            sample_dataset = dataset
        
        data_loader = torch.utils.data.DataLoader(sample_dataset, batch_size=64, shuffle=False, num_workers=get_cpu_core_num(), pin_memory=True)

        return self._get_representation_dataloader(model, data_loader, device)
    
    def _get_representation(self, model, device='cpu', **kwargs):
        if 'data_loader' in kwargs:
            return self._get_representation_dataloader(model, device=device, **kwargs)
        elif 'dataset' in kwargs:
            return self._get_representation_dataset(model, device=device, **kwargs)

    def compute(self, model, device='cpu', **kwargs):
        """Computes the disentanglement metrics for the given model and data loader.

        Args:
            model (torch.nn.Module): The model to evaluate.
            data_loader (torch.utils.data.DataLoader): The data loader containing the dataset.
            device (torch.device): The device to perform computations on.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        latent_reps, gt_factors = self._get_representation(model, device=device, **kwargs)

        results = {}
        progress_bar = tqdm(self.metrics, desc="Computing metrics")
        for metric in progress_bar:
            metric_name = metric['name']
            progress_bar.set_description(f"Computing {metric_name}")
            metric_args = metric.get('args', {})
            metric_obj = select_metric(metric_name, **metric_args)
            result = metric_obj(latent_reps, gt_factors)
            results[metric_name] = result
        
        return results




