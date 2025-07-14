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
    'reconstruction_error',
    'kld'
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
    if name == 'kld':
        return metrics.KLD(**kwargs)

class MetricAggregator:
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

    def _compute_reconstruction_metric_dataloader(self, model, data_loader, metric_obj, device='cpu'):
        """Compute reconstruction metric using data loader.

        Args:
            model (torch.nn.Module): The model to evaluate.
            data_loader (torch.utils.data.DataLoader): The data loader containing the dataset.
            metric_obj: The reconstruction metric object.
            device (torch.device): The device to perform computations on.

        Returns:
            float: Average reconstruction error across all batches.
        """
        model.to(device)
        model.eval()
        total_error = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Computing reconstruction error"):
                inputs, _ = batch
                inputs = inputs.to(device)
                
                # Get reconstructions using mean representation
                reconstructions = model.reconstruct(inputs, mode='mean')
                
                # Compute batch error with sum reduction to get total error for batch
                batch_error = metric_obj(reconstructions, inputs)
                total_error += batch_error.item()
                total_samples += inputs.size(0)

        return total_error / total_samples

    def _compute_reconstruction_metric_dataset(self, model, dataset, metric_obj, seed, sample_num=None, device='cpu'):
        """Compute reconstruction metric using dataset.

        Args:
            model (torch.nn.Module): The model to evaluate.
            dataset (torch.utils.data.Dataset): The dataset containing the data.
            metric_obj: The reconstruction metric object.
            seed (int): Random seed for reproducible sampling.
            sample_num (int, optional): Number of samples to use. If None, use entire dataset.
            device (torch.device): The device to perform computations on.

        Returns:
            float: Average reconstruction error across all batches.
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

        return self._compute_reconstruction_metric_dataloader(model, data_loader, metric_obj, device)

    def _compute_reconstruction_metric(self, model, metric_obj, device='cpu', **kwargs):
        """Compute reconstruction metric with either dataloader or dataset format.

        Args:
            model (torch.nn.Module): The model to evaluate.
            metric_obj: The reconstruction metric object.
            device (torch.device): The device to perform computations on.
            **kwargs: Additional arguments containing either 'data_loader' or 'dataset'.

        Returns:
            float: Average reconstruction error across all batches.
        """
        if 'data_loader' in kwargs:
            return self._compute_reconstruction_metric_dataloader(model, kwargs['data_loader'], metric_obj, device)
        elif 'dataset' in kwargs:
            return self._compute_reconstruction_metric_dataset(model, kwargs['dataset'], metric_obj, kwargs.get('seed', 42), kwargs.get('sample_num'), device)
        else:
            raise ValueError("Either 'data_loader' or 'dataset' must be provided for reconstruction metrics")

    def _compute_kld_metric_dataloader(self, model, data_loader, metric_obj, device='cpu'):
        """Compute KLD metric using data loader.

        Args:
            model (torch.nn.Module): The model to evaluate.
            data_loader (torch.utils.data.DataLoader): The data loader containing the dataset.
            metric_obj: The KLD metric object.
            device (torch.device): The device to perform computations on.

        Returns:
            dict: Dictionary with averaged KLD values for all dimensions and total KL.
        """
        model.to(device)
        model.eval()
        total_kld = {}
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Computing KLD"):
                inputs, _ = batch
                inputs = inputs.to(device)
                
                # Get encoder statistics (mean and logvar)
                stats = model.encoder(inputs)['stats_qzx']
                
                # Compute batch KLD - returns a dictionary with KL_i and KL keys
                batch_kld_dict = metric_obj(stats_qzx=stats)
                
                # Accumulate all KLD components
                for key, value in batch_kld_dict.items():
                    if key not in total_kld:
                        total_kld[key] = 0.0
                    total_kld[key] += value
                
                total_samples += inputs.size(0)

        # Average all KLD components across all samples
        averaged_kld = {}
        for key, total_value in total_kld.items():
            averaged_kld[key] = total_value / total_samples

        return averaged_kld

    def _compute_kld_metric_dataset(self, model, dataset, metric_obj, seed, sample_num=None, device='cpu'):
        """Compute KLD metric using dataset.

        Args:
            model (torch.nn.Module): The model to evaluate.
            dataset (torch.utils.data.Dataset): The dataset containing the data.
            metric_obj: The KLD metric object.
            seed (int): Random seed for reproducible sampling.
            sample_num (int, optional): Number of samples to use. If None, use entire dataset.
            device (torch.device): The device to perform computations on.

        Returns:
            dict: Dictionary with averaged KLD values for all dimensions and total KL.
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

        return self._compute_kld_metric_dataloader(model, data_loader, metric_obj, device)

    def _compute_kld_metric(self, model, metric_obj, device='cpu', **kwargs):
        """Compute KLD metric with either dataloader or dataset format.

        Args:
            model (torch.nn.Module): The model to evaluate.
            metric_obj: The KLD metric object.
            device (torch.device): The device to perform computations on.
            **kwargs: Additional arguments containing either 'data_loader' or 'dataset'.

        Returns:
            float: Average KLD across all batches.
        """
        if 'data_loader' in kwargs:
            return self._compute_kld_metric_dataloader(model, kwargs['data_loader'], metric_obj, device)
        elif 'dataset' in kwargs:
            return self._compute_kld_metric_dataset(model, kwargs['dataset'], metric_obj, kwargs.get('seed', 42), kwargs.get('sample_num'), device)
        else:
            raise ValueError("Either 'data_loader' or 'dataset' must be provided for KLD metrics")

    def compute(self, model, device='cpu', **kwargs):
        """Computes the disentanglement metrics for the given model and data loader.

        Args:
            model (torch.nn.Module): The model to evaluate.
            data_loader (torch.utils.data.DataLoader): The data loader containing the dataset.
            device (torch.device): The device to perform computations on.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        # Separate different types of metrics
        reconstruction_metric = next((m for m in self.metrics if m['name'] == 'reconstruction_error'), None)
        kld_metric = next((m for m in self.metrics if m['name'] == 'kld'), None)
        other_metrics = [m for m in self.metrics if m['name'] not in ['reconstruction_error', 'kld']]

        results = {}

        # Handle other metrics that require latent representations
        if other_metrics:
            latent_reps, gt_factors = self._get_representation(model, device=device, **kwargs)

            progress_bar = tqdm(other_metrics, desc="Computing metrics")
            for metric in progress_bar:
                metric_name = metric['name']
                progress_bar.set_description(f"Computing {metric_name}")
                metric_args = metric.get('args', {})
                metric_obj = select_metric(metric_name, **metric_args)
                result = metric_obj(latent_reps, gt_factors)
                results[metric_name] = result

        # Handle reconstruction metrics separately
        if reconstruction_metric:
            metric_name = reconstruction_metric['name']
            metric_args = reconstruction_metric.get('args', {})
            metric_obj = select_metric(metric_name, **metric_args)
            result = self._compute_reconstruction_metric(model, metric_obj, device, **kwargs)
            results[metric_name] = result
        
        # Handle KLD metric separately
        if kld_metric:
            metric_name = kld_metric['name']
            metric_args = kld_metric.get('args', {})
            metric_obj = select_metric(metric_name, **metric_args)
            result = self._compute_kld_metric(model, metric_obj, device, **kwargs)
            results[metric_name] = result

        return results




