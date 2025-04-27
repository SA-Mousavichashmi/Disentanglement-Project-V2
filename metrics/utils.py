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

from ..utils import math
from . import metrics

METRICS = [
    'mig',
    'aam',
    'sap_d',
    'dci_d',
    'fos',
    'kld',
    'rand_fos',
    'rand_kld',
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
    if name == 'dci_d':
        return metrics.DCId(**kwargs)
    if name == 'fos':
        return metrics.FoS(**kwargs)
    if name == 'kld':
        return metrics.KLD(**kwargs)
    if name == 'rand_fos':
        return metrics.randFoS(**kwargs)
    if name == 'rand_kld':
        return metrics.randKLD(**kwargs)
    if name == 'modularity_d':
        return metrics.Modularityd(**kwargs)
    if name == 'reconstruction_error':
        return metrics.ReconstructionError(**kwargs)

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
    
    def _get_representation(self, model, data_loader, device='cpu'):
        """Get the latent representation and ground truth factors from the model and data loader.

        Args:
            model (torch.nn.Module): The model to evaluate.
            data_loader (torch.utils.data.DataLoader): The data loader containing the dataset.
            device (torch.device): The device to perform computations on.

        Returns:
            tuple: A tuple containing the latent representations and ground truth factors in cpu
        """
        model.eval()
        latent_reps = []
        gt_factors = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Computing representations"):
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                z = model.encode(inputs)
                latent_reps.append(z.cpu())
                gt_factors.append(labels.cpu())

        return torch.cat(latent_reps), torch.cat(gt_factors)
    
    def compute(self, model, data_loader, device='cpu'):
        """Computes the disentanglement metrics for the given model and data loader.

        Args:
            model (torch.nn.Module): The model to evaluate.
            data_loader (torch.utils.data.DataLoader): The data loader containing the dataset.
            device (torch.device): The device to perform computations on.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        latent_reps, gt_factors = self._get_representation(model, data_loader, device)

        results = {}
        for metric in self.metrics:
            metric_name = metric['name']
            metric_args = metric.get('args', {})
            metric_obj = select_metric(metric_name, **metric_args)
            result = metric_obj(latent_reps, gt_factors)
            results[metric_name] = result
        
        return results




