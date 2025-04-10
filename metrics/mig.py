"""
Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
Copyright 2018 The DisentanglementLib Authors.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

ORIGINAL CODE WAS CHANGED AS FOLLOWS:
- Conversion from Tensorflow to PyTorch.
- Integration as a mergable BaseMetric that can be combined with multiple other metrics for efficient computation.
- Efficiency improvements through parallelization.
- Function and variable renaming.
- Moved mutual information and entropy calculation to utils.math module.
"""
from joblib import Parallel, delayed
import numpy as np
import sklearn.preprocessing
import torch
from tqdm import trange

from .basemetric import BaseMetric
from utils.math import calculate_mutual_information, calculate_entropy

class MIG(BaseMetric):
    def __init__(self, num_bins=20, num_workers=8, mi_method='sklearn', entropy_method='scipy', **kwargs):
        super().__init__(**kwargs)
        self.num_bins = num_bins
        self.num_workers = num_workers
        self.mi_method = mi_method
        self.entropy_method = entropy_method

    @property   
    def _requires(self):
        return ['latent_reps', 'gt_factors']

    @property
    def _mode(self):
        return 'full'
    
    def __call__(self, latent_reps, gt_factors, **kwargs):
        """Compute the mutual information gap as in [1].

        References
        ----------
           [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
           autoencoders." Advances in Neural Information Processing Systems. 2018.
        """
        if isinstance(latent_reps, torch.Tensor):
            latent_reps = latent_reps.detach().cpu().numpy()
        
        if isinstance(gt_factors, torch.Tensor):
            gt_factors = gt_factors.detach().cpu().numpy()

        num_latents = latent_reps.shape[-1]
        num_factors = gt_factors.shape[-1]

        latent_reps = sklearn.preprocessing.minmax_scale(latent_reps)
        gt_factors = sklearn.preprocessing.minmax_scale(gt_factors)

        bins = np.linspace(0, 1, self.num_bins + 1)
        latent_reps = np.digitize(latent_reps, bins[:-1], right=False).astype(int)        
        gt_factors = np.digitize(gt_factors, bins[:-1], right=False).astype(int)

        def compute_mutual_info(latent_reps, gt_factors):
            factor_mutual_info_scores = []
            for latent_id in range(num_latents):
                factor_mutual_info_scores.append(
                    calculate_mutual_information(latent_reps[:, latent_id], gt_factors, method=self.mi_method)
                )
            sorted_factor_mutual_info_scores = sorted(factor_mutual_info_scores)
            mutual_info_gap = sorted_factor_mutual_info_scores[-1] - sorted_factor_mutual_info_scores[-2]
            factor_entropy = calculate_entropy(gt_factors, method=self.entropy_method)
            normalized_mutual_info_gap = 1. / factor_entropy * mutual_info_gap
            return normalized_mutual_info_gap
        
        if self.num_workers == 1:
            mig = []
            for factor_id in trange(num_factors, desc='Computing MI for ground truth factors...', leave=False):
                mig.append(compute_mutual_info(latent_reps, gt_factors[:, factor_id])) 
        else:
            mig = Parallel(n_jobs=self.num_workers)(
                delayed(compute_mutual_info)(latent_reps, gt_factors[:, factor_id]) 
                for factor_id in trange(num_factors, desc='Computing MI for ground truth factors...', leave=False)
            )

        return np.mean(mig)
