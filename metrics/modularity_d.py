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
- Moved mutual information calculation to utils.math module.
"""
from joblib import Parallel, delayed
import numpy as np
import torch
import sklearn.preprocessing
from tqdm import trange

from .basemetric import BaseMetric
from utils.math import calculate_mutual_information

class Modularityd(BaseMetric): # TODO: check the correctness of this metric the result is not valid acroos different disentanglement level
    def __init__(self, num_bins=20, num_workers=8, mi_method='sklearn', **kwargs):
        super().__init__(**kwargs)
        self.num_bins = num_bins
        self.num_workers = num_workers
        self.mi_method = mi_method

    @property
    def _requires(self):
        return ['latent_reps', 'gt_factors']

    @property
    def _mode(self):
        return 'full'

    def __call__(self, latent_reps, gt_factors, **kwargs):
        """Compute Modularity Score as proposed in [1] (eq.2).

        References
        ----------
           [1] Ridgeway et al. "Learning deep disentangled embeddings with the f-statistic loss", 
           Advances in Neural Information Processing Systems. 2018.
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

        def compute_mutual_info_for_factor(factor_id):
            factor_mutual_info_scores = []
            for latent_id in range(num_latents):
                factor_mutual_info_scores.append(
                    calculate_mutual_information(latent_reps[:, latent_id], gt_factors[:, factor_id], method=self.mi_method)
                )
            return factor_mutual_info_scores
        
        if self.num_workers == 1:
            mutual_info_scores = []
            for factor_id in trange(num_factors, desc='Computing MI for ground truth factors...', leave=False):
                mutual_info_scores.append(compute_mutual_info_for_factor(factor_id))
        else:
            mutual_info_scores = Parallel(n_jobs=self.num_workers)(
                delayed(compute_mutual_info_for_factor)(factor_id) 
                for factor_id in trange(num_factors, desc='Computing MI for ground truth factors...', leave=False)
            )
            
        # Convert the results to the expected numpy array shape for further processing
        mutual_info_scores = np.array(mutual_info_scores)
        
        modularity = 0
        for latent_id in range(num_latents):
            factor_mutual_info_scores = mutual_info_scores[:, latent_id]
            max_mutual_info_idx = np.argmax(factor_mutual_info_scores)

            deviation_from_ideal = 0
            for factor_id, factor_mutual_info in enumerate(factor_mutual_info_scores):
                if factor_id != max_mutual_info_idx:
                    deviation_from_ideal += factor_mutual_info ** 2

            normalization = factor_mutual_info_scores[max_mutual_info_idx] ** 2 * (num_factors - 1)
            modularity += 1 - deviation_from_ideal / normalization
        
        return modularity / num_latents
