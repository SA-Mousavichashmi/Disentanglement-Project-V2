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
"""
from joblib import Parallel, delayed
import numpy as np
from pyitlib import discrete_random_variable as drv
import sklearn.preprocessing
import sklearn.metrics
import sklearn.feature_selection
import torch
from tqdm import trange
from scipy import stats

from .basemetric import BaseMetric

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
    
    # TODO the mutual information methods and entropy methods should be moved to a math file in the utils folder 
    def _mutual_information_pyitlib(self, x, y):
        """Calculate mutual information using pyitlib."""
        return drv.information_mutual(x, y, cartesian_product=True, base=np.e)
    
    def _mutual_information_sklearn(self, x, y):
        """Calculate mutual information using sklearn."""
        return sklearn.metrics.mutual_info_score(x, y)
    
    def _mutual_information_numpy(self, x, y):
        """Calculate mutual information using numpy-based implementation for already digitized data."""
        # Find unique values in x and y
        unique_x = np.unique(x)
        unique_y = np.unique(y)
        
        # Create a contingency table/joint counts matrix based on actual values
        joint_counts = np.zeros((len(unique_x), len(unique_y)), dtype=np.int64)
        
        # Create mappings from values to indices
        x_map = {val: idx for idx, val in enumerate(unique_x)}
        y_map = {val: idx for idx, val in enumerate(unique_y)}
        
        # Populate joint counts
        for i, j in zip(x, y):
            joint_counts[x_map[i], y_map[j]] += 1
        
        # Calculate joint probability
        total = len(x)
        joint_prob = joint_counts / total
        
        # Calculate marginal probabilities
        px = np.sum(joint_prob, axis=1)
        py = np.sum(joint_prob, axis=0)
        
        # Calculate mutual information
        mi = 0
        for i in range(len(px)):
            for j in range(len(py)):
                if joint_prob[i, j] > 0:
                    mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (px[i] * py[j]))
        return mi
    
    # TODO the mutual information methods and entropy methods should be moved to a math file in the utils folder
    def _entropy_pyitlib(self, x):
        """Calculate entropy using pyitlib."""
        return drv.entropy(x, base=np.e)
    
    def _entropy_scipy(self, x):
        """Calculate entropy using scipy."""
        value, counts = np.unique(x, return_counts=True)
        probs = counts / len(x)
        return stats.entropy(probs, base=np.e)
    
    def _entropy_numpy(self, x):
        """Calculate entropy using numpy-based implementation."""
        value, counts = np.unique(x, return_counts=True)
        probs = counts / len(x)
        return -np.sum(probs * np.log(probs + 1e-10))
    
    def calculate_mutual_information(self, x, y):
        """Calculate mutual information using the selected method."""
        if self.mi_method == 'pyitlib':
            return self._mutual_information_pyitlib(x, y)
        elif self.mi_method == 'sklearn':
            return self._mutual_information_sklearn(x, y)
        elif self.mi_method == 'numpy':
            return self._mutual_information_numpy(x, y)
        else:
            raise ValueError(f"Unknown mutual information method: {self.mi_method}")
    
    def calculate_entropy(self, x):
        """Calculate entropy using the selected method."""
        if self.entropy_method == 'pyitlib':
            return self._entropy_pyitlib(x)
        elif self.entropy_method == 'scipy':
            return self._entropy_scipy(x)
        elif self.entropy_method == 'numpy':
            return self._entropy_numpy(x)
        else:
            raise ValueError(f"Unknown entropy method: {self.entropy_method}")

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
                    self.calculate_mutual_information(latent_reps[:, latent_id], gt_factors))
            sorted_factor_mutual_info_scores = sorted(factor_mutual_info_scores)
            mutual_info_gap = sorted_factor_mutual_info_scores[-1] - sorted_factor_mutual_info_scores[-2]
            factor_entropy = self.calculate_entropy(gt_factors)
            normalized_mutual_info_gap = 1. / factor_entropy * mutual_info_gap
            return normalized_mutual_info_gap
        
        if self.num_workers == 1:
            mig = []
            for factor_id in trange(num_factors, desc='Computing MI for ground truth factors...', leave=False):
                mig.append(compute_mutual_info(latent_reps, gt_factors[:, factor_id])) 
        else:
            mig = Parallel(n_jobs=self.num_workers)(delayed(compute_mutual_info)(latent_reps, gt_factors[:, factor_id]) for factor_id in trange(num_factors, desc='Computing MI for ground truth factors...', leave=False))

        return np.mean(mig)
