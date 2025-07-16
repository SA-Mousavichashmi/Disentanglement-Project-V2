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
import scipy
import scipy.stats
import sklearn.metrics
import sklearn.preprocessing
import torch
from tqdm import trange
import math

from .basemetric import BaseMetric

class DCI(BaseMetric):
    def __init__(self, num_train=None, num_test=None, split_ratio=None, backend='sklearn', num_workers=8, **kwargs):
        super().__init__(**kwargs)

        if split_ratio is not None:
            if num_train is not None or num_test is not None:
                raise ValueError("Cannot specify both split_ratio and num_train/num_test.")
            if not (0 < split_ratio <= 1.0):
                 raise ValueError("split_ratio must be between 0 (exclusive) and 1 (inclusive).")
            self.num_train = None # Will be calculated later
            self.num_test = None # Will be calculated later
        elif num_train is not None and num_test is not None:
             if num_train <= 0 or num_test < 0:
                 raise ValueError("num_train must be positive and num_test must be non-negative.")
             self.num_train = num_train
             self.num_test = num_test
        else:
            raise ValueError("Must specify either split_ratio or both num_train and num_test.")

        self.split_ratio = split_ratio

        if backend == 'sklearn':
            import sklearn.ensemble
            self.prediction_model = sklearn.ensemble.GradientBoostingClassifier
        elif backend == 'sklearn_forest':
            import sklearn.ensemble
            self.prediction_model = sklearn.ensemble.RandomForestClassifier(n_jobs=num_workers)
        else:
            import xgboost
            self.prediction_model = xgboost.XGBClassifier
        self.num_workers = num_workers

    @property
    def _requires(self):
        return ['latent_reps', 'gt_factors']

    @property
    def _mode(self):
        return 'full'

    def __call__(self, latent_reps, gt_factors, **kwargs):
        """Compute Disentanglement, Completeness and Informativness [1].

        References
        ----------
           [1] Eastwood et al. "A Framework for the Quantitative Evaluation of Disentangled
           Representations", International Conferences on Learning Representations, 2018.

        Parameters
        ----------
        latent_reps: torch.Tensor or np.ndarray
            The latent representations of the data. shape: (num_samples, num_latents).
        gt_factors: torch.Tensor or np.ndarray
            The ground truth factors corresponding to the representations. shape: (num_samples, num_factors).
        """

        if isinstance(latent_reps, torch.Tensor):
            latent_reps = latent_reps.detach().cpu().numpy()

        if isinstance(gt_factors, torch.Tensor):
            gt_factors = gt_factors.detach().cpu().numpy()

        total_samples = len(latent_reps)

        if self.split_ratio is not None:
            num_train = math.ceil(total_samples * self.split_ratio)
            if self.split_ratio == 1.0:
                num_test = 0
            else:
                # Ensure at least one sample for testing if ratio is not 1
                num_test = max(1, total_samples - num_train)
                # Adjust num_train if necessary to not exceed total samples
                num_train = total_samples - num_test
        else:
            num_train = self.num_train
            num_test = self.num_test


        if total_samples < num_train + num_test:
            raise ValueError(
                f'Number of train- and test-samples {num_train}/{num_test} exceed total number of samples [{total_samples}]')


        total_idcs = list(range(total_samples))

        if num_test > 0:
            train_idcs = np.random.choice(total_idcs, num_train, replace=False)
            test_idcs_pool = list(set(total_idcs) - set(list(train_idcs)))
            # Ensure we don't request more test samples than available
            actual_num_test = min(num_test, len(test_idcs_pool))
            test_idcs = np.random.choice(test_idcs_pool, actual_num_test, replace=False)
        else: # num_test is 0
            train_idcs = np.random.choice(total_idcs, num_train, replace=False)
            test_idcs = train_idcs.copy() # Test on training data as per original logic when num_test=0

        gt_factors = sklearn.preprocessing.minmax_scale(gt_factors)
        latent_reps = sklearn.preprocessing.minmax_scale(latent_reps)

        for i in range(gt_factors.shape[-1]):
            uv = np.unique(gt_factors[:, i])
            dc = {val: i for i, val in enumerate(uv)}
            def dmap(val):
                return dc[val]
            out = list(map(dmap, gt_factors[:, i]))
            gt_factors[:, i] = np.array(out)
        gt_factors = gt_factors.astype(int)

        latent_reps_train = latent_reps[train_idcs]
        gt_factors_train = gt_factors[train_idcs]
        latent_reps_test = latent_reps[test_idcs]
        gt_factors_test = gt_factors[test_idcs]  

        num_latents = latent_reps.shape[-1]
        num_factors = gt_factors.shape[-1]
        importance_scores = np.zeros([num_latents, num_factors])

        def get_importance(factor_id):
            pred_model = self.prediction_model()
            
            pred_model.fit(latent_reps_train, gt_factors_train[:, factor_id])
            
            importance_scores = np.abs(pred_model.feature_importances_)
            train_preds = pred_model.predict(latent_reps_train)
            test_preds = pred_model.predict(latent_reps_test)

            train_score = np.mean(train_preds == gt_factors_train[:, factor_id])
            test_score = np.mean(test_preds == gt_factors_test[:, factor_id])

            train_err = 1 - train_score
            test_err = 1 - test_score
            return [factor_id, importance_scores, train_score, test_score, train_err, test_err]

        # Use sequential processing when num_workers is 1, otherwise use parallel processing
        if self.num_workers == 1:
            print('Using sequential processing for importance score computation.')
            res = [get_importance(factor_id) for factor_id in trange(num_factors, leave=False)]
        else:
            res = Parallel(n_jobs=self.num_workers)(delayed(get_importance)(factor_id) for factor_id in trange(num_factors, leave=False))

        importance_scores = np.zeros([num_latents, num_factors])
        informativeness_train_scores = []
        informativeness_test_scores = []
        informativeness_train_errors = []
        informativeness_test_errors = []
        for factor_id, importance_score_factor, train_score, test_score, train_err, test_err in res:
            importance_scores[:, factor_id] = importance_score_factor
            informativeness_train_scores.append(train_score)
            informativeness_test_scores.append(test_score)
            informativeness_train_errors.append(train_err)
            informativeness_test_errors.append(test_err)
        informativeness_train_scores = np.mean(informativeness_train_scores)
        informativeness_test_scores = np.mean(informativeness_test_scores)
        informativeness_train_errors = np.mean(informativeness_train_errors)
        informativeness_test_errors = np.mean(informativeness_test_errors)

        per_latent_disentanglement = 1. - scipy.stats.entropy(importance_scores.T + 1e-11, base=importance_scores.shape[1])
        per_factor_completeness = 1. - scipy.stats.entropy(importance_scores + 1e-11, base=importance_scores.shape[0])        
        if importance_scores.sum() == 0.:
            importance_scores = np.ones_like(importance_scores)
        total_latent_disentanglement = importance_scores.sum(axis=1) / importance_scores.sum()
        total_factor_completeness = importance_scores.sum(axis=0) / importance_scores.sum()
        disentanglement = np.sum(per_latent_disentanglement * total_latent_disentanglement)
        completeness = np.sum(per_factor_completeness * total_factor_completeness)

        # return {
        #     'disentanglement': disentanglement,
        #     'completeness': completeness,
        #     'informativeness_train_scores': informativeness_train_scores,
        #     'informativeness_test_scores': informativeness_test_scores,
        #     'informativeness_train_errors': informativeness_train_errors,
        #     'informativeness_test_errors': informativeness_test_errors,
        # }

        return {
            'd': disentanglement,
            'c': completeness,
            'i': informativeness_test_scores
        }
