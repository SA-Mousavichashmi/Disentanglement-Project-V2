# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .. import baseloss
from ..reconstruction import reconstruction_loss
from .kl_div import kl_normal_loss

class Loss(baseloss.BaseLoss):
    """
    Compute the Beta-VAE loss as in [1]

    Parameters
    ----------
    beta : float, optional
        Weight of the kl divergence.
    
    log_components : bool, optional
        Whether to log individual KL components.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
        [1] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with
        a constrained variational framework." (2016).
    """

    def __init__(self, beta=1.0, log_components=False, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.log_components = log_components

    def __call__(self, data, reconstructions, stats_qzx, is_train, **kwargs):   
        self._pre_call(is_train)
        if isinstance(stats_qzx, torch.Tensor):
            stats_qzx = stats_qzx.unbind(-1)     

        # 1. Calculate all values first
        rec_loss = reconstruction_loss(data, reconstructions, distribution=self.rec_dist)
        kl_components = kl_normal_loss(*stats_qzx, return_components=True)
        kl_total = kl_components.sum()
        loss = rec_loss + self.beta * kl_total

        # 2. Initialize the dictionary
        log_data = {}

        # 3. Add items in the desired order
        log_data['loss'] = loss.item()
        log_data['rec_loss'] = rec_loss.item()
        log_data['kl_loss'] = kl_total.item()

        if self.log_components:
            # Add individual components last (or wherever you prefer)
            for i, value in enumerate(kl_components):
                 log_data[f'kl_loss_{i}'] = value.item()

        return {'loss': loss, 'to_log': log_data}
