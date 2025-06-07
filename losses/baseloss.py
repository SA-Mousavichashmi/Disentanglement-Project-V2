# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2019 Yann Dubois, Aleco Kastanos, Dave Lines, Bart Melman
# Copyright (c) 2018 Schlumberger
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Dict, Optional, Any, List
from utils.scheduler import  get_scheduler

class BaseLoss(abc.ABC):
    """
    Base class for losses.

    Parameters
    ----------
    record_loss_every: int, optional
        Every how many steps to record the loss.

    rec_dist: {"bernoulli", "gaussian", "laplace"}, optional
        Reconstruction distribution of the likelihood on each pixel.
        Implicitly defines the reconstruction loss. Bernoulli corresponds to a
        binary cross entropy (bse), Gaussian corresponds to MSE, Laplace
        corresponds to L1.

    mode: str
        Determines how the loss operates: if 'post_forward', it only takes standard model forward outputs/data points
        and returns a loss. If 'pre_forward', it takes in the model and performs respective forward computations itself.
        If 'optimizes_internally', takes model, does forward computations AND backward updates.
        
    schedulers_kwargs: list of dict, optional
        List of dictionaries, where each dictionary contains keyword arguments for initializing a
        `BaseHyperparameterScheduler`.
    """

    def __init__(self, mode, rec_dist="bernoulli", schedulers_kwargs=None, **kwargs):
        
        assert mode in ["post_forward", "pre_forward", "optimizes_internally"], \
            f"Invalid mode: {mode}. Choose from 'post_forward', 'pre_forward', or 'optimizes_internally'."
        
        assert rec_dist in ["bernoulli", "gaussian", "laplace"], \
            f"Invalid rec_dist: {rec_dist}. Choose from 'bernoulli', 'gaussian', or 'laplace'."

        self.rec_dist = rec_dist
        self.mode = mode
        self.schedulers = self._init_schedulers(schedulers_kwargs) if schedulers_kwargs is not None else None
    
    def _init_schedulers(self, schedulers_kwargs: Optional[List[Dict[str, Any]]] = None):
        schedulers = {}
        if schedulers_kwargs:
            for kwargs in schedulers_kwargs:
                scheduler_name = kwargs['name']
                scheduler = get_scheduler(scheduler_name, **kwargs['kwargs'])
                schedulers[scheduler.param_name] = scheduler         
        return schedulers

    def step_schedulers(self):
        """Step all schedulers and update corresponding attributes."""
        for param_name, scheduler in self.schedulers.items():
            new_value = scheduler.step()
            setattr(self, param_name, new_value)
    
    def get_scheduler_values(self) -> Dict[str, float]:
        """Get current values from all schedulers."""
        return {param_name: scheduler.get_value() 
                for param_name, scheduler in self.schedulers.items()}

    @property
    @abc.abstractmethod
    def name(self):
        """A unique name for the loss function, to be implemented by subclasses."""
        pass

    @property
    @abc.abstractmethod
    def kwargs(self):
        """A dictionary of keyword arguments for the loss function, to be implemented by subclasses."""
        pass

    @abc.abstractmethod
    def state_dict(self):
        """Returns the state dictionary of the loss function."""
        pass

    @abc.abstractmethod
    def load_state_dict(self, state_dict):
        """Loads the state dictionary into the loss function."""
        pass

    @abc.abstractmethod
    def __call__(self, data, reconstructions, stats_qzx, is_train, **kwargs):
        """
        Calculates loss for a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            Input data (e.g. batch of images). Shape : (batch_size, n_chan,
            height, width).

        reconstructions : torch.Tensor
            Reconstructed data. Shape : (batch_size, n_chan, height, width).

        stats_qzx : torch.tensor
            sufficient statistics of the latent dimension. E.g. for gaussian
            (mean, log_var) each of shape : (batch_size, latent_dim).

        is_train : bool
            Whether currently in train mode.

        storer : dict
            Dictionary in which to store important variables for visualization.

        kwargs:
            Loss specific arguments
        """
        pass
