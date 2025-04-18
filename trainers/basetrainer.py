"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import trange
import collections

class BaseTrainer():
    """
    Class to handle training of model.

    Parameters
    ----------
    model: dent.vae.VAE

    optimizer: torch.optim.Optimizer

    scheduler: torch.optim.lr_scheduler._LRScheduler

    loss_f: dent.models.BaseLoss
        Loss function.

    device: torch.device, optional
        Device on which to run the code.

    is_progress_bar: bool, optional
        Whether to use a progress bar for training

    progress_bar_log_interval: int, optional
        Update the progress bar with losses every `progress_bar_log_interval` iterations.
    """

    def __init__(self,
                 model,
                 loss_fn,
                 optimizer,
                 scheduler,
                 device,
                 is_progress_bar=True,
                 progress_bar_log_interval=10  # Add log_interval parameter for updating progress bar based iterations
                 ): 

        self.device = device
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.is_progress_bar = is_progress_bar  # Store is_progress_bar
        self.progress_bar_log_interval = progress_bar_log_interval # Store log_interval

        if scheduler is None:
            ### Using constant scheduler with no warmup
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: 1.0,
            )
        else:
            self.scheduler = scheduler

    def train(self, data_loader, num_epochs): 
        """
        Trains the model.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        num_epochs: int
            Number of epochs to train the model for.
        """
        self.model.train()

        for epoch in range(num_epochs): 
            self.epoch = epoch
            mean_epoch_loss = self._train_epoch(data_loader, epoch)
            self.scheduler.step()

        self.model.eval()

    def _train_epoch(self, data_loader, epoch):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        epoch: int
            Epoch number

        Return
        ------
        epoch_to_log: dict
            Dictionary containing the mean of logged values for the epoch.
        """
        epoch_to_log = collections.defaultdict(list)
        num_batches = len(data_loader) # Get total number of batches

        # Added kwargs for trange
        kwargs = dict(desc="Epoch {}".format(epoch + 1),
                      leave=False,
                      disable=not self.is_progress_bar)
        # Wrap loop with trange
        with trange(num_batches, **kwargs) as t:
            for i, data_out in enumerate(data_loader): # Use enumerate to get iteration index 'i'
                data = data_out[0]
                iter_out = self._train_iteration(data)
                
                for key, item in iter_out['to_log'].items():
                    epoch_to_log[key].append(item)

                # Update progress bar postfix only at progress_bar_log_interval or last iteration
                if (i + 1) % self.progress_bar_log_interval == 0 or (i + 1) == num_batches:
                    t.set_postfix(**iter_out['to_log'])
                t.update()

        return {key: np.mean(item) for key, item in epoch_to_log.items()} # take the mean of the logged values for each epoch

    def _train_iteration(self, samples):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        samples: torch.Tensor
            A batch of data. Shape : (batch_size, channel, height, width).

        Return
        ------
        loss: float
            Loss for the current iteration.
        """
        samples = samples.to(self.device)
        loss = None 

        if self.loss_fn.mode == 'post_forward':
            model_out = self.model(samples)
            inputs = {
                'data': samples,
                'is_train': self.model.training,
                **model_out,
            }

            loss_out = self.loss_fn(**inputs)
            loss = loss_out['loss']

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        elif self.loss_fn.mode == 'pre_forward':
            inputs = {
                'model': self.model,
                'data': samples,
                'is_train': self.model.training
            }
            loss_out = self.loss_fn(**inputs)
            loss = loss_out['loss']

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        elif self.loss_fn.mode == 'optimizes_internally':
            loss_out = self.loss_fn(samples, self.model, self.optimizer)
            loss = loss_out['loss']
        else:
            raise ValueError(f"Unknown loss function mode: {self.loss_fn.mode}")

        # Extract any logged metrics and return both loss and logs
        to_log = loss_out.get('to_log', {})
        loss_val = loss.item() if loss is not None else 0.0
        
        return {"loss": loss_val, "to_log": to_log}
