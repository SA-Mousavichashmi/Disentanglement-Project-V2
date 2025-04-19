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

    train_step_unit: str, optional
        Specifies the unit for `max_steps`. Either 'epoch' or 'iteration'. Defaults to 'epoch'.

    max_steps: int
        The total number of steps (epochs or iterations) to train for, based on `train_step_unit`.

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
                 train_step_unit: str = 'epoch',  # Renamed from step_unit
                 is_progress_bar=True,
                 progress_bar_log_iter_interval=50,  # update the progress bar with losses every `progress_bar_log_iter_interval` iterations
                 ):

        self.device = device
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.is_progress_bar = is_progress_bar
        self.progress_bar_log_iter_interval = progress_bar_log_iter_interval

        if train_step_unit not in ['epoch', 'iteration']:
            raise ValueError("train_step_unit must be either 'epoch' or 'iteration'")
        self.train_step_unit = train_step_unit  # Renamed from step_unit

        if scheduler is None:
            ### Using constant scheduler with no warmup
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: 1.0,
            )
        else:
            self.scheduler = scheduler

    def train(self, data_loader, max_steps: int):
        """
        Trains the model based on the mode specified in the constructor.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        max_steps: int
            The total number of steps (epochs or iterations) to train for, based on `train_step_unit`.
        """
        self.model.train()

        if self.train_step_unit == 'epoch':  # Renamed from step_unit
            # --- Epoch-based training ---
            num_epochs = max_steps
            for epoch in range(num_epochs):
                self.epoch = epoch
                mean_epoch_loss = self._train_epoch(data_loader, epoch)
                # Assuming scheduler steps per epoch if epoch-based training
                self.scheduler.step()

        elif self.train_step_unit == 'iteration':  # Renamed from step_unit
            # --- Iteration-based training ---
            total_iterations = max_steps
            iteration_to_log = collections.defaultdict(list)
            # Get the initial iterator for the DataLoader
            data_iterator = iter(data_loader)
            num_batches_per_epoch = len(data_loader) # Get number of batches per epoch
            approx_epochs = total_iterations / num_batches_per_epoch

            kwargs = dict(desc=f"Training for {total_iterations} iterations ({approx_epochs:.1f} epochs)", 
                          total=total_iterations,
                          leave=False,
                          disable=not self.is_progress_bar)

            with trange(total_iterations, **kwargs) as t:
                for i in range(total_iterations):
                    try:
                        # Get next batch from the current iterator
                        data_out = next(data_iterator)
                    except StopIteration:
                        # DataLoader exhausted, get a new iterator to restart
                        data_iterator = iter(data_loader)
                        data_out = next(data_iterator)

                    data = data_out[0]  # Batch of data instead of label (factor values)
                    iter_out = self._train_iteration(data)

                    for key, item in iter_out['to_log'].items():
                        iteration_to_log[key].append(item)

                    if (i + 1) % self.progress_bar_log_iter_interval == 0 or (i + 1) == total_iterations:
                        recent_logs = {k: np.mean(v[-(self.progress_bar_log_iter_interval):])
                                       for k, v in iteration_to_log.items() if v}
                        t.set_postfix(**recent_logs)
                    t.update()

                    # Step the scheduler after each iteration if iteration-based training
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
        num_batches = len(data_loader)  # Get total number of batches

        # Added kwargs for trange
        kwargs = dict(desc="Epoch {}".format(epoch + 1),
                      leave=False,
                      disable=not self.is_progress_bar)
        # Wrap loop with trange
        with trange(num_batches, **kwargs) as t:
            for i, data_out in enumerate(data_loader):  # Use enumerate to get iteration index 'i'
                data = data_out[0]
                iter_out = self._train_iteration(data)

                for key, item in iter_out['to_log'].items():
                    epoch_to_log[key].append(item)

                # Update progress bar postfix only at progress_bar_log_iter_interval or last iteration
                if (i + 1) % self.progress_bar_log_iter_interval == 0 or (i + 1) == num_batches:
                    # Calculate mean for the last interval
                    interval_mean_logs = {key: np.mean(item[-(self.progress_bar_log_iter_interval):])
                                          for key, item in epoch_to_log.items() if item} # Calculate mean over the last interval
                    t.set_postfix(**interval_mean_logs) # Use interval mean for postfix
                t.update()

        return {key: np.mean(item) for key, item in epoch_to_log.items()}  # Return the overall epoch mean

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
