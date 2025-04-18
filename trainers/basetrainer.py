"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import trange  # Added import

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
        Whether to use a progress bar for training (Note: progress bar functionality removed).
    """

    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 loss_fn,
                 device,
                 is_progress_bar=True): 

        self.device = device
        self.model = model.to(self.device)
        self.loss_f = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.is_progress_bar = is_progress_bar  # Store is_progress_bar

    def train(self, data_loader, epochs): 
        """
        Trains the model.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        epochs: int
            Number of epochs to train the model for.
        """
        self.model.train()

        for epoch in range(epochs): 
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
        mean_epoch_loss: float
            Mean loss per image
        """
        epoch_losses = []
        # Added kwargs for trange
        kwargs = dict(desc="Epoch {}".format(epoch + 1),
                      leave=False,
                      disable=not self.is_progress_bar)
        # Wrap loop with trange
        with trange(len(data_loader), **kwargs) as t:
            for _, data_out in enumerate(data_loader):
                data = data_out[0]
                iter_loss = self._train_iteration(data)
                epoch_losses.append(iter_loss)
                # Update progress bar postfix and step
                t.set_postfix(loss=iter_loss)
                t.update()

        return np.mean(epoch_losses)

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

        if self.loss_f.mode == 'post_forward':
            model_out = self.model(samples)
            inputs = {
                'data': samples,
                'is_train': self.model.training,
                **model_out,
            }

            loss_out = self.loss_f(**inputs)
            loss = loss_out['loss']

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        elif self.loss_f.mode == 'pre_forward':
            inputs = {
                'model': self.model,
                'data': samples,
                'is_train': self.model.training
            }
            loss_out = self.loss_f(**inputs)
            loss = loss_out['loss']

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        elif self.loss_f.mode == 'optimizes_internally':
            loss_out = self.loss_f(samples, self.model, self.optimizer)
            loss = loss_out['loss']

        return loss.item() if loss is not None else 0.0
