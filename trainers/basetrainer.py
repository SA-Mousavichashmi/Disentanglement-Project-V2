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

    def __init__(self,
                 model,
                 loss_fn,
                 optimizer,
                 lr_scheduler,
                 device,
                 use_compile_model=False,
                 compile_kwargs={'mode': 'max-autotune', 'backend': 'inductor'},  # Compile options for torch.compile
                 train_step_unit: str = 'epoch',  # Renamed from step_unit
                 is_progress_bar=True,
                 progress_bar_log_iter_interval=50,  # update the progress bar with losses every `progress_bar_log_iter_interval` iterations
                 return_log_loss=False,
                 log_loss_interval_type='epoch', # 'epoch' or 'iteration' 
                 log_loss_iter_interval=50, # logged the losses every `log_loss_iter_interval` iterations if in 'iteration' mode
                 checkpoint_input_file_path=None,
                 checkpoint_output_dir='./',
                 checkpoint_every_n_steps = 0,
                 checkpoint_only_metric_improved = False,
                 ):
        """
        Initializes the BaseTrainer.

        Parameters
        ----------
        model : torch.nn.Module
            The neural network model to be trained.
        loss_fn : losses.baseloss.BaseLoss
            The loss function to be used for training.
        optimizer : torch.optim.Optimizer
            The optimizer for updating model parameters.
        lr_scheduler : torch.optim.lr_scheduler._LRScheduler
            The learning rate scheduler.
        device : torch.device
            The device (CPU or GPU) on which to perform training.
        use_compile_model : bool, optional
            Whether to compile the model using `torch.compile` for potential performance improvements.
            Defaults to False.
        compile_kwargs : dict, optional
            Keyword arguments to pass to `torch.compile` if `use_compile_model` is True.
            Defaults to `{'mode': 'max-autotune', 'backend': 'inductor'}`.
        train_step_unit : str, optional
            Specifies the unit for `max_steps` in the `train` method.
            Can be either 'epoch' or 'iteration'. Defaults to 'epoch'.
        is_progress_bar : bool, optional
            Whether to display a progress bar during training. Defaults to True.
        progress_bar_log_iter_interval : int, optional
            The interval (in iterations) at which to update the progress bar with loss information.
            Defaults to 50.
        return_log_loss : bool, optional
            Whether the `train` method should return a log of losses. Defaults to False.
        log_loss_interval_type : str, optional
            Specifies the interval type for logging losses if `return_log_loss` is True.
            Can be 'epoch' or 'iteration'. Defaults to 'epoch'.
        log_loss_iter_interval : int, optional
            The interval (in iterations) at which to log losses if `log_loss_interval_type` is 'iteration'.
            Defaults to 50.
        """
        self.device = device
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.use_compile_model = use_compile_model
        self.compile_kwargs = compile_kwargs

        self.is_progress_bar = is_progress_bar
        self.progress_bar_log_iter_interval = progress_bar_log_iter_interval
        self.return_log_loss = return_log_loss
        self.log_loss_interval_type = log_loss_interval_type
        self.log_loss_iter_interval = log_loss_iter_interval

        if self.use_compile_model:
            self.model = torch.compile(self.model, **self.compile_kwargs)

        if train_step_unit not in ['epoch', 'iteration']:
            raise ValueError("train_step_unit must be either 'epoch' or 'iteration'")
        self.train_step_unit = train_step_unit  # Renamed from step_unit

        if log_loss_interval_type not in ['epoch', 'iteration']:
            raise ValueError("log_loss_interval_type must be either 'epoch' or 'iteration'")

        if self.train_step_unit == 'iteration' and self.log_loss_interval_type == 'epoch':
             raise ValueError("When train_step_unit is 'iteration', log_loss_interval_type must also be 'iteration'")

        if lr_scheduler is None:  # Renamed from scheduler
            ### Using constant scheduler with no warmup
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: 1.0,
            )
        else:
            self.lr_scheduler = lr_scheduler  # Renamed from scheduler
    
    def load_checkpoint(self, checkpoint_input_file_path):
        """
        Loads a model checkpoint.

        Parameters
        ----------
        checkpoint_input_file_path : str
            Path to the checkpoint file.
        """
        checkpoint = torch.load(checkpoint_input_file_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from {checkpoint_input_file_path}")
    
    def save_checkpoint(self, checkpoint_output_dir, step):
        """
        Saves the model checkpoint.

        Parameters
        ----------
        checkpoint_output_dir : str
            Directory to save the checkpoint.
        step : int
            Current training step (epoch or iteration).
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'epoch': step,
        }
        torch.save(checkpoint, f"{checkpoint_output_dir}/checkpoint_{step}.pth")
        print(f"Checkpoint saved at {checkpoint_output_dir}/checkpoint_{step}.pth")
    
    def train(self, data_loader, max_steps: int):
        """
        Trains the model based on the mode specified in the constructor.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        max_steps: int
            The total number of steps (epochs or iterations) to train for, based on `train_step_unit`.

        Returns
        -------
        list or None:
            If `return_log_loss` is True, returns a list of dictionaries containing
            the mean logged losses at specified intervals. Otherwise, returns None.
        """
        self.model.train()
        all_logs = [] if self.return_log_loss else None

        if self.train_step_unit == 'epoch':  # Renamed from step_unit
            # --- Epoch-based training ---
            num_epochs = max_steps
            for epoch in range(num_epochs):
                self.epoch = epoch
                # _train_epoch returns dict if log_loss_interval_type=='epoch', list if 'iteration'
                epoch_logs_out = self._train_epoch(data_loader, epoch)

                if self.return_log_loss:
                    if self.log_loss_interval_type == 'epoch':
                        # epoch_logs_out already includes 'epoch' key from _train_epoch
                        all_logs.append(epoch_logs_out) # Append dict
                    elif self.log_loss_interval_type == 'iteration':
                        # epoch_logs_out is a list of dicts, each includes 'iteration' key
                        all_logs.extend(epoch_logs_out) # Extend with list of dicts

                # Assuming scheduler steps per epoch if epoch-based training
                self.lr_scheduler.step()

        elif self.train_step_unit == 'iteration':  # Renamed from step_unit
            # --- Iteration-based training ---
            total_iterations = max_steps
            iteration_to_log = collections.defaultdict(list) # For progress bar
            current_interval_logs = collections.defaultdict(list) # For return logs

            # Get the initial iterator for the DataLoader
            data_iterator = iter(data_loader)
            num_batches_per_epoch = len(data_loader) # Get number of batches per epoch
            approx_epochs = total_iterations / num_batches_per_epoch if num_batches_per_epoch > 0 else float('inf')

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

                    # Accumulate logs for progress bar
                    for key, item in iter_out['to_log'].items():
                        iteration_to_log[key].append(item)

                    # Accumulate logs for returning if needed
                    if self.return_log_loss:
                        # log_loss_interval_type must be 'iteration' here due to __init__ check
                        for key, item in iter_out['to_log'].items():
                            current_interval_logs[key].append(item)

                        # Check if interval is complete or it's the last iteration
                        if (i + 1) % self.log_loss_iter_interval == 0 or (i + 1) == total_iterations:
                            if current_interval_logs: # Ensure there are logs to process
                                mean_interval_logs = {k: np.mean(v) for k, v in current_interval_logs.items() if v}
                                mean_interval_logs['iteration'] = i + 1 # Add iteration number
                                all_logs.append(mean_interval_logs)
                                current_interval_logs = collections.defaultdict(list) # Reset for next interval

                    # Update progress bar
                    if (i + 1) % self.progress_bar_log_iter_interval == 0 or (i + 1) == total_iterations:
                        recent_logs = {k: np.mean(v[-(self.progress_bar_log_iter_interval):])
                                       for k, v in iteration_to_log.items() if v}
                        t.set_postfix(**recent_logs)
                    t.update()

                    # Step the scheduler after each iteration if iteration-based training
                    self.lr_scheduler.step()

        self.model.eval()
        return all_logs

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
        dict or list:
            If `return_log_loss` is True and `log_loss_interval_type` is 'iteration',
            returns a list of dictionaries containing mean logs per interval within the epoch.
            Otherwise, returns a single dictionary containing the mean of logged values
            for the entire epoch.
        """
        epoch_to_log = collections.defaultdict(list) # For overall epoch mean / progress bar
        num_batches = len(data_loader)

        # Variables for interval logging if needed
        log_intervals = self.return_log_loss and self.log_loss_interval_type == 'iteration'
        all_interval_logs = [] if log_intervals else None
        current_interval_logs = collections.defaultdict(list) if log_intervals else None

        kwargs = dict(desc="Epoch {}".format(epoch + 1),
                      leave=False,
                      disable=not self.is_progress_bar)

        with trange(num_batches, **kwargs) as t:
            for i, data_out in enumerate(data_loader):
                data = data_out[0]
                iter_out = self._train_iteration(data)

                # Accumulate logs for overall epoch / progress bar
                for key, item in iter_out['to_log'].items():
                    epoch_to_log[key].append(item)

                # Accumulate logs for intervals if needed
                if log_intervals:
                    for key, item in iter_out['to_log'].items():
                        current_interval_logs[key].append(item)

                    # Check if interval is complete or it's the last iteration of the epoch
                    if (i + 1) % self.log_loss_iter_interval == 0 or (i + 1) == num_batches:
                         if current_interval_logs: # Ensure there are logs to process
                            mean_interval_logs = {k: np.mean(v) for k, v in current_interval_logs.items() if v}
                            mean_interval_logs['iteration'] = i + 1 # Add iteration number for this interval
                            all_interval_logs.append(mean_interval_logs)
                            current_interval_logs = collections.defaultdict(list) # Reset

                # Update progress bar postfix
                if (i + 1) % self.progress_bar_log_iter_interval == 0 or (i + 1) == num_batches:
                    interval_mean_logs = {key: np.mean(item[-(self.progress_bar_log_iter_interval):])
                                          for key, item in epoch_to_log.items() if item}
                    t.set_postfix(**interval_mean_logs)
                t.update()

        if log_intervals:
            return all_interval_logs # Return list of interval log dicts
        else:
            # Return dict of overall epoch mean logs
            epoch_mean_logs = {key: np.mean(item) for key, item in epoch_to_log.items()}
            epoch_mean_logs['epoch'] = epoch + 1 # Add epoch number
            return epoch_mean_logs

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
    

# class TrainerWithCheckpoints(BaseTrainer):
#     def __init__(self,
#                  train_mode,
#                  checkpoint_input_dir=None,
#                  checkpoint_input_file_path=None,
#                  checkpoint_output_dir=None,
#                  **kwargs
#                  ):
        
#         super().__init__(
#             **kwargs
#         )

#         assert train_mode in ['from-scratch', 'resume-ckpt-new', 'resume-ckpt', 'resume-ckpt-dir'], \
#             f"train_mode must be one of ['from-scratch', 'resume-ckpt-new', 'resume-ckpt', 'resume-ckpt-dir'], but got {train_mode}"

#         assert train_mode == 'from-scratch' and checkpoint_input_dir is not None and checkpoint_input_file_path is not None

#         self.train_mode = train_mode
