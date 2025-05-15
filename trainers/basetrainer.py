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
from utils.io import load_chkpt, save_chkpt, create_chkpt
from utils.reproducibility import set_deterministic_run
import uuid
from utils.helpers import create_load_optimizer, create_load_lr_scheduler
from vae_models.utils import create_load_model
from losses.utils import create_load_loss


class BaseTrainer():

    def __init__(self,
                 model,
                 loss,  # renamed
                 optimizer,
                 lr_scheduler,
                 device,
                 train_id=None,
                 seed = None,
                 determinism_type = None,
                 use_compile_model=False,
                 compile_kwargs={'mode': 'max-autotune', 'backend': 'inductor'},  # Compile options for torch.compile
                 train_step_unit: str = 'epoch',  # Renamed from step_unit
                 ### logging ###
                 is_progress_bar=True,
                 progress_bar_log_iter_interval=50,  # update the progress bar with losses every `progress_bar_log_iter_interval` iterations
                 return_log_loss=False,
                 log_loss_interval_type='iteration', # 'epoch' or 'iteration' 
                 log_loss_iter_interval=100, # logged the losses every `log_loss_iter_interval` iterations if in 'iteration' mode
                ### save chkpt ### 
                 return_chkpt=False,
                 chkpt_save_output_dir=None,
                 chkpt_every_n_steps=None,
                 ):
        """
        Initializes the BaseTrainer.

        Parameters
        ----------
        model : torch.nn.Module
            The neural network model to be trained.
        loss : losses.baseloss.BaseLoss
            The loss function to be used for training.
        optimizer : torch.optim.Optimizer
            The optimizer for updating model parameters.
        lr_scheduler : torch.optim.lr_scheduler._LRScheduler
            The learning rate scheduler.
        device : torch.device
            The device (CPU or GPU) on which to perform training.
        seed : int, optional
            Random seed for reproducibility. If provided, `determinism_type` must also be provided.
            Defaults to None.
        determinism_type : str, optional
            Type of determinism to enforce ('full', 'seed_only', 'cudnn_only').
            Required if `seed` is provided. Defaults to None.
        use_compile_model : bool, optional
            Whether to compile the model using `torch.compile` for potential performance improvements.
            Defaults to False.
        compile_kwargs : dict, optional
            Keyword arguments to pass to `torch.compile` if `use_compile_model` is True.
            Defaults to `{'mode': 'max-autotune', 'backend': 'inductor'}`.
        train_step_unit : str, optional
            Specifies the unit for `max_steps` in the `train` method.
            Can be either 'epoch' or 'iteration'. Defaults to 'epoch'.
        
        ### Logging ###
        -----------
        is_progress_bar : bool, optional
            Whether to display a progress bar during training. Defaults to True.
        progress_bar_log_iter_interval : int, optional
            The interval (in iterations) at which to update the progress bar with loss information.
            Defaults to 50.
        return_log_loss : bool, optional
            Whether the `train` method should return a log of losses. Defaults to False.
        log_loss_interval_type : str, optional
            Specifies the interval type for logging losses if `return_log_losses` is True.
            Can be 'epoch' or 'iteration'. Defaults to 'epoch'.
        log_loss_iter_interval : int, optional
            The interval (in iterations) at which to log losses if `log_loss_interval_type` is 'iteration'.
            Defaults to 50. 
    
        ### Checkpointing ###
        -------------------
        return_chkpt : bool, optional
            Whether the `train` method should return a list of checkpoints from trainer. Defaults to False.
        chkpt_save_output_dir : str, optional
            Directory where checkpoint files will be saved. Defaults to None.
        chkpt_every_n_steps : int, optional
            Create checkpoint every N training steps (epochs or iterations, depending on `train_step_unit`).
            Including the last step.
            If set default to None, it will only create a checkpoint at the end of training.
            Defaults to None.
        """
        if train_id is None:
            # Generate a new UUID for the training session
            self.train_id = uuid.uuid4()
        else:
            self.train_id = train_id

        self.seed = seed
        self.determinism_type = determinism_type

        self.device = device
        self.model = model.to(self.device)
        self.loss = loss  # renamed
        self.optimizer = optimizer

        self.use_compile_model = use_compile_model
        self.compile_kwargs = compile_kwargs

        self.is_progress_bar = is_progress_bar
        self.progress_bar_log_iter_interval = progress_bar_log_iter_interval
        self.return_log_loss = return_log_loss
        self.log_loss_interval_type = log_loss_interval_type
        self.log_loss_iter_interval = log_loss_iter_interval

        self.return_chkpt = return_chkpt
        self.chkpt_save_output_dir = chkpt_save_output_dir
        self.chkpt_every_n_steps = chkpt_every_n_steps
        self.use_chkpt = return_chkpt or (chkpt_save_output_dir is not None)

        if seed is not None:
            if determinism_type is None:
                raise ValueError("If seed is provided, determinism_type must also be provided.")
            
        if seed is None and determinism_type is not None:
            raise ValueError("If determinism_type is provided, seed must also be provided.")

        if self.use_compile_model:
            if seed is not None and determinism_type is not None:
                raise ValueError("Determinism is not supported with torch.compile. Please set seed and determinism_type to None.")
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

        ## determinism ##
        if self.seed is not None:
            set_deterministic_run(self.seed, self.determinism_type)
        
        self.chkpt_list = []

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

                # checkpoint #
                self._save_checkpoint_if_needed(
                    step=epoch + 1,
                    total_steps=num_epochs
                )

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
                for iter in range(total_iterations):
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
                        if (iter + 1) % self.log_loss_iter_interval == 0 or (iter + 1) == total_iterations:
                            if current_interval_logs: # Ensure there are logs to process
                                mean_interval_logs = {k: np.mean(v) for k, v in current_interval_logs.items() if v}
                                mean_interval_logs['iteration'] = iter + 1 # Add iteration number
                                all_logs.append(mean_interval_logs)
                                current_interval_logs = collections.defaultdict(list) # Reset for next interval

                    # Update progress bar
                    if (iter + 1) % self.progress_bar_log_iter_interval == 0 or (iter + 1) == total_iterations:
                        recent_logs = {k: np.mean(v[-(self.progress_bar_log_iter_interval):])
                                       for k, v in iteration_to_log.items() if v}
                        t.set_postfix(**recent_logs)
                    t.update()

                    # checkpoints
                    self._save_checkpoint_if_needed(
                        step=iter + 1,
                        total_steps=total_iterations
                    )

                    # Step the scheduler after each iteration if iteration-based training
                    self.lr_scheduler.step()

        self.model.eval()
        return all_logs

    def _save_checkpoint_if_needed(self, step, total_steps):
        """
        Handles checkpoint creation and saving logic for both epoch and iteration training.

        Parameters
        ----------
        step : int
            Current step (epoch or iteration, 1-based).
        total_steps : int
            Total number of steps (epochs or iterations).
        """
        if not self.use_chkpt:
            return

        should_save = False
        if self.chkpt_every_n_steps is not None:
            if (step % self.chkpt_every_n_steps == 0) or (step == total_steps):
                should_save = True
        else:
            if step == total_steps:
                should_save = True

        if should_save:

            chkpt = create_chkpt(
                train_id=self.train_id,
                train_step_num=step,
                train_step_unit=self.train_step_unit,
                train_seed=self.seed,
                train_determinism_type=self.determinism_type,
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                loss=self.loss,
            )

            self.chkpt_list.append(chkpt)
            if self.chkpt_save_output_dir is not None:
                save_chkpt(chkpt, self.chkpt_save_output_dir, self.train_id, step)

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

        if self.loss.mode == 'post_forward':
            model_out = self.model(samples)
            inputs = {
                'data': samples,
                'is_train': self.model.training,
                **model_out,
            }

            loss_out = self.loss(**inputs)
            loss = loss_out['loss']

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        elif self.loss.mode == 'pre_forward':
            inputs = {
                'model': self.model,
                'data': samples,
                'is_train': self.model.training
            }
            loss_out = self.loss(**inputs)
            loss = loss_out['loss']

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        elif self.loss.mode == 'optimizes_internally':
            loss_out = self.loss(samples, self.model, self.optimizer)
            loss = loss_out['loss']
        else:
            raise ValueError(f"Unknown loss function mode: {self.loss.mode}")

        # Extract any logged metrics and return both loss and logs
        to_log = loss_out.get('to_log', {})
        loss_val = loss.item() if loss is not None else 0.0
        
        return {"loss": loss_val, "to_log": to_log}
    

def create_trainer_from_chkpt(ckpt, 
                              additional_trainer_kwargs=None, 
                              new_model=None, 
                              new_loss=None,
                              new_optimizer=None,
                              new_lr_scheduler=None,
                              new_device=None,
                              ):
    """
    Creates a trainer instance from a checkpoint, with optional replacement of components.

    Parameters
    ----------
    ckpt: dict
        A dictionary containing the checkpoint data.
    additional_trainer_kwargs: dict, optional
        Additional keyword arguments to pass to the BaseTrainer constructor.
    new_model: torch.nn.Module, optional
        If provided, use this model instead of loading from checkpoint.
    new_loss: losses.baseloss.BaseLoss, optional
        If provided, use this loss function instead of loading from checkpoint.
    new_optimizer: torch.optim.Optimizer, optional
        If provided, use this optimizer instead of loading from checkpoint.
    new_lr_scheduler: torch.optim.lr_scheduler._LRScheduler, optional
        If provided, use this learning rate scheduler instead of loading from checkpoint.
    new_device: torch.device, optional
        If provided, use this device instead of the one stored in the checkpoint.

    Returns
    -------
    BaseTrainer:
        An instance of the BaseTrainer class initialized with components from the checkpoint
        or with the provided new components.
        if the new_model, new_loss, new_optimizer, or new_lr_scheduler are not None, train_id will be set to a new UUID.
    """
    train_id = ckpt['train_id']
    model_chkpt = ckpt['model']
    loss_chkpt = ckpt['loss']
    optimizer_chkpt = ckpt['optimizer']
    lr_scheduler_chkpt = ckpt['lr_scheduler']
    device = new_device if new_device is not None else ckpt['device']

    if new_model is not None or new_loss is not None or new_optimizer is not None or new_lr_scheduler is not None:
        train_id = uuid.uuid4()
    else:
        train_id = ckpt['train_id']

    if new_model is not None:
        model = new_model
    else:
        model = create_load_model(
            model_chkpt['name'],
            model_chkpt['kwargs'],
            model_chkpt['state_dict']
        )

    if new_loss is not None:
        loss = new_loss
    else:
        loss = create_load_loss(
            loss_chkpt['name'],
            loss_chkpt['kwargs'],
            loss_chkpt['state_dict']
        )

    if new_optimizer is not None:
        optimizer = new_optimizer
    else:
        optimizer = create_load_optimizer(
            optimizer_chkpt['name'],
            optimizer_chkpt['kwargs'],
            optimizer_chkpt['state_dict']
        )

    if new_lr_scheduler is not None:
        lr_scheduler = new_lr_scheduler
    else:
        lr_scheduler = create_load_lr_scheduler(
            name=lr_scheduler_chkpt['name'],
            kwargs=lr_scheduler_chkpt['kwargs'],
            state_dict=lr_scheduler_chkpt['state_dict'],
            optimizer=optimizer
        )

    trainer = BaseTrainer(
        model=model,
        loss=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        train_id=train_id,
        seed=ckpt['train_seed'],
        determinism_type=ckpt['train_determinism_type'],
        use_compile_model=False,  # Assuming compile is not needed for loading
        train_step_unit=ckpt['train_step_unit'],
        **(additional_trainer_kwargs or {})
    )

    return trainer
