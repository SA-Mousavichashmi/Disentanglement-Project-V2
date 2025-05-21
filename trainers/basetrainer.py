"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import numpy as np
import torch
from tqdm import trange
import collections
from utils.io import create_chkpt
import uuid
from utils.helpers import get_model_device
from collections import OrderedDict

class BaseTrainer():

    def __init__(self,
                 model,
                 loss,
                 optimizer,
                 lr_scheduler=None,
                 train_id=None,
                 determinism_kwargs=None,
                 use_torch_compile=False,  # Renamed
                 torch_compile_kwargs={'mode': 'max-autotune', 'backend': 'inductor'},  # Renamed
                 prev_train_iter=0,
                 dataloader=None,
                 # logging args
                 is_progress_bar=True,
                 progress_bar_log_iter_interval=50,
                 log_loss_interval_type='iter',
                 use_train_logging=True,
                 log_loss_iter_interval=200,
                 return_log_loss=True,
                 prev_train_losses_log=None,
                 prev_train_metrics_log=None,
                 # checkpointing args
                 return_chkpt=False,
                 chkpt_every_n_steps=None,
                 chkpt_step_type='iter',   # <--- new parameter
                 chkpt_save_path=None,
                 chkpt_save_dir=None,
                 chkpt_save_master_dir=None,
                 ):
        """
        Initializes the BaseTrainer.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be trained.
        loss : losses.baseloss.BaseLoss
            The loss function.
        optimizer : torch.optim.Optimizer
            Optimizer for model parameters.
        lr_scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            Scheduler for learning rate updates. Defaults to None (constant LR).
        train_id : uuid.UUID or str, optional
            Unique identifier for this training run. Generated if None.
        determinism_kwargs : dict, optional
            Settings for reproducibility (e.g., seed, determinism_type).
        use_torch_compile : bool, optional
            If True, compiles the model with `torch.compile`. Defaults to False.
        torch_compile_kwargs : dict, optional
            Arguments passed to `torch.compile`. Defaults to {'mode': 'max-autotune', 'backend': 'inductor'}.
        prev_train_iter : int, optional
            The number of training iterations completed before this run. Used to resume training.
        dataloader : torch.utils.data.DataLoader, optional
            DataLoader to use for training.

        Logging Parameters
        ------------------
        is_progress_bar : bool, optional
            Enable progress bar. Defaults to True.
        progress_bar_log_iter_interval : int, optional
            Iterations between progress bar updates. Defaults to 50.
        return_log_loss : bool, optional
            Return logged losses from `train()`. Defaults to True.
        log_loss_interval_type : {'epoch','iter'}, optional
            Granularity for loss logging. Defaults to 'iter'.
        log_loss_iter_interval : int, optional
            Iterations between logged loss records when using iteration-level logging. Defaults to 100.

        Checkpointing Parameters
        ------------------------
        return_chkpt : bool, optional
            Return checkpoint dicts from `train()`. Defaults to False.
        chkpt_every_n_steps : int, optional
            Interval for checkpoint creation (epochs or iterations). None = only final. Defaults to None.
        chkpt_save_path : str, optional
            File path to save final checkpoint. Exclusive with other save options.
        chkpt_save_dir : str, optional
            Directory to save checkpoints. Exclusive with other save options.
        chkpt_save_master_dir : str, optional
            Master directory for organized checkpoints. Exclusive with other save options.
        """
        self._validate_init_params(
            use_torch_compile=use_torch_compile,  # Renamed
            determinism_kwargs=determinism_kwargs,
            log_loss_interval_type=log_loss_interval_type,
            chkpt_step_type=chkpt_step_type,
            chkpt_save_path=chkpt_save_path,
            chkpt_save_dir=chkpt_save_dir,
            chkpt_save_master_dir=chkpt_save_master_dir,
            chkpt_every_n_steps=chkpt_every_n_steps
        )

        if train_id is None:
            # Generate a new UUID for the training session
            self.train_id = uuid.uuid4()
        else:
            self.train_id = train_id

        self.determinism_kwargs = determinism_kwargs

        self.loss = loss  # renamed
        self.optimizer = optimizer
        self.model = model
        self.device = get_model_device(model)

        self.use_torch_compile = use_torch_compile  # Renamed
        self.torch_compile_kwargs = torch_compile_kwargs  # Renamed

        if self.use_torch_compile:  # Renamed
            self.model = torch.compile(self.model, **self.torch_compile_kwargs)  # Renamed
        
        self.dataloader = dataloader

        self.prev_train_iter = prev_train_iter
        self.current_train_iter = prev_train_iter if prev_train_iter is not None else 0
        
        if self.dataloader is not None:
            self.current_train_epoch = self.prev_train_iter / len(self.dataloader) 
        else:
            self.current_train_epoch = 0

        self.is_progress_bar = is_progress_bar
        self.progress_bar_log_iter_interval = progress_bar_log_iter_interval

        self.use_train_logging = use_train_logging
        self.log_loss_interval_type = log_loss_interval_type
        self.log_loss_iter_interval = log_loss_iter_interval
        self.return_log_loss = return_log_loss
        self.train_losses_log = prev_train_losses_log if prev_train_losses_log is not None else []
        self.train_metrics_log = prev_train_metrics_log if prev_train_metrics_log is not None else []

        self.return_chkpt = return_chkpt
        self.chkpt_save_path = chkpt_save_path
        self.chkpt_save_dir = chkpt_save_dir
        self.chkpt_save_master_dir = chkpt_save_master_dir
        self.chkpt_every_n_steps = chkpt_every_n_steps
        self.chkpt_step_type = chkpt_step_type           # <--- store it
        self.use_chkpt = return_chkpt or (chkpt_save_dir is not None) or (chkpt_save_master_dir is not None) or (chkpt_save_path is not None)
        self.chkpt_list = []
        self.chkpt_train_losses_log = []
        self.chkpt_train_metrics_log = []

        if lr_scheduler is None:  # Renamed from scheduler
            ### Using constant scheduler with no warmup
            self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer,
                factor=1.0,  # Factor of 1.0 means no adjustment to learning rate
                total_iters=0  # Apply the factor immediately and keep it constant
            )
        else:
            self.lr_scheduler = lr_scheduler  # Renamed from scheduler

    def _validate_init_params(
        self,
        use_torch_compile,  # Renamed
        determinism_kwargs,
        log_loss_interval_type,
        chkpt_step_type,
        chkpt_save_path,
        chkpt_save_dir,
        chkpt_save_master_dir,
        chkpt_every_n_steps
    ):
        """
        Validates the parameters passed to the BaseTrainer constructor.
        """
        ##### Assertions #####

        if use_torch_compile:  # Renamed
            if determinism_kwargs is not None:
                if determinism_kwargs.get('determinism_type') is not None:
                    raise ValueError("Determinism should be used with torch.compile.")
                if determinism_kwargs.get('cublas_workspace_config') is not None:
                    raise ValueError("CUBLAS_WORKSPACE_CONFIG should be used with torch.compile.")

        #### Logging assertions ####
        if log_loss_interval_type not in ['epoch', 'iter']:
            raise ValueError("log_loss_interval_type must be either 'epoch' or 'iter'")
        # validate step_type
        if chkpt_step_type not in ['epoch', 'iter']:           # <--- validate
            raise ValueError("chkpt_step_type must be either 'epoch' or 'iter'")

        #### Checkpointing assertions ####
        # Ensure at most one of chkpt_save_path, chkpt_save_dir, or chkpt_save_master_dir is set
        save_path_set = chkpt_save_path is not None
        save_dir_set = chkpt_save_dir is not None
        save_master_dir_set = chkpt_save_master_dir is not None

        if sum([save_path_set, save_dir_set, save_master_dir_set]) > 1:
             raise ValueError("At most one of chkpt_save_path, chkpt_save_dir, or chkpt_save_master_dir can be set.")

        # If chkpt_save_path is set, chkpt_every_n_steps must be None
        if chkpt_save_path is not None and chkpt_every_n_steps is not None:
            raise ValueError("chkpt_every_n_steps cannot be set when chkpt_save_path is used," \
            " as only the final checkpoint is saved at final step.")

    def train(self, step_unit, max_steps: int, dataloader=None):
        """
        Trains the model for a specified number of steps (epochs or iterations).

        Parameters
        ----------
        step_unit : str
            Unit of training steps, either 'epoch' or 'iteration'.
        max_steps : int
            Number of steps to train for, interpreted as epochs or iterations depending on `step_unit`.
        dataloader : torch.utils.data.DataLoader, optional
            DataLoader to use for training. If None, uses the dataloader provided at initialization.

        Returns
        -------
        dict:
            Dictionary containing training logs under the key 'logs'. If checkpointing is enabled and
            `self.return_chkpt` is True, also includes a list of checkpoints under the key 'chkpts'.
        """
        assert step_unit in ['epoch', 'iter'], "step_unit must be either 'epoch' or 'iter'"

        # Use the dataloader provided to train(), or fall back to self.dataloader
        if dataloader is None:
            dataloader = self.dataloader
        if dataloader is None:
            raise ValueError("Either dataloader in the constructor or data_loader in train() must be provided.")

        self.model.train()

        # Determine total iterations (epochs→iterations if needed)
        num_batches = len(dataloader)
        is_last_batch_full = len(dataloader.dataset) % dataloader.batch_size == 0
        total_iterations = max_steps * num_batches if step_unit == 'epoch' else max_steps
        # total_epoch = max_steps if step_unit == 'epoch' else max_steps // 

        iteration_to_log = collections.defaultdict(list)
        current_interval_logs = collections.defaultdict(list)
        data_iterator = iter(dataloader)
        approx_epochs = total_iterations / num_batches if num_batches > 0 else float('inf')
        
        kwargs = dict(
            desc=f"Training for {total_iterations} iter, {approx_epochs:.2f} epochs",
            total=total_iterations,
            leave=False,
            disable=not self.is_progress_bar
        )

        with trange(total_iterations, **kwargs) as t:
            for it in t:
                try:
                    data = next(data_iterator)[0]
                except StopIteration:
                    data_iterator = iter(dataloader)
                    data = next(data_iterator)[0]

                iter_out = self._train_iteration(data)

                prog_bar_log = OrderedDict()
                prog_bar_log['iter'] = f'{self.current_train_iter}/{self.prev_train_iter + total_iterations}'
                prog_bar_log['epoch'] = f'{(self.current_train_iter / num_batches):.2f}/{((self.prev_train_iter + total_iterations) / num_batches):.2f}'

                # accumulate logs
                for key, val in iter_out['to_log'].items():
                    iteration_to_log[key].append(val)
                    if self.use_train_logging and self.log_loss_interval_type == 'iter':
                        current_interval_logs[key].append(val)

                # progress bar update
                if (it + 1) % self.progress_bar_log_iter_interval == 0 or (it + 1) == total_iterations:
                    recent = {k: np.mean(v[-self.progress_bar_log_iter_interval:]) 
                              for k, v in iteration_to_log.items() if v}
                        
                    prog_bar_log.update(recent)
                    t.set_postfix(**prog_bar_log)
                    
                t.update()

                # scheduler step
                self.lr_scheduler.step()

                # checkpoint & logging at epoch or iteration boundaries based on chkpt_step_type
                if (it + 1) % num_batches == 0:
                    
                    epoch_num = (it + 1) // num_batches

                    if self.use_train_logging and self.log_loss_interval_type == 'epoch':
                        mean_epoch = {k: np.mean(v) for k, v in iteration_to_log.items()}
                        mean_epoch['epoch'] = epoch_num
                        mean_epoch['iter'] = it + 1
                        self.train_losses_log.append(mean_epoch)

                    if self.chkpt_step_type == 'epoch':                        # <--- gate here
                        self._save_checkpoint_if_needed(
                            step=epoch_num,
                            total_steps=max_steps,
                            dataloader=dataloader,
                            chkpt_train_losses_log=iter_out['to_log'],
                            chkpt_train_metrics_log=None # TODO: Add metrics logging
                        )
                
                if self.use_train_logging and self.log_loss_interval_type == 'iter' and \
                    ((it + 1) % self.log_loss_iter_interval == 0 or (it + 1) == total_iterations):
                    mean_it = {k: np.mean(v) for k, v in current_interval_logs.items() if v}
                    mean_it['iter'] = it + 1
                    mean_it['epoch'] = (it + 1) / num_batches

                    self.train_losses_log.append(mean_it)
                    current_interval_logs = collections.defaultdict(list)

                if self.chkpt_step_type == 'iter':                   # <--- gate here
                    self._save_checkpoint_if_needed(
                        step=it + 1,
                        total_steps=total_iterations,
                        dataloader=dataloader,
                        chkpt_train_losses_log=iter_out['to_log'],
                        chkpt_train_metrics_log=None # TODO: Add metrics logging
                    )

                if self.chkpt_step_type == 'epoch' and (not is_last_batch_full) and \
                     (it + 1) == total_iterations and self.use_chkpt:
                    
                    self._save_checkpoint(
                        dataloader=dataloader,
                        chkpt_train_losses_log=iter_out['to_log'],
                        chkpt_train_metrics_log=None # TODO: Add metrics logging
                    )
                      
        
        self.model.eval()

        return {'logs': {
                    'train_losses_log': self.train_losses_log,
                    'train_metrics_log': self.train_metrics_log,
                }, 
                'chkpts': self.chkpt_list}
    
    def _save_checkpoint_if_needed(self, 
                                   step, 
                                   total_steps, 
                                   dataloader, 
                                   chkpt_train_losses_log, 
                                   chkpt_train_metrics_log
                                   ):
        """
        Handles checkpoint creation and saving logic for both epoch and iteration training.

        Parameters
        ----------
        step : int
            Current step (epoch or iteration, 1-based).
        total_steps : int
            Total number of steps (epochs or iterations).
        dataloader : torch.utils.data.DataLoader
            DataLoader used for training.
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
            self._save_checkpoint(
                dataloader=dataloader, 
                chkpt_train_losses_log=chkpt_train_losses_log,
                chkpt_train_metrics_log=chkpt_train_metrics_log
                )

    def _save_checkpoint(self, dataloader, chkpt_train_losses_log, chkpt_train_metrics_log):
        """
        Saves the current state of the model, optimizer, and other components to a checkpoint.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            DataLoader used for training.
        """
        chkpt = create_chkpt(
            train_id=self.train_id,
            train_iter_num=self.current_train_iter,
            train_determinism_kwargs=self.determinism_kwargs,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            dataset=dataloader.dataset,  # Use dataloader.dataset directly
            dataloader=dataloader,
            loss=self.loss,
            use_torch_compile=self.use_torch_compile,  # Renamed
            train_losses_log=self.train_losses_log,
            train_metrics_log=self.train_metrics_log,
            chkpt_train_losses_log=chkpt_train_losses_log,
            chkpt_metrics_log=chkpt_train_metrics_log,
            # --- Pass checkpointing args ---
            chkpt_every_n_steps=self.chkpt_every_n_steps,
            chkpt_step_type=self.chkpt_step_type,
            chkpt_save_path=self.chkpt_save_path,
            chkpt_save_dir=self.chkpt_save_dir,
            chkpt_save_master_dir=self.chkpt_save_master_dir,
            # --- Pass logging args ---
            is_progress_bar=self.is_progress_bar,
            progress_bar_log_iter_interval=self.progress_bar_log_iter_interval,
            log_loss_interval_type=self.log_loss_interval_type,
            use_train_logging=self.use_train_logging,
            log_loss_iter_interval=self.log_loss_iter_interval,
            return_log_loss=self.return_log_loss,
        )

        if self.return_chkpt:
            self.chkpt_list.append(chkpt)

        if self.chkpt_save_path is not None:
            print(f"Saving checkpoint to {self.chkpt_save_path}")
            torch.save(chkpt, self.chkpt_save_path)

            if not self.return_chkpt:
                del chkpt

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
        
        self.current_train_iter += 1  # Increment cumulative iteration counter
        return {"loss": loss_val, "to_log": to_log}

