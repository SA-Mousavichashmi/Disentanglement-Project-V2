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
from utils.io import create_chkpt, check_dir_empty, check_compatibility_chkpt
import uuid
from utils.helpers import get_model_device
from collections import OrderedDict
from utils.reproducibility import set_deterministic_run
import os
import json
from utils.visualize import Visualizer
from pathlib import Path

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
                 # progress bar args
                 is_progress_bar=True,
                 progress_bar_log_iter_interval=50,
                 # train logging args
                 use_train_logging=True,
                 return_logs=True,
                 # loss logging args
                 log_loss_interval_type='iter',
                 log_loss_iter_interval=200,
                 prev_train_losses_log=None,
                 # metrics logging args
                 log_metrics_interval_type='iter', 
                 log_metrics_iter_interval=200,
                 prev_train_metrics_log=None, 
                 # checkpointing args
                 return_chkpt=False,
                 chkpt_every_n_steps=None,
                 chkpt_step_type='iter',
                 chkpt_save_path=None,
                 chkpt_save_dir=None,
                 chkpt_viz=False,  # Deprecated - use visualization config instead
                 # visualization args
                 visualization_config=None,
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
            The number of training iterations completed before this run. Used to resume training. Defaults to 0.
        dataloader : torch.utils.data.DataLoader, optional
            DataLoader to use for training. Defaults to None.

        Logging Parameters
        ------------------
        is_progress_bar : bool, optional
            Enable progress bar. Defaults to True.
        progress_bar_log_iter_interval : int, optional
            Iterations between progress bar updates. Defaults to 50.
        use_train_logging : bool, optional
            If True, enables logging of training losses and metrics. Defaults to True.
        return_logs : bool, optional
            If True, the `train()` method will return logged losses and metrics. Defaults to True.
        log_loss_interval_type : {'epoch','iter'}, optional
            Granularity for loss logging. Defaults to 'iter'.
        log_loss_iter_interval : int, optional
            Iterations between logged loss records when using iteration-level logging. Defaults to 200.
        prev_train_losses_log : list, optional
            A list of previously logged training losses to resume logging. Defaults to None.
        log_metrics_interval_type : {'epoch','iter'}, optional
            Granularity for metrics logging. Defaults to 'iter'.
        log_metrics_iter_interval : int, optional
            Iterations between logged metrics records when using iteration-level logging. Defaults to 200.
        prev_train_metrics_log : list, optional
            A list of previously logged training metrics to resume logging. Defaults to None.

        Checkpointing Parameters
        ------------------------
        return_chkpt : bool, optional
            If True, the `train()` method will return checkpoint dicts. Defaults to False.
        chkpt_every_n_steps : int, optional
            Interval for checkpoint creation (epochs or iterations). None = only final. Defaults to None.
        chkpt_step_type : {'epoch','iter'}, optional
            Granularity for checkpointing. Defaults to 'iter'.
        chkpt_save_path : str, optional
            File path to save final checkpoint. Exclusive with other save options. Defaults to None.
        chkpt_save_dir : str, optional
            Directory to save checkpoints. Exclusive with other save options. Defaults to None.
        chkpt_viz : bool, optional
            If True, saves visualizations (e.g., latent traversals, reconstructions) with checkpoints. Defaults to False.
            DEPRECATED: Use visualization_config instead.
            
        Visualization Parameters
        ------------------------
        visualization_config : VisualizationConfig, optional
            Configuration object for visualization settings. Controls reconstruction plots, latent traversals,
            and other visualizations during training. Defaults to None (no visualizations).
        """
        self._validate_init_params(
            use_torch_compile=use_torch_compile,  # Renamed
            determinism_kwargs=determinism_kwargs,
            log_loss_interval_type=log_loss_interval_type,
            log_metrics_interval_type=log_metrics_interval_type,
            chkpt_step_type=chkpt_step_type,
            chkpt_save_path=chkpt_save_path,
            chkpt_save_dir=chkpt_save_dir,
            chkpt_every_n_steps=chkpt_every_n_steps,
            chkpt_viz=chkpt_viz,  # Pass chkpt_viz here
            visualization_config=visualization_config
        )

        if train_id is None:
            # Generate a new UUID for the training session
            self.train_id = str(uuid.uuid4())
        else:
            self.train_id = train_id

        self.determinism_kwargs = determinism_kwargs

        self.loss = loss
        self.optimizer = optimizer
        self.model = model
        self.device = get_model_device(model)

        self.use_torch_compile = use_torch_compile
        self.torch_compile_kwargs = torch_compile_kwargs

        if self.use_torch_compile:
            self.model = torch.compile(self.model, **self.torch_compile_kwargs)
        
        self.dataloader = dataloader

        self.prev_train_iter = prev_train_iter
        self.current_train_iter = prev_train_iter if prev_train_iter is not None else 0
        
        if self.dataloader is not None:
            self.current_train_epoch = self.prev_train_iter / len(self.dataloader) 
        else:
            self.current_train_epoch = 0

        # logging parameters
        self.is_progress_bar = is_progress_bar
        self.progress_bar_log_iter_interval = progress_bar_log_iter_interval

        self.use_train_logging = use_train_logging

        self.log_loss_interval_type = log_loss_interval_type
        self.log_loss_iter_interval = log_loss_iter_interval

        self.log_metrics_interval_type = log_metrics_interval_type
        self.log_metrics_iter_interval = log_metrics_iter_interval

        self.return_logs = return_logs
        self.train_losses_log = prev_train_losses_log if prev_train_losses_log is not None else []
        self.train_metrics_log = prev_train_metrics_log if prev_train_metrics_log is not None else []

        # checkpointing parameters
        self.return_chkpt = return_chkpt
        self.chkpt_save_path = chkpt_save_path
        self.chkpt_save_dir = chkpt_save_dir
        self.chkpt_every_n_steps = chkpt_every_n_steps
        self.chkpt_step_type = chkpt_step_type 
        self.chkpt_viz = chkpt_viz  # Deprecated
        
        # visualization parameters
        self.visualization_config = visualization_config
        self.visualizer = None  # Will be initialized when needed
        
        # Setup visualization if enabled
        if self.visualization_config is not None and self.visualization_config.enabled:
            self._setup_visualization()

        self.use_chkpt = return_chkpt or (chkpt_save_dir is not None) or (chkpt_save_path is not None)
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
            self.lr_scheduler = lr_scheduler
        
        
        if self.chkpt_save_dir is not None:

            self.base_subfolder_chkpt_name = "chkpt_{chkpt_num}"

            if check_dir_empty(self.chkpt_save_dir):
                self.chkpt_num = 0
                self.chkpt_result_file_name = 'chkpt_result.json'
                Path(os.path.join(self.chkpt_save_dir, self.chkpt_result_file_name)).write_text("{}")
            else:
                # Find the maximum existing checkpoint number
                existing_chkpt_nums = []
                for item in os.listdir(self.chkpt_save_dir):
                    if item.startswith("chkpt_") and os.path.isdir(os.path.join(self.chkpt_save_dir, item)):
                        try:
                            num = int(item.split('_')[1])
                            existing_chkpt_nums.append(num)
                        except (ValueError, IndexError):
                            # Skip items that don't follow the expected naming pattern
                            continue
                
                self.chkpt_num = max(existing_chkpt_nums) if existing_chkpt_nums else 0

    def _validate_init_params(
        self,
        use_torch_compile,
        determinism_kwargs,
        log_loss_interval_type,
        log_metrics_interval_type,
        chkpt_step_type,
        chkpt_save_path,
        chkpt_save_dir,
        chkpt_every_n_steps,
        chkpt_viz,  # Add chkpt_viz here
        visualization_config
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
        
        if log_metrics_interval_type not in ['epoch', 'iter']:
            raise ValueError("log_metrics_interval_type must be either 'epoch' or 'iter'")

        # validate step_type
        if chkpt_step_type not in ['epoch', 'iter']:           # <--- validate
            raise ValueError("chkpt_step_type must be either 'epoch' or 'iter'")

        #### Checkpointing assertions ####
        # Ensure at most one of chkpt_save_path or chkpt_save_dir is set
        save_path_set = chkpt_save_path is not None
        save_dir_set = chkpt_save_dir is not None

        if sum([save_path_set, save_dir_set]) > 1:
             raise ValueError("At most one of chkpt_save_path or chkpt_save_dir can be set.")

        # If chkpt_save_path is set, chkpt_every_n_steps must be None
        if chkpt_save_path is not None and chkpt_every_n_steps is not None:
            raise ValueError("chkpt_every_n_steps cannot be set when chkpt_save_path is used," \
            " as only the final checkpoint is saved at final step.")
        
        if chkpt_viz and chkpt_save_dir is None:
            raise ValueError("chkpt_viz is enabled, but no chkpt_save_dir is set. " \
                             "Please provide a directory to save visualizations.")
                             
        # Validate visualization config
        if visualization_config is not None and visualization_config.enabled:
            if (visualization_config.save_dir is None and 
                chkpt_save_dir is None and 
                not visualization_config.show_plots):
                raise ValueError("Visualization is enabled but no save_dir is provided and " \
                               "show_plots is False. Either provide a save_dir or enable show_plots.")
        

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
        if dataloader is None and self.dataloader is None:
            raise ValueError("No dataloader provided for training. Please provide a dataloader.")
        elif self.dataloader is None:
            self.dataloader = dataloader

        self.model.train()

        # Determine total iterations (epochs→iterations if needed)
        num_batches = len(self.dataloader)
        is_last_batch_full = len(self.dataloader.dataset) % self.dataloader.batch_size == 0
        total_iterations = max_steps * num_batches if step_unit == 'epoch' else max_steps
        # total_epoch = max_steps if step_unit == 'epoch' else max_steps // 

        iteration_to_log = collections.defaultdict(list)
        current_interval_logs = collections.defaultdict(list)
        epoch_accumulated_logs = collections.defaultdict(list) # Accumulates logs for the current epoch

        data_iterator = iter(self.dataloader)
        approx_epochs = total_iterations / num_batches if num_batches > 0 else float('inf')
        
        kwargs = dict(
            desc=f"Training for {total_iterations} iter, {approx_epochs:.2f} epochs",
            total=total_iterations,
            leave=True,  # Changed from False to True
            disable=not self.is_progress_bar
        )

        with trange(total_iterations, **kwargs) as t:
            for it in t:
                try:
                    data = next(data_iterator)[0]
                except StopIteration:
                    data_iterator = iter(self.dataloader)
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
                    if self.use_train_logging and self.log_loss_interval_type == 'epoch':
                        epoch_accumulated_logs[key].append(val) # Accumulate for epoch-level logging

                # progress bar update
                if (it + 1) % self.progress_bar_log_iter_interval == 0 or (it + 1) == total_iterations:
                    recent = {k: np.mean(v[-self.progress_bar_log_iter_interval:]) 
                              for k, v in iteration_to_log.items() if v}
                    
                    if self.loss.schedulers:
                        for param_name in self.loss.schedulers.keys():
                            recent['sched_' + param_name] = getattr(self.loss, param_name)
                        
                    prog_bar_log.update(recent)
                    t.set_postfix(**prog_bar_log)

                # scheduler step
                self.lr_scheduler.step()

                # Step schedulers at the beginning of each iteration
                if  self.loss.schedulers and self.model.training:
                    self.loss.step_schedulers()

                # checkpoint & logging at epoch or iteration boundaries based on chkpt_step_type
                if (it + 1) % num_batches == 0:
                    
                    epoch_num = (it + 1) // num_batches 

                    if self.use_train_logging:

                        if self.log_loss_interval_type == 'epoch':
                            # Calculate mean for the completed epoch and reset accumulator
                            mean_epoch = {k: np.mean(v) for k, v in epoch_accumulated_logs.items() if v}
                            mean_epoch['iter'] = self.current_train_iter
                            mean_epoch['epoch'] = self.current_train_epoch
                            self.train_losses_log.append(mean_epoch)
                            epoch_accumulated_logs = collections.defaultdict(list) # Reset for the next epoch
                        
                        if self.log_metrics_interval_type == 'epoch':
                            metric_log = {}
                            pass  # TODO: Add metrics logging here

                    if self.chkpt_step_type == 'epoch':                        # <--- gate here
                        self._save_checkpoint_if_needed(
                            step=epoch_num,
                            total_steps=max_steps,
                            dataloader=self.dataloader,
                            chkpt_train_losses_log=iter_out['to_log'],
                            chkpt_train_metrics_log=None # TODO: Add metrics logging
                        )
                        
                        # Save visualization if needed (for epoch-based training)
                        if self._should_save_visualization(epoch_num, max_steps, 'epoch'):
                            self._save_visualization()
                
                if self.use_train_logging and self.log_loss_interval_type == 'iter' and \
                    ((it + 1) % self.log_loss_iter_interval == 0 or (it + 1) == total_iterations):
                    mean_it = {k: np.mean(v) for k, v in current_interval_logs.items() if v}
                    mean_it['iter'] = self.current_train_iter
                    mean_it['epoch'] = self.current_train_epoch

                    self.train_losses_log.append(mean_it)
                    current_interval_logs = collections.defaultdict(list)

                if self.chkpt_step_type == 'iter':                   # <--- gate here
                    self._save_checkpoint_if_needed(
                        step=it + 1,
                        total_steps=total_iterations,
                        dataloader=self.dataloader,
                        chkpt_train_losses_log=iter_out['to_log'],
                        chkpt_train_metrics_log=None # TODO: Add metrics logging
                    )
                    
                    # Save visualization if needed (for iteration-based training)
                    if self._should_save_visualization(it + 1, total_iterations, 'iter'):
                        self._save_visualization()

                if self.chkpt_step_type == 'epoch' and (not is_last_batch_full) and \
                     (it + 1) == total_iterations and self.use_chkpt:
                    
                    self._save_checkpoint(
                        dataloader=self.dataloader,
                        chkpt_train_losses_log=iter_out['to_log'],
                        chkpt_train_metrics_log=None # TODO: Add metrics logging
                    )
                      
        
        self.model.eval()
        
        # Save visualization at training end if configured
        if (self.visualization_config is not None and 
            self.visualization_config.enabled and 
            self.visualization_config.save_at_training_end):
            self._save_visualization()

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
                                   chkpt_train_metrics_log,
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
                chkpt_train_metrics_log=chkpt_train_metrics_log,
                )

    def _save_checkpoint(self, 
                         dataloader, 
                         chkpt_train_losses_log, 
                         chkpt_train_metrics_log,
                         ):
        """
        Saves the current state of the model, optimizer, and other components to a checkpoint.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            DataLoader used for training.
        """
        chkpt_save_dir = None
        chkpt_save_path = None

        if self.chkpt_save_path is not None:
            chkpt_save_path = self.chkpt_save_path

        elif self.chkpt_save_dir is not None:
            self.chkpt_num += 1
            subfolder_chkpt_name = self.base_subfolder_chkpt_name.format(chkpt_num=self.chkpt_num)
            chkpt_file_name = f"{subfolder_chkpt_name}.pth"

            # Create the full checkpoint save path
            chkpt_save_dir = os.path.join(self.chkpt_save_dir, subfolder_chkpt_name)
            chkpt_save_path = os.path.join(chkpt_save_dir, chkpt_file_name)
            
            os.makedirs(chkpt_save_dir, exist_ok=True)

        else:
            chkpt_save_path = None

        # Print checkpoint message
        if chkpt_save_path is not None:
            print(f"Creating and saving checkpoint at {chkpt_save_path} at iteration {self.current_train_iter}")
        else:
            print(f"Creating checkpoint at iteration {self.current_train_iter}, but not saving it to disk.")

        chkpt = create_chkpt(
            train_id=self.train_id,
            train_iter_num=self.current_train_iter,
            train_determinism_kwargs=self.determinism_kwargs,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            dataset=dataloader.dataset,  # Use dataloader.dataset directly
            dataloader=dataloader,            loss=self.loss,
            use_torch_compile=self.use_torch_compile,  # Renamed
            torch_compile_kwargs=self.torch_compile_kwargs,  # Added
            train_losses_log=self.train_losses_log,
            train_metrics_log=self.train_metrics_log,
            chkpt_train_losses_log=chkpt_train_losses_log,
            chkpt_metrics_log=chkpt_train_metrics_log,
            # --- Pass checkpointing args ---
            chkpt_every_n_steps=self.chkpt_every_n_steps,
            chkpt_step_type=self.chkpt_step_type,
            chkpt_save_path=self.chkpt_save_path,
            chkpt_save_dir=self.chkpt_save_dir,
            return_chkpt=self.return_chkpt, # Added
            chkpt_viz=self.chkpt_viz, # Added
            # --- Pass logging args ---
            is_progress_bar=self.is_progress_bar,
            progress_bar_log_iter_interval=self.progress_bar_log_iter_interval,
            log_loss_interval_type=self.log_loss_interval_type,
            use_train_logging=self.use_train_logging,
            log_loss_iter_interval=self.log_loss_iter_interval,
            log_metrics_interval_type=self.log_metrics_interval_type, # Added
            log_metrics_iter_interval=self.log_metrics_iter_interval, # Added
            return_logs=self.return_logs,
        )

        if chkpt_save_path is not None:
            torch.save(
                chkpt,
                chkpt_save_path
            ) 

        if self.chkpt_viz and chkpt_save_dir is not None:
            # Deprecated: use visualization_config instead
            print("Warning: chkpt_viz is deprecated. Use visualization_config instead.")
            self._save_visualization(chkpt_save_dir)
            
        # Save visualization with checkpoint if configured
        if (self.visualization_config is not None and 
            self.visualization_config.enabled and 
            self.visualization_config.save_with_checkpoints and 
            chkpt_save_dir is not None):
            self._save_visualization(chkpt_save_dir)

        if self.chkpt_save_dir is not None:
            chkpt_result_filepath = os.path.join(self.chkpt_save_dir, self.chkpt_result_file_name)
            with open(chkpt_result_filepath, 'r') as f:
                chkpt_result = json.load(f)

            # Save only the current iteration's logs instead of entire history
            chkpt_result[self.chkpt_num] = {
                'train_losses': chkpt_train_losses_log,  # Current iteration's loss log
                'train_metrics': chkpt_train_metrics_log  # Current iteration's metrics log
            }

            with open(chkpt_result_filepath, 'w') as f:
                json.dump(chkpt_result, f, indent=4)

        if self.return_chkpt:
            self.chkpt_list.append(chkpt)
        else:
            del chkpt
    
    def _save_visualization(self, dir=None):
        """
        Save visualizations based on the visualization configuration.
        
        Args:
            dir: Directory to save visualizations. If None, uses config save_dir.
        """
        if not self.visualization_config or not self.visualization_config.enabled:
            return
            
        # Use provided directory or fall back to config directory
        save_dir = dir if dir is not None else self.visualization_config.save_dir
        if save_dir is None:
            return
            
        # Ensure visualizer is initialized
        if self.visualizer is None:
            self._setup_visualization()
            
        if self.visualizer is None:
            return
            
        # Update visualizer's save directory
        self.visualizer.save_dir = save_dir
        self.visualizer.is_save = True
        
        config = self.visualization_config
        
        try:
            # Save reconstructions if enabled
            if config.reconstructions:
                if config.reconstruction_indices is not None:
                    # Use specific indices
                    self.visualizer.plot_reconstructions_sub_dataset(
                        img_indices=config.reconstruction_indices,
                        mode=config.reconstruction_mode,
                        figsize=config.figure_size
                    )
                else:
                    # Use random samples
                    self.visualizer.plot_random_reconstructions(
                        num_samples=config.num_reconstruction_samples,
                        mode=config.reconstruction_mode,
                        figsize=config.figure_size
                    )
            
            # Save latent traversals if enabled
            if config.latent_traversals:
                model_type = self._detect_model_type()
                
                # Get model-specific settings
                if model_type == "sn_vae":
                    settings = config.sn_vae_settings
                    self.visualizer.plot_all_latent_traversals(
                        r1_max_traversal_type=settings.get("r1_traversal_range_type", "probability"),
                        r1_max_traversal=settings.get("r1_traversal_range_value", 0.99),
                        s1_max_traversal_type=settings.get("s1_traversal_range_type", "fraction"),
                        s1_max_traversal=settings.get("s1_traversal_range_value", 1.0),
                        num_samples=config.num_traversal_samples,
                        use_ref_img=config.use_reference_image,
                        ref_img=None,
                        reg_img_idx=config.reference_image_index,
                        figsize=config.figure_size
                    )
                elif model_type == "s_vae":
                    settings = config.s_vae_settings
                    self.visualizer.plot_all_latent_traversals(
                        max_traversal_type=settings.get("traversal_range_type", "fraction"),
                        max_traversal=settings.get("traversal_range_value", 1.0),
                        num_samples=config.num_traversal_samples,
                        use_ref_img=config.use_reference_image,
                        ref_img=None,
                        reg_img_idx=config.reference_image_index,
                        figsize=config.figure_size
                    )
                else:  # n_vae
                    settings = config.n_vae_settings
                    self.visualizer.plot_all_latent_traversals(
                        max_traversal_type=settings.get("traversal_range_type", "probability"),
                        max_traversal=settings.get("traversal_range_value", 0.99),
                        num_samples=config.num_traversal_samples,
                        use_ref_img=config.use_reference_image,
                        ref_img=None,
                        reg_img_idx=config.reference_image_index,
                        figsize=config.figure_size
                    )
                    
            # Clear memory if configured
            if config.clear_memory_after_viz:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            print(f"Warning: Visualization failed with error: {e}")
            # Continue training even if visualization fails
    
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
        self.current_train_epoch = self.current_train_iter / len(self.dataloader)

        return {"loss": loss_val, "to_log": to_log}

    def _setup_visualization(self):
        """
        Initialize the appropriate visualizer based on the model type and visualization config.
        """
        if not self.visualization_config or not self.visualization_config.enabled:
            return
            
        # Import visualizers here to avoid circular imports
        from utils.visualize import NVAEVisualizer, SVAEVisualizer, SNVAEVisualizer
        
        # Create visualization directory if specified and doesn't exist
        viz_dir = self.visualization_config.save_dir
        if viz_dir is None and self.chkpt_save_dir is not None:
            # Auto-create visualization directory within checkpoint directory
            viz_dir = os.path.join(self.chkpt_save_dir, "visualizations")
        
        if viz_dir is not None:
            os.makedirs(viz_dir, exist_ok=True)
            self.visualization_config.save_dir = viz_dir  # Update config with actual path
        
        # Determine the appropriate visualizer based on model type
        model_type = self._detect_model_type()
        
        if model_type == "sn_vae":
            self.visualizer = SNVAEVisualizer(
                vae_model=self.model,
                dataset=self.dataloader.dataset if self.dataloader else None,
                is_plot=self.visualization_config.show_plots,
                save_dir=viz_dir
            )
        elif model_type == "s_vae":
            self.visualizer = SVAEVisualizer(
                vae_model=self.model,
                dataset=self.dataloader.dataset if self.dataloader else None,
                is_plot=self.visualization_config.show_plots,
                save_dir=viz_dir
            )
        else:  # Default to N-VAE (normal VAE)
            self.visualizer = NVAEVisualizer(
                vae_model=self.model,
                dataset=self.dataloader.dataset if self.dataloader else None,
                is_plot=self.visualization_config.show_plots,
                save_dir=viz_dir
            )
    
    def _detect_model_type(self):
        """
        Detect the type of VAE model to determine which visualizer to use.
        
        Returns:
            str: One of "n_vae", "s_vae", or "sn_vae"
        """
        # Check for S-N-VAE (mixed topology) - has latent_factor_topologies
        if hasattr(self.model, 'latent_factor_topologies'):
            return "sn_vae"
        
        # Check for S-VAE (toroidal) - has latent_factor_num instead of latent_dim
        if hasattr(self.model, 'latent_factor_num') and not hasattr(self.model, 'latent_dim'):
            return "s_vae"
        
        # Default to N-VAE (normal VAE with Gaussian latent space)
        return "n_vae"
    
    def _should_save_visualization(self, step, total_steps, step_type):
        """
        Determine if visualization should be saved based on configuration and current step.
        
        Args:
            step: Current step number
            total_steps: Total number of steps
            step_type: Either "epoch" or "iter"
            
        Returns:
            bool: True if visualization should be saved
        """
        if not self.visualization_config or not self.visualization_config.enabled:
            return False
            
        config = self.visualization_config
        
        # Save at training end
        if step == total_steps and config.save_at_training_end:
            return True
            
        # Save with checkpoints
        if config.save_with_checkpoints and self._should_save_checkpoint(step, total_steps, step_type):
            return True
            
        # Save at epoch end (only if step_type is epoch)
        if step_type == "epoch" and config.save_at_epoch_end:
            return True
            
        # Save every N epochs (only if step_type is epoch)
        if (step_type == "epoch" and 
            config.save_every_n_epochs is not None and 
            step % config.save_every_n_epochs == 0):
            return True
            
        return False
    
    def _should_save_checkpoint(self, step, total_steps, step_type):
        """Helper method to determine if checkpoint should be saved."""
        if not self.use_chkpt:
            return False
            
        # Always save at the end
        if step == total_steps:
            return True
            
        # Save every N steps if configured
        if (self.chkpt_every_n_steps is not None and 
            self.chkpt_step_type == step_type and 
            step % self.chkpt_every_n_steps == 0):
            return True
            
        return False

