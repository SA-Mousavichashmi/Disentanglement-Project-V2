import torch
import time
import os
import psutil  # Added import
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data import IterableDataset
from utils.helpers import create_load_optimizer, create_load_lr_scheduler, get_model_device
from vae_models.utils import create_load_model
from losses.utils import create_load_loss
# from utils.io import get_dataloader_from_chkpt  # Removed to fix circular import
from utils.reproducibility import set_deterministic_run
import uuid
import json


def find_optimal_num_workers(
    dataset: Dataset,
    batch_size: int,
    max_workers: int | None = None,
    num_batches_to_test: int | str = 200, # Use 'all' to process the entire dataset
    num_warmup_batches: int = 5,
    pin_memory: bool = True,
    **kwargs
) -> int:
    """
    Finds the optimal number of workers for a PyTorch DataLoader by timing
    data loading for different worker counts.

    Args:
        dataset: The dataset to load from.
        batch_size: The batch size for the DataLoader.
        max_workers: The maximum number of workers to test. Defaults to the
                     number of physical CPU cores if psutil is available,
                     otherwise falls back to logical cores (os.cpu_count()),
                     and finally to 4 if cpu_count fails.
        num_batches_to_test: The number of batches to load for timing each
                             worker count.
        num_warmup_batches: The number of batches to load before starting timer
                           to warm up caches and worker processes.
        pin_memory: Whether to use pin_memory in the DataLoader.
        **kwargs: Additional arguments to pass to the DataLoader constructor.

    Returns:
        The optimal number of workers found.
    """
    if max_workers is None:
        try:
            # Try getting physical cores first
            max_workers = psutil.cpu_count(logical=False)
            if max_workers is None:  # If psutil returns None
                raise TypeError("psutil.cpu_count(logical=False) returned None")
            print(f"Defaulting max_workers to physical core count: {max_workers}")
        except (ImportError, AttributeError, TypeError, NotImplementedError):
            # Fallback to logical cores if psutil fails or not installed
            print("psutil not found or failed, falling back to os.cpu_count() for max_workers.")
            max_workers = os.cpu_count()
            if max_workers is None:
                print("os.cpu_count() failed, defaulting max_workers to 4.")
                max_workers = 4  # Final fallback
            else:
                print(f"Defaulting max_workers to logical core count: {max_workers}")

    print(f"Finding optimal num_workers (testing 0 to {max_workers})...")
    results = {}

    for num_workers in range(max_workers + 1):
        try:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                **kwargs
            )
            
            # Perform warm-up phase first
            loader_iter = iter(loader)
            warmup_batches_loaded = 0
            print(f"  num_workers={num_workers}: Warming up with {num_warmup_batches} batches...")
            while warmup_batches_loaded < num_warmup_batches:
                try:
                    next(loader_iter)
                    warmup_batches_loaded += 1
                except StopIteration:
                    # If dataset too small for warm-up, reset iterator and continue to timing
                    print(f"  Warning: Dataset exhausted during warm-up after {warmup_batches_loaded} batches.")
                    loader_iter = iter(loader)  # Reset iterator
                    break
            
            # Start actual timing after warm-up
            start_time = time.time()
            batches_loaded = 0
            # Process all batches if num_batches_to_test is 'all', otherwise process the specified number
            if num_batches_to_test == 'all':
                while True:
                    try:
                        next(loader_iter)
                        batches_loaded += 1
                    except StopIteration:
                        print(f"  num_workers={num_workers}: Processed {batches_loaded} batches (entire dataset).")
                        break # Dataset exhausted
            else:
                print(f"  num_workers={num_workers}: Processing {num_batches_to_test} batches...")
                while batches_loaded < num_batches_to_test:
                    try:
                        next(loader_iter)
                        batches_loaded += 1
                    except StopIteration:
                        print(f"  Warning: Dataset exhausted after {batches_loaded} batches for num_workers={num_workers}. Requested {num_batches_to_test} batches.")
                        break  # Dataset is smaller than num_batches_to_test
                if batches_loaded == num_batches_to_test:
                    print(f"  num_workers={num_workers}: Successfully processed {batches_loaded} batches.")
            end_time = time.time()
            elapsed_time = end_time - start_time

            if batches_loaded > 0:
                results[num_workers] = elapsed_time / batches_loaded
                print(f"  num_workers={num_workers}: {results[num_workers]:.5f} sec/batch")
            else:
                print(f"  num_workers={num_workers}: Could not load any batches.")
                results[num_workers] = float('inf')  # Penalize if no batches loaded

            # Clean up loader resources if possible (helps prevent resource leaks)
            del loader_iter
            del loader

        except Exception as e:
            print(f"  Error testing num_workers={num_workers}: {e}")
            results[num_workers] = float('inf')  # Assign high cost on error

    # Find the num_workers with the minimum time per batch
    if not results:
        print("Warning: No results obtained. Defaulting to 0 workers.")
        return 0

    optimal_num_workers = min(results, key=results.get)
    print(f"Optimal num_workers: {optimal_num_workers} ({results[optimal_num_workers]:.5f} sec/batch)")

    return optimal_num_workers

def has_shuffle(loader) -> bool:
    """
    Return True if the DataLoader was configured with shuffle=True.
    Returns False for iterable-style datasets or if no sampler is set.
    """
    # IterableDataset doesn’t support meaningful shuffle
    if isinstance(loader.dataset, IterableDataset):
        return False

    # If there's no sampler attribute, no shuffle
    if not hasattr(loader, "sampler"):
        return False

    sampler = loader.sampler

    # A SequentialSampler means shuffle=False
    if isinstance(sampler, SequentialSampler):
        return False

    # RandomSampler indicates shuffle=True
    return isinstance(sampler, RandomSampler)

def check_dir_empty(dir):
    """
    Checks if a directory is empty.

    Args:
        dir: The directory path to check.

    Returns:
        bool: True if the directory is empty, False otherwise.
    """
    if not os.path.exists(dir):
        raise FileNotFoundError(f"Directory {dir} does not exist.")
    
    return len(os.listdir(dir)) == 0  # Check if the directory has any files or subdirectories

##################### Checkpoint utils #####################

CHKPT_DIR = 'checkpoints' # Default directory for checkpoints

def create_chkpt(
                train_id,
                train_iter_num, # Replaced train_step_unit and train_step_num
                train_determinism_kwargs,
                model,
                optimizer,
                lr_scheduler,
                dataset,
                dataloader,
                loss,
                use_torch_compile=False,
                torch_compile_kwargs=None,
                train_losses_log=None,
                train_metrics_log=None,
                chkpt_train_losses_log=None,
                chkpt_metrics_log=None,
                return_chkpt=None,
                # --- Add checkpointing args ---
                chkpt_every_n_steps=None,
                chkpt_step_type=None,
                chkpt_save_path=None,
                chkpt_save_dir=None,
                chkpt_save_master_dir=None,
                # --- Add logging args ---
                is_progress_bar=None,
                progress_bar_log_iter_interval=None,
                log_loss_interval_type=None,
                use_train_logging=None,
                log_loss_iter_interval=None,
                return_logs=None, # Renamed from return_log_loss
                      ):
    """
    Creates a checkpoint dictionary.

    Args:
        train_id: Identifier for the training run.
        train_iter_num: Current number of training iterations completed.
        train_determinism_kwargs: Dictionary containing determinism settings used for training.
        model: The model to save. Must have 'name', 'kwargs', and 'state_dict' attributes/methods.
        optimizer: The optimizer to save. Must have '__class__.__name__', 'defaults', and 'state_dict' attributes/methods.
        lr_scheduler: The learning rate scheduler to save. Must have 'state_dict' method.
                     Optionally, 'name' and 'kwargs' attributes.
        dataset: The dataset configuration/object to save. Must have 'name' and 'kwargs' attributes.
        dataloader: The dataloader configuration/object to save. Must be a StatefulDataLoader and have relevant attributes and 'state_dict'.
        loss: The loss function (or its configuration) to save. Must have 'name' and 'kwargs' attributes.
        use_torch_compile: Boolean indicating whether torch.compile was used on the model.
        train_losses_log: Dictionary containing loss results over training.
        train_metrics_log: Dictionary containing metric results over training.
        chkpt_train_losses_log: Dictionary containing loss results at checkpoint time.
        chkpt_metrics_log: Dictionary containing metric results at checkpoint time.
        return_chkpt: 
        chkpt_every_n_steps:
        chkpt_step_type:
        chkpt_save_path:
        chkpt_save_dir:
        chkpt_save_master_dir:
        is_progress_bar:
        progress_bar_log_iter_interval:
        log_loss_interval_type:
        use_train_logging:
        log_loss_iter_interval:
        return_logs: # Renamed from return_log_loss

    Returns:
        A dictionary containing the checkpoint data.
    """

    checkpoint = {
        'train_id': train_id,
        'train_iter_num': train_iter_num, # Updated from train_step_unit and train_step_num
        'train_epoch_num': train_iter_num / len(dataloader), 
        'train_determinism_kwargs': train_determinism_kwargs,
        'train_device': get_model_device(model),
        'use_torch_compile': use_torch_compile,
        'torch_compile_kwargs': torch_compile_kwargs,
        'model': {
            'name': model.name, # Assuming model has a 'name' attribute
            'kwargs': model.kwargs, # Assuming model has a 'kwargs' attribute
            'state_dict': model.state_dict(),
        },
        'loss': {
            'name': loss.name, # Assuming loss object has a 'name' attribute
            'kwargs': loss.kwargs, # Assuming loss object has a 'kwargs' attribute
            'state_dict': loss.state_dict() if hasattr(loss, 'state_dict') else None,
        },
        'optimizer': {
            'name': optimizer.__class__.__name__,
            'kwargs': optimizer.defaults,
            'state_dict': optimizer.state_dict(),
        },
        'lr_scheduler': { # Assuming lr_scheduler has 'name' and 'kwargs' attributes
            'name': lr_scheduler.name if hasattr(lr_scheduler, 'name') else lr_scheduler.__class__.__name__,
            'kwargs': lr_scheduler.kwargs if hasattr(lr_scheduler, 'kwargs') else {}, # Provide default if not present
            'state_dict': lr_scheduler.state_dict() if hasattr(lr_scheduler, 'state_dict') else None,
        },
        'dataset': {
            'name': dataset.name, # Assuming dataset has a 'name' attribute
            'kwargs': dataset.kwargs, # Assuming dataset has a 'kwargs' attribute
        },
        'dataloader': {
           'kwargs': {
                'batch_size': dataloader.batch_size,
                'shuffle': True, #TODO fix this, the True value is common but a solution for that must be implemented
                'num_workers': dataloader.num_workers,
                'pin_memory': dataloader.pin_memory,
                # Add specific args from get_deterministic_dataloader
                'seed': dataloader.generator.initial_seed() if dataloader.generator else None, # Assuming generator is set and has initial_seed
                'persistent_workers': dataloader.persistent_workers,
                'in_order': dataloader.in_order,
                'snapshot_every_n_steps': dataloader.snapshot_every_n_steps,
           },
           'state_dict': dataloader.state_dict() # Assuming using StatefulDataLoader
        },
        'chkpt': {
            'return_chkpt': return_chkpt,
            'chkpt_every_n_steps': chkpt_every_n_steps,
            'chkpt_step_type': chkpt_step_type,
            'chkpt_save_path': chkpt_save_path,
            'chkpt_save_dir': chkpt_save_dir,
            'chkpt_save_master_dir': chkpt_save_master_dir
        },
        'logging': {
            'is_progress_bar': is_progress_bar,
            'progress_bar_log_iter_interval': progress_bar_log_iter_interval,
            'log_loss_interval_type': log_loss_interval_type,
            'use_train_logging': use_train_logging,
            'log_loss_iter_interval': log_loss_iter_interval,
            'return_logs': return_logs, # Renamed from return_log_loss
        },
        
        'logs':{
            'train': {
                'loss_results': train_losses_log,
                'metrics': train_metrics_log,
            },
            'chkpt': {
                'loss_results': chkpt_train_losses_log,
                'metrics': chkpt_metrics_log,
            }
        }
    }

    return checkpoint

def print_chkpt_info(checkpoint):
    """
    Prints information about the checkpoint.

    Args:
        checkpoint: The checkpoint data.
    """
    print("Checkpoint Information:")
    print(f"  Train ID: {checkpoint['train_id']}")
    print(f"  Train Iteration Number: {checkpoint['train_iter_num']}") # Updated
    print(f"  Train Epoch Number: {checkpoint['train_epoch_num']}")
    print(f"  Train determinism kwargs: {checkpoint['train_determinism_kwargs']}")
    print(f"  Use Torch Compile: {checkpoint['use_torch_compile']}")
    print(f"#### Model ####")
    print(f"  Model Name: {checkpoint['model']['name']}")
    print(f"  Model kwargs: {checkpoint['model']['kwargs']}")
    print(f"#### Loss ####")
    print(f"  Loss Name: {checkpoint['loss']['name']}")
    print(f"  Loss kwargs: {checkpoint['loss']['kwargs']}")
    print(f"#### Dataset ####")
    print(f"  Dataset Name: {checkpoint['dataset']['name']}")
    print(f"  Dataset kwargs: {checkpoint['dataset']['kwargs']}")
    print(f"#### Dataloader ####")
    print(f"  Dataloader kwargs: {checkpoint['dataloader']['kwargs']}")
    print(f"#### Optimizer ####")
    print(f"  Optimizer Name: {checkpoint['optimizer']['name']}")
    print(f"  LR Scheduler Name: {checkpoint['lr_scheduler']['name']}")


def save_chkpt(
        save_path,
        train_id,
        train_iter_num, # Replaced train_step_unit and train_step_num
        train_determinism_kwargs,
        model,
        optimizer,
        lr_scheduler,
        dataset,
        dataloader,
        loss,
        train_losses_log=None,
        train_metrics_log=None,
        chkpt_train_losses_log=None,
        chkpt_metrics_log=None,
        use_torch_compile=False,
        ):
    """
    Saves a training checkpoint.

    Args:
        save_path: Path to save the checkpoint file.
        train_id: Identifier for the training run.
        train_iter_num: Current number of training iterations completed.
        train_determinism_kwargs: Dictionary containing determinism settings used for training.
        model: The model to save. Must have 'name', 'kwargs', and 'state_dict' attributes/methods.
        optimizer: The optimizer to save. Must have '__class__.__name__', 'defaults', and 'state_dict' attributes/methods.
        lr_scheduler: The learning rate scheduler to save. Must have 'state_dict' method.
                      Optionally, 'name' and 'kwargs' attributes.
        dataset: The dataset configuration/object to save. Must have 'name' and 'kwargs' attributes.
        dataloader: The dataloader configuration/object to save. Must be a StatefulDataLoader and have relevant attributes and 'state_dict'.
        loss: The loss function (or its configuration) to save. Must have 'name' and 'kwargs' attributes.
        train_losses_log: Dictionary containing loss results over training.
        train_metrics_log: Dictionary containing metric results over training.
        chkpt_train_losses_log: Dictionary containing loss results at checkpoint time.
        chkpt_metrics_log: Dictionary containing metric results at checkpoint time.
        use_torch_compile: Boolean indicating whether torch.compile was used on the model.
    """
    checkpoint_data = create_chkpt(
        train_id=train_id,
        train_iter_num=train_iter_num,
        train_determinism_kwargs=train_determinism_kwargs,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataset=dataset,
        dataloader=dataloader,
        loss=loss,
        use_torch_compile=use_torch_compile,
        train_losses_log=train_losses_log,
        train_metrics_log=train_metrics_log,
        chkpt_train_losses_log=chkpt_train_losses_log,
        chkpt_metrics_log=chkpt_metrics_log,
        # Add the new parameter to the call
        return_logs=True # Assuming default to True when saving
    )
    torch.save(checkpoint_data, save_path)
    print(f"Checkpoint saved to {save_path}")

def load_chkpt(path: str, device: str = 'original'):
    """
    Loads a training checkpoint.

    Args:
        path: Path to the checkpoint file.
        device: Device to map the checkpoint to ('cpu', 'cuda', 'original').

    Returns:
        A dictionary containing the checkpoint data.
    
    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file {path} does not exist.")
    
    assert device in ['cpu', 'cuda', 'original'], "Device must be 'cpu', 'cuda', or 'original'."
    # map loaded tensors to the requested device
    checkpoint = None

    if device == 'original':
        # Load the checkpoint on the original device
        checkpoint = torch.load(path)
    elif device == 'cuda':
        # Load the checkpoint on GPU
        map_location = torch.device('cuda')
        checkpoint = torch.load(path, map_location=map_location)
    else:
        # Load the checkpoint on CPU
        map_location = torch.device('cpu')
        checkpoint = torch.load(path, map_location=map_location)
    
    print(f"Checkpoint loaded from {path} on {device}.")
    
    return checkpoint

def check_compatibility_chkpt(checkpoint, model, optimizer, lr_scheduler, loss):
    """
    Checks if the checkpoint is compatible with the current model, optimizer, lr_scheduler, and loss.

    Args:
        checkpoint: The checkpoint data.
        model: The current model.
        optimizer: The current optimizer.
        lr_scheduler: The current learning rate scheduler.
        loss: The current loss function.

    Returns:
        bool: True if compatible, False otherwise.
    """
    # Check model compatibility
    if checkpoint['model']['name'] != model.name:
        print(f"Model mismatch: {checkpoint['model']['name']} vs {model.name}")
        return False

    # Check optimizer compatibility
    if checkpoint['optimizer']['name'] != optimizer.__class__.__name__:
        print(f"Optimizer mismatch: {checkpoint['optimizer']['name']} vs {optimizer.__class__.__name__}")
        return False

    # Check lr_scheduler compatibility
    if checkpoint['lr_scheduler']['name'] != lr_scheduler.__class__.__name__:
        print(f"LR Scheduler mismatch: {checkpoint['lr_scheduler']['name']} vs {lr_scheduler.__class__.__name__}")
        return False

    # Check loss compatibility
    if checkpoint['loss']['name'] != loss.name:
        print(f"Loss mismatch: {checkpoint['loss']['name']} vs {loss.name}")
        return False

    return True

def get_dataloader_from_chkpt(checkpoint):
    """
    Reconstructs a deterministic DataLoader (StatefulDataLoader) from a checkpoint dict.
    Returns the DataLoader with its state restored.
    """
    # Import here to avoid circular imports
    from datasets.utils import get_dataset
    from utils.reproducibility import get_deterministic_dataloader

    # 1. Reconstruct the dataset
    dataset_info = checkpoint['dataset']
    dataset_name = dataset_info['name']
    dataset_kwargs = dataset_info['kwargs']
    dataset_class = get_dataset(dataset_name)
    dataset = dataset_class(**dataset_kwargs)

    # 2. Prepare dataloader kwargs
    dl_kwargs = checkpoint['dataloader']['kwargs']
    batch_size = dl_kwargs['batch_size']
    shuffle = dl_kwargs['shuffle']
    num_workers = dl_kwargs['num_workers']
    pin_memory = dl_kwargs['pin_memory']
    seed = dl_kwargs['seed']
    persistent_workers = dl_kwargs['persistent_workers']
    in_order = dl_kwargs['in_order']
    snapshot_every_n_steps = dl_kwargs['snapshot_every_n_steps']

    # 3. Reconstruct the DataLoader
    loader = get_deterministic_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        seed=seed,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        in_order=in_order,
        snapshot_every_n_steps=snapshot_every_n_steps
    )

    # 4. Restore DataLoader state if present (stateful dataloader)
    if 'state_dict' in checkpoint['dataloader'] and checkpoint['dataloader']['state_dict'] is not None:
        loader.load_state_dict(checkpoint['dataloader']['state_dict'])

    return loader

def create_trainer_from_chkpt_exact(chkpt, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Creates a trainer instance from a checkpoint with exact settings.
    
    This function precisely recreates the training environment from a checkpoint,
    using all stored parameters and settings without any modifications.
    
    Parameters
    ----------
    chkpt: dict
        A dictionary containing the checkpoint data.
    device: torch.device or str, optional
        If provided, use this device instead of the one stored in the checkpoint.
        Defaults to CUDA if available, else CPU.
        
    Returns
    -------
    BaseTrainer:
        An instance of the BaseTrainer class initialized with exactly the
        same components and settings as stored in the checkpoint.
    """

    if chkpt['train_determinism_kwargs'] is not None:
        set_deterministic_run(**chkpt['train_determinism_kwargs'])
        print(f"Determinism settings applied from checkpoint: {chkpt['train_determinism_kwargs']}")

    # Load model with original settings
    model = create_load_model(
        chkpt['model']['name'],
        chkpt['model']['kwargs'],
        chkpt['model']['state_dict']
    )
    model = model.to(device)
    
    # Load loss function with original settings
    loss = create_load_loss(
        chkpt['loss']['name'],
        chkpt['loss']['kwargs'],
        chkpt['loss']['state_dict']
    )
    
    # Load optimizer with original settings
    optimizer = create_load_optimizer(
        chkpt['optimizer']['name'],
        chkpt['optimizer']['kwargs'],
        chkpt['optimizer']['state_dict'],
        model_params=model.parameters()
    )
    
    # Load lr_scheduler with original settings
    lr_scheduler = create_load_lr_scheduler(
        name=chkpt['lr_scheduler']['name'],
        kwargs=chkpt['lr_scheduler']['kwargs'],
        state_dict=chkpt['lr_scheduler']['state_dict'],
        optimizer=optimizer
    )
    
    # Recreate dataloader with exact settings
    dataloader = get_dataloader_from_chkpt(chkpt)
    
    # Extract training logs
    train_iter_num = chkpt['train_iter_num']
    train_losses_log = chkpt['logs']['train']['loss_results']
    train_metrics_log = chkpt['logs']['train']['metrics']
    
    # Extract checkpointing settings
    chkpt_settings = chkpt['chkpt']
    return_chkpt = chkpt_settings['return_chkpt']
    chkpt_every_n_steps = chkpt_settings['chkpt_every_n_steps']
    chkpt_step_type = chkpt_settings['chkpt_step_type']
    chkpt_save_path = chkpt_settings['chkpt_save_path']
    chkpt_save_dir = chkpt_settings['chkpt_save_dir']
    chkpt_save_master_dir = chkpt_settings['chkpt_save_master_dir']
    
    # Extract logging settings
    logging_settings = chkpt['logging']
    is_progress_bar = logging_settings['is_progress_bar']
    progress_bar_log_iter_interval = logging_settings['progress_bar_log_iter_interval']
    log_loss_interval_type = logging_settings['log_loss_interval_type']
    use_train_logging = logging_settings['use_train_logging']
    log_loss_iter_interval = logging_settings['log_loss_iter_interval']
    return_log_loss = logging_settings['return_log_loss']
    
    # Create trainer with exact settings from checkpoint
    from trainers.basetrainer import BaseTrainer  # Import here to avoid circular import
    trainer = BaseTrainer(
        model=model,
        loss=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_id=chkpt['train_id'],  # Use original train_id
        prev_train_iter=train_iter_num,
        determinism_kwargs=chkpt['train_determinism_kwargs'],
        dataloader=dataloader,
        # Set torch compile settings
        use_torch_compile=chkpt['use_torch_compile'],
        torch_compile_kwargs=chkpt['torch_compile_kwargs'],
        # Set logging parameters
        is_progress_bar=is_progress_bar,
        progress_bar_log_iter_interval=progress_bar_log_iter_interval,
        log_loss_interval_type=log_loss_interval_type,
        use_train_logging=use_train_logging,
        log_loss_iter_interval=log_loss_iter_interval,
        return_log_loss=return_log_loss,
        # Set checkpointing parameters
        return_chkpt=return_chkpt,
        chkpt_every_n_steps=chkpt_every_n_steps,
        chkpt_step_type=chkpt_step_type,
        chkpt_save_path=chkpt_save_path,
        chkpt_save_dir=chkpt_save_dir,
        chkpt_save_master_dir=chkpt_save_master_dir,
    )
    
    return trainer


def create_trainer_from_chkpt(ckpt,
                              create_exact=False,
                              new_model=None, 
                              new_loss=None,
                              new_optimizer=None,
                              new_dataloader=None,
                              new_lr_scheduler=None,
                              additional_trainer_kwargs=None,
                              device='cuda' if torch.cuda.is_available() else 'cpu',
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
    device: torch.device, optional
        If provided, use this device instead of the one stored in the checkpoint.
        Defaults to CUDA if available, else CPU.

    Returns
    -------
    BaseTrainer:
        An instance of the BaseTrainer class initialized with components from the checkpoint
        or with the provided new components.
        If any of new_model, new_loss, new_optimizer, or new_lr_scheduler are provided,
        train_id will be set to a new UUID.
    """

    if create_exact:
        return create_trainer_from_chkpt_exact(ckpt, device=device)

    if ckpt['train_determinism_kwargs'] is not None:
        set_deterministic_run(
            seed=ckpt['train_determinism_kwargs']['seed'],
            use_cuda_det=ckpt['train_determinism_kwargs']['use_cuda_deterministic'],
            cublas_workspace_config=ckpt['train_determinism_kwargs']['cublas_workspace_config']
        )

    if new_model is not None and new_optimizer is None:
        raise ValueError("If new_model is provided, new_optimizer must also be provided.")

    train_id = ckpt['train_id']
    model_chkpt = ckpt['model']
    loss_chkpt = ckpt['loss']
    optimizer_chkpt = ckpt['optimizer']
    lr_scheduler_chkpt = ckpt['lr_scheduler']

    resume_logging = True if dataloader is None else False

    if new_model is not None or new_loss is not None or new_optimizer is not None or new_lr_scheduler is not None:
        train_id = uuid.uuid4()
    else:
        continue_training_exact = True
        train_id = ckpt['train_id']
    
    model = new_model if new_model is not None else create_load_model(
        model_chkpt['name'],
        model_chkpt['kwargs'],
        model_chkpt['state_dict']
    )
    model = model.to(device)

    loss = new_loss if new_loss is not None else create_load_loss(
        loss_chkpt['name'],
        loss_chkpt['kwargs'],
        loss_chkpt['state_dict']
    )

    optimizer = new_optimizer if new_optimizer is not None else create_load_optimizer(
        optimizer_chkpt['name'],
        optimizer_chkpt['kwargs'],
        optimizer_chkpt['state_dict'],
        model_params=model.parameters()
    )

    lr_scheduler = new_lr_scheduler if new_lr_scheduler is not None else create_load_lr_scheduler(
        name=lr_scheduler_chkpt['name'],
        kwargs=lr_scheduler_chkpt['kwargs'],
        state_dict=lr_scheduler_chkpt['state_dict'],
        optimizer=optimizer
    )
    
    dataloader = new_dataloader if new_dataloader is not None else get_dataloader_from_chkpt(ckpt) 

    if resume_logging:
        # Load the training logs from the checkpoint
        train_iter_num = ckpt['train_iter_num']
        train_losses_log = ckpt['train_logs']['loss_results']
        train_metrics_log = ckpt['train_logs']['metrics']
    else:
        train_iter_num = None
        train_losses_log = None
        train_metrics_log = None
        
    from trainers.basetrainer import BaseTrainer  # Import here to avoid circular import
    trainer = BaseTrainer(
        model=model,
        loss=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_id=train_id,
        prev_train_iter=train_iter_num, 
        determinism_kwargs=ckpt['train_determinism_kwargs'],
        dataloader=dataloader,
        prev_train_losses_log=train_losses_log,
        prev_train_metrics_log=train_metrics_log,
        chkpt_save_dir=additional_trainer_kwargs.get('chkpt_save_dir', None) if additional_trainer_kwargs else None,
        **(additional_trainer_kwargs or {}) # Spread remaining kwargs, allows overriding any previous args
    )

    return trainer

def is_chkpt_dir_compatible(chkpt_dir: str, train_id):
    """
    Checks if the checkpoint directory is compatible with the given train_id
    by checking the 'train_id' in the 'train_metadata.json' file.

    Args:
        chkpt_dir (str): The path to the checkpoint directory.
        train_id (str): The train_id to check against.

    Returns:
        bool: True if the checkpoint directory is compatible, False otherwise.
    """
    if not os.path.exists(chkpt_dir):
        return False

    # Check if the train_metadata.json file exists in the checkpoint directory
    metadata_file = os.path.join(chkpt_dir, 'train_metadata.json')

    metadata = None
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    stored_train_id = metadata.get('train_id')

    # Convert UUID to string for consistent comparison if train_id is a UUID object
    if isinstance(train_id, uuid.UUID):
        train_id = str(train_id)
    
    if stored_train_id != train_id:
        return False

    return True