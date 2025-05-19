import torch
import time
import os
import psutil  # Added import
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data import IterableDataset
from utils.helpers import get_model_device


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
                train_loss_results=None, 
                train_metrics=None,
                chkpt_train_results=None,
                chkpt_metrics=None,
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
        train_loss_results: Dictionary containing loss results over training.
        train_metrics: Dictionary containing metric results over training.
        chkpt_train_results: Dictionary containing loss results at checkpoint time.
        chkpt_metrics: Dictionary containing metric results at checkpoint time.

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
        'train_logs': {
            'loss_results': train_loss_results,
            'metrics': train_metrics,
        },
        'chkpt_logs': {
            'loss_results': chkpt_train_results,
            'metrics': chkpt_metrics,
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
        loss_results=None,
        metrics=None,
        ):
    """
    Saves a training checkpoint.

    Args:
        save_path: Path to save the checkpoint file.
        train_id: Identifier for the training run.
        train_iter_num: Current number of training iterations completed.
        train_determinism_kwargs: Dictionary containing determinism settings used for training. # Added
        model: The model to save. Must have 'name', 'kwargs', and 'state_dict' attributes/methods.
        optimizer: The optimizer to save. Must have '__class__.__name__', 'defaults', and 'state_dict' attributes/methods.
        lr_scheduler: The learning rate scheduler to save. Must have 'state_dict' method.
                      Optionally, 'name' and 'kwargs' attributes.
        dataset: The dataset configuration/object to save. Must have 'name' and 'kwargs' attributes. # Added
        dataloader: The dataloader configuration/object to save. Must have '_dict_' and 'state_dict' (if stateful) attributes/methods. # Added
        loss: The loss function (or its configuration) to save. Must have 'name' and 'kwargs' attributes.
        loss_results: Dictionary containing loss results (optional). # Added
        metrics: Dictionary containing metric results (optional). # Added
    """
    checkpoint_data = create_chkpt(
        train_id=train_id,
        train_iter_num=train_iter_num, # Updated
        train_determinism_kwargs=train_determinism_kwargs,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataset=dataset,
        dataloader=dataloader,
        loss=loss,
        loss_results=loss_results,
        metrics=metrics
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