import torch
import time
import os
import psutil  # Added import
from torch.utils.data import DataLoader, Dataset

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


##################### Checkpoint utils #####################

CHKPT_DIR = 'checkpoints' # Default directory for checkpoints

def create_chkpt(
                train_id,
                train_step_unit,
                train_step_num,
                train_seed,
                train_determinism_type,
                model,
                optimizer,
                lr_scheduler,
                dataset,
                dataloader,
                loss,
                loss_results=None, # TODO Check them later
                metrics=None, # TODO check them later
                # TODO: Add device argument if needed.
                      ):
    """
    Creates a checkpoint dictionary.

    Args:
        train_id: Identifier for the training run.
        train_step_unit: the unit of the current training step (e.g., 'epoch', 'iteration').
        train_step_num: Current number of training steps completed.
        model: The model to save. Must have 'name', 'kwargs', and 'state_dict' attributes/methods.
        optimizer: The optimizer to save. Must have '__class__.__name__', 'defaults', and 'state_dict' attributes/methods.
        lr_scheduler: The learning rate scheduler to save. Must have 'state_dict' method.
                      Optionally, 'name' and 'kwargs' attributes.
        loss: The loss function (or its configuration) to save. Must have 'name' and 'kwargs' attributes.

    Returns:
        A dictionary containing the checkpoint data.
    """

    checkpoint = {
        'train_id': train_id,
        'train_step_unit': train_step_unit,
        'train_step_num': train_step_num,
        'train_seed': train_seed,
        'train_determinism_type': train_determinism_type,
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
           'kwargs': dataloader._dict_,
           'state_dict': dataloader.state_dict() # Assuming using StatefulDataLoader
        },
        'metrics': {
            'loss_results': loss_results,
            'metrics': metrics
        }
    }
    return checkpoint

def save_chkpt(
        train_id,
        train_step_unit,
        train_step_num,
        model,
        optimizer,
        lr_scheduler,
        loss,
        save_path
        ):
    """
    Saves a training checkpoint.

    Args:
        train_id: Identifier for the training run.
        train_step_unit: the unit of the current training step (e.g., 'epoch', 'iteration').
        train_step_num: Current number of training steps completed.
        model: The model to save. Must have 'name', 'kwargs', and 'state_dict' attributes/methods.
        optimizer: The optimizer to save. Must have '__class__.__name__', 'defaults', and 'state_dict' attributes/methods.
        lr_scheduler: The learning rate scheduler to save. Must have 'state_dict' method.
                      Optionally, 'name' and 'kwargs' attributes.
        loss: The loss function (or its configuration) to save. Must have 'name' and 'kwargs' attributes.
        save_path: Path to save the checkpoint file.
    """
    checkpoint_data = create_chkpt(
        train_id=train_id,
        train_step_unit=train_step_unit,
        train_step_num=train_step_num,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss=loss
    )
    torch.save(checkpoint_data, save_path)
    print(f"Checkpoint saved to {save_path}")

def load_chkpt(path: str, device: str = 'cpu'):
    """
    Loads a training checkpoint.

    Args:
        path: Path to the checkpoint file.
        device: Device to map the checkpoint to ('cpu' or 'cuda').

    Returns:
        A dictionary containing the checkpoint data.
    
    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file {path} does not exist.")
    
    # map loaded tensors to the requested device
    map_location = torch.device(device)
    checkpoint_data = torch.load(path, map_location=map_location)
    print(f"Checkpoint loaded from {path} on {device}")
    return checkpoint_data

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