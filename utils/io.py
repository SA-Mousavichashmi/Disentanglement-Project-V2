import torch
import time
import os
import psutil  # Added import
from torch.utils.data import DataLoader, Dataset

def find_optimal_num_workers(
    dataset: Dataset,
    batch_size: int,
    max_workers: int = None,
    num_batches_to_test: int = 100,
    num_warmup_batches: int = 5,  # Added warm-up parameter
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
            while batches_loaded < num_batches_to_test:
                try:
                    next(loader_iter)
                    batches_loaded += 1
                except StopIteration:
                    print(f"  Warning: Dataset exhausted after {batches_loaded} batches for num_workers={num_workers}.")
                    break  # Dataset is smaller than num_batches_to_test
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