import os
import random
import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader # type: ignore

def set_all_random_seed(seed: int):
    """
    Fix all relevant RNG seeds for Python, NumPy, and PyTorch (CPU & GPU).
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_cudnn_determinism():
    """
    Disable cuDNN benchmark/autotuner and force deterministic algorithms.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    # For PyTorch ≥1.8, will error if a nondeterministic op is used
    torch.use_deterministic_algorithms(True)


def set_cublas_workspace(config: str = ":16:8"):
    """
    Set the CUBLAS_WORKSPACE_CONFIG env var so that CUDA kernel
    workspace sizes are fixed for reproducible reductions.
    Must be called before any torch.cuda calls.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = config


def set_deterministic_run(seed: int = 0, mode: str = 'full'):
    """
    Configure settings for deterministic runs based on the specified mode.

    Args:
        seed (int): The random seed to use for all libraries. Defaults to 0.
        mode (str): The level of determinism to apply. Options:
                    'full': Apply all determinism settings (seed, cuBLAS, cuDNN).
                    'seed_only': Only set the random seeds for Python, NumPy, and PyTorch.
                    'cudnn_only': Only configure cuDNN for deterministic algorithms.
                    Defaults to 'full'.
    """
    if mode == 'full':
        set_cublas_workspace()
        set_all_random_seed(seed)
        configure_cudnn_determinism()
    elif mode == 'seed_only':
        set_all_random_seed(seed)
    elif mode == 'cudnn_only':
        configure_cudnn_determinism()
    else:
        raise ValueError(f"Unknown determinism mode: {mode}. Options are 'full', 'seed_only', 'cudnn_only'.")


def get_deterministic_dataloader(dataset,
                                 batch_size: int,
                                 shuffle: bool,
                                 num_workers: int,
                                 seed: int,
                                 pin_memory: bool = True,
                                 persistent_workers: bool = True,
                                 in_order: bool = True,
                                 snapshot_every_n_steps: int = 1,
                                 ) -> StatefulDataLoader:
    """
    Returns a StatefulDataLoader whose RNGs and shuffle order
    are seeded for determinism, with state_dict checkpoints.
    """
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    return StatefulDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,       # keep workers alive
        in_order=in_order,                 # strict batch ordering
        snapshot_every_n_steps=snapshot_every_n_steps       # checkpoint every batch
    )
