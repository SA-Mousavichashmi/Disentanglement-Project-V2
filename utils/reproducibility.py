import os
import random
import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader # type: ignore

def set_all_random_seed(seed: int):
    """
    Fix all relevant RNG seeds for Python, NumPy, and PyTorch (CPU & GPU).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_cudnn_determinism(enforce_det):
    """
    Configure cuDNN and PyTorch settings for deterministic behavior.

    Disables cuDNN benchmark/autotuner and enables deterministic algorithms.
    If enforce_det is True, PyTorch will raise an error if a nondeterministic
    operation is used (requires PyTorch ≥1.8).

    Args:
        enforce_det (bool): If True, enforce deterministic algorithms and error on nondeterministic ops.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    if enforce_det:
        # For PyTorch ≥1.8, will error if a nondeterministic op is used
        torch.use_deterministic_algorithms(True)


def set_deterministic_run(seed,
                          use_cuda_det,
                          enforce_det,
                          **kwargs 
                          ):
    """
    Set up reproducible and deterministic behavior for experiments.

    Args:
        seed (int): Random seed for Python, NumPy, and PyTorch (CPU & GPU).
        use_cuda_det (bool): If True, configure cuDNN and PyTorch for deterministic CUDA operations.
        enforce_det (bool): If True, PyTorch will raise an error if a nondeterministic operation is used (requires PyTorch ≥1.8).
    """
    # set all RNG seeds
    set_all_random_seed(seed)
    # optionally enforce CUDA determinism
    if use_cuda_det:
        configure_cudnn_determinism(enforce_det=enforce_det)


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
