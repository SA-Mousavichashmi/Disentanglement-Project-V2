import random
import numpy as np
import torch


#-------------- TRAINING HELPERS -----------------
def get_device(on_cpu=False, gpu_id=0): # Added parameters to signature
    """Return the correct device"""
    return torch.device(
        f"cuda:{gpu_id}" if torch.cuda.is_available() and not on_cpu else "cpu")

def get_model_device(model):
    """Return the device on which a model is."""
    return next(model.parameters()).device

def get_n_param(model):
    """Return the number of parameters."""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nParams = sum([np.prod(p.size()) for p in model_parameters])
    return nParams

def get_optimizer(name, model_params, **kwargs):
    """Return an optimizer instance from torch.optim given its name."""
    try:
        optimizer_cls = getattr(torch.optim, name)
    except AttributeError:
        raise ValueError(f'Optimizer option [{name}] not available!')
    return optimizer_cls(model_params, **kwargs)

def create_load_optimizer(optimizer_name, optimizer_kwargs, optimizer_state_dict, model_params):
    """Creates an optimizer and loads its state dictionary.

    Parameters
    ----------
    optimizer_name : str
        The name of the optimizer to create.
    optimizer_kwargs : dict
        The keyword arguments to pass to the optimizer constructor.
    optimizer_state_dict : dict
        The state dictionary containing the parameters of the optimizer.
    model_params : iterable
        The parameters to optimize (e.g., model.parameters()).
    """
    if model_params is None:
        raise ValueError("'model_params' argument (model parameters) must be provided to create the optimizer.")
    optimizer = get_optimizer(optimizer_name, model_params, **optimizer_kwargs)
    optimizer.load_state_dict(optimizer_state_dict)
    return optimizer

def get_lr_scheduler(name, kwargs, optimizer): # Added parameters to signature
    """Return a scheduler instance from torch.optim.lr_scheduler given its name."""
    # if no scheduler requested, return None
    if not name:
        return None

    try:
        sched_cls = getattr(torch.optim.lr_scheduler, name)
    except AttributeError:
        raise ValueError(f'Scheduler option [{name}] not available!')

    return sched_cls(optimizer, **kwargs)

def create_load_lr_scheduler(name, kwargs, state_dict, optimizer):
    """Creates an LR scheduler and loads its state dictionary.

    Parameters
    ----------
    name : str
        Name of the scheduler to create.
    kwargs : dict
        Keyword arguments for the scheduler constructor.
    state_dict : dict
        State dictionary to load into the scheduler.
    optimizer : torch.optim.Optimizer
        Optimizer to which the scheduler is attached.
    """
    # Create the scheduler
    lr_scheduler = get_lr_scheduler(name=name, kwargs=kwargs, optimizer=optimizer)
    # Load the state dictionary into the scheduler
    lr_scheduler.load_state_dict(state_dict)
    
    return lr_scheduler