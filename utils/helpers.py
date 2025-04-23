
import random
import numpy as np
import torch


#-------------- TRAINING HELPERS -----------------
def get_scheduler(optimizer, name='none', tau=[], gamma=0.1): # Added parameters to signature
    if name == 'none':
        #If no learning rate scheduler is desired, we simply use a constant scheduler.
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=0)
    if name == 'multistep':
        # Ensure gamma is a float if it was passed as a list (e.g., from config)
        if isinstance(gamma, list): gamma = gamma[0]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=tau, gamma=gamma)
    raise ValueError(f'Scheduler option [{name}] not available!')

def get_optimizer(parameters, lr, name='adam'): # Added parameters to signature
    if name == 'adam':
        return torch.optim.Adam(parameters, lr=lr)
    # Add other optimizers here if needed
    raise ValueError(f'Optimizer option [{name}] not available!')

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
