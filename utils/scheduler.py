import abc
import torch
from typing import Any, Dict, Union

class BaseHyperparameterScheduler(abc.ABC):
    """Base class for hyperparameter schedulers."""
    
    def __init__(self, param_name: str, initial_value: float, **kwargs):
        self.param_name = param_name
        self.initial_value = initial_value
        self.current_step = 0
        self.current_value = initial_value
    
    @abc.abstractmethod
    def step(self) -> float:
        """Update and return the current parameter value."""
        pass
    
    @abc.abstractmethod
    def get_value(self) -> float:
        """Get current parameter value without stepping."""
        pass
    
    def state_dict(self) -> Dict[str, Any]:
        return {
            'param_name': self.param_name,
            'initial_value': self.initial_value,
            'current_step': self.current_step,
            'current_value': self.current_value
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.param_name = state_dict['param_name']
        self.initial_value = state_dict['initial_value']
        self.current_step = state_dict['current_step']
        self.current_value = state_dict['current_value']

class LinearScheduler(BaseHyperparameterScheduler):
    """Linear annealing scheduler."""
    
    def __init__(self, param_name: str, initial_value: float, final_value: float, 
                 total_steps: int, **kwargs):
        super().__init__(param_name, initial_value, **kwargs)
        self.final_value = final_value
        self.total_steps = total_steps
    
    def step(self) -> float:
        if self.total_steps == 0:
            self.current_value = self.final_value
        else:
            progress = min(self.current_step / self.total_steps, 1.0)
            self.current_value = self.initial_value + (self.final_value - self.initial_value) * progress
        
        self.current_step += 1
        return self.current_value
    
    def get_value(self) -> float:
        if self.total_steps == 0:
            return self.final_value
        progress = min(self.current_step / self.total_steps, 1.0)
        return self.initial_value + (self.final_value - self.initial_value) * progress

class ExponentialScheduler(BaseHyperparameterScheduler):
    """Exponential decay scheduler."""
    
    def __init__(self, param_name: str, initial_value: float, decay_rate: float, **kwargs):
        super().__init__(param_name, initial_value, **kwargs)
        self.decay_rate = decay_rate
    
    def step(self) -> float:
        self.current_value = self.initial_value * (self.decay_rate ** self.current_step)
        self.current_step += 1
        return self.current_value
    
    def get_value(self) -> float:
        return self.initial_value * (self.decay_rate ** self.current_step)

class CosineScheduler(BaseHyperparameterScheduler):
    """Cosine annealing scheduler."""
    
    def __init__(self, param_name: str, initial_value: float, final_value: float, 
                 total_steps: int, **kwargs):
        super().__init__(param_name, initial_value, **kwargs)
        self.final_value = final_value
        self.total_steps = total_steps
    
    def step(self) -> float:
        if self.total_steps == 0:
            self.current_value = self.final_value
        else:
            progress = min(self.current_step / self.total_steps, 1.0)
            cosine_factor = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
            self.current_value = self.final_value + (self.initial_value - self.final_value) * cosine_factor
        
        self.current_step += 1
        return self.current_value
    
    def get_value(self) -> float:
        if self.total_steps == 0:
            return self.final_value
        progress = min(self.current_step / self.total_steps, 1.0)
        cosine_factor = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
        return self.final_value + (self.initial_value - self.final_value) * cosine_factor

class CyclicalAnnealingScheduler(BaseHyperparameterScheduler):
    """Cyclical annealing scheduler based on https://github.com/haofuml/cyclical_annealing"""
    
    def __init__(self, param_name: str, initial_value: float, final_value: float, 
                 total_steps: int, n_cycle: int = 4, ratio: float = 0.5, **kwargs):
        super().__init__(param_name, initial_value, **kwargs)
        self.final_value = final_value
        self.total_steps = total_steps
        self.n_cycle = n_cycle
        self.ratio = ratio
        self.period = total_steps / n_cycle
        self.step_size = (final_value - initial_value) / (self.period * ratio)
    
    def step(self) -> float:
        if self.total_steps == 0:
            self.current_value = self.final_value
        else:
            cycle = int(self.current_step / self.period)
            pos_in_cycle = self.current_step - cycle * self.period
            
            if pos_in_cycle < (self.period * self.ratio):
                self.current_value = self.initial_value + self.step_size * pos_in_cycle
            else:
                self.current_value = self.final_value
        
        self.current_step += 1
        return self.current_value
    
    def get_value(self) -> float:
        if self.total_steps == 0:
            return self.final_value
        
        cycle = int(self.current_step / self.period)
        pos_in_cycle = self.current_step - cycle * self.period
        
        if pos_in_cycle < (self.period * self.ratio):
            return self.initial_value + self.step_size * pos_in_cycle
        return self.final_value
    
    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state.update({
            'final_value': self.final_value,
            'total_steps': self.total_steps,
            'n_cycle': self.n_cycle,
            'ratio': self.ratio,
            'period': self.period,
            'step_size': self.step_size
        })
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        super().load_state_dict(state_dict)
        self.final_value = state_dict['final_value']
        self.total_steps = state_dict['total_steps']
        self.n_cycle = state_dict['n_cycle']
        self.ratio = state_dict['ratio']
        self.period = state_dict['period']
        self.step_size = state_dict['step_size']

def get_scheduler(scheduler_type: str, **kwargs) -> BaseHyperparameterScheduler:
    """Factory function to create schedulers."""
    schedulers = {
        'linear': LinearScheduler,
        'exponential': ExponentialScheduler,
        'cosine': CosineScheduler,
        'cyclical': CyclicalAnnealingScheduler,
    }
    
    if scheduler_type not in schedulers:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Available: {list(schedulers.keys())}")
    
    return schedulers[scheduler_type](**kwargs)