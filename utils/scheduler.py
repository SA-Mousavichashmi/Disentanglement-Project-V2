import abc
import torch
from typing import Any, Dict, Union

class BaseHyperparameterScheduler(abc.ABC):
    """Base class for hyperparameter schedulers."""
    
    def __init__(self, param_name: str, initial_value: float, start_step: int = 0, **kwargs):
        if start_step < 0:
            raise ValueError("start_step must be non-negative")
        self.param_name = param_name
        self.initial_value = initial_value
        self.start_step = start_step
        self.current_step = 0
        self.current_value = initial_value
    
    @property
    @abc.abstractmethod
    def name(self):
        """A unique name for the scheduler, to be implemented by subclasses."""
        pass

    @property
    @abc.abstractmethod
    def kwargs(self):
        """A dictionary of keyword arguments for the scheduler, to be implemented by subclasses."""
        pass

    @abc.abstractmethod
    def step(self) -> float:
        """Update and return the current parameter value."""
        pass
    
    @abc.abstractmethod
    def get_value(self) -> float:
        """Get current parameter value without stepping."""
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Return the state of the scheduler (current_step only)."""
        return {
            'current_step': self.current_step
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load the state of the scheduler (current_step only)."""
        self.current_step = state_dict['current_step']
        self.current_value = self.get_value() # Recalculate current_value

class LinearScheduler(BaseHyperparameterScheduler):
    """Linear annealing scheduler."""
    
    def __init__(self, param_name: str, initial_value: float, final_value: float, 
                 total_steps: int, **kwargs):
        super().__init__(param_name, initial_value, **kwargs)
        self.final_value = final_value
        self.total_steps = total_steps
    
    @property
    def name(self):
        return "LinearScheduler"

    @property
    def kwargs(self):
        return {
            'param_name': self.param_name,
            'initial_value': self.initial_value,
            'final_value': self.final_value,
            'start_step': self.start_step,
            'total_steps': self.total_steps
        }
    
    def step(self) -> float:
        if self.current_step < self.start_step:
            self.current_step += 1
            return self.current_value
            
        adjusted_step = self.current_step - self.start_step
        adjusted_total_steps = self.total_steps
        
        if adjusted_total_steps == 0:
            self.current_value = self.final_value
        else:
            progress = min(adjusted_step / adjusted_total_steps, 1.0)
            self.current_value = self.initial_value + (self.final_value - self.initial_value) * progress
        
        self.current_step += 1
        return self.current_value
    
    def get_value(self) -> float:
        if self.current_step < self.start_step:
            return self.current_value
            
        adjusted_step = self.current_step - self.start_step
        adjusted_total_steps = self.total_steps
        
        if adjusted_total_steps == 0:
            return self.final_value
        progress = min(adjusted_step / adjusted_total_steps, 1.0)
        return self.initial_value + (self.final_value - self.initial_value) * progress

    # state_dict and load_state_dict now handled by base class

class ExponentialScheduler(BaseHyperparameterScheduler):
    """Exponential decay scheduler."""
    
    def __init__(self, param_name: str, initial_value: float, decay_rate: float, start_step: int = 0, **kwargs):
        super().__init__(param_name, initial_value, start_step, **kwargs)
        self.decay_rate = decay_rate
    
    @property
    def name(self):
        return "ExponentialScheduler"

    @property
    def kwargs(self):
        return {
            'param_name': self.param_name,
            'initial_value': self.initial_value,
            'start_step': self.start_step,
            'decay_rate': self.decay_rate
        }
    
    def step(self) -> float:
        if self.current_step < self.start_step:
            self.current_step += 1
            return self.current_value
            
        adjusted_step = self.current_step - self.start_step
        self.current_value = self.initial_value * (self.decay_rate ** adjusted_step)
        self.current_step += 1
        return self.current_value
    
    def get_value(self) -> float:
        if self.current_step < self.start_step:
            return self.current_value
        adjusted_step = self.current_step - self.start_step
        return self.initial_value * (self.decay_rate ** adjusted_step)

    # state_dict and load_state_dict now handled by base class

class CosineScheduler(BaseHyperparameterScheduler):
    """Cosine annealing scheduler."""
    
    def __init__(self, param_name: str, initial_value: float, final_value: float,
                 total_steps: int, start_step: int = 0, **kwargs):
        super().__init__(param_name, initial_value, start_step, **kwargs)
        self.final_value = final_value
        self.total_steps = total_steps
    
    @property
    def name(self):
        return "CosineScheduler"

    @property
    def kwargs(self):
        return {
            'param_name': self.param_name,
            'initial_value': self.initial_value,
            'final_value': self.final_value,
            'start_step': self.start_step,
            'total_steps': self.total_steps
        }
    
    def step(self) -> float:
        if self.current_step < self.start_step:
            self.current_step += 1
            return self.current_value
            
        adjusted_step = self.current_step - self.start_step
        adjusted_total_steps = self.total_steps
        
        if adjusted_total_steps == 0:
            self.current_value = self.final_value
        else:
            progress = min(adjusted_step / adjusted_total_steps, 1.0)
            cosine_factor = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
            self.current_value = self.final_value + (self.initial_value - self.final_value) * cosine_factor
        
        self.current_step += 1
        return self.current_value
    
    def get_value(self) -> float:
        if self.current_step < self.start_step:
            return self.current_value
            
        adjusted_step = self.current_step - self.start_step
        adjusted_total_steps = self.total_steps
        
        if adjusted_total_steps == 0:
            return self.final_value
        progress = min(adjusted_step / adjusted_total_steps, 1.0)
        cosine_factor = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
        return self.final_value + (self.initial_value - self.final_value) * cosine_factor

    # state_dict and load_state_dict now handled by base class

class CyclicalAnnealingScheduler(BaseHyperparameterScheduler):
    """Cyclical annealing scheduler based on https://github.com/haofuml/cyclical_annealing"""
    
    def __init__(self, param_name: str, initial_value: float, final_value: float,
                 total_steps: int, n_cycle: int = 4, ratio: float = 0.5, start_step: int = 0, **kwargs):
        super().__init__(param_name, initial_value, start_step, **kwargs)
        self.final_value = final_value
        self.total_steps = total_steps
        self.n_cycle = n_cycle
        self.ratio = ratio
        self.period = total_steps / n_cycle
        self.step_size = (final_value - initial_value) / (self.period * ratio)
    
    @property
    def name(self):
        return "CyclicalAnnealingScheduler"

    @property
    def kwargs(self):
        return {
            'param_name': self.param_name,
            'initial_value': self.initial_value,
            'final_value': self.final_value,
            'start_step': self.start_step,
            'total_steps': self.total_steps,
            'n_cycle': self.n_cycle,
            'ratio': self.ratio
        }
    
    def step(self) -> float:
        if self.current_step < self.start_step:
            self.current_step += 1
            return self.current_value
            
        adjusted_step = self.current_step - self.start_step
        adjusted_total_steps = self.total_steps
        adjusted_period = adjusted_total_steps / self.n_cycle if self.n_cycle > 0 else 0
        adjusted_step_size = (self.final_value - self.initial_value) / (adjusted_period * self.ratio) if adjusted_period * self.ratio > 0 else 0
        
        if adjusted_total_steps == 0:
            self.current_value = self.final_value
        else:
            cycle = int(adjusted_step / adjusted_period) if adjusted_period > 0 else 0
            pos_in_cycle = adjusted_step - cycle * adjusted_period
            
            if pos_in_cycle < (adjusted_period * self.ratio) and adjusted_period > 0:
                self.current_value = self.initial_value + adjusted_step_size * pos_in_cycle
            else:
                self.current_value = self.final_value
        
        self.current_step += 1
        return self.current_value
    
    def get_value(self) -> float:
        if self.current_step < self.start_step:
            return self.current_value
            
        adjusted_step = self.current_step - self.start_step
        adjusted_total_steps = self.total_steps
        adjusted_period = adjusted_total_steps / self.n_cycle if self.n_cycle > 0 else 0
        adjusted_step_size = (self.final_value - self.initial_value) / (adjusted_period * self.ratio) if adjusted_period * self.ratio > 0 else 0
        
        if adjusted_total_steps == 0:
            return self.final_value
        
        cycle = int(adjusted_step / adjusted_period) if adjusted_period > 0 else 0
        pos_in_cycle = adjusted_step - cycle * adjusted_period
        
        if pos_in_cycle < (adjusted_period * self.ratio) and adjusted_period > 0:
            return self.initial_value + adjusted_step_size * pos_in_cycle
        return self.final_value
    
    # state_dict and load_state_dict now handled by base class

def get_scheduler(name: str, **kwargs) -> BaseHyperparameterScheduler:
    """Factory function to create schedulers."""
    schedulers = {
        'linear': LinearScheduler,
        'exponential': ExponentialScheduler,
        'cosine': CosineScheduler,
        'cyclical': CyclicalAnnealingScheduler,
    }
    
    if name not in schedulers:
        raise ValueError(f"Unknown scheduler type: {name}. Available: {list(schedulers.keys())}")
    
    return schedulers[name](**kwargs)