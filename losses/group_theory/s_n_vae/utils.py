import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def generate_group_action_parameters(data_num,
                                    selected_component_indices,
                                    latent_factor_topologies,
                                    r1_range=1,
                                    s1_range= 2 * torch.pi,
                                    r1_dist='normal',
                                    s1_dist='uniform'
                                    ):
    """
    Generates parameters for group actions based on specified topologies.

    Supports both R1 (translation) and S1 (rotation) topologies.

    Args:
        data_num (int): The number of transformation sets to generate.
                        Should match selected_component_indices.size(0).
        selected_component_indices (torch.Tensor): Indices of the selected components to transform for each data point.
                                                 Shape (data_num, component_order).
        latent_factor_topologies (list): List of topology types for each latent factor (e.g., 'R1', 'S1').
                                       Length must be equal to the number of latent factors.
        r1_range (float): For uniform distribution: the range [-r1_range, r1_range] for R1.
                         For normal distribution: the standard deviation for R1.
        s1_range (float): For uniform distribution: the range [0, s1_range] for S1 angles.
                         For normal distribution: the standard deviation for S1.
        r1_dist (str): The distribution to sample R1 parameters from. Either 'uniform' or 'normal'.
        s1_dist (str): The distribution to sample S1 parameters from. Either 'uniform' or 'normal'.

    Returns:
        list: A list of lists. The outer list has length `data_num`.
              Each inner list corresponds to a data sample and contains dictionaries,
              where each dictionary represents a specific transformation to be applied.
              Each transformation dictionary has the following keys:
              - 'component_index' (int): The global index (0 to factor_num-1) of the
                                         latent component to be transformed.
              - 'topology' (str): The topology of the component ('R1' or 'S1').
              - 'value' (torch.Tensor): A scalar tensor representing the transformation magnitude.
                                        For 'R1', this is the translation amount.
                                        For 'S1', this is the rotation angle in radians.
    """
    device = selected_component_indices.device
    
    all_samples_transform_params = [] # List to be returned

    for batch_idx in range(data_num):
        # List of specific transformation operations for the current data sample
        current_sample_specific_transforms = []
        
        # selected_component_indices[batch_idx] is a 1D tensor of component indices
        # that were selected to be transformed for this specific data sample.
        for selected_comp_tensor in selected_component_indices[batch_idx]:
            selected_comp = selected_comp_tensor.item() # Global component index
            
            if not (0 <= selected_comp < len(latent_factor_topologies)):
                # This check ensures selected_comp is a valid index for latent_factor_topologies.
                # It should ideally not be triggered if inputs are correctly formed.
                raise IndexError(
                    f"Selected component index {selected_comp} is out of bounds "
                    f"for latent_factor_topologies with length {len(latent_factor_topologies)}."
                )
            
            topology = latent_factor_topologies[selected_comp]
            transform_value = None # Placeholder for the generated transformation value
            
            if topology == 'R1':
                if r1_dist == 'uniform':
                    transform_value = (2 * torch.rand(1, device=device) - 1) * r1_range
                elif r1_dist == 'normal':
                    transform_value = torch.randn(1, device=device) * r1_range
                else:
                    raise ValueError(f"Unsupported R^1 distribution '{r1_dist}'. Must be 'uniform' or 'normal'."
                                    )
                
            elif topology == 'S1':
                # For S^1, the value generated is an angle.
                if s1_dist == 'uniform':
                    transform_value = torch.rand(1, device=device) * s1_range
                elif s1_dist == 'normal':
                    # s1_range acts as standard deviation for normal distribution of angles
                    transform_value = torch.randn(1, device=device) * s1_range 
                else:
                    raise ValueError(f"Unsupported S^1 distribution '{s1_dist}'. Must be 'uniform' or 'normal'.")
            else:
                # This case handles unknown topologies, which implies an issue with input `latent_factor_topologies`.
                raise ValueError(f"Unsupported topology '{topology}' encountered for component {selected_comp}.")

            # Ensure a value was generated (e.g., topology was valid)
            if transform_value is not None:
                current_sample_specific_transforms.append({
                    'component_index': selected_comp,
                    'topology': topology,
                    'value': transform_value.squeeze() # Store as a scalar tensor
                })
        
        all_samples_transform_params.append(current_sample_specific_transforms)
        
    return all_samples_transform_params


def apply_group_action_latent_space(group_action_params, latent_rep, latent_factor_topologies):
    """Applies a group action to the latent space representation.

    Supports both R1 (translation) and S1 (rotation) topologies.

    Args:
        group_action_params (list): A list of lists. The outer list has length `batch_size`.
                                  Each inner list corresponds to a data sample and contains dictionaries,
                                  where each dictionary represents a specific transformation to be applied.
                                  Each transformation dictionary has the following keys:
                                  - 'component_index' (int): The logical index (0 to num_logical_factors-1) of the
                                                             latent component to be transformed.
                                  - 'topology' (str): The topology of the component ('R1' or 'S1').
                                  - 'value' (torch.Tensor): A scalar tensor representing the transformation magnitude.
                                                            For 'R1', this is the translation amount.
                                                            For 'S1', this is the rotation angle in radians.
        latent_rep (torch.Tensor): Input latent representations.
            Shape (batch_size, factor_num).
        latent_factor_topologies (list): List of topology types for each logical latent factor.
                                       Length must be num_logical_factors.

    Returns:
        torch.Tensor: Transformed latent space with same shape as input.
    """
    transformed_latent = latent_rep.clone()
    
    # Pre-calculate the mapping from logical component index to actual starting dimension
    logical_to_actual_dim_map = {}
    current_actual_dim = 0
    for logical_idx, topology in enumerate(latent_factor_topologies):
        logical_to_actual_dim_map[logical_idx] = current_actual_dim
        if topology == 'R1':
            current_actual_dim += 1
        elif topology == 'S1':
            current_actual_dim += 2
        else:
            raise ValueError(f"Unsupported topology '{topology}' in latent_factor_topologies.")

    for batch_idx, sample_transforms in enumerate(group_action_params):
        for transform_dict in sample_transforms:
            logical_component_index = transform_dict['component_index']
            topology = transform_dict['topology']
            value = transform_dict['value']
            
            actual_start_dim = logical_to_actual_dim_map[logical_component_index]
            
            if topology == 'R1':
                transformed_latent[batch_idx, actual_start_dim] += value
            elif topology == 'S1':
                # Get the 2D pair for S^1 rotation
                cos_sin_pair = transformed_latent[batch_idx, actual_start_dim : actual_start_dim + 2]
                # Apply rotation and update the segment in transformed_latent
                transformed_latent[batch_idx, actual_start_dim : actual_start_dim + 2] = _rotate_s1(cos_sin_pair, value)
            else:
                # This should be caught by the initial map creation, but as a safeguard
                raise ValueError(f"Unsupported topology '{topology}' encountered for component {logical_component_index}.")
    
    return transformed_latent


def _rotate_s1(cos_sin_tensor, rotation_angle):
    """
    Helper function to rotate a 2D (cos, sin) representation on S1.
    
    Args:
        cos_sin_tensor (torch.Tensor): Tensor of shape (..., 2) containing (cos, sin) values.
        rotation_angle (torch.Tensor): Rotation angles to apply (scalar tensor).
    
    Returns:
        torch.Tensor: The rotated 2D (cos, sin) tensor.
    """
    cos_val = cos_sin_tensor[..., 0]
    sin_val = cos_sin_tensor[..., 1]
    
    # Convert to angle, apply rotation, convert back
    current_angle = torch.atan2(sin_val, cos_val)
    new_angle = current_angle + rotation_angle
    
    # Return new tensor with rotated values
    return torch.stack((torch.cos(new_angle), torch.sin(new_angle)), dim=-1)
