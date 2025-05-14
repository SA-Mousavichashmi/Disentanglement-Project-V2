from losses import select


def create_load_loss(loss_name, loss_kwargs, loss_state_dict):
    """Creates a loss function and loads its state dictionary.

    Parameters
    ----------
    loss_name : str
        The name of the loss function to create.
    loss_kwargs : dict
        The keyword arguments to pass to the loss function constructor.
    loss_state_dict : dict
        The state dictionary containing the parameters of the loss function.
    """
    # Create the loss function
    loss = select(loss_name, **loss_kwargs)
    
    # Load the state dictionary into the loss function
    loss.load_state_dict(loss_state_dict)
    
    return loss
