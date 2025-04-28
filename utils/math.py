import numpy as np
from scipy import stats
import sklearn.metrics
import math
import torch

# Import pyitlib if available for advanced information theory computations
try:
    from pyitlib import discrete_random_variable as drv
except ImportError:
    drv = None



######################### Mutual Information #########################

def mutual_information_pyitlib(x, y):
    """Calculate mutual information using pyitlib.
    
    Note: The inputs x and y are expected to be discrete random variables.
    
    Args:
        x: First variable (array-like)
        y: Second variable (array-like)
    
    Returns:
        float: Mutual information between x and y
    """
    if drv is None:
        raise ImportError("pyitlib is required for this function. Install with 'pip install pyitlib'")
    return drv.information_mutual(x, y, cartesian_product=True, base=np.e)

def mutual_information_sklearn(x, y):
    """Calculate mutual information using sklearn.
    
    Note: The inputs x and y are expected to be discrete random variables.
    
    Args:
        x: First variable (array-like)
        y: Second variable (array-like)
    
    Returns:
        float: Mutual information between x and y
    """
    return sklearn.metrics.mutual_info_score(x, y)

def mutual_information_numpy(x, y):
    """Calculate mutual information using numpy-based implementation for already digitized data.
    
    Note: The inputs x and y are expected to be discrete random variables.
    
    Args:
        x: First variable (array-like)
        y: Second variable (array-like)
    
    Returns:
        float: Mutual information between x and y
    """
    # Find unique values in x and y
    unique_x = np.unique(x)
    unique_y = np.unique(y)
    
    # Create a contingency table/joint counts matrix based on actual values
    joint_counts = np.zeros((len(unique_x), len(unique_y)), dtype=np.int64)
    
    # Create mappings from values to indices
    x_map = {val: idx for idx, val in enumerate(unique_x)}
    y_map = {val: idx for idx, val in enumerate(unique_y)}
    
    # Populate joint counts
    for i, j in zip(x, y):
        joint_counts[x_map[i], y_map[j]] += 1
    
    # Calculate joint probability
    total = len(x)
    joint_prob = joint_counts / total
    
    # Calculate marginal probabilities
    px = np.sum(joint_prob, axis=1)
    py = np.sum(joint_prob, axis=0)
    
    # Calculate mutual information
    mi = 0
    for i in range(len(px)):
        for j in range(len(py)):
            if joint_prob[i, j] > 0:
                mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (px[i] * py[j]))
    return mi

def calculate_mutual_information(x, y, method='sklearn'):
    """Calculate mutual information using the selected method.
    
    Note: All methods assume that the input variables are discrete random variables.
          The inputs x and y should be provided as such.
    
    Args:
        x: First discrete random variable (array-like)
        y: Second discrete random variable (array-like)
        method: Method to use ('pyitlib', 'sklearn', or 'numpy')
    
    Returns:
        float: Mutual information between x and y
    """
    if method == 'pyitlib':
        return mutual_information_pyitlib(x, y)
    elif method == 'sklearn':
        return mutual_information_sklearn(x, y)
    elif method == 'numpy':
        return mutual_information_numpy(x, y)
    else:
        raise ValueError(f"Unknown mutual information method: {method}")

############################### Entropy ##############################

def entropy_pyitlib(x):
    """Calculate entropy using pyitlib.
    
    Note: The input x is expected to be a discrete random variable.
    
    Args:
        x: Input variable (array-like)
    
    Returns:
        float: Entropy of x
    """
    if drv is None:
        raise ImportError("pyitlib is required for this function. Install with 'pip install pyitlib'")
    return drv.entropy(x, base=np.e)

def entropy_scipy(x):
    """Calculate entropy using scipy.
    
    Note: The input x is expected to be a discrete random variable.
    
    Args:
        x: Input variable (array-like)
    
    Returns:
        float: Entropy of x
    """
    value, counts = np.unique(x, return_counts=True)
    probs = counts / len(x)
    return stats.entropy(probs, base=np.e)

def entropy_numpy(x):
    """Calculate entropy using numpy-based implementation.
    
    Note: The input x is expected to be a discrete random variable.
    
    Args:
        x: Input variable (array-like)
    
    Returns:
        float: Entropy of x
    """
    value, counts = np.unique(x, return_counts=True)
    probs = counts / len(x)
    return -np.sum(probs * np.log(probs + 1e-10))

def calculate_entropy(x, method='scipy'):
    """Calculate entropy using the selected method.
    
    Note: All methods assume that the input variable is a discrete random variable.
          The input x should be provided as such.
    
    Args:
        x: Input discrete random variable (array-like)
        method: Method to use ('pyitlib', 'scipy', or 'numpy')
    
    Returns:
        float: Entropy of x
    """
    if method == 'pyitlib':
        return entropy_pyitlib(x)
    elif method == 'scipy':
        return entropy_scipy(x)
    elif method == 'numpy':
        return entropy_numpy(x)
    else:
        raise ValueError(f"Unknown entropy method: {method}")

############################## Log Density #############################

def matrix_log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian for all combination of batch pairs of
    `x` and `mu`. I.e. return tensor of shape `(batch_size, batch_size, dim)`
    instead of (batch_size, dim) in the usual log density.

    Parameters
    ----------
    x: torch.Tensor
        Value at which to compute the density. Shape: (batch_size, dim).

    mu: torch.Tensor
        Mean. Shape: (batch_size, dim).

    logvar: torch.Tensor
        Log variance. Shape: (batch_size, dim).

    batch_size: int
        number of training images in the batch
    """
    batch_size, dim = x.shape
    x = x.view(batch_size, 1, dim)
    mu = mu.view(1, batch_size, dim)
    logvar = logvar.view(1, batch_size, dim)
    return log_density_gaussian(x, mu, logvar)


def log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian.

    Parameters
    ----------
    x: torch.Tensor or np.ndarray or float
        Value at which to compute the density.

    mu: torch.Tensor or np.ndarray or float
        Mean.

    logvar: torch.Tensor or np.ndarray or float
        Log variance.
    """
    normalization = -0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)
    return log_density


def log_importance_weight_matrix(batch_size, dataset_size):
    """
    Calculates a log importance weight matrix

    Parameters
    ----------
    batch_size: int
        number of training images in the batch

    dataset_size: int
    number of training images in the dataset
    """
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M + 1] = 1 / N
    W.view(-1)[1::M + 1] = strat_weight
    W[M - 1, 0] = strat_weight
    return W.log()
