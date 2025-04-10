import numpy as np
from scipy import stats
import sklearn.metrics

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