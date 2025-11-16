# utils.py
import numpy as np


def h(x):
    """
    True function h(x) = sin(2*x1) + x1 * cos(x1 - 1)
    Only the first feature x1 is relevant; others are irrelevant.
    
    Parameters
    ----------
    x : ndarray of shape (n_samples, p)
    
    Returns
    -------
    h_x : ndarray of shape (n_samples,)
    """
    x1 = x[:, 0]
    return np.sin(2 * x1) + x1 * np.cos(x1 - 1)

def generate_X(n, p, low=-10, high=10, seed=None):
    """
    Generate random input matrix X ∈ R^(n×p) uniformly in [low, high]
    
    Parameters
    ----------
    n : int
        Number of samples
    p : int
        Number of features
    low : float
        Lower bound
    high : float
        Upper bound
    seed : int or None
        Random seed for reproducibility
    
    Returns
    -------
    X : ndarray of shape (n, p)
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n, p))

def generate_data(n, p, sigma=1.0, seed=None):
    """
    Generate a dataset (X, y) with noise
    
    Parameters
    ----------
    n : int
        Number of samples
    p : int
        Number of features
    sigma : float
        Standard deviation of Gaussian noise
    seed : int or None
        Random seed for reproducibility
    
    Returns
    -------
    X : ndarray of shape (n, p)
    y : ndarray of shape (n,)
    """
    X = generate_X(n, p, seed=seed)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, size=n)
    y = h(X) + noise
    return X, y


def display_results(name, results):
    """
    Nicely print bias-variance decomposition results for a model.

    Parameters
    ----------
    name : str
        Name of the model
    results : dict
        Dictionary returned by estimate_bias_variance containing keys:
        'bias2', 'variance', 'residual', 'total', 'y_pred_mean', 'y_true', 'X_test'
    """
    print(f"--- {name} ---")
    print(f"Bias²      : {results['bias2']:.4f}")
    print(f"Variance   : {results['variance']:.4f}")
    print(f"Residual   : {results['residual']:.4f}")
    print(f"Total error: {results['total']:.4f}")
    print("\n")
