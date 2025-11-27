# utils.py
import numpy as np
from sklearn.ensemble import BaggingRegressor  # <--- add this
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


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

import numpy as np

def estimate_bias_variance(model_class, model_params, N=80, p=5, M=100, sigma=1.0, seed=None, generate_data=None, h=None):

    if generate_data is None or h is None:
        raise ValueError("generate_data and h functions must be provided")

    rng = np.random.default_rng(seed)

    # Generate test set
    X_test, _ = generate_data(500, p, sigma=0, seed=seed)
    n_test = X_test.shape[0]
    all_preds = np.zeros((M, n_test))

    for m in range(M):
        X_train, y_train = generate_data(N, p, sigma=sigma,
                                         seed=None if seed is None else seed + m)
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        all_preds[m] = model.predict(X_test)

    y_pred_mean = all_preds.mean(axis=0)
    y_true = h(X_test)

    bias2 = np.mean((y_true - y_pred_mean)**2)
    variance = np.mean(np.var(all_preds, axis=0))
    residual = sigma**2
    total = bias2 + variance + residual

    return bias2, variance, residual, total



def bagging_bias_variance(base_model_class, model_params, N=80, p=5, M=100, sigma=1.0, seed=None, n_estimators_list=[1,5,10,20]):
    results = []

    for n_models in n_estimators_list:
        all_preds = np.zeros((M, 500))  # fixed test set size
        X_test, _ = generate_data(500, p, sigma=0, seed=seed)

        for m in range(M):
            X_train, y_train = generate_data(N, p, sigma=sigma,
                                             seed=None if seed is None else seed + m)
            base_model = base_model_class(**model_params)
            bag_model = BaggingRegressor(base_model, n_estimators=n_models, random_state=seed)
            bag_model.fit(X_train, y_train)
            all_preds[m] = bag_model.predict(X_test)

        y_pred_mean = all_preds.mean(axis=0)
        y_true = h(X_test)

        bias2 = np.mean((y_true - y_pred_mean)**2)
        variance = np.mean(np.var(all_preds, axis=0))
        residual = sigma**2
        total = bias2 + variance + residual

        results.append([n_models, bias2, variance, residual, total])

    return np.array(results)  # columns: n_models, bias², variance, residual, total
