"""
Integrand registry for the 2D GMM experiment.

Three integrands evaluated under the same target pi:
    - first  : f(x) = x                 (vector-valued, true = E_pi[X])
    - second : f(x) = x ** 2            (vector-valued, true = diag E_pi[X X^T])
    - complex: f(x) = -Phi(1/sqrt(2)) + Phi(1 + sum_j x_j / sqrt(d))
                                        (scalar-valued, analytic via target_function.py)

Each spec exposes:
    fx           : torch callable, (N, d) -> (N,) or (N, d)
    is_vector    : bool, controls RMSE aggregation
    true_value() : np.ndarray (vector) or float (scalar)
"""

import numpy as np
import torch

from target_function import f_torch as _complex_f_torch
from target_function import true_expectation_2d


MEANS_2D = np.array([[-2., -2.],
                     [ 2., -2.],
                     [-2.,  2.],
                     [ 2.,  2.]], dtype=np.float64)
COV_2D = np.array([[0.5, 0.1],
                   [0.1, 0.5]], dtype=np.float64)
WEIGHTS_2D = np.full(4, 0.25, dtype=np.float64)


def _true_first_2d():
    return MEANS_2D.mean(axis=0)                                   # (2,)


def _true_second_2d():
    s = np.zeros(2, dtype=np.float64)
    for k in range(4):
        s += WEIGHTS_2D[k] * (np.diag(COV_2D) + MEANS_2D[k] ** 2)
    return s                                                       # E[X^2] per coord


def _fx_first(x: torch.Tensor) -> torch.Tensor:
    return x                                                       # (N, d)


def _fx_second(x: torch.Tensor) -> torch.Tensor:
    return x ** 2                                                  # (N, d)


def _fx_complex(x: torch.Tensor) -> torch.Tensor:
    return _complex_f_torch(x)                                     # (N,)


INTEGRANDS = {
    "first": {
        "fx": _fx_first,
        "is_vector": True,
        "true_value": _true_first_2d,
        "label": "First moment",
    },
    "second": {
        "fx": _fx_second,
        "is_vector": True,
        "true_value": _true_second_2d,
        "label": "Second moment",
    },
    "complex": {
        "fx": _fx_complex,
        "is_vector": False,
        "true_value": true_expectation_2d,
        "label": "Complex f",
    },
}


def reduce_mean(values: torch.Tensor) -> np.ndarray:
    """
    Average f(x) over the sample dimension.
    values: (N,) or (N, d) torch tensor.
    Returns numpy array of shape () or (d,).
    """
    return values.mean(dim=0).detach().cpu().numpy()


def weighted_sum(values: torch.Tensor, w: torch.Tensor) -> np.ndarray:
    """
    SNIS estimator:  sum_i w_i * f(x_i).
    values: (N,) or (N, d); w: (N,).
    Returns numpy array of shape () or (d,).
    """
    if values.dim() == 1:
        return float(torch.sum(w * values).item())
    return torch.sum(w.unsqueeze(1) * values, dim=0).detach().cpu().numpy()


def rmse(estimates, true_value, is_vector):
    """
    estimates: list/array of shape (R,) or (R, d)
    Returns (mean_est, rmse, bias_rms, var_rms).
    """
    estimates = np.asarray(estimates)
    if is_vector:
        mean_est = estimates.mean(axis=0)
        bias = mean_est - np.asarray(true_value)
        bias_sq = float(np.sum(bias ** 2))
        var = float(np.mean(np.sum((estimates - mean_est) ** 2, axis=1)))
    else:
        mean_est = float(estimates.mean())
        bias_sq = (mean_est - float(true_value)) ** 2
        var = float(np.mean((estimates - mean_est) ** 2))
    return mean_est, float(np.sqrt(bias_sq + var)), float(np.sqrt(bias_sq)), float(np.sqrt(var))
