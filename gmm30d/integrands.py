"""
Integrand registry for the 30D GMM experiment.

cov = 0.5 * I_30. Means are nonzero only on the first two dims.
"""

import numpy as np
import torch

from target_function import f_torch as _complex_f_torch
from target_function import true_expectation_30d


DIM = 30
K = 4
SIGMA2 = 0.5

MEANS_30D = np.zeros((K, DIM), dtype=np.float64)
MEANS_30D[0, :2] = [-2., -2.]
MEANS_30D[1, :2] = [ 2., -2.]
MEANS_30D[2, :2] = [-2.,  2.]
MEANS_30D[3, :2] = [ 2.,  2.]
WEIGHTS_30D = np.full(K, 1.0 / K, dtype=np.float64)


def _true_first_30d():
    return MEANS_30D.mean(axis=0)                                  # all zero


def _true_second_30d():
    s = np.zeros(DIM, dtype=np.float64)
    for k in range(K):
        s += WEIGHTS_30D[k] * (SIGMA2 + MEANS_30D[k] ** 2)
    return s                                                       # 4.5 on first two dims, 0.5 elsewhere


def _fx_first(x: torch.Tensor) -> torch.Tensor:
    return x


def _fx_second(x: torch.Tensor) -> torch.Tensor:
    return x ** 2


def _fx_complex(x: torch.Tensor) -> torch.Tensor:
    return _complex_f_torch(x)


INTEGRANDS = {
    "first": {
        "fx": _fx_first,
        "is_vector": True,
        "true_value": _true_first_30d,
        "label": "First moment",
    },
    "second": {
        "fx": _fx_second,
        "is_vector": True,
        "true_value": _true_second_30d,
        "label": "Second moment",
    },
    "complex": {
        "fx": _fx_complex,
        "is_vector": False,
        "true_value": lambda: true_expectation_30d(dim=DIM),
        "label": "Complex f",
    },
}


def reduce_mean(values: torch.Tensor) -> np.ndarray:
    return values.mean(dim=0).detach().cpu().numpy()


def weighted_sum(values: torch.Tensor, w: torch.Tensor) -> np.ndarray:
    if values.dim() == 1:
        return float(torch.sum(w * values).item())
    return torch.sum(w.unsqueeze(1) * values, dim=0).detach().cpu().numpy()


def rmse(estimates, true_value, is_vector):
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
