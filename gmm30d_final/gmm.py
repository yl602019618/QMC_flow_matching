"""
Target distribution, dataset, analytic log-density, and the two test
integrands (with closed-form expectations) for the 30D GMM experiment.

Target pi : 4-component Gaussian mixture in d = 30 dimensions
    pi(x) = (1/4) sum_{k=1}^4 N(x | mu_k, SIGMA),   SIGMA = VAR * I_30
    means nonzero only on the first two coordinates:
        mu_1=(-2,-2,0,..), mu_2=(2,-2,..), mu_3=(-2,2,..), mu_4=(2,2,..)
    VAR = 5.0  (large-variance / "iso-large" regime).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.special import ndtr
from scipy.stats import norm

DIM = 30
K = 4
VAR = 5.0                                     # isotropic per-coordinate variance
WEIGHTS = np.full(K, 1.0 / K, dtype=np.float64)
PHI_INV_SQRT2 = float(norm.cdf(1.0 / np.sqrt(2.0)))   # constant offset in complex f

MEANS = np.zeros((K, DIM), dtype=np.float64)
MEANS[0, :2] = [-2., -2.]
MEANS[1, :2] = [ 2., -2.]
MEANS[2, :2] = [-2.,  2.]
MEANS[3, :2] = [ 2.,  2.]


# ---------------------------------------------------------------------------
# Dataset for training the flow (vectorized sampler from pi)
# ---------------------------------------------------------------------------

class GMMDataset(Dataset):
    def __init__(self, num_samples=1_000_000, seed=0):
        rng = np.random.default_rng(seed)
        comp = rng.integers(0, K, size=num_samples)
        z = rng.standard_normal((num_samples, DIM)).astype(np.float32) * np.sqrt(VAR)
        self.data = (MEANS[comp] + z).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])


# ---------------------------------------------------------------------------
# Analytic log-density of pi (used by SNIS to form importance weights)
# ---------------------------------------------------------------------------

def log_prob_torch(x: torch.Tensor) -> torch.Tensor:
    """log pi(x) for cov = VAR * I_30, 4 components, equal weights. x: (N, d)."""
    device = x.device
    means = torch.tensor(MEANS, dtype=x.dtype, device=device)
    norm_const = (2 * np.pi) ** (-DIM / 2) * (VAR ** (-DIM / 2))
    diff = x.unsqueeze(0) - means.unsqueeze(1)        # (K, N, d)
    md = diff.pow(2).sum(dim=2) / VAR                 # (K, N)
    comp = norm_const * torch.exp(-0.5 * md)
    return torch.log(torch.clamp(0.25 * comp.sum(dim=0), min=1e-38))


# ---------------------------------------------------------------------------
# Integrands  f : R^d -> R (complex) or R^d -> R^d (second moment)
# ---------------------------------------------------------------------------

def f_complex_torch(x: torch.Tensor) -> torch.Tensor:
    """f(x) = -Phi(1/sqrt 2) + Phi(1 + (1/sqrt d) sum_j x_j).  Returns (N,)."""
    s = 1.0 + x.sum(dim=1) / np.sqrt(DIM)
    return torch.special.ndtr(s) - PHI_INV_SQRT2


def f_second_torch(x: torch.Tensor) -> torch.Tensor:
    """f(x) = x^2 (per coordinate).  Returns (N, d)."""
    return x ** 2


def true_complex() -> float:
    """E_pi[f_complex] = -Phi(1/sqrt2) + sum_k w_k Phi((1+a_k)/sqrt(1+b_k^2)),
    a_k = 1^T mu_k / sqrt d, b_k^2 = 1^T Sigma_k 1 / d = VAR."""
    a = MEANS.sum(axis=1) / np.sqrt(DIM)
    return float(-PHI_INV_SQRT2 + np.sum(WEIGHTS * ndtr((1.0 + a) / np.sqrt(1.0 + VAR))))


def true_second() -> np.ndarray:
    """E_pi[x_j^2] = VAR + sum_k w_k mu_{k,j}^2  (per coordinate)."""
    return VAR + WEIGHTS @ (MEANS ** 2)


INTEGRANDS = {
    "complex": {"fn": f_complex_torch, "is_vector": False, "true": true_complex,
                "label": "Complex f"},
    "second":  {"fn": f_second_torch,  "is_vector": True,  "true": true_second,
                "label": "Second moment"},
}
