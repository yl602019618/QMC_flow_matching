"""
Complex test integrand and exact GMM expectation.

f(x) = -Phi(1/sqrt(2)) + Phi(1 + (1/sqrt(d)) * sum_j x_j)

For X ~ N(mu, Sigma), c = 1/sqrt(d) * 1_d:
  c^T X ~ N(a, b^2),  a = 1^T mu / sqrt(d),  b^2 = 1^T Sigma 1 / d
  E[Phi(1 + c^T X)] = Phi((1 + a) / sqrt(1 + b^2))

Therefore for a GMM pi = sum_k w_k N(mu_k, Sigma_k):
  E_pi[f] = -Phi(1/sqrt(2)) + sum_k w_k * Phi((1 + a_k) / sqrt(1 + b_k^2))
"""

import numpy as np
import torch
from scipy.stats import norm

SQRT2 = float(np.sqrt(2.0))
PHI_INV_SQRT2 = float(norm.cdf(1.0 / SQRT2))  # constant offset


def f_torch(x: torch.Tensor) -> torch.Tensor:
    """
    x: (N, d) tensor
    returns: (N,) tensor with f(x) = -Phi(1/sqrt(2)) + Phi(1 + sum_j x_j / sqrt(d))
    """
    d = x.shape[1]
    s = 1.0 + x.sum(dim=1) / float(np.sqrt(d))         # (N,)
    return torch.special.ndtr(s) - PHI_INV_SQRT2       # (N,)


def exact_expectation_gmm(means: np.ndarray,
                          covs,                       # (K,d,d) array or list of (d,d)
                          weights: np.ndarray) -> float:
    """
    means: (K, d), covs: (K, d, d) or list of K (d,d), weights: (K,)
    Returns scalar E_pi[f].
    """
    means = np.asarray(means, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    K, d = means.shape
    sqrt_d = np.sqrt(d)

    if isinstance(covs, list):
        covs = np.stack([np.asarray(c, dtype=np.float64) for c in covs], axis=0)
    else:
        covs = np.asarray(covs, dtype=np.float64)
        if covs.ndim == 2:
            covs = np.broadcast_to(covs, (K, d, d)).copy()

    a = means.sum(axis=1) / sqrt_d                       # (K,)
    b2 = covs.sum(axis=(1, 2)) / d                       # 1^T Sigma 1 / d, (K,)
    contrib = norm.cdf((1.0 + a) / np.sqrt(1.0 + b2))    # (K,)
    return float(-PHI_INV_SQRT2 + np.sum(weights * contrib))


# ---------- Specific test cases ----------

def gmm_2d_params():
    means = np.array([[-2., -2.],
                      [ 2., -2.],
                      [-2.,  2.],
                      [ 2.,  2.]], dtype=np.float64)
    cov = np.array([[0.5, 0.1],
                    [0.1, 0.5]], dtype=np.float64)
    covs = np.stack([cov] * 4, axis=0)
    weights = np.full(4, 0.25, dtype=np.float64)
    return means, covs, weights


def gmm_30d_params(dim=30):
    means = np.zeros((4, dim), dtype=np.float64)
    means[0, :2] = [-2., -2.]
    means[1, :2] = [ 2., -2.]
    means[2, :2] = [-2.,  2.]
    means[3, :2] = [ 2.,  2.]
    cov = 0.5 * np.eye(dim, dtype=np.float64)
    covs = np.stack([cov] * 4, axis=0)
    weights = np.full(4, 0.25, dtype=np.float64)
    return means, covs, weights


def true_expectation_2d() -> float:
    return exact_expectation_gmm(*gmm_2d_params())


def true_expectation_30d(dim=30) -> float:
    return exact_expectation_gmm(*gmm_30d_params(dim))


if __name__ == "__main__":
    print(f"PHI(1/sqrt(2)) = {PHI_INV_SQRT2:.8f}")
    print(f"True E_pi[f]  (2D GMM)  = {true_expectation_2d():.8f}")
    print(f"True E_pi[f] (30D GMM)  = {true_expectation_30d():.8f}")

    # quick sanity: empirical vs analytical
    rng = np.random.default_rng(0)
    means, covs, weights = gmm_2d_params()
    K, d = means.shape
    Nmc = 2_000_000
    comp = rng.choice(K, size=Nmc, p=weights)
    Ls = np.linalg.cholesky(covs)
    z = rng.standard_normal((Nmc, d))
    x = means[comp] + np.einsum('nij,nj->ni', Ls[comp], z)
    s = 1.0 + x.sum(axis=1) / np.sqrt(d)
    fx = norm.cdf(s) - PHI_INV_SQRT2
    print(f"Empirical (2D, N={Nmc}) = {fx.mean():.6f} +/- {fx.std()/np.sqrt(Nmc):.2e}")
