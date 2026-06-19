"""
The two direct (non-flow) RQMC estimators compared against the flow methods.

Both draw N samples sequentially as  x_i = mu_{k_i} + sqrt(VAR) * z_i  and use
the plain sample mean  I_hat = (1/N) sum_i f(x_i)  (no importance weights,
since they sample pi exactly). They differ only in HOW the bucket k_i and the
Gaussian z_i are generated:

  Direct-MCbk-RQMC  (MC bucket + RQMC Gaussian):
      k_i ~ Uniform{1..K}  i.i.d. (Monte-Carlo),
      z_i = Phi^{-1}(b_i),  {b_i} a d-dim scrambled Sobol' point set.
    The discrete mixture index is plain MC -> the estimator is ~ O(N^{-1/2}).

  Direct-Joint-RQMC  (M1, joint (d+1)-dim RQMC):
      one (d+1)-dim scrambled Sobol' point set {u_i};
      k_i = floor(K * u_{i,0})        (first coordinate -> bucket),
      z_i = Phi^{-1}(u_{i,1:d+1})     (remaining d coordinates -> Gaussian).
    A single global low-discrepancy point set drives both the bucket and the
    Gaussian, so the joint (d+1)-cube is filled with low discrepancy and the
    estimator approaches O(N^{-1}) for smooth integrands.

  Direct-MC  (pure MC baseline):
      k_i ~ Uniform{1..K}  i.i.d.,
      z_i ~ N(0, I_d)  i.i.d.
    The plain Monte Carlo estimator from the target mixture; ~ O(N^{-1/2}).
"""

import numpy as np
from scipy.stats import qmc, norm

from gmm import DIM, K, VAR, MEANS

SCALE = float(np.sqrt(VAR))


def _seed(p, exp_id, tag=0):
    return (1_000_003 * p + 9_973 * exp_id + 31 * tag + 7) & 0x7FFFFFFF


def sample_estimator1(N, exp_id, eps=1e-10):
    """MC bucket + scrambled-Sobol' Gaussian. Returns x: (N, DIM)."""
    p = int(np.log2(N))
    rng = np.random.default_rng(_seed(p, exp_id, tag=0))
    k = rng.integers(0, K, size=N)
    sob = qmc.Sobol(d=DIM, scramble=True, seed=_seed(p, exp_id, tag=1))
    b = np.clip(sob.random(N), eps, 1.0 - eps)
    z = norm.ppf(b)
    return MEANS[k] + SCALE * z


def sample_estimator2(N, exp_id, eps=1e-10):
    """M1: single (d+1)-dim scrambled Sobol'. Returns x: (N, DIM)."""
    p = int(np.log2(N))
    sob = qmc.Sobol(d=DIM + 1, scramble=True, seed=_seed(p, exp_id, tag=10))
    u = np.clip(sob.random(N), eps, 1.0 - eps)
    k = np.minimum((u[:, 0] * K).astype(np.int64), K - 1)
    z = norm.ppf(u[:, 1:])
    return MEANS[k] + SCALE * z


def sample_estimator3(N, exp_id):
    """Pure MC: i.i.d. bucket + i.i.d. standard Gaussian. Returns x: (N, DIM)."""
    p = int(np.log2(N))
    rng = np.random.default_rng(_seed(p, exp_id, tag=20))
    k = rng.integers(0, K, size=N)
    z = rng.standard_normal(size=(N, DIM))
    return MEANS[k] + SCALE * z
