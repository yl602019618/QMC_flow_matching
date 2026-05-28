"""
Advanced pure-RQMC estimators for the 2D GMM target.

All randomness comes from scrambled Sobol' sequences (NO i.i.d. uniform MC for
the bucket assignment). We compare several designs:

  M1 (joint_d+1)        : Single (d+1)-dim scrambled Sobol'. First column ->
                          bucket k = floor(K * u_1), remaining d columns ->
                          Gaussian z = Phi^{-1}(u_{2:d+1}). One global RQMC
                          stream, no index-coupling leakage.

  M2 (oracle_strat)     : Oracle stratified bucket assignment: each bucket gets
                          exactly N/K samples (N is a power of 2 and K=4 so
                          this is exact). Within each bucket use an independent
                          scrambled Sobol' d-dim stream. Bucket is deterministic
                          (not from MC), so the bucket dimension contributes 0
                          MC noise; only the Gaussian residual is RQMC.

  M3 (oracle_strat_pad) : Same oracle stratification as M2 but uses ONE single
                          (d+1)-dim Sobol' point set: first column maps to
                          bucket by RANK (the rank-based stratification turns
                          the bucket assignment into a Latin permutation; the
                          remaining d columns give the Gaussian). Tests whether
                          tying bucket-rank and Gaussian indices inside ONE
                          Sobol' stream is benign.

  M4 (joint_cv)         : M1 with a control variate: subtract the cluster mean
                          mu_k from x before squaring; this gives the analytic
                          decomposition E[X^2] = E[mu_k^2] + 2 E[mu_k . L z]
                          + E[(L z)^2] where mu_k . z are component-wise. We
                          Rao-Blackwellise: substitute exact E[mu_k^2] =
                          (1/K) sum_k mu_k^2 and exact E[(L z)^2] = diag(Sigma);
                          the cross-term 2 mu_k . L z is the only Monte-Carlo
                          piece. Strongly reduces variance for second moment.

  M5 (joint_antithetic) : M1 with antithetic pairing within the same Sobol'
                          stream (u, 1-u) -> Gaussian -> (z, -z). Halves the
                          stream length: 2*(N/2) samples form N total.

For RMSE evaluation we use the registry in `integrands.py`.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc, norm
from tqdm import tqdm
import torch

from integrands import (INTEGRANDS, MEANS_2D, COV_2D, rmse)


HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results_logi")
os.makedirs(OUT_DIR, exist_ok=True)

K = 4
D = 2
LS = np.stack([np.linalg.cholesky(COV_2D)] * K, axis=0)            # (K, D, D)
DIAG_COV_2D = np.diag(COV_2D).copy()                                # (D,)


def _seed(p, exp_id, tag=0):
    return (1_000_003 * p + 9_973 * exp_id + 31 * tag + 7) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Samplers (each returns x: (N, D) numpy, plus optional extras for CV).
# Extras dict carries pre-computed control-variate pieces when applicable.
# ---------------------------------------------------------------------------


def sample_joint(N, exp_id, eps=1e-10):
    """M1: single (d+1)-dim scrambled Sobol'."""
    p = int(np.log2(N))
    sob = qmc.Sobol(d=D + 1, scramble=True, seed=_seed(p, exp_id, tag=10))
    u = sob.random(N)
    u = np.clip(u, eps, 1.0 - eps)
    k_idx = np.minimum((u[:, 0] * K).astype(np.int64), K - 1)
    z = norm.ppf(u[:, 1:])
    L_per_sample = LS[k_idx]
    x = MEANS_2D[k_idx] + np.einsum('nij,nj->ni', L_per_sample, z)
    return x, {"k_idx": k_idx, "z": z}


def sample_oracle_strat(N, exp_id, eps=1e-10):
    """M2: exact N/K per bucket; independent scrambled Sobol' d-dim per bucket."""
    p = int(np.log2(N))
    assert N % K == 0, f"N={N} not divisible by K={K}"
    n_per = N // K

    z = np.empty((N, D), dtype=np.float64)
    k_idx = np.empty(N, dtype=np.int64)
    for k in range(K):
        sob = qmc.Sobol(d=D, scramble=True, seed=_seed(p, exp_id, tag=100 + k))
        u_k = sob.random(n_per)
        u_k = np.clip(u_k, eps, 1.0 - eps)
        s, e = k * n_per, (k + 1) * n_per
        z[s:e] = norm.ppf(u_k)
        k_idx[s:e] = k

    L_per_sample = LS[k_idx]
    x = MEANS_2D[k_idx] + np.einsum('nij,nj->ni', L_per_sample, z)
    return x, {"k_idx": k_idx, "z": z}


def sample_oracle_strat_pad(N, exp_id, eps=1e-10):
    """
    M3: one (d+1)-dim Sobol' point set. Rank-based stratification: sort by
    u[:, 0] and assign the first N/K samples to bucket 0, next N/K to bucket 1,
    etc. This is *oracle* (deterministic counts) but the assignment-to-rank is
    driven by the same Sobol' stream that drives z.
    """
    p = int(np.log2(N))
    assert N % K == 0
    sob = qmc.Sobol(d=D + 1, scramble=True, seed=_seed(p, exp_id, tag=200))
    u = sob.random(N)
    u = np.clip(u, eps, 1.0 - eps)
    order = np.argsort(u[:, 0], kind='stable')
    k_idx = np.empty(N, dtype=np.int64)
    n_per = N // K
    for k in range(K):
        k_idx[order[k * n_per:(k + 1) * n_per]] = k
    z = norm.ppf(u[:, 1:])
    L_per_sample = LS[k_idx]
    x = MEANS_2D[k_idx] + np.einsum('nij,nj->ni', L_per_sample, z)
    return x, {"k_idx": k_idx, "z": z}


def sample_joint_antithetic(N, exp_id, eps=1e-10):
    """
    M5: (d+1)-dim Sobol' of length N/2, antithetic pair (u, 1-u). Bucket from
    first column, Gaussian from the remaining d columns.
    """
    assert N % 2 == 0
    p = int(np.log2(N))
    half = N // 2
    sob = qmc.Sobol(d=D + 1, scramble=True, seed=_seed(p, exp_id, tag=300))
    u = sob.random(half)
    u = np.clip(u, eps, 1.0 - eps)
    u_full = np.vstack([u, 1.0 - u])
    k_idx = np.minimum((u_full[:, 0] * K).astype(np.int64), K - 1)
    z = norm.ppf(u_full[:, 1:])
    L_per_sample = LS[k_idx]
    x = MEANS_2D[k_idx] + np.einsum('nij,nj->ni', L_per_sample, z)
    return x, {"k_idx": k_idx, "z": z}


SAMPLERS = {
    "M1_Joint_d+1":           sample_joint,
    "M2_Oracle_Strat":        sample_oracle_strat,
    "M3_Oracle_Strat_Pad":    sample_oracle_strat_pad,
    "M5_Joint_Antithetic":    sample_joint_antithetic,
}


# ---------------------------------------------------------------------------
# Estimators (each takes a sample produced by SAMPLERS[*] and returns
# a dict {key: estimate} matching INTEGRANDS keys).
# Two flavors: plain mean (works for any sampler) and M4 (CV on second moment
# layered on top of M1 sampler).
# ---------------------------------------------------------------------------


def _eval_plain(x_np):
    x = torch.from_numpy(x_np).float()
    out = {}
    for key, spec in INTEGRANDS.items():
        fx = spec["fx"](x)
        if fx.dim() == 1:
            out[key] = float(fx.mean().item())
        else:
            out[key] = fx.mean(dim=0).cpu().numpy()
    return out


# E_pi[mu_k^2] (per coord)  +  diag(Sigma): analytical second-moment baseline
_EXACT_SECOND_2D = (np.full(K, 1.0 / K) @ (MEANS_2D ** 2)) + DIAG_COV_2D
# i.e. equals true E[X^2] per coord; the CV residual is the per-sample
# deviation 2 mu_k . (L z)_per-coord term + (L z)^2 - diag(Sigma).


def _eval_cv_second(x_np, extras):
    """Control-variate second-moment estimator.

    X^2_j = mu_{k,j}^2 + 2 mu_{k,j} (L z)_j + (L z)_j^2

    The empirical second moment (1/N) sum_i X_{i,j}^2 decomposes into the same
    three sample averages. We REPLACE the first piece's sample average by its
    exact value E[mu_{k,j}^2] = (1/K) sum_k mu_{k,j}^2, keeping the cross-term
    and (Lz)^2 sample averages. The replacement removes the (Sobol'-residual)
    variance contributed by the bucket-frequency fluctuation. Bias = 0 always
    because (1/N) sum_i mu_{k_i,j}^2 is itself an unbiased estimator of the
    same constant (mu_k^2 has the same expectation under the joint Sobol'
    sampler as under MC; we are subtracting that residual and adding back the
    exact constant).
    """
    k_idx = extras["k_idx"]
    z = extras["z"]
    L = LS[k_idx]
    Lz = np.einsum('nij,nj->ni', L, z)
    mu = MEANS_2D[k_idx]
    # empirical pieces
    mu2_emp = (mu ** 2).mean(axis=0)            # (D,)
    cross_emp = (2.0 * mu * Lz).mean(axis=0)    # (D,)
    Lz2_emp = (Lz ** 2).mean(axis=0)            # (D,)
    # replace mu^2 empirical with exact constant: (1/K) sum_k mu_k^2
    mu2_exact = np.full(K, 1.0 / K) @ (MEANS_2D ** 2)   # (D,)
    return mu2_exact + cross_emp + Lz2_emp


def _eval_with_cv(x_np, extras):
    """Plain for first / complex; CV for second."""
    out = _eval_plain(x_np)
    out["second"] = _eval_cv_second(x_np, extras)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_methods(p_values, num_experiments=10, methods=None):
    """
    Returns res[method_name][key] = {sample_sizes, mse_errors, estimated_means}.

    Methods include the four samplers in SAMPLERS plus "M4_Joint_CV" which
    wraps M1_Joint_d+1 with the control-variate evaluator on the second moment.
    """
    if methods is None:
        methods = list(SAMPLERS.keys()) + ["M4_Joint_CV"]

    keys = list(INTEGRANDS.keys())
    res = {m: {k: {"sample_sizes": [], "mse_errors": [], "estimated_means": []}
               for k in keys} for m in methods}

    for p in tqdm(p_values, desc="RQMC-advanced 2D"):
        N = 2 ** p
        per_method = {m: {k: [] for k in keys} for m in methods}

        for r in range(num_experiments):
            # produce all samples once per method
            sampler_outputs = {}
            for sampler_name, fn in SAMPLERS.items():
                sampler_outputs[sampler_name] = fn(N, exp_id=r)

            for m in methods:
                if m == "M4_Joint_CV":
                    x, extras = sampler_outputs["M1_Joint_d+1"]
                    est = _eval_with_cv(x, extras)
                else:
                    x, _ = sampler_outputs[m]
                    est = _eval_plain(x)
                for k in keys:
                    per_method[m][k].append(est[k])

        for m in methods:
            for k in keys:
                spec = INTEGRANDS[k]
                tv = spec["true_value"]()
                mean_est, err, _, _ = rmse(per_method[m][k], tv, spec["is_vector"])
                res[m][k]["sample_sizes"].append(N)
                res[m][k]["mse_errors"].append(err)
                res[m][k]["estimated_means"].append(mean_est)

    return res


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _slope(N, err):
    N = np.asarray(N, dtype=float); err = np.asarray(err, dtype=float)
    # Mask out near-zero errors (e.g. antithetic exactly cancelling)
    m = np.isfinite(N) & np.isfinite(err) & (N > 0) & (err > 1e-12)
    if m.sum() < 2:
        return np.nan
    s, _ = np.polyfit(np.log(N[m]), np.log(err[m]), 1)
    return float(s)


_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']


def plot_methods(res, key, save_path, title, baseline_curves=None):
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 13})

    slopes = {}
    N_ref = None
    err_ref = None

    for i, (m, data) in enumerate(res.items()):
        N = data[key]["sample_sizes"]
        err = data[key]["mse_errors"]
        s = _slope(N, err)
        slopes[m] = s
        plt.loglog(N, err, '-' + _MARKERS[i % len(_MARKERS)],
                   label=f'{m} (fit={s:.2f})')
        if N_ref is None:
            N_ref = np.asarray(N, dtype=float)
            err_ref = err[0]

    # optional baseline curves (dict name -> (N, err))
    if baseline_curves:
        for j, (name, (N, err)) in enumerate(baseline_curves.items()):
            s = _slope(N, err)
            slopes[name] = s
            plt.loglog(N, err, '--' + _MARKERS[(j + len(res)) % len(_MARKERS)],
                       alpha=0.6, label=f'{name} (fit={s:.2f})')

    for ref_s in (-0.5, -1.0):
        plt.loglog(N_ref, err_ref * (N_ref / N_ref[0]) ** ref_s, ':',
                   linewidth=1, alpha=0.5, label=f'Slope = {ref_s}')

    plt.xlabel("Sample Size N")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.legend(fontsize=9, loc='lower left')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return slopes


def _load_baseline_npz():
    """Load baseline A (MC-bk) and B (Sobol-bk) curves for overlay from
    gmm/results_logi/all_methods_2d.npz (produced by is_logi_all.py).
    Returns (sample_sizes, {key: {'A': err, 'B': err}}) or None if missing."""
    npz_path = os.path.join(OUT_DIR, "all_methods_2d.npz")
    if not os.path.exists(npz_path):
        return None
    return np.load(npz_path)


def main():
    p_values = list(range(2, 14))      # N = 4..8192 (we need N % K == 0 for M2/M3)
    num_exp = 10
    res = run_methods(p_values, num_experiments=num_exp)

    # build overlay from is_logi_all.py output (if available)
    base_npz = _load_baseline_npz()
    overlays = {key: {} for key in INTEGRANDS.keys()}
    if base_npz is not None and "sample_sizes" in base_npz.files:
        Ns = base_npz["sample_sizes"]
        for key in INTEGRANDS.keys():
            keymap = {
                f"rmse_direct_mc_{key}":  "Baseline-A (MC-bk)",
                f"rmse_direct_qmc_{key}": "Baseline-B (1D-Sobol-bk)",
            }
            for k_arr, name in keymap.items():
                if k_arr in base_npz.files:
                    overlays[key][name] = (Ns, base_npz[k_arr])

    # save plots
    all_slopes = {}
    for key in INTEGRANDS.keys():
        save_path = os.path.join(OUT_DIR, f"rqmc_advanced_2d_{key}.pdf")
        title = f"2D GMM advanced RQMC: {INTEGRANDS[key]['label']}"
        slopes = plot_methods(res, key, save_path, title,
                              baseline_curves=overlays.get(key))
        all_slopes[key] = slopes
        print(f"\n[{title}]")
        for m, s in slopes.items():
            print(f"   {m:32s} slope = {s:+.3f}")

    # save npz
    npz = {"sample_sizes": np.array(res[list(res.keys())[0]]["first"]["sample_sizes"])}
    for m in res.keys():
        for k in INTEGRANDS.keys():
            npz[f"rmse_{m}_{k}"] = np.array(res[m][k]["mse_errors"])
            npz[f"est_{m}_{k}"] = np.array(res[m][k]["estimated_means"])
    np.savez(os.path.join(OUT_DIR, "rqmc_advanced_2d.npz"), **npz)
    print(f"\nSaved npz + {len(INTEGRANDS)} pdfs to {OUT_DIR}")

    return all_slopes


if __name__ == "__main__":
    main()
