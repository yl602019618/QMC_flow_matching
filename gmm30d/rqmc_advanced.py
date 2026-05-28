"""
Advanced pure-RQMC estimators for the 30D GMM target.

Mirror of gmm/rqmc_advanced.py with dim=30. See that file for the method
descriptions. Diagonal cov = SIGMA2 * I_30 makes L = sqrt(SIGMA2) * I.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc, norm
from tqdm import tqdm
import torch

from integrands import (INTEGRANDS, MEANS_30D, DIM, K, SIGMA2, rmse)


HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results30d")
os.makedirs(OUT_DIR, exist_ok=True)

SCALE = float(np.sqrt(SIGMA2))   # diag-Cholesky scalar

# Householder-like rotation matrix Q with Q[:,0] = 1/sqrt(d) * 1_d (the
# "main direction" of the complex integrand f(x) = Phi(1 + 1_d^T x / sqrt(d))).
# Using Q maps the FIRST Gaussian Sobol' column directly onto the all-ones
# direction; high-discrepancy reduction on the most important projection.
_ONES = np.ones(DIM) / np.sqrt(DIM)
_Q = np.eye(DIM)
_Q[:, 0] = _ONES
# Gram-Schmidt against e_1 ... e_{d-1} (Householder reflection): build an
# orthonormal basis with first column = 1_d / sqrt(d).
_q, _ = np.linalg.qr(_Q)
# qr may flip signs; fix so first col aligned with _ONES
if _q[0, 0] * _ONES[0] < 0:
    _q[:, 0] *= -1
ROT_MAIN_DIR = _q  # (DIM, DIM) orthonormal; col 0 = 1_d / sqrt(d)


def _seed(p, exp_id, tag=0):
    return (1_000_003 * p + 9_973 * exp_id + 31 * tag + 7) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


def sample_joint(N, exp_id, eps=1e-10):
    """M1: single (D+1)-dim scrambled Sobol'."""
    p = int(np.log2(N))
    sob = qmc.Sobol(d=DIM + 1, scramble=True, seed=_seed(p, exp_id, tag=10))
    u = sob.random(N)
    u = np.clip(u, eps, 1.0 - eps)
    k_idx = np.minimum((u[:, 0] * K).astype(np.int64), K - 1)
    z = norm.ppf(u[:, 1:])
    x = MEANS_30D[k_idx] + SCALE * z
    return x, {"k_idx": k_idx, "z": z}


def sample_oracle_strat(N, exp_id, eps=1e-10):
    """M2: oracle stratification, per-bucket independent scrambled Sobol' D-dim."""
    p = int(np.log2(N))
    assert N % K == 0
    n_per = N // K
    z = np.empty((N, DIM), dtype=np.float64)
    k_idx = np.empty(N, dtype=np.int64)
    for k in range(K):
        sob = qmc.Sobol(d=DIM, scramble=True, seed=_seed(p, exp_id, tag=100 + k))
        u_k = sob.random(n_per)
        u_k = np.clip(u_k, eps, 1.0 - eps)
        s, e = k * n_per, (k + 1) * n_per
        z[s:e] = norm.ppf(u_k)
        k_idx[s:e] = k
    x = MEANS_30D[k_idx] + SCALE * z
    return x, {"k_idx": k_idx, "z": z}


def sample_oracle_strat_pad(N, exp_id, eps=1e-10):
    """M3: single (D+1) Sobol', rank-based oracle stratification on first column."""
    p = int(np.log2(N))
    assert N % K == 0
    sob = qmc.Sobol(d=DIM + 1, scramble=True, seed=_seed(p, exp_id, tag=200))
    u = sob.random(N)
    u = np.clip(u, eps, 1.0 - eps)
    order = np.argsort(u[:, 0], kind='stable')
    k_idx = np.empty(N, dtype=np.int64)
    n_per = N // K
    for k in range(K):
        k_idx[order[k * n_per:(k + 1) * n_per]] = k
    z = norm.ppf(u[:, 1:])
    x = MEANS_30D[k_idx] + SCALE * z
    return x, {"k_idx": k_idx, "z": z}


def sample_joint_rotated(N, exp_id, eps=1e-10):
    """M6: (D+1)-dim joint Sobol' as in M1, but rotate the Gaussian so the
    LOW-ORDER Sobol' axes target the all-ones direction (dominant projection
    for the complex integrand). x = mu_k + SCALE * (Q z) where z is the
    Phi^{-1} of the (D+1)-dim Sobol' from columns 1..D."""
    p = int(np.log2(N))
    sob = qmc.Sobol(d=DIM + 1, scramble=True, seed=_seed(p, exp_id, tag=400))
    u = sob.random(N)
    u = np.clip(u, eps, 1.0 - eps)
    k_idx = np.minimum((u[:, 0] * K).astype(np.int64), K - 1)
    z = norm.ppf(u[:, 1:])                # (N, D)
    z_rot = z @ ROT_MAIN_DIR.T            # (N, D), still N(0, I)
    x = MEANS_30D[k_idx] + SCALE * z_rot
    return x, {"k_idx": k_idx, "z": z_rot}


def sample_joint_antithetic(N, exp_id, eps=1e-10):
    """M5: (D+1) Sobol' of length N/2, antithetic pair (u, 1-u)."""
    assert N % 2 == 0
    p = int(np.log2(N))
    half = N // 2
    sob = qmc.Sobol(d=DIM + 1, scramble=True, seed=_seed(p, exp_id, tag=300))
    u = sob.random(half)
    u = np.clip(u, eps, 1.0 - eps)
    u_full = np.vstack([u, 1.0 - u])
    k_idx = np.minimum((u_full[:, 0] * K).astype(np.int64), K - 1)
    z = norm.ppf(u_full[:, 1:])
    x = MEANS_30D[k_idx] + SCALE * z
    return x, {"k_idx": k_idx, "z": z}


SAMPLERS = {
    "M1_Joint_d+1":           sample_joint,
    "M2_Oracle_Strat":        sample_oracle_strat,
    "M3_Oracle_Strat_Pad":    sample_oracle_strat_pad,
    "M5_Joint_Antithetic":    sample_joint_antithetic,
    "M6_Joint_Rotated":       sample_joint_rotated,
}


# ---------------------------------------------------------------------------
# Estimators
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


def _eval_cv_second(x_np, extras):
    """Control-variate second-moment estimator (see gmm/rqmc_advanced.py)."""
    k_idx = extras["k_idx"]
    z = extras["z"]
    mu = MEANS_30D[k_idx]
    Lz = SCALE * z
    cross_emp = (2.0 * mu * Lz).mean(axis=0)
    Lz2_emp = (Lz ** 2).mean(axis=0)
    mu2_exact = np.full(K, 1.0 / K) @ (MEANS_30D ** 2)
    return mu2_exact + cross_emp + Lz2_emp


def _eval_with_cv(x_np, extras):
    out = _eval_plain(x_np)
    out["second"] = _eval_cv_second(x_np, extras)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_methods(p_values, num_experiments=10, methods=None):
    if methods is None:
        methods = list(SAMPLERS.keys()) + ["M4_Joint_CV"]

    keys = list(INTEGRANDS.keys())
    res = {m: {k: {"sample_sizes": [], "mse_errors": [], "estimated_means": []}
               for k in keys} for m in methods}

    for p in tqdm(p_values, desc="RQMC-advanced 30D"):
        N = 2 ** p
        per_method = {m: {k: [] for k in keys} for m in methods}

        for r in range(num_experiments):
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
    """Load baseline A/B curves from all_methods_30d.npz (is_logi_all output)."""
    npz_path = os.path.join(OUT_DIR, "all_methods_30d.npz")
    if not os.path.exists(npz_path):
        return None
    return np.load(npz_path)


def main():
    p_values = list(range(2, 15))      # N = 4..16384
    num_exp = 10
    res = run_methods(p_values, num_experiments=num_exp)

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

    all_slopes = {}
    for key in INTEGRANDS.keys():
        save_path = os.path.join(OUT_DIR, f"rqmc_advanced_30d_{key}.pdf")
        title = f"30D GMM advanced RQMC: {INTEGRANDS[key]['label']}"
        slopes = plot_methods(res, key, save_path, title,
                              baseline_curves=overlays.get(key))
        all_slopes[key] = slopes
        print(f"\n[{title}]")
        for m, s in slopes.items():
            print(f"   {m:32s} slope = {s:+.3f}")

    npz = {"sample_sizes": np.array(res[list(res.keys())[0]]["first"]["sample_sizes"])}
    for m in res.keys():
        for k in INTEGRANDS.keys():
            npz[f"rmse_{m}_{k}"] = np.array(res[m][k]["mse_errors"])
            npz[f"est_{m}_{k}"] = np.array(res[m][k]["estimated_means"])
    np.savez(os.path.join(OUT_DIR, "rqmc_advanced_30d.npz"), **npz)
    print(f"\nSaved npz + {len(INTEGRANDS)} pdfs to {OUT_DIR}")
    return all_slopes


if __name__ == "__main__":
    main()
