"""
Direct (non-FM) RQMC baselines for the 30D GMM experiment.

Sequential per-sample procedure (one global N-sized RQMC point set, shared
across the four mixture components via the bucket assignment):

  A. Direct-MCbk-RQMC  : bucket k_i ~ Uniform{0..K-1} (i.i.d. MC),
                         z_i = Phi^{-1}(scrambled-Sobol' 30D point i).
  B. Direct-RQMCbk-RQMC: a_i ~ 1D scrambled-Sobol', k_i = floor(K * a_i);
                         z_i = Phi^{-1}(scrambled-Sobol' 30D point i),
                         the two Sobol' streams are independently scrambled.

Sample: x_i = mu_{k_i} + sqrt(0.5) * z_i   (cov = 0.5 * I_30)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc, norm
from tqdm import tqdm
import torch

from integrands import INTEGRANDS, MEANS_30D, DIM, K, SIGMA2, rmse


HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results30d")
os.makedirs(OUT_DIR, exist_ok=True)

SCALE = float(np.sqrt(SIGMA2))                                     # diag Cholesky


def _seed(p, exp_id, tag=0):
    return (1_000_003 * p + 9_973 * exp_id + 31 * tag + 7) & 0x7FFFFFFF


def rqmc_sample_30d(N, exp_id, bucket_mode="qmc", eps=1e-10):
    """Two-stream sampler. Returns x of shape (N, 30)."""
    p = int(np.log2(N))

    if bucket_mode == "qmc":
        sob_a = qmc.Sobol(d=1, scramble=True, seed=_seed(p, exp_id, tag=0))
        a = sob_a.random(N).reshape(-1)
        a = np.clip(a, eps, 1.0 - eps)
        k_idx = np.minimum((a * K).astype(np.int64), K - 1)
    elif bucket_mode == "mc":
        rng = np.random.default_rng(_seed(p, exp_id, tag=0))
        k_idx = rng.integers(0, K, size=N)
    else:
        raise ValueError(bucket_mode)

    sob_b = qmc.Sobol(d=DIM, scramble=True, seed=_seed(p, exp_id, tag=1))
    b = sob_b.random(N)
    b = np.clip(b, eps, 1.0 - eps)
    z = norm.ppf(b)                                                # (N, 30)

    x = MEANS_30D[k_idx] + SCALE * z                               # (N, 30)
    return x


def _eval_all(x_np):
    x = torch.from_numpy(x_np).float()
    out = {}
    for key, spec in INTEGRANDS.items():
        fx = spec["fx"](x)
        if fx.dim() == 1:
            out[key] = float(fx.mean().item())
        else:
            out[key] = fx.mean(dim=0).cpu().numpy()
    return out


def run_all(p_values, num_experiments=10, bucket_modes=("mc", "qmc")):
    keys = list(INTEGRANDS.keys())
    res = {m: {k: {"sample_sizes": [], "mse_errors": [], "estimated_means": []}
               for k in keys} for m in bucket_modes}

    for p in tqdm(p_values, desc="RQMC-baseline 30D"):
        N = 2 ** p
        per_mode_per_key = {m: {k: [] for k in keys} for m in bucket_modes}
        for r in range(num_experiments):
            for m in bucket_modes:
                x = rqmc_sample_30d(N, exp_id=r, bucket_mode=m)
                est = _eval_all(x)
                for k in keys:
                    per_mode_per_key[m][k].append(est[k])
        for m in bucket_modes:
            for k in keys:
                spec = INTEGRANDS[k]
                tv = spec["true_value"]()
                mean_est, err, _, _ = rmse(per_mode_per_key[m][k], tv,
                                           spec["is_vector"])
                res[m][k]["sample_sizes"].append(N)
                res[m][k]["mse_errors"].append(err)
                res[m][k]["estimated_means"].append(mean_est)

    return res


def to_baseline_curves(res):
    name_map = {"mc": "Direct-MCbk-RQMC", "qmc": "Direct-RQMCbk-RQMC"}
    out = {}
    for k in INTEGRANDS.keys():
        out[k] = {}
        for m in res.keys():
            out[k][name_map[m]] = (res[m][k]["sample_sizes"], res[m][k]["mse_errors"])
    return out


def _slope(N, err):
    N = np.asarray(N, dtype=float); err = np.asarray(err, dtype=float)
    m = np.isfinite(N) & np.isfinite(err) & (N > 0) & (err > 0)
    if m.sum() < 2: return np.nan
    s, _ = np.polyfit(np.log(N[m]), np.log(err[m]), 1)
    return float(s)


def plot_baseline_only(res, key, save_path, title):
    N_mc = res["mc"][key]["sample_sizes"]; err_mc = res["mc"][key]["mse_errors"]
    N_qmc = res["qmc"][key]["sample_sizes"]; err_qmc = res["qmc"][key]["mse_errors"]
    s_mc = _slope(N_mc, err_mc); s_qmc = _slope(N_qmc, err_qmc)

    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 14})
    plt.loglog(N_mc, err_mc, 'm-D', label=f'Direct-MCbk-RQMC (fit={s_mc:.2f})')
    plt.loglog(N_qmc, err_qmc, 'y-v', label=f'Direct-RQMCbk-RQMC (fit={s_qmc:.2f})')
    N_arr = np.array(N_mc, dtype=float)
    for ref_s in (-0.5, -1.0):
        plt.loglog(N_arr, err_mc[0] * (N_arr / N_arr[0]) ** ref_s, '--',
                   linewidth=1, alpha=0.6, label=f'Slope = {ref_s}')
    plt.xlabel("Sample Size"); plt.ylabel("RMSE"); plt.title(title)
    plt.legend(); plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200); plt.close()
    print(f"[{title}] mc-bk={s_mc:.3f}, qmc-bk={s_qmc:.3f}")
    return s_mc, s_qmc


def main():
    p_values = list(range(2, 15))              # N = 4..16384
    num_exp = 10
    res = run_all(p_values, num_experiments=num_exp)
    for k in INTEGRANDS.keys():
        save_path = os.path.join(OUT_DIR, f"rqmc_baseline_30d_{k}.pdf")
        title = f"30D GMM RQMC baselines: {INTEGRANDS[k]['label']}"
        plot_baseline_only(res, k, save_path, title)

    npz = {"sample_sizes": np.array(res["mc"]["first"]["sample_sizes"])}
    for m in res.keys():
        for k in INTEGRANDS.keys():
            npz[f"rmse_{m}_{k}"] = np.array(res[m][k]["mse_errors"])
    np.savez(os.path.join(OUT_DIR, "rqmc_all_30d.npz"), **npz)
    print(f"Saved npz + 3 baseline pdfs to {OUT_DIR}")


if __name__ == "__main__":
    main()
