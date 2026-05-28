"""
2D GMM unified driver: runs three integrands (first / second / complex f)
under four FM-based methods (MC, QMC, ISMC=SNIS-MC, ISQMC=SNIS-QMC) in a
single loop, sharing the heavy operations (sample + log_prob).

Output:
    results_logi/all_methods_2d.npz       -- 12 RMSE curves + sample_sizes
    results_logi/all_methods_2d_<key>.pdf -- one figure per integrand (4 lines)
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from flow_matching_logistic import FlowMatchingOT
from integrands import INTEGRANDS, reduce_mean, weighted_sum, rmse
from rqmc_baseline import run_all as run_rqmc_baselines, to_baseline_curves


HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "results_logi", "fm_model_logi.pt")
OUT_DIR = os.path.join(HERE, "results_logi")
os.makedirs(OUT_DIR, exist_ok=True)


def load_model(device):
    model = FlowMatchingOT(
        dim=2, hidden_dim=512, num_blocks=4, sigma=0.0, lr=1e-3, device=device,
        base_dist="logistic", base_loc=0.0, base_scale=1.0,
    )
    model.load(CKPT)
    model.eval()
    return model


def true_gmm_log_prob_torch(x: torch.Tensor) -> torch.Tensor:
    """Analytic log-density of the 2D 4-component GMM (shared cov).

    cov = [[0.5, 0.1], [0.1, 0.5]] ; det = 0.24 ; inv = [[25,-5],[-5,25]]/12.
    Hardcoded to avoid the torch.det CUDA / nvrtc JIT bug.
    """
    device = x.device
    means = torch.tensor([[-2., -2.], [2., -2.], [-2., 2.], [2., 2.]],
                         dtype=x.dtype, device=device)
    cov_inv = torch.tensor([[25., -5.], [-5., 25.]],
                           dtype=x.dtype, device=device) / 12.0
    det = 0.24
    norm_const = 1.0 / (2 * np.pi * float(np.sqrt(det)))
    diff = x.unsqueeze(0) - means.unsqueeze(1)                    # (4, N, 2)
    md = torch.einsum('kni,ij,knj->kn', diff, cov_inv, diff)      # (4, N)
    comp = norm_const * torch.exp(-0.5 * md)                      # (4, N)
    p = 0.25 * comp.sum(dim=0)                                    # (N,)
    return torch.log(torch.clamp(p, min=1e-38))


# ----------------------------
# Per-experiment helpers
# ----------------------------

def _eval_integrands_plain(x: torch.Tensor):
    """Return dict {key: np.ndarray} of plain MC averages over samples."""
    out = {}
    for key, spec in INTEGRANDS.items():
        fx = spec["fx"](x)
        out[key] = reduce_mean(fx)
    return out


def _eval_integrands_snis(x: torch.Tensor, w: torch.Tensor):
    """Return dict {key: ndarray or float} of SNIS estimates with weights w."""
    out = {}
    for key, spec in INTEGRANDS.items():
        fx = spec["fx"](x)
        out[key] = weighted_sum(fx, w)
    return out


# ----------------------------
# Main estimation loop
# ----------------------------

def run(model, p_values, num_experiments=10,
        sampling_steps=64, integrator="heun",
        logprob_steps=64, logprob_batch_size=256):
    """
    Returns nested dict:
        results[method][integrand_key] = {
            "sample_sizes": [...],
            "mse_errors":   [...],
            "estimated_means": [...],
        }
    Methods: MC, QMC, ISMC, ISQMC
    """
    device = next(model.model.parameters()).device
    methods = ["MC", "QMC", "ISMC", "ISQMC"]
    keys = list(INTEGRANDS.keys())

    res = {m: {k: {"sample_sizes": [], "mse_errors": [], "estimated_means": []}
               for k in keys} for m in methods}
    ess_record = {"MC": [], "QMC": []}     # average ESS per N for SNIS methods

    for p in tqdm(p_values, desc="2D all"):
        N = 2 ** p
        per_method_per_key = {m: {k: [] for k in keys} for m in methods}
        ess_mc_list, ess_qmc_list = [], []

        for r in range(num_experiments):
            # ---- sample (model.sample / sample_qmc are already under no_grad) ----
            x_mc = model.sample(N, sampling_steps=sampling_steps,
                                integrator=integrator).to(device)
            x_qmc = model.sample_qmc(N, sampling_steps=sampling_steps,
                                     exp=r, integrator=integrator).to(device)

            # plain MC / QMC estimates (no grad needed)
            with torch.no_grad():
                ests_mc = _eval_integrands_plain(x_mc)
                ests_qmc = _eval_integrands_plain(x_qmc)

            # log q / log p for SNIS (batched_log_prob needs autograd internally)
            logq_mc = model.batched_log_prob(x_mc, steps=logprob_steps,
                                             batch_size=logprob_batch_size,
                                             integrator=integrator)
            logq_qmc = model.batched_log_prob(x_qmc, steps=logprob_steps,
                                              batch_size=logprob_batch_size,
                                              integrator=integrator)

            with torch.no_grad():
                logp_mc = true_gmm_log_prob_torch(x_mc)
                lw_mc = logp_mc - logq_mc.detach()
                w_mc = torch.exp(lw_mc - torch.logsumexp(lw_mc, dim=0))

                logp_qmc = true_gmm_log_prob_torch(x_qmc)
                lw_qmc = logp_qmc - logq_qmc.detach()
                w_qmc = torch.exp(lw_qmc - torch.logsumexp(lw_qmc, dim=0))

                ess_mc_list.append(float((1.0 / torch.sum(w_mc * w_mc)).item()))
                ess_qmc_list.append(float((1.0 / torch.sum(w_qmc * w_qmc)).item()))

                ests_ismc = _eval_integrands_snis(x_mc, w_mc)
                ests_isqmc = _eval_integrands_snis(x_qmc, w_qmc)

            for k in keys:
                per_method_per_key["MC"][k].append(ests_mc[k])
                per_method_per_key["QMC"][k].append(ests_qmc[k])
                per_method_per_key["ISMC"][k].append(ests_ismc[k])
                per_method_per_key["ISQMC"][k].append(ests_isqmc[k])

        # aggregate
        for m in methods:
            for k in keys:
                spec = INTEGRANDS[k]
                tv = spec["true_value"]()
                mean_est, err, _, _ = rmse(per_method_per_key[m][k], tv,
                                           spec["is_vector"])
                res[m][k]["sample_sizes"].append(N)
                res[m][k]["mse_errors"].append(err)
                res[m][k]["estimated_means"].append(mean_est)

        ess_record["MC"].append(float(np.mean(ess_mc_list)))
        ess_record["QMC"].append(float(np.mean(ess_qmc_list)))

    return res, ess_record


# ----------------------------
# Plot
# ----------------------------

def _fit_loglog_slope(N, err, tail_k=None):
    x = np.asarray(N, dtype=float); y = np.asarray(err, dtype=float)
    if tail_k is not None and 2 <= tail_k <= len(x):
        x = x[-tail_k:]; y = y[-tail_k:]
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[m], y[m]
    if x.size < 2:
        return np.nan
    s, _ = np.polyfit(np.log(x), np.log(y), 1)
    return float(s)


def plot_one_integrand(res, integrand_key, save_path, title,
                       baseline_curves=None):
    """
    Plot RMSE vs N for one integrand:
        4 FM lines + (optional) 2 baseline lines + 3 reference/fit dashes.
    `baseline_curves`: optional dict like
        { "Direct-MCbk-RQMC": (N_list, err_list),
          "Direct-RQMCbk-RQMC": (N_list, err_list) }
    """
    keys = ["MC", "QMC", "ISMC", "ISQMC"]
    styles = {"MC": ('r-o', 'FM-MC'),
              "QMC": ('b-s', 'FM-QMC'),
              "ISMC": ('g-^', 'FM-ISMC'),
              "ISQMC": ('k-^', 'FM-ISQMC')}

    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 14})

    slopes = {}
    N_arr = np.array(res["MC"][integrand_key]["sample_sizes"], dtype=float)

    for k in keys:
        N = res[k][integrand_key]["sample_sizes"]
        err = res[k][integrand_key]["mse_errors"]
        style, label = styles[k]
        plt.loglog(N, err, style, label=label)
        slopes[k] = _fit_loglog_slope(N, err)

    if baseline_curves:
        baseline_styles = {
            "Direct-MCbk-RQMC":   ('m-D', 'Direct-MCbk-RQMC'),
            "Direct-RQMCbk-RQMC": ('y-v', 'Direct-RQMCbk-RQMC'),
        }
        for name, (Nb, errb) in baseline_curves.items():
            style, label = baseline_styles.get(name, ('c-x', name))
            plt.loglog(Nb, errb, style, label=label)
            slopes[name] = _fit_loglog_slope(Nb, errb)

    # reference / fit dashes anchored at FM-ISQMC's first point
    isqmc_err0 = res["ISQMC"][integrand_key]["mse_errors"][0]
    N0 = N_arr[0]
    plt.loglog(N_arr, isqmc_err0 * (N_arr / N0) ** (-0.5), '--',
               linewidth=1, alpha=0.7, label='Slope = -0.5')
    plt.loglog(N_arr, isqmc_err0 * (N_arr / N0) ** (-1.0), '--',
               linewidth=1, alpha=0.7, label='Slope = -1')
    if np.isfinite(slopes["ISQMC"]):
        plt.loglog(N_arr, isqmc_err0 * (N_arr / N0) ** slopes["ISQMC"], '--',
                   linewidth=1, alpha=0.7,
                   label=f'Slope = {slopes["ISQMC"]:.2f}')

    plt.xlabel("Sample Size"); plt.ylabel("RMSE")
    plt.title(title)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200); plt.close()
    print(f"[{title}] slopes: " +
          ", ".join(f"{k}={slopes[k]:.3f}" for k in slopes))
    return slopes


# ----------------------------
# Main
# ----------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = load_model(device)

    p_values = list(range(1, 14))    # N = 2..8192
    num_exp = 10

    print("Integrands:", list(INTEGRANDS.keys()))
    for k, spec in INTEGRANDS.items():
        print(f"  {k}: true = {spec['true_value']()}")

    res, ess = run(model, p_values, num_experiments=num_exp,
                   sampling_steps=64, integrator="heun",
                   logprob_steps=64, logprob_batch_size=256)

    # ---- direct RQMC baselines (no FM, no IS) ----
    res_base = run_rqmc_baselines(p_values, num_experiments=num_exp)
    baseline_curves_by_key = to_baseline_curves(res_base)

    slope_table = {}
    for k in INTEGRANDS.keys():
        title = f"2D GMM, {INTEGRANDS[k]['label']}: 4 FM + 2 direct baselines"
        save_path = os.path.join(OUT_DIR, f"all_methods_2d_{k}.pdf")
        slope_table[k] = plot_one_integrand(
            res, k, save_path, title,
            baseline_curves=baseline_curves_by_key[k]
        )

    # save npz (both FM methods and direct baselines)
    npz_payload = {"sample_sizes": np.array(res["MC"]["first"]["sample_sizes"])}
    for m in ["MC", "QMC", "ISMC", "ISQMC"]:
        for k in INTEGRANDS.keys():
            npz_payload[f"rmse_{m}_{k}"] = np.array(res[m][k]["mse_errors"])
    for m in ["mc", "qmc"]:
        for k in INTEGRANDS.keys():
            npz_payload[f"rmse_direct_{m}_{k}"] = np.array(res_base[m][k]["mse_errors"])
    npz_payload["ess_mc"] = np.array(ess["MC"])
    npz_payload["ess_qmc"] = np.array(ess["QMC"])
    np.savez(os.path.join(OUT_DIR, "all_methods_2d.npz"), **npz_payload)
    print(f"Saved npz + 3 combined pdfs to {OUT_DIR}")


if __name__ == "__main__":
    main()
