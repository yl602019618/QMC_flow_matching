# eval_banana_transport_expectation.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import qmc
from tqdm import tqdm

# ----------------------------
# Banana params (a=0.5, b=1/sqrt(2), c=-1.0)
# ----------------------------
A = 0.3
B = 1.0 / np.sqrt(2.0)
C = -1.0

LOG2PI = np.log(2.0 * np.pi)

# ----------------------------
# Load trained model (flow -> transport map)
# ----------------------------
# 如果你的类文件名不同，请改这里的 import
from flow_matching_logistic import FlowMatchingOT

def load_model(checkpoint_path, device="cpu"):
    model = FlowMatchingOT(dim=2, hidden_dim=512, num_blocks=6, sigma=0.0, lr=1e-3, device=device)
    model.load(checkpoint_path)
    model.eval()
    return model

# ----------------------------
# True banana mean (analytic)
# ----------------------------
def compute_true_banana_mean(a=A, b=B, c=C):
    """
    x1 = z1
    x2 = a z1^2 + c + b z2
    E[x1]=0, E[z1^2]=1, E[z2]=0  =>  E[x2]=a + c
    """
    return np.array([0.0, a + c], dtype=np.float64)

# ----------------------------
# True banana log-density (torch)
# ----------------------------
def stdnorm_logpdf_torch(x: torch.Tensor) -> torch.Tensor:
    # x: (N,)
    return -0.5 * x * x - 0.5 * LOG2PI

def true_banana_log_prob_torch(x: torch.Tensor, a=A, b=B, c=C) -> torch.Tensor:
    """
    x: (N,2) on any device
    p(x1, x2) = (1/|b|) * N(x1;0,1) * N( (x2 - a x1^2 - c)/b ; 0,1 )
    return: log p(x) as (N,)
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    u = (x2 - a * x1 * x1 - c) / b
    logp = stdnorm_logpdf_torch(x1) + stdnorm_logpdf_torch(u) - np.log(abs(b))
    return logp

# ----------------------------
# MC (transport via learned flow)
# ----------------------------
def estimate_first_moment(model, p_values, true_mean=None, num_experiments=10,
                          sampling_steps=64, integrator="heun"):
    model.eval()
    results = {
        'p_values': p_values,
        'sample_sizes': [],
        'estimated_means': [],
        'mse_errors': [],
        'bias_squared': [],
        'variance': [],
        'std_errors': [],
        'true_mean': true_mean
    }

    with torch.no_grad():
        for p in tqdm(p_values):
            N = 2**p
            results['sample_sizes'].append(N)
            estimates = []
            for exp_id in range(num_experiments):
                x = model.sample(N, sampling_steps=sampling_steps, integrator=integrator)
                estimates.append(torch.mean(x, dim=0).cpu().numpy())
            estimates = np.stack(estimates)              # (num_exp, 2)
            mean_est = estimates.mean(axis=0)
            results['estimated_means'].append(mean_est)
            if true_mean is not None:
                bias = mean_est - true_mean
                bias_sq = np.sum(bias**2)
                var = np.mean(np.sum((estimates - mean_est)**2, axis=1))
                mse = bias_sq + var
                results['mse_errors'].append(np.sqrt(mse))
                results['bias_squared'].append(np.sqrt(bias_sq))
                results['variance'].append(np.sqrt(var))
                results['std_errors'].append(np.mean(np.std(estimates, axis=0, ddof=1)))
    return results

# ----------------------------
# QMC (transport via learned flow)
# ----------------------------
def estimate_first_moment_qmc(model, p_values, true_mean=None, num_experiments=10,
                              sampling_steps=128, integrator="heun"):
    model.eval()
    results = {
        'p_values': p_values,
        'sample_sizes': [],
        'estimated_means': [],
        'mse_errors': [],
        'bias_squared': [],
        'variance': [],
        'std_errors': [],
        'true_mean': true_mean
    }

    with torch.no_grad():
        for p in tqdm(p_values):
            N = 2**p
            results['sample_sizes'].append(N)
            estimates = []
            for exp_id in range(num_experiments):
                x = model.sample_qmc(N, sampling_steps=sampling_steps,
                                     integrator=integrator, exp=exp_id)
                estimates.append(torch.mean(x, dim=0).cpu().numpy())
            estimates = np.stack(estimates)
            mean_est = estimates.mean(axis=0)
            results['estimated_means'].append(mean_est)
            if true_mean is not None:
                bias = mean_est - true_mean
                bias_sq = np.sum(bias**2)
                var = np.mean(np.sum((estimates - mean_est)**2, axis=1))
                mse = bias_sq + var
                results['mse_errors'].append(np.sqrt(mse))
                results['bias_squared'].append(np.sqrt(bias_sq))
                results['variance'].append(np.sqrt(var))
                results['std_errors'].append(np.mean(np.std(estimates, axis=0, ddof=1)))
    return results

# ----------------------------
# Self-normalized IS (MC/QMC) with model as proposal q
# ----------------------------
def estimate_first_moment_snis(model,
                               p_values,
                               true_mean=None,
                               num_experiments=10,
                               sampling_steps=64,
                               integrator="rk4",
                               logprob_steps=64,
                               logprob_batch_size=256,
                               method='mc'):
    """
    从 q_theta (flow 模型) 采样（MC或QMC），用 SNIS 纠正到真实香蕉分布 p。
    权重: w_i ∝ p_true(x_i) / q_model(x_i)，归一化后估计 E_p[X] = Σ w_i x_i
    """
    model.eval()
    device = next(model.model.parameters()).device

    results = {
        'p_values': p_values,
        'sample_sizes': [],
        'estimated_means': [],
        'mse_errors': [],
        'bias_squared': [],
        'variance': [],
        'std_errors': [],
        'ess': [],
        'true_mean': true_mean
    }

    for p in tqdm(p_values):
        N = 2 ** p
        results['sample_sizes'].append(N)

        exp_estimates = []
        exp_ess = []

        for exp_id in range(num_experiments):
            # 1) 从 q_theta 采样
            if method == 'mc':
                x = model.sample(N, sampling_steps=sampling_steps, integrator=integrator).to(device)
            elif method == 'qmc':
                x = model.sample_qmc(N, sampling_steps=sampling_steps,
                                     integrator=integrator, exp=exp_id).to(device)
            else:
                raise ValueError("method must be 'mc' or 'qmc'")

            # 2) log q_model(x)
            logq = model.batched_log_prob(
                x, steps=logprob_steps, batch_size=logprob_batch_size, integrator=integrator
            )  # (N,)

            # 3) log p_true(x)  (banana)
            logp = true_banana_log_prob_torch(x)  # (N,)

            # 4) 归一化权重
            logw = (logp - logq)
            lse = torch.logsumexp(logw, dim=0)
            w = torch.exp(logw - lse)  # (N,), sum(w)=1

            # 5) 一阶矩
            mu_hat = torch.sum(w.unsqueeze(1) * x, dim=0)    # (2,)
            exp_estimates.append(mu_hat.detach().cpu().numpy())

            # 6) ESS
            ess = 1.0 / torch.sum(w * w)
            exp_ess.append(ess.item())

        estimates = np.stack(exp_estimates)              # (num_exp, 2)
        mean_est = estimates.mean(axis=0)
        results['estimated_means'].append(mean_est)
        results['ess'].append(float(np.mean(exp_ess)))

        if true_mean is not None:
            bias = mean_est - true_mean
            bias_sq = float(np.sum(bias ** 2))
            diffs = estimates - mean_est[None, :]
            var = float(np.mean(np.sum(diffs ** 2, axis=1)))
            mse = bias_sq + var
            results['mse_errors'].append(np.sqrt(mse))
            results['bias_squared'].append(np.sqrt(bias_sq))
            results['variance'].append(np.sqrt(var))
            results['std_errors'].append(float(np.mean(np.std(estimates, axis=0, ddof=1))))

    return results

# ----------------------------
# Helpers: slope on log-log
# ----------------------------
def _fit_loglog_slope(sample_sizes, errors, tail_k=None):
    x = np.asarray(sample_sizes, dtype=float)
    y = np.asarray(errors, dtype=float)
    if tail_k is not None and 2 <= tail_k <= len(x):
        x = x[-tail_k:]; y = y[-tail_k:]
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[m], y[m]
    if x.size < 2: return np.nan, np.nan
    lx, ly = np.log(x), np.log(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    return float(slope), float(intercept)

# ----------------------------
# Plotting
# ----------------------------
def plot_mc_vs_qmc(mc_results, qmc_results, snis_results_mc=None, snis_results_qmc=None,
                   save_path="results_logi/banana_mc_qmc_snis.png", tail_k=None):
    sample_sizes = mc_results['sample_sizes']
    plt.figure(figsize=(10, 6))

    # MC
    mc_err = mc_results['mse_errors']
    plt.loglog(sample_sizes, mc_err, 'r-o', label='MC (transport via flow)')
    plt.loglog(sample_sizes, mc_err[0] * (sample_sizes[0] / np.array(sample_sizes))**0.5,
               'r--', label='MC O(N^-0.5)')
    mc_slope, _ = _fit_loglog_slope(sample_sizes, mc_err, tail_k=tail_k)
    plt.text(sample_sizes[-1]*1.05, mc_err[-1], f"slope≈{mc_slope:.2f}",
             color='r', ha='left', va='center', fontsize=10)

    # QMC
    qmc_err = qmc_results['mse_errors']
    plt.loglog(sample_sizes, qmc_err, 'b-s', label='QMC (transport via flow)')
    plt.loglog(sample_sizes, qmc_err[0] * (sample_sizes[0] / np.array(sample_sizes)),
               'b--', label='QMC O(N^-1)')
    qmc_slope, _ = _fit_loglog_slope(sample_sizes, qmc_err, tail_k=tail_k)
    plt.text(sample_sizes[-1]*1.05, qmc_err[-1], f"slope≈{qmc_slope:.2f}",
             color='b', ha='left', va='center', fontsize=10)

    # SNIS（可选）
    if snis_results_mc is not None and snis_results_mc.get('mse_errors'):
        snis_mc_err = snis_results_mc['mse_errors']
        plt.loglog(sample_sizes, snis_mc_err, 'g-^', label='SNIS (MC, q=flow)')
        snis_mc_slope, _ = _fit_loglog_slope(sample_sizes, snis_mc_err, tail_k=tail_k)
        plt.text(sample_sizes[-1]*1.05, snis_mc_err[-1], f"slope≈{snis_mc_slope:.2f}",
                 color='g', ha='left', va='center', fontsize=10)

    if snis_results_qmc is not None and snis_results_qmc.get('mse_errors'):
        snis_qmc_err = snis_results_qmc['mse_errors']
        plt.loglog(sample_sizes, snis_qmc_err, 'k-^', label='SNIS (QMC, q=flow)')
        snis_qmc_slope, _ = _fit_loglog_slope(sample_sizes, snis_qmc_err, tail_k=tail_k)
        plt.text(sample_sizes[-1]*1.05, snis_qmc_err[-1], f"slope≈{snis_qmc_slope:.2f}",
                 color='k', ha='left', va='center', fontsize=10)

    plt.xlabel("Sample Size")
    plt.ylabel("RMSE (Euclidean) of mean")
    plt.title("Banana mean: MC vs QMC vs SNIS")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"[Slope] MC: {mc_slope:.3f}, QMC: {qmc_slope:.3f}" +
          (f", SNIS-MC: {snis_mc_slope:.3f}" if snis_results_mc is not None else "") +
          (f", SNIS-QMC: {snis_qmc_slope:.3f}" if snis_results_qmc is not None else ""))

def plot_snis_ess(snis_results, save_path="results_logi/banana_snis_ess.png"):
    if snis_results is None or not snis_results.get('ess'): return
    N = snis_results['sample_sizes']
    ess = snis_results['ess']
    plt.figure(figsize=(8,5))
    plt.loglog(N, ess, 'g-^', label='SNIS ESS')
    plt.xlabel("Sample Size")
    plt.ylabel("ESS")
    plt.title("SNIS Effective Sample Size vs N (Banana)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

# ----------------------------
# Main
# ----------------------------
def main():
    # 按你的训练脚本保存的 ckpt 路径来
    checkpoint_path = "results_logi/fm_model_banana.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device=device)

    true_mean = compute_true_banana_mean(a=A, b=B, c=C)   # [0, -0.5]
    print("True Banana Mean:", true_mean)

    # 样本规模（2^p）
    p_values = list(range(1, 14))  # 64 ... 1024
    # 统一数值设置
    sampling_steps = 64
    integrator = "heun"
    logprob_steps = 64
    logprob_batch_size = 256

    # MC / QMC（直接把 base 高斯通过 learned flow 传输到模型分布）
    mc_res = estimate_first_moment(model, p_values, true_mean, num_experiments=30,
                                   sampling_steps=sampling_steps, integrator=integrator)
    qmc_res = estimate_first_moment_qmc(model, p_values, true_mean, num_experiments=30,
                                        sampling_steps=2*sampling_steps, integrator=integrator)

    # SNIS（用 model 作为提议分布 q，权重用真香蕉 p）
    snis_res_mc = estimate_first_moment_snis(
        model,
        p_values,
        true_mean=true_mean,
        num_experiments=30,
        sampling_steps=sampling_steps,
        integrator=integrator,          # 反向 log_prob 也可用 rk4，数值更稳
        logprob_steps=logprob_steps,
        logprob_batch_size=logprob_batch_size,
        method='mc'
    )
    snis_res_qmc = estimate_first_moment_snis(
        model,
        p_values,
        true_mean=true_mean,
        num_experiments=30,
        sampling_steps=sampling_steps,
        integrator=integrator,
        logprob_steps=logprob_steps,
        logprob_batch_size=logprob_batch_size,
        method='qmc'
    )

    # 画图
    plot_mc_vs_qmc(mc_res, qmc_res, snis_results_mc=snis_res_mc,
                   snis_results_qmc=snis_res_qmc, save_path="results_logi/banana_mc_qmc_snis.png")
    plot_snis_ess(snis_res_mc, save_path="results_logi/banana_snis_ess_mc.png")
    plot_snis_ess(snis_res_qmc, save_path="results_logi/banana_snis_ess_qmc.png")

    print("Evaluation complete. Plots saved under results_logi/.")

if __name__ == "__main__":
    main()
