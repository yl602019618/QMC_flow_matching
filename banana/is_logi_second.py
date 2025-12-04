
# eval_banana_transport_expectation_second_moment.py
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
from flow_matching_logistic import FlowMatchingOT

def load_model(checkpoint_path, device="cpu"):
    model = FlowMatchingOT(dim=2, hidden_dim=512, num_blocks=6, sigma=0.0, lr=1e-3, device=device)
    model.load(checkpoint_path)
    model.eval()
    return model

# ----------------------------
# True banana *second* moment (analytic)
# ----------------------------
def compute_true_banana_second_moment(a=A, b=B, c=C):
    """
    x1 = z1
    x2 = a z1^2 + c + b z2
    E[z1] = E[z2] = 0, Var(z1) = Var(z2) = 1, E[z1^4] = 3

    E[x1^2] = E[z1^2] = 1
    E[x2^2] = E[(a z1^2 + c + b z2)^2]
            = a^2 E[z1^4] + c^2 + b^2 E[z2^2] + 2ac E[z1^2]
            = 3 a^2 + c^2 + b^2 + 2 a c
    """
    m2_x1 = 1.0
    m2_x2 = 3.0 * a * a + b * b + c * c + 2.0 * a * c
    return np.array([m2_x1, m2_x2], dtype=np.float64)

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
# MC (transport via learned flow) — second moment
# ----------------------------
def estimate_second_moment(model, p_values, true_mean=None, num_experiments=10,
                           sampling_steps=64, integrator="heun"):
    """
    估计原始二阶矩 E[X^2]（逐坐标）：
        m2[i] = E[X_i^2]
    MC 估计：m2_hat = (1/N) Σ x_i^2
    """
    model.eval()
    results = {
        'p_values': p_values,
        'sample_sizes': [],
        'estimated_means': [],   # 这里沿用字段名 estimated_means，含义是二阶矩
        'mse_errors': [],
        'bias_squared': [],
        'variance': [],
        'std_errors': [],
        'true_mean': true_mean   # 这里存的是 true second moment
    }

    with torch.no_grad():
        for p in tqdm(p_values):
            N = 2**p
            results['sample_sizes'].append(N)
            estimates = []
            for exp_id in range(num_experiments):
                x = model.sample(N, sampling_steps=sampling_steps, integrator=integrator)  # (N,2)
                m2 = torch.mean(x ** 2, dim=0)  # 二阶矩
                estimates.append(m2.cpu().numpy())
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
# QMC (transport via learned flow) — second moment
# ----------------------------
def estimate_second_moment_qmc(model, p_values, true_mean=None, num_experiments=10,
                               sampling_steps=128, integrator="heun"):
    """
    QMC 版本的二阶矩估计：
        m2[i] = E[X_i^2]
    """
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
                                     integrator=integrator, exp=exp_id)  # (N,2)
                m2 = torch.mean(x ** 2, dim=0)
                estimates.append(m2.cpu().numpy())
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
# Self-normalized IS (MC/QMC) with model as proposal q — second moment
# ----------------------------
def estimate_second_moment_snis(model,
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
    目标量是原始二阶矩：
        m2[i] = E_p[X_i^2]
    SNIS 估计： m2_hat = Σ w_i g(x_i)，其中 g(x)=x^2，w_i 是归一化权重。
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

            # 5) 二阶矩：g(x) = x^2
            g = x ** 2
            mu_hat = torch.sum(w.unsqueeze(1) * g, dim=0)    # (2,)
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
    N_arr = np.array(sample_sizes, dtype=float)

    plt.figure(figsize=(8, 6))
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 14,
    })

    # --------- FM-MC ----------
    mc_err = mc_results['mse_errors']
    plt.loglog(sample_sizes, mc_err, 'r-o', label='FM-MC')
    mc_slope, _ = _fit_loglog_slope(sample_sizes, mc_err, tail_k=tail_k)

    # --------- FM-QMC ----------
    qmc_err = qmc_results['mse_errors']
    plt.loglog(sample_sizes, qmc_err, 'b-s', label='FM-QMC')
    qmc_slope, _ = _fit_loglog_slope(sample_sizes, qmc_err, tail_k=tail_k)

    # 预先定义，避免没有 SNIS 时变量未定义
    snis_mc_slope = np.nan
    snis_qmc_slope = np.nan

    # --------- FM-ISMC ----------
    if snis_results_mc is not None and snis_results_mc.get('mse_errors'):
        snis_mc_err = snis_results_mc['mse_errors']
        plt.loglog(sample_sizes, snis_mc_err, 'g-^', label='FM-ISMC')
        snis_mc_slope, _ = _fit_loglog_slope(sample_sizes, snis_mc_err, tail_k=tail_k)

        # 参考线：斜率 -0.5，以 FM-ISMC 起点为起点
        N0_mc = N_arr[0]
        err0_mc = snis_mc_err[0]
        ref_mc_minus05 = err0_mc * (N_arr / N0_mc) ** (-0.5)
        plt.loglog(
            sample_sizes, ref_mc_minus05,
            '--', linewidth=1, alpha=0.7,
            label='Slope = -0.5'
        )

    # --------- FM-ISQMC ----------
    if snis_results_qmc is not None and snis_results_qmc.get('mse_errors'):
        snis_qmc_err = snis_results_qmc['mse_errors']
        plt.loglog(sample_sizes, snis_qmc_err, 'k-^', label='FM-ISQMC')
        snis_qmc_slope, _ = _fit_loglog_slope(sample_sizes, snis_qmc_err, tail_k=tail_k)

        # 参考线 1：斜率 -1，以 FM-ISQMC 起点为起点
        N0_qmc = N_arr[0]
        err0_qmc = snis_qmc_err[0]
        ref_qmc_minus1 = err0_qmc * (N_arr / N0_qmc) ** (-1.0)
        plt.loglog(
            sample_sizes, ref_qmc_minus1,
            '--', linewidth=1, alpha=0.7,
            label='Slope = -1 '
        )

        # 参考线 2：斜率为 FM-ISQMC 拟合斜率，以 FM-ISQMC 起点为起点
        ref_qmc_slope = err0_qmc * (N_arr / N0_qmc) ** (snis_qmc_slope)
        plt.loglog(
            sample_sizes, ref_qmc_slope,
            '--', linewidth=1, alpha=0.7,
            label=f'Slope = {snis_qmc_slope:.2f} '
        )

    plt.xlabel("Sample Size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    # 控制台打印斜率，方便对比
    print(f"[Slope] MC: {mc_slope:.3f}, QMC: {qmc_slope:.3f}"
          + (f", SNIS-MC: {snis_mc_slope:.3f}" if not np.isnan(snis_mc_slope) else "")
          + (f", SNIS-QMC: {snis_qmc_slope:.3f}" if not np.isnan(snis_qmc_slope) else ""))


def plot_snis_ess(snis_results, save_path="results_logi/banana_snis_ess_second_moment.png"):
    if snis_results is None or not snis_results.get('ess'): return
    N = snis_results['sample_sizes']
    ess = snis_results['ess']
    plt.figure(figsize=(8,5))
    plt.loglog(N, ess, 'g-^', label='SNIS ESS')
    plt.xlabel("Sample Size")
    plt.ylabel("ESS")
    plt.title("SNIS Effective Sample Size vs N (Banana, Second Moment)")
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

    true_second_moment = compute_true_banana_second_moment(a=A, b=B, c=C)
    print("True Banana Second Moment:", true_second_moment)

    # 样本规模（2^p）
    p_values = list(range(1, 14))
    # 统一数值设置
    sampling_steps = 64
    integrator = "heun"
    logprob_steps = 64
    logprob_batch_size = 256

    # MC / QMC（二阶矩）
    mc_res = estimate_second_moment(
        model,
        p_values,
        true_second_moment,
        num_experiments=30,
        sampling_steps=sampling_steps,
        integrator=integrator,
    )
    qmc_res = estimate_second_moment_qmc(
        model,
        p_values,
        true_second_moment,
        num_experiments=30,
        sampling_steps=2 * sampling_steps,
        integrator=integrator,
    )

    # SNIS（二阶矩，model 作为提议分布 q，权重用真香蕉 p）
    snis_res_mc = estimate_second_moment_snis(
        model,
        p_values,
        true_mean=true_second_moment,
        num_experiments=30,
        sampling_steps=sampling_steps,
        integrator=integrator,
        logprob_steps=logprob_steps,
        logprob_batch_size=logprob_batch_size,
        method='mc'
    )
    snis_res_qmc = estimate_second_moment_snis(
        model,
        p_values,
        true_mean=true_second_moment,
        num_experiments=30,
        sampling_steps=sampling_steps,
        integrator=integrator,
        logprob_steps=logprob_steps,
        logprob_batch_size=logprob_batch_size,
        method='qmc'
    )

    # 画图
    plot_mc_vs_qmc(
        mc_res,
        qmc_res,
        snis_results_mc=snis_res_mc,
        snis_results_qmc=snis_res_qmc,
        save_path="results_logi/banana_mc_qmc_snis_second_moment.pdf",
    )
    plot_snis_ess(snis_res_mc, save_path="results_logi/banana_snis_ess_mc_second_moment.png")
    plot_snis_ess(snis_res_qmc, save_path="results_logi/banana_snis_ess_qmc_second_moment.png")

    print("Evaluation complete (SECOND moment). Plots saved under results_logi/.")

if __name__ == "__main__":
    main()

