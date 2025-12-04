import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import qmc
from tqdm import tqdm

# ----------------------------
# Load trained model (30D)
# ----------------------------
# 注意：这里假定你的 30D FlowMatchingOT 类与训练时一致，并且提供：
# - sample / sample_qmc
# - batched_log_prob（用于 SNIS 的 q(x) 计算）
# 若你的类名/文件名不同，请按实际调整 import。
from flow_matching_logi import FlowMatchingOT  # （若你把30D模型放在别处，请改成对应模块名）


def load_model(checkpoint_path, device="cpu"):
    """
    与 30D 训练时保持一致的配置（尤其是 dim/base_dist/网络深度）
    """
    model = FlowMatchingOT(
        dim=30, hidden_dim=512, num_blocks=8, sigma=0.0, lr=1e-3, device=device,
        base_dist="logistic", base_loc=0.0, base_scale=1.0
    )
    model.load(checkpoint_path)
    model.eval()
    return model


# ----------------------------
# Compute true 30D GMM mean & log-prob
# ----------------------------

def compute_true_gmm_mean_30d(dim=30):
    """
    30D GMM 的真实均值：四个分量的均值在前两维为 (+/-2, +/-2)，其余维 0；
    权重均等 => 总体均值为全 0（长度 dim）。
    """
    return np.zeros(dim, dtype=np.float32)


def true_gmm_log_prob_torch_30d(x: torch.Tensor) -> torch.Tensor:
    """
    30D 目标分布：4 个分量，均值只在前两维是 (+/-2, +/-2)，其余维=0。
    协方差为 0.5 * I_30（对角阵），权重均等 0.25。
    返回 log p(x) 形状 (N,).
    """
    device = x.device
    N, D = x.shape
    assert D == 30, f"Expecting 30D input, got {D}."

    # 四个均值 (4,30)
    means = torch.zeros((4, D), dtype=torch.float32, device=device)
    means[0, :2] = torch.tensor([-2.0, -2.0], device=device)
    means[1, :2] = torch.tensor([ 2.0, -2.0], device=device)
    means[2, :2] = torch.tensor([-2.0,  2.0], device=device)
    means[3, :2] = torch.tensor([ 2.0,  2.0], device=device)

    # 协方差：0.5 * I（各维独立）
    var = 0.5
    inv_var = 1.0 / var
    # 归一化常数：(2π)^{-D/2} * var^{-D/2}
    norm_const = (2 * np.pi) ** (-D / 2) * (var ** (-D / 2))
    norm_const = torch.tensor(norm_const, dtype=torch.float32, device=device)

    # (4,N,D)
    diff = x.unsqueeze(0) - means.unsqueeze(1)
    md = (diff.pow(2).sum(dim=2)) * inv_var     # (4,N)
    comp = norm_const * torch.exp(-0.5 * md)    # (4,N)
    weights = torch.full((4, 1), 0.25, dtype=torch.float32, device=device)  # (4,1)
    p = torch.sum(weights * comp, dim=0)        # (N,)
    p = torch.clamp(p, min=1e-38)
    return torch.log(p)


# ----------------------------
# MC Estimation (proposal = model,直接取样均值)
# ----------------------------

def estimate_first_moment(model, p_values, true_mean=None, num_experiments=10):
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
            for _ in range(num_experiments):
                x = model.sample(N, sampling_steps=32)         # (N,30)
                estimates.append(torch.mean(x, dim=0).numpy()) # (30,)
            estimates = np.stack(estimates)                    # (num_experiments,30)
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
                results['std_errors'].append(np.mean(np.std(estimates, axis=0)))
    return results


# ----------------------------
# QMC Estimation (proposal = model.sample_qmc)
# ----------------------------

def estimate_first_moment_qmc(model, p_values, true_mean=None, num_experiments=10):
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
                x = model.sample_qmc(N, sampling_steps=128, exp=exp_id)  # (N,30)
                estimates.append(torch.mean(x, dim=0).numpy())
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
                results['std_errors'].append(np.mean(np.std(estimates, axis=0)))
    return results


# ----------------------------
# SNIS (proposal = model, target = true 30D GMM)
# ----------------------------

def estimate_first_moment_snis(model,
                               p_values,
                               true_mean=None,
                               num_experiments=10,
                               sampling_steps=32,
                               integrator="rk4",
                               logprob_steps=64,
                               logprob_batch_size=256,
                               method='mc'):
    """
    用模型 q_theta 采样（提议分布），用 SNIS 纠正到真实目标 p（30D GMM）。
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
            # 1) 从 q_theta 采样 (N,30)
            if method == 'mc':
                x = model.sample(N, sampling_steps=sampling_steps, integrator=integrator).to(device)
            elif method == 'qmc':
                x = model.sample_qmc(N, sampling_steps=sampling_steps, integrator=integrator, exp=exp_id).to(device)
            else:
                raise ValueError("method must be 'mc' or 'qmc'")

            # 2) log q(x)（模型密度）
            logq = model.batched_log_prob(
                x, steps=logprob_steps, batch_size=logprob_batch_size, integrator=integrator
            )  # (N,)

            # 3) log p(x)（30D 真 GMM）
            logp = true_gmm_log_prob_torch_30d(x)  # (N,)

            # 4) 归一化权重
            logw = (logp - logq)
            lse = torch.logsumexp(logw, dim=0)
            w = torch.exp(logw - lse)  # (N,)

            # 5) 一阶矩 μ = E_p[X] ≈ Σ w_i x_i
            mu_hat = torch.sum(w.unsqueeze(1) * x, dim=0)    # (30,)
            exp_estimates.append(mu_hat.detach().cpu().numpy())

            # 6) ESS
            ess = 1.0 / torch.sum(w * w)
            exp_ess.append(ess.item())

        estimates = np.stack(exp_estimates)              # (num_experiments, 30)
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


def _fit_loglog_slope(sample_sizes, errors, tail_k=None):
    x = np.asarray(sample_sizes, dtype=float)
    y = np.asarray(errors, dtype=float)
    if tail_k is not None and 2 <= tail_k <= len(x):
        x = x[-tail_k:]
        y = y[-tail_k:]
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[m], y[m]
    if x.size < 2:
        return np.nan, np.nan
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


def plot_snis_ess(snis_results, save_path="results30d/snis_ess.pdf"):
    if snis_results is None: return
    N = snis_results['sample_sizes']
    ess = snis_results['ess']
    plt.figure(figsize=(8,5))
    plt.loglog(N, ess, 'g-^', label='SNIS ESS')
    plt.xlabel("Sample Size")
    plt.ylabel("ESS")
    plt.title("SNIS Effective Sample Size vs N (30D)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


# ----------------------------
# Main
# ----------------------------

def main():
    checkpoint_path = "results30d/fm_model_30d.pt"  # 30D checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device=device)

    true_mean = compute_true_gmm_mean_30d(dim=30)   # 30D 全 0
    p_values = list(range(1, 15))  # N = 2^1 ... 2^13 = 8192
    print("True Mean (30D):", true_mean)

    # 统一数值设置（与 10D 一致）
    sampling_steps = 64
    integrator = "heun"
    logprob_steps = 64
    logprob_batch_size = 256

    # MC / QMC
    mc_res = estimate_first_moment(model, p_values, true_mean, num_experiments=10)
    qmc_res = estimate_first_moment_qmc(model, p_values, true_mean, num_experiments=10)

    # SNIS（提议 = flow 模型，目标 = 30D 真 GMM）
    snis_res_mc = estimate_first_moment_snis(
        model,
        p_values,
        true_mean=true_mean,
        num_experiments=10,
        sampling_steps=sampling_steps,
        integrator=integrator,
        logprob_steps=logprob_steps,
        logprob_batch_size=logprob_batch_size,
        method='mc'
    )
    snis_res_qmc = estimate_first_moment_snis(
        model,
        p_values,
        true_mean=true_mean,
        num_experiments=10,
        sampling_steps=sampling_steps,
        integrator=integrator,
        logprob_steps=logprob_steps,
        logprob_batch_size=logprob_batch_size,
        method='qmc'
    )

    # 可视化：RMSE 对比 & ESS
    plot_mc_vs_qmc(mc_res, qmc_res,
                   snis_results_mc=snis_res_mc,
                   snis_results_qmc=snis_res_qmc,
                   save_path="results30d/mc_qmc_snis_30d.pdf")
    plot_snis_ess(snis_res_mc, save_path="results30d/snis_ess_mc_30d.png")
    plot_snis_ess(snis_res_qmc, save_path="results30d/snis_ess_qmc_30d.png")

    print("Evaluation complete. Plots saved to results30d/.")

if __name__ == "__main__":
    main()
