import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import qmc
from tqdm import tqdm
# ----------------------------
# Load trained model
# ----------------------------
from flow_matching_logistic import FlowMatchingOT


def load_model(checkpoint_path, device="cpu"):
    # 与训练时保持一致：Logistic 基分布（若你训练时用的别的 loc/scale，也在这里一并填上）
    model = FlowMatchingOT(
        dim=2, hidden_dim=512, num_blocks=4, sigma=0.0, lr=1e-3, device=device,
        base_dist="logistic", base_loc=0.0, base_scale=1.0
    )
    model.load(checkpoint_path)
    model.eval()
    return model


# ----------------------------
# Compute true GMM *second* moment E[X_i^2]
# ----------------------------

def compute_true_gmm_second_moment():
    """
    返回的是每个坐标的二阶原始矩：
        m2[i] = E[X_i^2], i=1,2
    对于 GMM：E[X X^T] = Σ_k w_k (Σ_k + μ_k μ_k^T)
    这里返回的是 diag(E[X X^T])。
    """
    means = np.array([[-2, -2], [2, -2], [-2, 2], [2, 2]], dtype=np.float32)
    covs = [np.array([[0.5, 0.1], [0.1, 0.5]], dtype=np.float32) for _ in range(4)]
    weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

    d = means.shape[1]
    m2 = np.zeros(d, dtype=np.float32)
    for k in range(len(weights)):
        # diag(Σ_k) + μ_k^2
        m2 += weights[k] * (np.diag(covs[k]) + means[k] ** 2)
    return m2


def true_gmm_log_prob_torch(x: torch.Tensor) -> torch.Tensor:
    """
    x: (N,2) on any device
    returns: log p(x) as (N,) torch tensor
    """
    device = x.device
    means = torch.tensor([[-2., -2.], [ 2., -2.], [-2.,  2.], [ 2.,  2.]], device=device)
    cov = torch.tensor([[0.5, 0.1], [0.1, 0.5]], device=device)  # all components equal
    cov_inv = torch.inverse(cov)
    det = torch.det(cov)
    norm_const = 1.0 / (2 * np.pi * torch.sqrt(det))
    weights = torch.tensor([0.25, 0.25, 0.25, 0.25], device=device)

    # compute each component density
    # exponent_k = (x - m_k)^T cov_inv (x - m_k)
    # broadcasting: (K, N, 2)
    diff = x.unsqueeze(0) - means.unsqueeze(1)  # (4,N,2)
    # (4,N,2) @ (2,2) -> (4,N,2), then * (4,N,2) -> sum over last dim
    md = torch.einsum('kni,ij,knj->kn', diff, cov_inv, diff)  # (4,N)
    comp = norm_const * torch.exp(-0.5 * md)                   # (4,N)
    p = torch.sum(weights.unsqueeze(1) * comp, dim=0)          # (N,)
    # numerical floor
    p = torch.clamp(p, min=1e-38)
    return torch.log(p)

# ----------------------------
# MC Estimation of second moment
# ----------------------------

def estimate_second_moment(model, p_values, true_mean=None, num_experiments=10):
    """
    估计 E[X^2]（逐坐标）：
        m2[i] = E[X_i^2]
    MC 估计：m2_hat = (1/N) Σ x_i^2
    """
    model.eval()
    results = {
        'p_values': p_values,
        'sample_sizes': [],
        'estimated_means': [],   # 这里依然沿用字段名 estimated_means，但含义是二阶矩
        'mse_errors': [],
        'bias_squared': [],
        'variance': [],
        'std_errors': [],
        'true_mean': true_mean   # 同理，这里存的是 true second moment
    }

    with torch.no_grad():
        for p in tqdm(p_values):
            N = 2**p
            results['sample_sizes'].append(N)
            estimates = []
            for _ in range(num_experiments):
                x = model.sample(N, sampling_steps=32)      # (N,2)
                m2 = torch.mean(x ** 2, dim=0)              # (2,) 二阶矩
                estimates.append(m2.numpy())
            estimates = np.stack(estimates)                 # (num_exp, 2)
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
# QMC Estimation (second moment)
# ----------------------------

def estimate_second_moment_qmc(model, p_values, true_mean=None, num_experiments=10):
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
                x = model.sample_qmc(N, sampling_steps=128, exp=exp_id)   # (N,2)
                m2 = torch.mean(x ** 2, dim=0)
                estimates.append(m2.numpy())
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


def estimate_second_moment_snis(model,
                                p_values,
                                true_mean=None,
                                num_experiments=10,
                                sampling_steps=32,
                                integrator="rk4",
                                logprob_steps=64,
                                logprob_batch_size=256,
                                method='mc'):
    """
    用模型 q_theta 采样（提议分布），用 SNIS 纠正到真实目标 p 来估计二阶矩。
    目标量是：
        m2[i] = E_p[X_i^2]
    这里沿用原来的未归一化形式：
        m2_hat ≈ Σ w_i * (x_i^2)
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
        'ess': [],                 # 平均 ESS（跨实验的均值）
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
                x = model.sample(N, sampling_steps=sampling_steps, integrator=integrator).to(device)  # (N,2)
            elif method == 'qmc':
                x = model.sample_qmc(N, sampling_steps=sampling_steps, integrator=integrator, exp=exp_id).to(device)
            else:
                raise ValueError(f"Unknown method: {method}")

            # 2) 计算 log q(x)  (模型密度；反向时间/trace 积分)
            logq = model.batched_log_prob(
                x, steps=logprob_steps, batch_size=logprob_batch_size, integrator=integrator
            )  # (N,)
            # 3) 计算 log p(x)  (真实 GMM 解析)
            logp = true_gmm_log_prob_torch(x)  # (N,)
            logw = (logp - logq)
            lse = torch.logsumexp(logw, dim=0)
            w = torch.exp(logw - lse)

            # 4) 二阶矩的 SNIS 估计：m2 ≈ Σ w_i x_i^2
            g = x ** 2
            mu_hat = torch.sum(w.unsqueeze(1) * g, dim=0)  # (2,)
            exp_estimates.append(mu_hat.detach().cpu().numpy())

            # 5) 有效样本量 ESS
            ess = 1.0 / torch.sum(w * w)
            exp_ess.append(ess.item())

        estimates = np.stack(exp_estimates)              # (num_experiments, 2)
        mean_est = estimates.mean(axis=0)
        results['estimated_means'].append(mean_est)
        results['ess'].append(float(np.mean(exp_ess)))

        if true_mean is not None:
            bias = mean_est - true_mean
            bias_sq = float(np.sum(bias ** 2))
            # 方差：用实验间的样本均值波动
            diffs = estimates - mean_est[None, :]
            var = float(np.mean(np.sum(diffs ** 2, axis=1)))
            mse = bias_sq + var
            results['mse_errors'].append(np.sqrt(mse))
            results['bias_squared'].append(np.sqrt(bias_sq))
            results['variance'].append(np.sqrt(var))
            results['std_errors'].append(float(np.mean(np.std(estimates, axis=0, ddof=1))))

    return results


def _fit_loglog_slope(sample_sizes, errors, tail_k=None):
    """
    在 log-log 空间对 y = errors, x = sample_sizes 做线性拟合，返回 (slope, intercept)。
    tail_k: 若给定，则只用最后 tail_k 个点做拟合（更贴近渐近收敛阶）。
    """
    x = np.asarray(sample_sizes, dtype=float)
    y = np.asarray(errors, dtype=float)
    if tail_k is not None and 2 <= tail_k <= len(x):
        x = x[-tail_k:]
        y = y[-tail_k:]
    # 过滤无效/非正数据
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[m], y[m]
    if x.size < 2:
        return np.nan, np.nan
    lx, ly = np.log(x), np.log(y)  # 自然对数或 log10 都可，斜率不变
    slope, intercept = np.polyfit(lx, ly, 1)
    return float(slope), float(intercept)

# ----------------------------
# Plotting
# ----------------------------
def plot_mc_vs_qmc(mc_results, qmc_results, snis_results_mc=None, snis_results_qmc=None,
                   save_path="results_logi/mc_vs_qmc_comparison_second_moment.png", tail_k=None):
    """
    画 MC / QMC / SNIS 在二阶矩估计上的 RMSE 收敛阶。
    """
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

    # --------- FM-ISMC ----------
    if snis_results_mc is not None and 'mse_errors' in snis_results_mc and snis_results_mc['mse_errors']:
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
    else:
        snis_mc_slope = np.nan

    # --------- FM-ISQMC ----------
    if snis_results_qmc is not None and 'mse_errors' in snis_results_qmc and snis_results_qmc['mse_errors']:
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
    else:
        snis_qmc_slope = np.nan

    plt.xlabel("Sample Size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    # 控制台打印斜率，方便对比
    print(f"[Slope] MC: {mc_slope:.3f}, QMC: {qmc_slope:.3f}"
          + (f", SNIS-MC: {snis_mc_slope:.3f}" if not np.isnan(snis_mc_slope) else "")
          + (f", SNIS-QMC: {snis_qmc_slope:.3f}" if not np.isnan(snis_qmc_slope) else ""))


def plot_snis_ess(snis_results, save_path="results_logi/snis_ess_second_moment.png"):
    if snis_results is None:
        return
    N = snis_results['sample_sizes']
    ess = snis_results['ess']
    plt.figure(figsize=(8,5))
    plt.loglog(N, ess, 'g-^', label='SNIS ESS')
    plt.xlabel("Sample Size")
    plt.ylabel("ESS")
    plt.title("SNIS Effective Sample Size vs N (Second Moment)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


# ----------------------------
# Main
# ----------------------------

def main():
    checkpoint_path = "results_logi/fm_model_logi.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device=device)

    true_second_moment = compute_true_gmm_second_moment()   # [E[X1^2], E[X2^2]]
    p_values = list(range(1, 14))
    print("True Second Moment:", true_second_moment)

    # 统一的数值设置，便于公平对比
    sampling_steps = 64
    integrator = "heun"
    logprob_steps = 64
    logprob_batch_size = 256

    # MC / QMC（二阶矩）
    mc_res = estimate_second_moment(model, p_values, true_second_moment, num_experiments=10)
    qmc_res = estimate_second_moment_qmc(model, p_values, true_second_moment, num_experiments=10)

    # SNIS（基于模型作为提议分布）——二阶矩
    snis_res_mc = estimate_second_moment_snis(
        model,
        p_values,
        true_mean=true_second_moment,
        num_experiments=10,
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
        num_experiments=10,
        sampling_steps=sampling_steps,
        integrator=integrator,
        logprob_steps=logprob_steps,
        logprob_batch_size=logprob_batch_size,
        method='qmc'
    )

    # 画 RMSE 对比（二阶矩）
    plot_mc_vs_qmc(
        mc_res,
        qmc_res,
        snis_results_mc=snis_res_mc,
        snis_results_qmc=snis_res_qmc,
        save_path="results_logi/mc_qmc_snis_second_moment.pdf"
    )

    # 可选：画 ESS
    plot_snis_ess(snis_res_mc, save_path="results_logi/snis_ess_mc_second_moment.png")
    plot_snis_ess(snis_res_qmc, save_path="results_logi/snis_ess_qmc_second_moment.png")

    print("Evaluation complete. Plots for SECOND moment saved.")

if __name__ == "__main__":
    main()
