import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import qmc
from tqdm import tqdm
# ----------------------------
# Load trained model
# ----------------------------
from flow_matching import FlowMatchingOT

def load_model(checkpoint_path, device="cpu"):
    model = FlowMatchingOT(dim=2, hidden_dim=512, num_blocks=4, sigma=0.0, lr=1e-3, device=device)
    model.load(checkpoint_path)
    model.eval()
    return model

# ----------------------------
# Compute true GMM mean
# ----------------------------

def compute_true_gmm_mean():
    means = np.array([[-2, -2], [2, -2], [-2, 2], [2, 2]])
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    return np.sum(weights[:, None] * means, axis=0)
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
# MC Estimation
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
                x = model.sample(N, sampling_steps=32)
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
# QMC Estimation (using same sample function)
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
            for _ in range(num_experiments):
                x = model.sample_qmc(N, sampling_steps=128,exp = _)
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
def estimate_first_moment_snis(model,
                               p_values,
                               true_mean=None,
                               num_experiments=10,
                               sampling_steps=32,
                               integrator="rk4",
                               logprob_steps=64,
                               logprob_batch_size=256,
                               method = 'mc'):
    """
    用模型 q_theta 采样（提议分布），用 SNIS 纠正到真实目标 p。
    - sampling_steps/integrator: 前向采样的数值积分设置
    - logprob_steps/logprob_batch_size: 计算 q(x)=model 的 batched_log_prob 的设置
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
            if method == 'qmc':
                x = model.sample_qmc(N, sampling_steps=sampling_steps, integrator=integrator,exp = exp_id ).to(device)  # (N,2)
            # 2) 计算 log q(x)  (模型密度；反向时间/trace 积分)
            logq = model.batched_log_prob(
                x, steps=logprob_steps, batch_size=logprob_batch_size, integrator=integrator
            )  # (N,)
            # 3) 计算 log p(x)  (真实 GMM 解析)
            logp = true_gmm_log_prob_torch(x)  # (N,)
            logw = (logp - logq)
            lse = torch.logsumexp(logw, dim=0)
            w = torch.exp(logw - lse)
            
            # 5) 估计一阶矩 μ = E_p[X] ≈ Σ w_i x_i
            mu_hat = torch.sum(w.unsqueeze(1) * x, dim=0)    # (2,)
            exp_estimates.append(mu_hat.detach().cpu().numpy())

            # 6) 有效样本量 ESS
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
                   save_path="results/mc_vs_qmc_comparison.png", tail_k=None):
    sample_sizes = mc_results['sample_sizes']
    plt.figure(figsize=(10, 6))

    # MC
    mc_err = mc_results['mse_errors']
    plt.loglog(sample_sizes, mc_err, 'r-o', label='MC')
    plt.loglog(sample_sizes, mc_err[0] * (sample_sizes[0] / np.array(sample_sizes))**0.5,
               'r--', label='MC O(N^-0.5)')
    mc_slope, _ = _fit_loglog_slope(sample_sizes, mc_err, tail_k=tail_k)
    # 在末端附近标注斜率
    plt.text(sample_sizes[-1]*1.05, mc_err[-1],
             f"slope≈{mc_slope:.2f}", color='r', ha='left', va='center', fontsize=10)

    # QMC
    qmc_err = qmc_results['mse_errors']
    plt.loglog(sample_sizes, qmc_err, 'b-s', label='QMC')
    plt.loglog(sample_sizes, qmc_err[0] * (sample_sizes[0] / np.array(sample_sizes)),
               'b--', label='QMC O(N^-1)')
    qmc_slope, _ = _fit_loglog_slope(sample_sizes, qmc_err, tail_k=tail_k)
    plt.text(sample_sizes[-1]*1.05, qmc_err[-1],
             f"slope≈{qmc_slope:.2f}", color='b', ha='left', va='center', fontsize=10)

    # SNIS（可选）
    if snis_results_mc is not None and 'mse_errors' in snis_results_mc and snis_results_mc['mse_errors']:
        snis_mc_err = snis_results_mc['mse_errors']
        plt.loglog(sample_sizes, snis_mc_err, 'g-^', label='SNIS MC (q=flow)')
        snis_mc_slope, _ = _fit_loglog_slope(sample_sizes, snis_mc_err, tail_k=tail_k)
        plt.text(sample_sizes[-1]*1.05, snis_mc_err[-1],
                 f"slope≈{snis_mc_slope:.2f}", color='g', ha='left', va='center', fontsize=10)

    if snis_results_qmc is not None and 'mse_errors' in snis_results_qmc and snis_results_qmc['mse_errors']:
        snis_qmc_err = snis_results_qmc['mse_errors']
        plt.loglog(sample_sizes, snis_qmc_err, 'k-^', label='SNIS QMC (q=flow)')
        snis_qmc_slope, _ = _fit_loglog_slope(sample_sizes, snis_qmc_err, tail_k=tail_k)
        plt.text(sample_sizes[-1]*1.05, snis_qmc_err[-1],
                 f"slope≈{snis_qmc_slope:.2f}", color='k', ha='left', va='center', fontsize=10)

    plt.xlabel("Sample Size")
    plt.ylabel("RMSE (Euclidean)")
    plt.title("MC vs QMC vs SNIS (First Moment)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

    # 可选：把斜率打印出来，方便日志查看
    print(f"[Slope] MC: {mc_slope:.3f}, QMC: {qmc_slope:.3f}" +
          (f", SNIS-MC: {snis_mc_slope:.3f}" if snis_results_mc is not None else "") +
          (f", SNIS-QMC: {snis_qmc_slope:.3f}" if snis_results_qmc is not None else ""))

def plot_snis_ess(snis_results, save_path="results/snis_ess.png"):
    if snis_results is None: return
    N = snis_results['sample_sizes']
    ess = snis_results['ess']
    plt.figure(figsize=(8,5))
    plt.loglog(N, ess, 'g-^', label='SNIS ESS')
    plt.xlabel("Sample Size")
    plt.ylabel("ESS")
    plt.title("SNIS Effective Sample Size vs N")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


# ----------------------------
# Main
# ----------------------------

def main():
    checkpoint_path = "results/fm_model1.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device=device)

    true_mean = compute_true_gmm_mean()   # [0,0]
    p_values = list(range(6, 11))
    print("True Mean:", true_mean)

    # 统一的数值设置，便于公平对比
    sampling_steps = 64
    integrator = "heun"
    logprob_steps = 64
    logprob_batch_size = 256

    # MC / QMC
    mc_res = estimate_first_moment(model, p_values, true_mean, num_experiments=10)
    qmc_res = estimate_first_moment_qmc(model, p_values, true_mean, num_experiments=10)

    # SNIS（基于模型作为提议分布）
    snis_res_mc = estimate_first_moment_snis(
        model,
        p_values,
        true_mean=true_mean,
        num_experiments=10,
        sampling_steps=sampling_steps,
        integrator=integrator,
        logprob_steps=logprob_steps,
        logprob_batch_size=logprob_batch_size,
        method = 'mc'
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
        method = 'qmc'
    )

    # 画 RMSE 对比（新增 SNIS）
    plot_mc_vs_qmc(mc_res, qmc_res, snis_results_mc=snis_res_mc,snis_results_qmc=snis_res_qmc, save_path="results/mc_qmc_snis_test.png")


    print("Evaluation complete. Plots saved.")

if __name__ == "__main__":
    main()
    
