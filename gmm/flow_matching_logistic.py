import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import qmc
from scipy.stats import norm

# ------------------------------
# Utils
# ------------------------------

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------
# Dataset
# ------------------------------

class GMMDataset(Dataset):
    def __init__(self, num_samples=5000):
        self.means = np.array([[-2, -2], [2, -2], [-2, 2], [2, 2]], dtype=np.float32)
        self.covs = [np.array([[0.5, 0.1], [0.1, 0.5]], dtype=np.float32) for _ in range(4)]
        self.weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        self.data = self._generate_samples(num_samples).astype(np.float32)

    def _generate_samples(self, num_samples):
        samples = []
        for _ in range(num_samples):
            c = np.random.choice(4, p=self.weights)
            s = np.random.multivariate_normal(self.means[c], self.covs[c])
            samples.append(s)
        return np.array(samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])


# ------------------------------
# Model
# ------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, multiplier: int = 2):
        super().__init__()
        hidden_dim = int(dim * multiplier)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return x + self.mlp(x)


class FourierEmbedding(nn.Module):
    def __init__(self, dim: int, scale: float = 16.0):
        super().__init__()
        assert dim % 2 == 0, "FourierEmbedding dim must be even."
        freqs = torch.randn(dim // 2) * scale * 2 * np.pi
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        x = x.unsqueeze(-1)
        freqs = self.freqs.view(1, -1).expand(x.shape[0], -1)
        x = x * freqs
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


class Network(nn.Module):
    def __init__(self, dim=2, hidden_dim=256, num_blocks=3):
        super().__init__()
        self.t_proj = FourierEmbedding(hidden_dim)
        self.x_proj = nn.Linear(dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.out_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, dim))

    def forward(self, x, t):
        h = self.x_proj(x) + self.t_proj(t)
        for block in self.blocks:
            h = block(h)
        return self.out_proj(h)


class FlowMatchingOT(nn.Module):
    def __init__(
        self,
        dim=2,
        hidden_dim=512,
        num_blocks=4,
        sigma=0.0,
        lr=1e-3,
        device="cpu",
        # ------- 新增：基分布配置 -------
        base_dist="logistic",   # "logistic" 或 "gaussian"
        base_loc=0.0,           # 可以是标量或 shape=[dim] 的向量
        base_scale=1.0          # 可以是标量或 shape=[dim] 的向量
    ):
        super().__init__()
        self.model = Network(dim, hidden_dim, num_blocks).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.sigma = sigma
        self.dim = dim
        self.device = device

        # ------- 新增：基分布参数缓存 -------
        assert base_dist in ("logistic", "gaussian")
        self.base_dist = base_dist

        # 归一化为 (1, dim) 的 tensor，便于广播
        self.base_loc = torch.as_tensor(base_loc, dtype=torch.float32, device=device)
        self.base_scale = torch.as_tensor(base_scale, dtype=torch.float32, device=device)

        if self.base_loc.ndim == 0:
            self.base_loc = self.base_loc.repeat(dim).view(1, -1)
        else:
            self.base_loc = self.base_loc.view(1, -1)
        if self.base_scale.ndim == 0:
            self.base_scale = self.base_scale.repeat(dim).view(1, -1)
        else:
            self.base_scale = self.base_scale.view(1, -1)

    # ------- 新增：从基分布采样 x0 -------
    def sample_from_base(self, shape):
        """
        shape: (B, dim)
        """
        if self.base_dist == "gaussian":
            return torch.randn(shape, device=self.device)
        else:
            # Logistic 逆变换采样：x = mu + s * log(u/(1-u)), u ~ U(0,1)
            eps = 1e-6
            U = torch.rand(shape, device=self.device).clamp_(eps, 1.0 - eps)
            return self.base_loc + self.base_scale * torch.log(U / (1.0 - U))

    def conditional_prob_path(self, t, z, x0):
        t = t.unsqueeze(1)
        mean = t * z + (1 - t) * x0
        if self.sigma > 0:
            mean = mean + torch.randn_like(z) * self.sigma
        return mean

    def conditional_vel_field(self, t, z, x0):
        return z - x0

    def forward(self, z):
        t = torch.rand((z.shape[0],), device=z.device, dtype=z.dtype)
        x0 = self.sample_from_base(z.shape)   # << 替换：从基分布采样 x0（原来是 torch.randn_like）
        x = self.conditional_prob_path(t, z, x0)
        u = self.conditional_vel_field(t, z, x0)
        v = self.model(x, t)
        return (u - v).pow(2).mean()

    def sample(self, N, sampling_steps, integrator="heun"):
        """
        integrator: "euler" | "heun"
        """
        self.model.eval()
        with torch.no_grad():
            x = self.sample_from_base((N, self.dim))  # << 替换：以基分布为起点（原来是 torch.randn）
            t_schedule = torch.linspace(0.0, 1.0, sampling_steps + 1, device=self.device)

            for i in range(sampling_steps):
                t_i = t_schedule[i].repeat(N)
                dt = (t_schedule[i + 1] - t_schedule[i])

                if integrator == "euler":
                    v = self.model(x, t_i)
                    x = x + v * dt
                elif integrator == "heun":
                    v1 = self.model(x, t_i)
                    x_pred = x + v1 * dt
                    t_ip1 = t_schedule[i + 1].repeat(N)
                    v2 = self.model(x_pred, t_ip1)
                    x = x + 0.5 * dt * (v1 + v2)
                else:
                    raise ValueError(f"Unknown integrator: {integrator}")
        return x.detach().cpu()

    # def sample_qmc(self, N, sampling_steps, exp, integrator="heun"):
    #     # 仅 Gaussian 基分布实现了 QMC；Logistic 时直接报错避免误用
    #     if self.base_dist != "gaussian":
    #         raise NotImplementedError("QMC sampler is only implemented for Gaussian base.")
    #     sampler = qmc.MultivariateNormalQMC(mean=[0, 0], cov=[[1, 0], [0, 1]], seed=42+exp)
    #     qmc_gaussian = sampler.random(N)
    #     x = torch.tensor(qmc_gaussian, dtype=torch.float32).to(self.device)
    #     self.model.eval()
    #     with torch.no_grad():
    #         t_schedule = torch.linspace(0.0, 1.0, sampling_steps + 1, device=self.device)
    #         for i in range(sampling_steps):
    #             t_i = t_schedule[i].repeat(N)
    #             dt = (t_schedule[i + 1] - t_schedule[i])

    #             if integrator == "euler":
    #                 v = self.model(x, t_i)
    #                 x = x + v * dt
    #             elif integrator == "heun":
    #                 v1 = self.model(x, t_i)
    #                 x_pred = x + v1 * dt
    #                 t_ip1 = t_schedule[i + 1].repeat(N)
    #                 v2 = self.model(x_pred, t_ip1)
    #                 x = x + 0.5 * dt * (v1 + v2)
    #             else:
    #                 raise ValueError(f"Unknown integrator: {integrator}")

    #     return x.detach().cpu()
    def sample_qmc(self, N, sampling_steps, exp, integrator="heun"):
        """
        QMC 起点：
        - gaussian: 用 MultivariateNormalQMC（保留原逻辑）
        - logistic: 用 Sobol 在 (0,1)^d 采样，然后做 x0 = mu + s*log(u/(1-u))
        """
        self.model.eval()
        with torch.no_grad():
            if self.base_dist == "gaussian":
                # 原实现：高斯 QMC
                sampler = qmc.MultivariateNormalQMC(
                    mean=[0]*self.dim,
                    cov=np.eye(self.dim),
                    seed=42 + exp
                )
                u = sampler.random(N)                       # (N,dim) numpy
                x = torch.tensor(u, dtype=torch.float32, device=self.device)

            else:  # logistic
                # 用 Sobol 给出 (0,1)^dim 的低差异序列
                sobol = qmc.Sobol(d=self.dim, scramble=True, seed=42 + exp)
                u = sobol.random(N)                         # (N,dim) numpy in [0,1)
                x = torch.tensor(u, dtype=torch.float32, device=self.device)
                # 避免 log(0) 数值问题
                eps = 1e-6
                x = x.clamp(eps, 1.0 - eps)
                # 逆 CDF：logit，再缩放/平移到 (mu, s)
                x = self.base_loc + self.base_scale * torch.log(x / (1.0 - x))  # (N,dim)

            # 从 t=0 起点（QMC）沿学习到的速度场前向积分到 t=1
            t_schedule = torch.linspace(0.0, 1.0, sampling_steps + 1, device=self.device)
            for i in range(sampling_steps):
                t_i = t_schedule[i].repeat(N)
                dt = (t_schedule[i + 1] - t_schedule[i])
                if integrator == "euler":
                    v = self.model(x, t_i)
                    x = x + v * dt
                elif integrator == "heun":
                    v1 = self.model(x, t_i)
                    x_pred = x + v1 * dt
                    t_ip1 = t_schedule[i + 1].repeat(N)
                    v2 = self.model(x_pred, t_ip1)
                    x = x + 0.5 * dt * (v1 + v2)
                else:
                    raise ValueError(f"Unknown integrator: {integrator}")
        return x.detach().cpu()

    def save(self, path):
        torch.save({"model": self.model.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])

    def divergence_exact(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        B, d = x.shape
        div = torch.zeros(B, device=x.device, dtype=x.dtype)
        for i in range(d):
            retain = i < d - 1
            grad = torch.autograd.grad(v[:, i].sum(), x, retain_graph=retain)[0]
            div += grad[:, i]
        return div

    @staticmethod
    def _divergence_hutchinson(v: torch.Tensor, x: torch.Tensor, num_probe: int = 4, rademacher: bool = True) -> torch.Tensor:
        B, d = x.shape
        div = 0.0
        for _ in range(num_probe):
            if rademacher:
                eps = torch.randint(0, 2, (B, d), device=x.device, dtype=x.dtype) * 2 - 1
            else:
                eps = torch.randn_like(x)
            v_dot = (v * eps).sum()  # scalar
            (grad_x,) = torch.autograd.grad(v_dot, x, retain_graph=True, create_graph=False)
            div += (grad_x * eps).sum(dim=1)
        return div / num_probe

    def divergence(self, v: torch.Tensor, x: torch.Tensor, t_scalar: float,
                   method: str = "hutch", num_probe: int = 4) -> torch.Tensor:
        if method == "exact":
            return self.divergence_exact(v, x)
        elif method == "hutch":
            return self._divergence_hutchinson(v, x, num_probe=num_probe, rademacher=True)
        else:
            raise ValueError(f"Unknown div method: {method}")

    def _rk4_step(self, x: torch.Tensor, t: float, dt: float,
                  div_method: str = "hutch", num_probe: int = 4):
        """
        单步 RK4：反向时间 t -> t - dt（dt > 0）
        返回: (x_new, ds)，其中 ds 是本步的 log-Jacobian 增量（负散度积分）
        """
        B = x.size(0)
        device = x.device
        dtype = x.dtype

        def f(x_, t_):
            t_vec = torch.full((B,), t_, device=device, dtype=dtype)
            v_ = self.model(x_, t_vec)
            d_ = self.divergence(v_, x_, t_, method=div_method, num_probe=num_probe)
            return v_, d_

        # k1 @ t
        v1, d1 = f(x, t)
        # k2 @ t - dt/2
        x2 = (x - 0.5 * dt * v1).detach(); x2.requires_grad_(True)
        v2, d2 = f(x2, t - 0.5 * dt)
        # k3 @ t - dt/2
        x3 = (x - 0.5 * dt * v2).detach(); x3.requires_grad_(True)
        v3, d3 = f(x3, t - 0.5 * dt)
        # k4 @ t - dt
        x4 = (x - dt * v3).detach(); x4.requires_grad_(True)
        v4, d4 = f(x4, t - dt)

        x_new = x - (dt / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
        ds    = -(dt / 6.0) * (d1 + 2 * d2 + 2 * d3 + d4)
        return x_new, ds

    # ---------- 基分布 log-prob ----------
    def log_p_base(self, x):
        if self.base_dist == "gaussian":
            return -0.5 * (x ** 2).sum(dim=1) - 0.5 * x.shape[1] * np.log(2 * np.pi)
        else:
            # logistic 独立同分布：sum_i [ -log s_i - y_i - 2*softplus(-y_i) ], y_i = (x_i - mu_i)/s_i
            y = (x - self.base_loc) / self.base_scale
            term1 = -torch.log(self.base_scale).sum(dim=1)
            term2 = -y.sum(dim=1)
            term3 = -2.0 * torch.nn.functional.softplus(-y).sum(dim=1)
            return term1 + term2 + term3

    def _log_prob_single(self,
                         x1: torch.Tensor,
                         steps: int = 128,
                         integrator: str = "rk4",
                         div_method: str = "exact",
                         num_probe: int = 4,
                         t_eps: float = 1e-4):
        """
        计算一批点 x1 的 log q(x1)。
        - steps: 反向时间积分步数（越大越准）
        - integrator: "euler" | "heun" | "rk4"
        - div_method: "exact" | "hutch"
        - num_probe: Hutchinson 估计的 probe 数
        - t_eps: 端点避开 [0,1] 的小缓冲（如 1e-3），可减小边界奇异
        """
        assert integrator in ("euler", "heun", "rk4")
        assert div_method in ("exact", "hutch")

        self.model.eval()
        device = self.device

        # 初始化
        x_t = x1.clone().detach().to(device)
        x_t.requires_grad_(True)

        # 从 t≈1 积分到 t≈0（反向）
        t_start = 1.0 - float(t_eps)
        t_end   = 0.0 + float(t_eps)
        ts = torch.linspace(t_start, t_end, steps, device=device)  # 递减
        s = torch.zeros(x_t.shape[0], device=device, dtype=x_t.dtype)  # 累积 log-Jacobian: s = -∫ div v dt

        if integrator == "rk4":
            for k in range(steps - 1):
                t = ts[k].item()
                t_next = ts[k + 1].item()
                dt = t - t_next  # > 0

                x_t, ds = self._rk4_step(x_t, t, dt, div_method=div_method, num_probe=num_probe)
                s = s + ds
                x_t = x_t.detach()
                x_t.requires_grad_(True)

        else:
            for k in range(steps - 1):
                t = ts[k]
                t_next = ts[k + 1]
                dt = (t - t_next)  # > 0, 标量张量

                # 第一次评估
                t_vec = torch.full((x_t.size(0),), t.item(), device=device, dtype=x_t.dtype)
                v1 = self.model(x_t, t_vec)
                div1 = self.divergence(v1, x_t, t.item(), method=div_method, num_probe=num_probe)

                if integrator == "euler":
                    s = s - dt * div1
                    x_t = x_t - dt * v1

                elif integrator == "heun":
                    # 预测到 t_next
                    x_pred = (x_t - dt * v1).detach()
                    x_pred.requires_grad_(True)
                    t_next_vec = torch.full((x_t.size(0),), t_next.item(), device=device, dtype=x_t.dtype)
                    v2 = self.model(x_pred, t_next_vec)
                    div2 = self.divergence(v2, x_pred, t_next.item(), method=div_method, num_probe=num_probe)

                    # 梯形法积分类散度 + Heun 位置校正
                    s = s - 0.5 * dt * (div1 + div2)
                    x_t = x_t - 0.5 * dt * (v1 + v2)

                x_t = x_t.detach()
                x_t.requires_grad_(True)

        log_px0 = self.log_p_base(x_t)
        log_px1 = log_px0 + s
        return log_px1

    def batched_log_prob(self, x, steps=64, batch_size=256, integrator="rk4"):
        out = []
        for i in range(0, x.shape[0], batch_size):
            out.append(self._log_prob_single(x[i:i + batch_size], steps, integrator=integrator))
        return torch.cat(out, dim=0)



def compare_samples_with_truth(model,
                               n_samples=1000,
                               sampling_steps=32,
                               integrator="rk4",
                               save_path="results/sample_compare.png"):
    """
    采样模型 n_samples 个点，与真实 GMM 采样对比散点图。
    左：真实 GMM；右：模型采样
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 真实 GMM 样本
    gmm_dataset = GMMDataset(num_samples=n_samples)
    gmm_samples = gmm_dataset.data[:n_samples]

    # 模型样本
    with torch.no_grad():
        gen = model.sample(n_samples, sampling_steps, integrator=integrator).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(gmm_samples[:, 0], gmm_samples[:, 1], s=4, alpha=0.7)
    axes[0].set_title("True GMM Samples")
    axes[0].set_xlim(-4, 4); axes[0].set_ylim(-4, 4)

    axes[1].scatter(gen[:, 0], gen[:, 1], s=4, alpha=0.7)
    axes[1].set_title(f"Model Samples ({integrator})")
    axes[1].set_xlim(-4, 4); axes[1].set_ylim(-4, 4)

    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Sample comparison figure saved at: {save_path}")


# ------------------------------
# Training
# ------------------------------

def plot_distributions(gaussian_data, target_data, generated_data=None, epoch=None, save_dir="results_logi"):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3 if generated_data is not None else 2, figsize=(15, 5))

    axes[0].scatter(gaussian_data[:, 0], gaussian_data[:, 1], alpha=0.6, s=1)
    axes[0].set_title('Standard Gaussian')
    axes[0].set_xlim(-4, 4); axes[0].set_ylim(-4, 4)

    axes[1].scatter(target_data[:, 0], target_data[:, 1], alpha=0.6, s=1)
    axes[1].set_title('Target GMM')
    axes[1].set_xlim(-4, 4); axes[1].set_ylim(-4, 4)

    if generated_data is not None:
        axes[2].scatter(generated_data[:, 0], generated_data[:, 1], alpha=0.6, s=1)
        axes[2].set_title(f'Generated (Epoch {epoch})')
        axes[2].set_xlim(-4, 4); axes[2].set_ylim(-4, 4)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"gmm_flow_epoch_{epoch}.png"))
    plt.close(fig)


def train_model():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- data ---
    gmm_dataset = GMMDataset(num_samples=10000)
    gaussian_data = np.random.randn(1000, 2).astype(np.float32)  # 只是图里做参照
    gmm_data = gmm_dataset.data[:1000]

    loader = DataLoader(
        gmm_dataset,
        batch_size=256,
        shuffle=True,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        num_workers=0
    )

    # ------- 这里选择 Logistic 作为基分布 -------
    model = FlowMatchingOT(
        dim=2,
        hidden_dim=512,
        num_blocks=4,
        sigma=0.0,
        lr=1e-3,
        device=device,
        base_dist="logistic",   # 切换基分布："logistic" or "gaussian"
        base_loc=0.0,           # 标准 Logistic：mu=0
        base_scale=1.0          # 标准 Logistic：s=1
    )
    model.to(device)

    steps = 50000
    vis_interval = 500
    sample_N = 1000
    sampling_steps = 32
    save_dir = "results_logi"
    os.makedirs(save_dir, exist_ok=True)

    it = iter(loader)
    model.train()
    for step in tqdm(range(1, steps + 1)):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        batch = batch.to(device, non_blocking=True)

        loss = model.forward(batch)
        model.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        model.optimizer.step()

        if step % vis_interval == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
            model.eval()
            with torch.no_grad():
                generated = model.sample(sample_N, sampling_steps).numpy()
                plot_distributions(gaussian_data, gmm_data, generated_data=generated, epoch=step)
            model.train()

    # Save checkpoint after training
    model.eval()
    ckpt_path = os.path.join(save_dir, "fm_model_logi.pt")
    model.save(ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    return model


def visualize_density(model, sampling_steps=64, save_path="results/density.png"):
    grid_size = 100
    x = torch.linspace(-4, 4, grid_size)
    y = torch.linspace(-4, 4, grid_size)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    coords = torch.stack([xx.flatten(), yy.flatten()], dim=1).to(model.device)

    log_probs = model.batched_log_prob(coords,
                                       steps=sampling_steps,
                                       batch_size=128)

    log_probs = log_probs.detach().cpu().numpy().reshape(grid_size, grid_size)
    
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, np.exp(log_probs), levels=100)
    plt.colorbar(label="Density")
    plt.title("Estimated Density from Flow Matching Model")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def load_model(checkpoint_path, device="cpu"):
    model = FlowMatchingOT(dim=2, hidden_dim=512, num_blocks=4, sigma=0.0, lr=1e-3, device=device,
                           base_dist="logistic", base_loc=0.0, base_scale=1.0)
    model.load(checkpoint_path)
    model.eval()
    return model


def compute_true_gmm_density(samples):
    """计算真实GMM分布的 log 密度"""
    means = np.array([[-2, -2], [2, -2], [-2, 2], [2, 2]])
    covs = [np.array([[0.5, 0.1], [0.1, 0.5]]) for _ in range(4)]
    weights = np.array([0.25, 0.25, 0.25, 0.25])

    N = samples.shape[0]
    densities = np.zeros(N)

    for k in range(4):
        diff = samples - means[k]
        cov_inv = np.linalg.inv(covs[k])
        exponent = np.einsum('ij,jk,ik->i', diff, cov_inv, diff)
        normalization = 1.0 / (2 * np.pi * np.sqrt(np.linalg.det(covs[k])))
        component_density = normalization * np.exp(-0.5 * exponent)
        densities += weights[k] * component_density

    return torch.log(torch.tensor(densities + 1e-8, dtype=torch.float32))


def visualize_density_with_truth(model, sampling_steps=64,
                                 save_path="results/density_compare.png",
                                 batch_size=256):
    """
    画 1x3 图：True density（解析）、Estimated density（模型）、误差 |p_est - p_true|
    """
    model.eval()

    # ---- 网格 ----
    grid_size = 100
    x = torch.linspace(-4, 4, grid_size)
    y = torch.linspace(-4, 4, grid_size)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    coords = torch.stack([xx.flatten(), yy.flatten()], dim=1).to(model.device)

    # ---- 模型 log p（分批算，避免爆显存）----
    logp_est = model.batched_log_prob(coords,
                                      steps=sampling_steps,
                                      batch_size=batch_size)
    logp_est = logp_est.detach().cpu().numpy().reshape(grid_size, grid_size)

    # ---- 解析真值 log p（GMM）----
    true_logp = compute_true_gmm_density(coords.detach().cpu().numpy())
    true_logp = true_logp.numpy().reshape(grid_size, grid_size)

    # ---- 误差（概率域的绝对误差；如需 log 域误差可改成 np.abs(logp_est - true_logp)）----
    err = np.abs(np.exp(logp_est) - np.exp(true_logp))

    import matplotlib.pyplot as plt
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # True density
    im0 = axes[0].contourf(xx.numpy(), yy.numpy(), np.exp(true_logp), levels=100)
    axes[0].set_title("True Density (GMM)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Estimated density
    im1 = axes[1].contourf(xx.numpy(), yy.numpy(), np.exp(logp_est), levels=100)
    axes[1].set_title("Estimated Density (Model)")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Error map
    im2 = axes[2].contourf(xx.numpy(), yy.numpy(), err, levels=100)
    axes[2].set_title("Error |p_est - p_true|")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("y")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Density comparison figure saved at: {save_path}")


if __name__ == '__main__':
    trained_model = train_model()
    checkpoint_path = "results_logi/fm_model_logi.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device=device)
    model.model.eval()

    # 用 Heun 做密度估计（反向积分）
    visualize_density_with_truth(model,
                                 sampling_steps=128,
                                 save_path="results_logi/density_compare_logi.png",
                                 batch_size=256)

    # 采样对比散点图（前向 Heun）
    compare_samples_with_truth(model,
                               n_samples=1000,
                               sampling_steps=32,
                               integrator="heun",
                               save_path="results_logi/sample_compare_logi.png")

    print("Density plot saved at: results/density_plot_logi.png")
