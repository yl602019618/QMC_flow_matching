import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import qmc

# ------------------------------
# Utils
# ------------------------------

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------
# Dataset (30D GMM with 4 components)
# ------------------------------

class GMMDataset30D(Dataset):
    """
    30 维、4 分量的 GMM：
    - 四个均值：只在前两维取 (+/-2, +/-2)，其余 28 维为 0
    - 协方差：对角阵 0.5 * I_30
    - 权重：均等 0.25
    """
    def __init__(self, num_samples=20000, dim=30):
        assert dim == 30, "This dataset is fixed to 30D for this experiment."
        self.dim = dim
        self.means = np.zeros((4, dim), dtype=np.float32)
        self.means[0, :2] = [-2.0, -2.0]
        self.means[1, :2] = [ 2.0, -2.0]
        self.means[2, :2] = [-2.0,  2.0]
        self.means[3, :2] = [ 2.0,  2.0]

        self.covs = [np.eye(dim, dtype=np.float32) * 0.5 for _ in range(4)]
        self.weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

        self.data = self._generate_samples(num_samples).astype(np.float32)

    def _generate_samples(self, num_samples):
        samples = []
        for _ in tqdm(range(num_samples)):
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
        # x: (B,) -> (B,1)
        x = x.unsqueeze(-1)
        freqs = self.freqs.view(1, -1).expand(x.shape[0], -1)
        x = x * freqs
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


class Network(nn.Module):
    def __init__(self, dim=30, hidden_dim=512, num_blocks=4):  # 改：dim=30
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
        dim=30,                 # 改：默认 30 维
        hidden_dim=512,
        num_blocks=4,
        sigma=0.0,
        lr=1e-3,
        device="cpu",
        base_dist="logistic",   # "logistic" or "gaussian"
        base_loc=0.0,
        base_scale=1.0
    ):
        super().__init__()
        self.model = Network(dim, hidden_dim, num_blocks).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.sigma = sigma
        self.dim = dim
        self.device = device

        assert base_dist in ("logistic", "gaussian")
        self.base_dist = base_dist

        # 将 loc/scale 归整成 (1, dim) 便于广播
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

    # 从基分布采样 x0
    def sample_from_base(self, shape):
        """
        shape: (B, dim)
        """
        if self.base_dist == "gaussian":
            return torch.randn(shape, device=self.device)
        else:
            eps = 1e-6
            U = torch.rand(shape, device=self.device).clamp_(eps, 1.0 - eps)
            return self.base_loc + self.base_scale * torch.log(U / (1.0 - U))

    # 条件路径与速度（线性 OT 路径）
    def conditional_prob_path(self, t, z, x0):
        t = t.unsqueeze(1)              # (B,) -> (B,1)
        mean = t * z + (1 - t) * x0
        if self.sigma > 0:
            mean = mean + torch.randn_like(z) * self.sigma
        return mean

    def conditional_vel_field(self, t, z, x0):
        return z - x0

    # 训练一步的 loss
    def forward(self, z):
        t = torch.rand((z.shape[0],), device=z.device, dtype=z.dtype)
        x0 = self.sample_from_base(z.shape)     # 基分布采样
        x = self.conditional_prob_path(t, z, x0)
        u = self.conditional_vel_field(t, z, x0)
        v = self.model(x, t)
        return (u - v).pow(2).mean()

    # 前向采样（t: 0 -> 1）
    def sample(self, N, sampling_steps=32, integrator="heun"):
        self.model.eval()
        with torch.no_grad():
            x = self.sample_from_base((N, self.dim))  # 起点：基分布
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

    # QMC 起点
    def sample_qmc(self, N, sampling_steps=32, exp=0, integrator="heun"):
        """
        QMC 起点：
        - gaussian: MultivariateNormalQMC
        - logistic: Sobol in (0,1)^d -> logit -> (mu, s)
        """
        self.model.eval()
        with torch.no_grad():
            if self.base_dist == "gaussian":
                sampler = qmc.MultivariateNormalQMC(
                    mean=[0]*self.dim,
                    cov=np.eye(self.dim),
                    seed=42 + exp
                )
                u = sampler.random(N)  # (N,dim)
                x = torch.tensor(u, dtype=torch.float32, device=self.device)
            else:
                sobol = qmc.Sobol(d=self.dim, scramble=True, seed=42 + exp)
                u = sobol.random(N)    # (N,dim) in [0,1)
                x = torch.tensor(u, dtype=torch.float32, device=self.device)
                eps = 1e-6
                x = x.clamp(eps, 1.0 - eps)
                x = self.base_loc + self.base_scale * torch.log(x / (1.0 - x))

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

    # ------------------------------
    # 密度估计（保持与 10D 版一致；对维度不敏感）
    # ------------------------------

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
            v_dot = (v * eps).sum()
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
        B = x.size(0)
        device = x.device
        dtype = x.dtype

        def f(x_, t_):
            t_vec = torch.full((B,), t_, device=device, dtype=dtype)
            v_ = self.model(x_, t_vec)
            d_ = self.divergence(v_, x_, t_, method=div_method, num_probe=num_probe)
            return v_, d_

        v1, d1 = f(x, t)
        x2 = (x - 0.5 * dt * v1).detach(); x2.requires_grad_(True)
        v2, d2 = f(x2, t - 0.5 * dt)
        x3 = (x - 0.5 * dt * v2).detach(); x3.requires_grad_(True)
        v3, d3 = f(x3, t - 0.5 * dt)
        x4 = (x - dt * v3).detach(); x4.requires_grad_(True)
        v4, d4 = f(x4, t - dt)

        x_new = x - (dt / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
        ds    = -(dt / 6.0) * (d1 + 2 * d2 + 2 * d3 + d4)
        return x_new, ds

    def log_p_base(self, x):
        if self.base_dist == "gaussian":
            return -0.5 * (x ** 2).sum(dim=1) - 0.5 * x.shape[1] * np.log(2 * np.pi)
        else:
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
        assert integrator in ("euler", "heun", "rk4")
        assert div_method in ("exact", "hutch")

        self.model.eval()
        device = self.device

        x_t = x1.clone().detach().to(device)
        x_t.requires_grad_(True)

        t_start = 1.0 - float(t_eps)
        t_end   = 0.0 + float(t_eps)
        ts = torch.linspace(t_start, t_end, steps, device=device)
        s = torch.zeros(x_t.shape[0], device=device, dtype=x_t.dtype)

        if integrator == "rk4":
            for k in range(steps - 1):
                t = ts[k].item()
                t_next = ts[k + 1].item()
                dt = t - t_next
                x_t, ds = self._rk4_step(x_t, t, dt, div_method=div_method, num_probe=num_probe)
                s = s + ds
                x_t = x_t.detach()
                x_t.requires_grad_(True)
        else:
            for k in range(steps - 1):
                t = ts[k]
                t_next = ts[k + 1]
                dt = (t - t_next)

                t_vec = torch.full((x_t.size(0),), t.item(), device=device, dtype=x_t.dtype)
                v1 = self.model(x_t, t_vec)
                div1 = self.divergence(v1, x_t, t.item(), method=div_method, num_probe=num_probe)

                if integrator == "euler":
                    s = s - dt * div1
                    x_t = x_t - dt * v1
                elif integrator == "heun":
                    x_pred = (x_t - dt * v1).detach()
                    x_pred.requires_grad_(True)
                    t_next_vec = torch.full((x_t.size(0),), t_next.item(), device=device, dtype=x_t.dtype)
                    v2 = self.model(x_pred, t_next_vec)
                    div2 = self.divergence(v2, x_pred, t_next.item(), method=div_method, num_probe=num_probe)
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


# ------------------------------
# Visualization (Samples only, PCA)
# ------------------------------

def pca_fit_transform(X, k=2):
    X_np = X if isinstance(X, np.ndarray) else X.detach().cpu().numpy()
    mean = X_np.mean(axis=0, keepdims=True)
    Xc = X_np - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:k]
    Z = Xc @ comps.T
    return Z, (mean, comps)

def pca_transform_with_basis(X, basis):
    mean, comps = basis
    X_np = X if isinstance(X, np.ndarray) else X.detach().cpu().numpy()
    return (X_np - mean) @ comps.T

def visualize_samples_pca(true_samples, gen_samples, save_path="results30d/sample_pca.png", title_suffix=""):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Z_true, basis = pca_fit_transform(true_samples, k=2)
    Z_gen = pca_transform_with_basis(gen_samples, basis)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(Z_true[:, 0], Z_true[:, 1], s=4, alpha=0.6)
    axes[0].set_title(f"True GMM (PCA){title_suffix}")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")

    axes[1].scatter(Z_gen[:, 0], Z_gen[:, 1], s=4, alpha=0.6)
    axes[1].set_title(f"Generated (PCA){title_suffix}")
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"PCA sample comparison saved at: {save_path}")


# ------------------------------
# Training
# ------------------------------

def train_model_30d():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gmm_dataset = GMMDataset30D(num_samples=1000000, dim=30)  # 改：30D 数据集
    loader = DataLoader(
        gmm_dataset,
        batch_size=256,
        shuffle=True,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        num_workers=0
    )

    model = FlowMatchingOT(
        dim=30,                 # 改：30D 模型
        hidden_dim=512,
        num_blocks=8,           # 你原来 10D 用了 6 层；30D 维度更高，6 层也合适
        sigma=0.0,
        lr=1e-3,
        device=device,
        base_dist="logistic",
        base_loc=0.0,
        base_scale=1.0
    )
    model.to(device)

    steps = 800000
    vis_interval = 1000
    sample_N = 2000
    sampling_steps = 32
    save_dir = "results30d"     # 改：输出目录
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
                idx = np.random.choice(len(gmm_dataset), size=sample_N, replace=False)
                true_samples = gmm_dataset.data[idx]
                generated = model.sample(sample_N, sampling_steps, integrator="heun").numpy()
                visualize_samples_pca(
                    true_samples=true_samples,
                    gen_samples=generated,
                    save_path=os.path.join(save_dir, f"sample_pca_step_{step}.png"),
                    title_suffix=f" (step {step})"
                )
            model.train()

    model.eval()
    ckpt_path = os.path.join(save_dir, "fm_model_30d.pt")   # 改：checkpoint 名称
    torch.save({"model": model.model.state_dict()}, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    return model, gmm_dataset


def load_model_30d(checkpoint_path, device="cpu"):
    model = FlowMatchingOT(dim=30, hidden_dim=512, num_blocks=8, sigma=0.0, lr=1e-3,
                           device=device, base_dist="logistic", base_loc=0.0, base_scale=1.0)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def compare_samples_after_training(checkpoint_path="results30d/fm_model_30d.pt",
                                   save_dir="results30d",
                                   sample_N=5000,
                                   sampling_steps=32):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_30d(checkpoint_path, device=device)

    gmm_dataset = GMMDataset30D(num_samples=100000, dim=30)   # 改：30D 数据集
    idx = np.random.choice(len(gmm_dataset), size=sample_N, replace=False)
    true_samples = gmm_dataset.data[idx]

    with torch.no_grad():
        gen = model.sample(sample_N, sampling_steps, integrator="heun").numpy()

    visualize_samples_pca(
        true_samples=true_samples,
        gen_samples=gen,
        save_path=os.path.join(save_dir, "sample_pca_final.png"),
        title_suffix=" (final)"
    )


if __name__ == "__main__":
    trained_model, dataset = train_model_30d()
    compare_samples_after_training(
        checkpoint_path="results30d/fm_model_30d.pt",
        save_dir="results30d",
        sample_N=5000,
        sampling_steps=32
    )
