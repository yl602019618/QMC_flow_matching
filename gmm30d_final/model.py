"""
Flow-Matching (Optimal-Transport) model for the 30D GMM experiment.

Self-contained: network architecture + FlowMatchingOT (training step, MC/QMC
forward sampling, and reverse-time exact-divergence log-prob for SNIS). The
target distribution / dataset / integrands / estimators live in gmm.py and
estimators.py.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import qmc


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
