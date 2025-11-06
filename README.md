# QMC_flow_matching

用 **Flow Matching** 学到的可微流当作**提议分布**，在多种目标分布上做 **Monte Carlo (MC)**、**Quasi–Monte Carlo (QMC)** 与 **Self-Normalized Importance Sampling (SNIS)** 的对比评测。支持**可配置基分布**（Gaussian 或 Logistic），并提供对 Logistic 的 **QMC 起点**（Sobol + 逆 CDF）。

---

## ✨ 特性

- **Flow Matching (OT 路径)**：线性路径 \(x_t=(1-t)x_0+t z\)，监督速度 \(u=z-x_0\)。
- **基分布可切换**：`gaussian` / `logistic`（多维独立，支持 per-dim `loc/scale`）。
- **数值积分器**：Euler / Heun / RK4（前向采样 & 反向似然）。
- **QMC 起点**  
  - Gaussian：`MultivariateNormalQMC`  
  - Logistic：Sobol 低差异序列 + 逐维逆 CDF（logit）
- **评测输出**：MC vs QMC 收敛曲线、SNIS（MC/QMC）与 ESS、2D 可视化与误差热力图。

---

## 📦 依赖

```bash
python >= 3.9
pytorch >= 2.0
numpy
matplotlib
tqdm
scipy
```

## 🧪 三个实验

1. **GMM2D**

   * 目标：四模态 GMM（固定均值/协方差/权重）。
   * 可视化：散点对比、密度与误差图。

2. **Banana（2D）**

   * 目标：banana-shaped 非高斯基准分布。
   * 可视化：散点/等高线与误差图。

3. **GMM30D**

   * 目标：30 维多峰 GMM。
   * 评测：以数值指标为主（MSE、SNIS-ESS、loglik 稳定性）；可选降维可视化。

> 三个实验共享同一 `FlowMatchingOT` 实现，仅数据生成与真实密度不同。

---

## ⚙️ 常用参数

* **基分布**：`base_dist in {"logistic","gaussian"}`
  `base_loc`, `base_scale` 支持标量或长度为 `dim` 的向量。
* **采样**：`integrator=("euler"|"heun"|"rk4")`, `sampling_steps`
* **似然**：`logprob_steps`, `logprob_batch_size`
* **QMC**：Gaussian 用 `MultivariateNormalQMC`；Logistic 用 **Sobol + logit** 映射。

---

## 📊 指标

* **RMSE vs N（log-log）**：MC / QMC / SNIS-MC / SNIS-QMC 收敛斜率。
* **ESS vs N**（SNIS）：权重退化度量。
* **2D 可视化**：真实密度 vs 模型估计、误差热力图、采样散点图。

---