# QMC-Flow-Matching：基于流匹配的（准）蒙特卡洛采样

本项目研究在四模态高斯混合 (GMM) 目标分布下，将**流匹配 (Flow Matching)** 模型作为重要性采样的**提议分布**，对比四种 FM 估计量与若干直接从目标 $\pi$ 采样的 RQMC/MC 估计量在二阶矩与一个非线性"复杂被积函数"上的收敛阶。

## 估计量总览

设目标 $\pi = \tfrac{1}{4}\sum_{k=1}^{4}\mathcal{N}(\mu_k, \Sigma_k)$（2D 与 30D 两套参数）。所有估计量都是 $\hat I_N(f) = N^{-1}\sum_i f(x_i)$（FM-ISMC / FM-ISQMC 用 SNIS 权重），$N$ 取 $2^p$，每 $N$ 用 10 次独立 RNG / scramble 重复算 RMSE。

| 估计量 | 提议分布 | 桶分配 | Gaussian 源 | 重要性纠偏 |
|---|---|---|---|---|
| FM-MC | 学到的流 $q_\theta$ | — | $\Phi^{-1}$ on i.i.d. uniform | 否 |
| FM-QMC | $q_\theta$ | — | $\Phi^{-1}$ on scrambled Sobol' | 否 |
| FM-ISMC | $q_\theta$ | — | i.i.d. uniform | SNIS $w_i\propto\pi/q_\theta$ |
| FM-ISQMC | $q_\theta$ | — | scrambled Sobol' | SNIS |
| **Direct-MCbk-RQMC** | $\pi$ 直接采 | i.i.d. MC | $\Phi^{-1}$ on scrambled Sobol' (d 维) | — |
| **Direct-Joint-RQMC** | $\pi$ 直接采 | 一份 $(d{+}1)$ 维 scrambled Sobol' 的第 1 维 | 同一份 Sobol' 的后 d 维 | — |
| **Direct-MC** | $\pi$ 直接采 | i.i.d. MC | i.i.d. 标准高斯 | — |

Direct-MCbk-RQMC、Direct-Joint-RQMC（= M1 Joint $(d{+}1)$-Sobol'）与 Direct-MC 都不依赖 flow matching。前两者用于 2D/30D 主实验；**Direct-MC 作为纯 MC baseline 额外加入 `gmm30d_final/` 的最终实验**。Direct-Joint-RQMC 的数学构造见 [`gmm30d_final/README.md`](gmm30d_final/README.md)。

> **最终大方差实验见 [`gmm30d_final/`](gmm30d_final/)**：在 30D、cov $=5I_{30}$ 的 GMM 上,自包含地给出网络训练 + 采样推理 + 两张收敛率图(complex f / 二阶矩),并在其 README 中完整写出分布、被积函数与**三个**直接估计量的数学定义。

## 主要结果

30D 实验使用 5× 训练长度 (4M 步) 的 ckpt。每条曲线 10 次独立实验的 RMSE，log-log 斜率：

| 被积 | FM-MC | FM-QMC | FM-ISMC | FM-ISQMC | Direct-MCbk | **Direct-Joint** |
|---|---|---|---|---|---|---|
| 2D second moment | -0.49 | -0.58 | -0.48 | **-0.65** | -0.46 | **-1.13** |
| 2D complex $f$ | -0.45 | -0.31 | -0.52 | **-0.76** | -0.59 | **-1.04** |
| 30D second moment | -0.45 | -0.42 | -0.51 | **-0.65** | -0.51 | **-1.02** |
| 30D complex $f$ | -0.49 | -0.47 | -0.51 | **-0.60** | -0.60 | **-0.72** |

复杂被积函数定义为

$$
f(x) = -\Phi\!\left(\tfrac{1}{\sqrt 2}\right) + \Phi\!\left(1 + \tfrac{1}{\sqrt d}\sum_{j=1}^d x_j\right).
$$

闭式真值 $E_\pi[f]$ 在 `target_function.py:exact_expectation_gmm` 实现。

最终 4 张论文图（无 title、统一 legend 顺序、3 条参考斜率）在 `figures/` 下，由根目录 `plot_final.py` 一键生成，数据缓存为 `figures/plot_data_cache.npz`。

## 仓库结构

```
.
├── plot_final.py                  # 4 张最终图 + 数据缓存（已存档结果）
├── figures/
│   ├── final_2d_second.pdf
│   ├── final_2d_complex.pdf
│   ├── final_30d_second.pdf
│   ├── final_30d_complex.pdf
│   └── plot_data_cache.npz        # 48 个数组，6 条曲线 × 4 张图
│
├── gmm/                           # 2D 实验
│   ├── flow_matching_logistic.py  # FM 训练与推理
│   ├── target_function.py         # 复杂 f + 闭式真值 (2D)
│   ├── integrands.py              # 3 个被积函数注册
│   ├── is_logi_all.py             # 4 个 FM 方法 × 3 被积函数（一次性算完）
│   ├── rqmc_baseline.py           # Direct-MCbk-RQMC
│   ├── rqmc_advanced.py           # Direct-Joint-RQMC (M1) + 其他探索方法
│   └── results_logi/
│       ├── fm_model_logi.pt
│       └── all_methods_2d*.{pdf,npz}, rqmc_advanced_2d*.{pdf,npz}
│
└── gmm30d/                        # 30D 实验
    ├── flow_matching_logi.py      # 30D FM 训练与推理
    ├── train_30d.py               # 参数化训练脚本（--steps --ckpt --seed --gpu）
    ├── target_function.py         # 复杂 f + 闭式真值 (30D)
    ├── integrands.py              # 3 个被积函数注册
    ├── is_logi_all.py             # 支持 --ckpt --tag，可评估不同 ckpt
    ├── rqmc_baseline.py           # Direct-MCbk-RQMC
    ├── rqmc_advanced.py           # Direct-Joint-RQMC (M1)
    ├── plot_ckpt_comparison.py    # 1× / 2× / 5× ckpt 对比图
    └── results30d/
        ├── fm_model_30d.pt        # 1× ckpt (800k steps)
        ├── fm_model_30d_2x.pt     # 2× ckpt (1.6M steps)
        ├── fm_model_30d_5x.pt     # 5× ckpt (4M steps，主结果)
        ├── all_methods_30d{,_2x,_5x}*.{pdf,npz}
        ├── rqmc_advanced_30d*.{pdf,npz}
        ├── ckpt_comparison_*.pdf
        └── ckpt_summary.txt

gmm30d_final/                      # 大方差 (cov=5I_30) 最终实验，自包含
├── model.py                       # FM 网络 + 训练步 + MC/QMC 采样 + 反向 log-prob
├── gmm.py                         # 分布 / 数据集 / 解析 log 密度 / 被积函数 + 真值
├── estimators.py                  # Direct-MCbk-RQMC 与 Direct-Joint-RQMC (M1)
├── train.py                       # 训练 q_theta
├── evaluate.py                    # 7 种方法 → final_{complex,second}.pdf
├── README.md                      # 分布 / 被积 / 两个 estimator 的数学定义
└── results/
    ├── fm_model.pt                # 已训练 ckpt (800k)
    └── final_{complex,second}.pdf + *_cache.npz
```

## 复现实验

### 依赖

```bash
# python >= 3.9, pytorch >= 2.0
pip install torch numpy scipy matplotlib tqdm
```

### 训练（30D，可选）

```bash
cd gmm30d
python train_30d.py --steps 4000000 --ckpt fm_model_30d_5x.pt --seed 43 --gpu 0
```

throughput 在 RTX 4090 上约 80 steps/s（5× = 4M 步约 14 小时）。

### 评估 FM 与 Direct-MCbk-RQMC（4 FM 方法 + Direct-MCbk-RQMC 同时跑）

```bash
cd gmm  &&  python is_logi_all.py                                                  # 2D, p=1..13
cd gmm30d && python is_logi_all.py --ckpt results30d/fm_model_30d_5x.pt --tag _5x  # 30D, p=1..14
```

### 评估 Direct-Joint-RQMC

```bash
cd gmm    && python rqmc_advanced.py
cd gmm30d && python rqmc_advanced.py
```

### 出最终 4 张图（基于上面跑出的 npz）

```bash
python plot_final.py
```

要在不重新跑实验的情况下调整图风格，直接读 `figures/plot_data_cache.npz`：

```python
import numpy as np
z = np.load("figures/plot_data_cache.npz")
N   = z["30d_complex_FM_ISQMC_N"]
rmse = z["30d_complex_FM_ISQMC_rmse"]
# key 格式: {setting}_{integrand}_{series}_{N|rmse}
# setting    ∈ {2d, 30d}
# integrand  ∈ {second, complex}
# series     ∈ {FM_MC, FM_QMC, FM_ISMC, FM_ISQMC, Direct_MCbk_RQMC, Direct_Joint_RQMC}
```

## 关键设计选择

* **基分布**：Logistic（逐维独立），通过 logit 变换从 Sobol' 起点构造 RQMC 初值。
* **流匹配**：OT 路径 $x_t = (1-t)x_0 + tz$，监督 $u = z - x_0$。
* **数值积分器**：Heun (二阶)，前向 64 步，反向 (log-prob) 64 步、batch 256。
* **重要性纠偏**：自归一化 (SNIS) $w_i \propto \pi(x_i)/q_\theta(x_i)$。

## 主要发现

1. **Direct-Joint-RQMC (M1) 在所有四个 hard target 上都接近 $O(N^{-1})$**，远超 FM-ISQMC 在 30D complex $f$ 上的 $-0.60$。M1 用单条 $(d{+}1)$ 维 scrambled Sobol' 把"抽桶 + 抽高斯"建成一个联合低差异积分，消除了"两条独立流"baseline 的 cross-term plateau（数学定义见 [`gmm30d_final/README.md`](gmm30d_final/README.md)）。
2. **SNIS 重要性纠偏对长训练是必要的**。30D 上不带 IS 的 FM-QMC 在 5× ckpt 下反而比 2× 差（撞到 ckpt 偏差地板），而 FM-ISQMC 单调改善。

## 许可证

见 [LICENSE](LICENSE)。
