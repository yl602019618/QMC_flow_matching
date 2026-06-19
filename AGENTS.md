# AGENTS.md

> 本文件面向 AI 编程助手。读者应被假设对项目一无所知。下文基于仓库实际内容编写，不做推测。

## 1. 项目概述

本仓库实现论文《QMC-Flow-Matching》的数值实验：在**四模态高斯混合模型 (GMM)** 目标分布上，将**流匹配 (Flow Matching, FM)** 模型作为重要性采样的**提议分布 (proposal)**，比较四类 FM 估计量与三类直接 RQMC/MC 估计量的收敛阶。

核心比较对象：

| 估计量 | 提议分布 | 基分布 / 源 | 是否重要性纠偏 |
|---|---|---|---|
| FM-MC | 学到的流 $q_\\theta$ | Logistic / i.i.d. uniform | 否 |
| FM-QMC | 学到的流 $q_\\theta$ | Logistic / scrambled Sobol' | 否 |
| FM-ISMC | 学到的流 $q_\\theta$ | Logistic / i.i.d. uniform | SNIS |
| FM-ISQMC | 学到的流 $q_\\theta$ | Logistic / scrambled Sobol' | SNIS |
| Direct-MCbk-RQMC | 直接从 $\\pi$ 采样 | MC 分桶 + RQMC 高斯 | — |
| Direct-Joint-RQMC (M1) | 直接从 $\\pi$ 采样 | 单条 $(d+1)$ 维 scrambled Sobol' | — |
| Direct-MC | 直接从 $\\pi$ 采样 | MC 分桶 + MC 高斯（纯 MC baseline） | — |

实验覆盖 2D GMM 与 30D GMM 两套参数，另有一个自包含的 `gmm30d_final/` 子模块用于 30D 大方差 (cov = $5I_{30}$) 的最终实验。

许可证：Apache License 2.0（见 `LICENSE`）。

## 2. 目录结构

```
QMC_flow_matching/
├── README.md                 # 项目总览、复现命令、结果表
├── LICENSE                   # Apache 2.0
├── .gitignore                # Python 临时文件
├── plot_final.py             # 生成 4 张最终论文图 + 数据缓存
├── figures/                  # 最终 4 张 PDF + plot_data_cache.npz
│
├── gmm/                      # 2D 实验
│   ├── flow_matching_logistic.py   # FM 训练、采样、密度估计
│   ├── target_function.py          # complex f 与闭式真值
│   ├── integrands.py               # 三个被积函数注册表
│   ├── is_logi_all.py              # 4 FM 方法 + Direct-MCbk-RQMC 统一驱动
│   ├── rqmc_baseline.py            # Direct-MCbk-RQMC + Direct-RQMCbk-RQMC
│   ├── rqmc_advanced.py            # Direct-Joint-RQMC (M1) 与其他高级 RQMC
│   └── results_logi/               # ckpt、npz、pdf
│
├── gmm30d/                   # 30D 实验（cov = 0.5 I_30）
│   ├── flow_matching_logi.py       # 30D FM 训练与推理
│   ├── train_30d.py                # 参数化训练脚本
│   ├── target_function.py
│   ├── integrands.py
│   ├── is_logi_all.py              # 支持 --ckpt --tag
│   ├── rqmc_baseline.py
│   ├── rqmc_advanced.py
│   ├── plot_ckpt_comparison.py     # 1x/2x/5x ckpt 对比图
│   └── results30d/                 # 三个 ckpt、npz、pdf
│
└── gmm30d_final/             # 30D 大方差 (cov = 5 I_30) 最终实验，自包含
    ├── README.md                     # 分布/被积函数/estimator 数学定义
    ├── model.py                      # FM 网络 + 训练步 + 采样 + log-prob
    ├── gmm.py                        # 目标分布、数据集、解析 log 密度、被积函数
    ├── estimators.py                 # Direct-MCbk-RQMC、Direct-Joint-RQMC 与 Direct-MC
    ├── train.py                      # 训练 q_theta
    ├── evaluate.py                   # 跑 7 种方法并出图
    └── results/                      # ckpt、pdf、缓存 npz
```

## 3. 技术栈

- **语言**：Python 3.9+
- **深度学习框架**：PyTorch 2.0+
- **科学计算**：NumPy、SciPy（`scipy.stats.qmc` 用于 Sobol'/MultivariateNormalQMC；`scipy.stats.norm` 用于 $\\Phi^{-1}$）
- **绘图**：matplotlib
- **进度条**：tqdm
- **无构建系统**：没有 `pyproject.toml`、`setup.py`、`requirements.txt`、`Makefile` 或 `package.json`。依赖在 `README.md` 与 `gmm30d_final/README.md` 中说明。

## 4. 环境安装

```bash
pip install torch numpy scipy matplotlib tqdm
```

建议在 GPU 环境运行训练脚本；纯评估与出图可在 CPU 运行，但 30D log-prob 反向积分较慢。

## 5. 代码组织约定

### 5.1 每个子目录自包含

- `gmm/`、`gmm30d/`、`gmm30d_final/` 各自有独立的模型、数据集、被积函数与估计量实现，子目录之间不互相 import。
- `plot_final.py` 在根目录，读取 `gmm/results_logi/` 与 `gmm30d/results30d/` 的 `.npz` 结果生成论文图。

### 5.2 共同模式

- **FM 网络**：`Network` 由 `FourierEmbedding`（时间编码）+ `ResidualBlock` MLP 残差块 + 输出投影组成。
- **FlowMatchingOT**：封装训练 loss、前向采样 (`sample` / `sample_qmc`)、反向 log-prob (`batched_log_prob`)。
- **基分布**：默认使用 **Logistic**（通过 logit 从 Sobol' 构造 QMC 初值）；也可切换为 Gaussian。
- **路径**：线性 OT 路径 $x_t = (1-t)x_0 + tz$，监督速度 $u = z - x_0$。
- **积分器**：前向采样默认 **Heun**，反向 log-prob 默认 **RK4** + 精确散度 (`exact`) 或 Hutchinson 估计 (`hutch`)。
- **重要性权重**：自归一化 SNIS，$w_i \\propto \\exp(\\log \\pi(x_i) - \\log q_\\theta(x_i))$。

### 5.3 命名约定

- 类名使用 `PascalCase`（如 `FlowMatchingOT`、`GMMDataset`）。
- 函数与变量使用 `snake_case`。
- 常量常以大写形式放在模块顶部（如 `DIM`、`K`、`VAR`）。
- 注释多为中英混合；关键数学实现（复杂被积函数、闭式真值、estimator）有详细中文注释。

### 5.4 输入 / 输出

- `.pt`：仅保存 `model.state_dict()`，字典键为 `"model"`。
- `.npz`：保存 RMSE 曲线、样本量、ESS 等数组。
- `.pdf`：matplotlib 生成，dpi 通常为 200。

## 6. 复现命令

### 6.1 2D 实验

```bash
cd gmm
python flow_matching_logistic.py          # 训练 2D FM ckpt（约 5 万步）
python is_logi_all.py                     # 评估 4 FM 方法 + Direct-MCbk-RQMC
python rqmc_advanced.py                   # 评估 Direct-Joint-RQMC (M1) 等高级方法
```

### 6.2 30D 实验（cov = 0.5 I_30）

```bash
cd gmm30d
python train_30d.py --steps 4000000 --ckpt fm_model_30d_5x.pt --seed 43 --gpu 0
python is_logi_all.py --ckpt results30d/fm_model_30d_5x.pt --tag _5x
python rqmc_advanced.py
python plot_ckpt_comparison.py            # 可选：1x/2x/5x 训练长度对比
```

### 6.3 30D 大方差最终实验（cov = 5 I_30）

```bash
cd gmm30d_final
python train.py --steps 800000 --ckpt results/fm_model.pt --gpu 0
python evaluate.py --ckpt results/fm_model.pt
```

### 6.4 生成最终 4 张论文图

```bash
python plot_final.py
```

输出到 `figures/`：`final_2d_second.pdf`、`final_2d_complex.pdf`、`final_30d_second.pdf`、`final_30d_complex.pdf`，以及 `plot_data_cache.npz`。

## 7. 测试策略

- **没有单元测试框架**（无 `pytest`、`unittest`、CI 配置）。
- 正确性主要依靠：
  1. **解析真值**：`target_function.py` / `gmm.py` 提供复杂被积函数与二阶矩的闭式期望。
  2. **数值自检脚本**：如 `target_function.py:__main__` 会用 $N=2{,}000{,}000$ 的蒙特卡洛验证解析真值。
  3. **可视化**：`flow_matching_logistic.py` 训练结束后会绘制真实样本与生成样本散点图、密度对比图。
- 如果你新增 estimator 或被积函数，建议：
  - 在对应 `integrands.py` 注册并给出 `true_value()`。
  - 用小的 `N`（如 $N=2^4$）和少量重复快速跑一次，对比解析真值检查偏差。
  - 对高维模型，先确认 `batched_log_prob` 在 1-2 个已知点上的 log 密度合理。

## 8. 性能与资源提示

- 训练 30D 模型 800k 步在 RTX 4090 上约 80 steps/s，耗时约 2.8 小时；4M 步约 14 小时。
- 评估脚本会重复 10 次独立实验，对每个 $N=2^p$ 都调用 `batched_log_prob`，是主要耗时来源。
- 内存：log-prob 默认 batch_size=256；若显存不足可降低 `logprob_batch_size`。
- 随机种子：训练与评估各自有 `set_seed`；Sobol' scramble 的 seed 由 `_seed()` 函数按 $p$ 与实验编号生成。

## 9. 修改时需注意

- **不要跨目录修改 imports**：`gmm/`、`gmm30d/`、`gmm30d_final/` 彼此独立。
- **保持 `.npz` key 一致**：`plot_final.py` 依赖固定的 key 格式读取结果。若修改 `is_logi_all.py` 的输出 key，需要同步更新 `plot_final.py`。
- **ckpt 维度匹配**：加载 `.pt` 时，必须保证 `FlowMatchingOT` 的 `dim`、`hidden_dim`、`num_blocks`、`base_dist` 与训练时一致。
- **基分布**：当前论文结果全部使用 Logistic 基分布；若改为 Gaussian，`sample_qmc` 路径会走 `MultivariateNormalQMC`，需要重新训练并重新评估，不能直接复用 Logistic ckpt。

## 10. 安全与合规

- 本项目不监听网络端口、不读取环境敏感变量、不执行外部命令。
- 运行脚本只在本仓库目录读写 `.pt`、`.npz`、`.pdf`、`.png` 文件。
- 训练脚本接受 `--gpu` 参数并通过 `CUDA_VISIBLE_DEVICES` 选择 GPU，属于常规做法。
- 不收集用户数据。

## 11. 常见入口速查

| 想做的事 | 入口脚本 |
|---|---|
| 看项目总览 | `README.md`、`gmm30d_final/README.md` |
| 训练 2D FM | `gmm/flow_matching_logistic.py` |
| 训练 30D FM（参数化） | `gmm30d/train_30d.py` |
| 训练 30D 大方差 FM | `gmm30d_final/train.py` |
| 评估 2D 全部方法 | `gmm/is_logi_all.py` + `gmm/rqmc_advanced.py` |
| 评估 30D 全部方法 | `gmm30d/is_logi_all.py` + `gmm30d/rqmc_advanced.py` |
| 评估 30D 大方差 | `gmm30d_final/evaluate.py` |
| 生成最终论文图 | `plot_final.py` |
