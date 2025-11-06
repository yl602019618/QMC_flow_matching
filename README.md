# QMC-Flow-Matching：基于流匹配的（准）蒙特卡洛采样

本项目实现了一种将 **流匹配 (Flow Matching)** 模型作为**提议分布 (Proposal Distribution)** 的新方法，用于高效估计多维目标分布的统计数据。我们对比了该方法在标准**蒙特卡洛 (MC)**、**准蒙特卡洛 (QMC)** 以及**自归一化重要性采样 (SNIS)** 框架下的表现。

## 核心思想

在复杂的分布上进行采样和积分是许多科学计算领域的关键挑战。传统 MCMC 方法可能收敛缓慢，而标准重要性采样的效率则严重依赖于提议分布的质量。

本项目探索了利用**流匹配**（一种新兴的生成模型）来学习一个可微的、从简单基分布到复杂目标分布的流。这个学习到的流可以作为一个高质量的提议分布，显著提升 QMC 和 SNIS-QMC 方法的收敛速度和估计精度。

## ✨ 主要特性

* **先进的提议分布**：使用基于**最优传输 (OT)** 路径的流匹配模型 (`x_t = (1-t)x_0 + t*z`) 作为提议分布，通过监督速度场 (`u = z - x_0`) 进行训练。
* **灵活的基分布**：支持两种可配置的基分布：
    * `gaussian`：标准高斯分布。
    * `logistic`：多维独立 Logistic 分布（支持逐维设置 `loc` 和 `scale`）。
* **QMC 优化**：为不同的基分布提供了专属的 QMC 起点（低差异序列）：
    * **Gaussian**：使用 `torch.quasirandom.MultivariateNormalQMC`。
    * **Logistic**：使用 Sobol 序列结合逆 CDF (Logit) 变换生成。
* **高精度积分器**：内置多种数值积分器，用于前向采样（生成样本）和反向似然估计（计算概率）：
    * Euler
    * Heun (二阶)
    * RK4 (四阶)
* **综合性能评测**：提供了丰富的评估指标和可视化工具，用于全面分析模型性能。

## 📦 依赖与安装

本项目基于 Python 3.9+ 和 PyTorch 2.0+。

1.  克隆本仓库：
    ```bash
    git clone [https://github.com/yl602019618/QMC_flow_matching.git](https://github.com/yl602019618/QMC_flow_matching.git)
    cd QMC_flow_matching
    ```

2.  安装所需依赖 (建议您在仓库根目录创建 `requirements.txt` 文件)：
    ```bash
    pip install -r requirements.txt 
    ```

    **`requirements.txt` 示例内容：**
    ```
    # python >= 3.9
    pytorch >= 2.0
    numpy
    scipy
    matplotlib
    tqdm
    ```

## 🧪 实验设置

代码中包含了三个基准实验，均共享相同的 `FlowMatchingOT` 核心实现，仅目标分布的定义和数据生成不同：

1.  **`gmm`：二维高斯混合模型 (GMM2D)**
    * **目标**：一个固定的四模态 GMM 分布。
    * **评测**：2D 散点图、密度等高线图、估计误差热力图。

2.  **`banana`：二维香蕉分布 (Banana-2D)**
    * **目标**：一个经典的非高斯香蕉状基准分布。
    * **评测**：2D 散点图、等高线对比、估计误差热力图。

3.  **`gmm30d`：三十维高斯混合模型 (GMM30D)**
    * **目标**：一个高维（30D）多峰 GMM 分布。
    * **评测**：以数值指标为主，如 MSE、SNIS 的有效样本大小 (ESS)、Log-Likelihood 稳定性。

## ⚙️ 关键参数配置

在运行实验时，您可以调整以下关键参数：

* **基分布**
    * `base_dist`：`"logistic"` 或 `"gaussian"`。
    * `base_loc`, `base_scale`：基分布的位置和尺度参数（可为标量或 `dim` 维向量）。
* **采样与积分**
    * `integrator`：`"euler"`, `"heun"` 或 `"rk4"`。
    * `sampling_steps`：前向采样（生成）时的积分步数。
* **似然估计**
    * `logprob_steps`：反向积分（计算似然）时的步数。
    * `logprob_batch_size`：计算似然时的批处理大小。

## 📊 评估指标

本项目通过以下指标评估不同采样方法的性能：

* **RMSE vs N (log-log 图)**：比较 MC / QMC / SNIS-MC / SNIS-QMC 四种方法的均方根误差 (RMSE) 随样本数量 (N) 增加的收敛斜率。
* **ESS vs N**：评估 SNIS 方法的有效样本大小 (ESS)，用于衡量重要性权重的退化程度。
* **2D 可视化 (适用于 2D 实验)**：
    * 真实密度 vs. 模型估计密度。
    * 采样点分布散点图。
    * 逐点的误差热力图。