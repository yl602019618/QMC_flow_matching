# gmm30d_final — 30D GMM 下 Flow-Matching 与 RQMC 估计量对比

本模块在一个 **30 维、大方差高斯混合** 目标上,对比基于流匹配 (Flow Matching) 的
四种估计量与两种纯 RQMC 直接估计量,产出两张收敛率图(complex f 与二阶矩)。

```
gmm30d_final/
├── model.py        # Flow-Matching (OT) 网络 + 训练步 + MC/QMC 采样 + 反向 log-prob
├── gmm.py          # 目标分布、数据集、解析 log 密度、两个被积函数及其闭式真值
├── estimators.py   # 三个直接估计量 (Direct-MCbk-RQMC, Direct-Joint-RQMC, Direct-MC)
├── train.py        # 训练 flow 提议分布 q_theta
├── evaluate.py     # 跑全部 7 种方法,产出 results/final_{complex,second}.pdf
└── results/
    ├── fm_model.pt          # 已训练 ckpt (800k 步)
    ├── final_complex.pdf    # complex f 的 RMSE-vs-N
    ├── final_second.pdf     # 二阶矩的 RMSE-vs-N
    └── final_{complex,second}_cache.npz   # 画图数据缓存
```

---

## 1. 采样分布(目标 π)

四分量高斯混合,维度 $d=30$,**等权重** $w_k=\tfrac14$:

$$
\pi(x)=\frac14\sum_{k=1}^{4}\mathcal N(x\mid \mu_k,\ \Sigma),
\qquad \Sigma=\sigma^2 I_{30},\quad \sigma^2 = 5.0 .
$$

均值只在前两维非零,其余 28 维为 0:

$$
\mu_1=(-2,-2,0,\dots),\quad \mu_2=(2,-2,0,\dots),\quad
\mu_3=(-2,2,0,\dots),\quad \mu_4=(2,2,0,\dots).
$$

采用**各向同性大方差** $\sigma^2=5.0$(每维方差为 5)。沿"全 1 方向"
$v=\mathbf 1/\sqrt d$ 的方差为 $v^\top\Sigma v=\sigma^2=5.0$ —— 这个量决定了下面
complex f 的难度(大方差把有界的 $\Phi$ 推向饱和)。

层级采样形式:$k\sim\mathrm{Uniform}\{1,\dots,4\}$,$z\sim\mathcal N(0,I_{30})$,
$x=\mu_k+\sqrt{\sigma^2}\,z$。

---

## 2. 被积函数与闭式真值

估计 $\mathbb E_\pi[f]$。$\Phi$ 为标准正态 CDF,$d=30$。

### (a) complex f(标量)

$$
f(x)=-\Phi\!\Big(\tfrac{1}{\sqrt2}\Big)+\Phi\!\Big(1+\tfrac{1}{\sqrt d}\sum_{j=1}^d x_j\Big).
$$

记 $a_k=\tfrac{\mathbf 1^\top\mu_k}{\sqrt d}$,$b_k^2=\tfrac{\mathbf 1^\top\Sigma_k\mathbf 1}{d}=\sigma^2$,
利用 $\mathbb E_{X\sim\mathcal N(\mu,\Sigma)}[\Phi(1+\mathbf c^\top X)]=\Phi\!\big(\tfrac{1+a}{\sqrt{1+b^2}}\big)$:

$$
\mathbb E_\pi[f]=-\Phi\!\Big(\tfrac{1}{\sqrt2}\Big)
+\sum_{k=1}^4 w_k\,\Phi\!\Big(\frac{1+a_k}{\sqrt{1+\sigma^2}}\Big)
\;=\; -0.105057 .
$$

### (b) 二阶矩(逐坐标向量)

$$
f(x)=x^2,\qquad
\mathbb E_\pi[x_j^2]=\sigma^2+\sum_{k=1}^4 w_k\,\mu_{k,j}^2
=\begin{cases}\sigma^2+4=9.0 & j=1,2\\[2pt] \sigma^2=5.0 & j\ge 3.\end{cases}
$$

误差用向量 RMSE:$\sqrt{\|\,\overline{\hat I}-\mathbb E_\pi[f]\,\|^2+\tfrac1R\sum_r\|\hat I^{(r)}-\overline{\hat I}\|^2}$。

---

## 3. 三个直接估计量的数学定义

三者都**不使用流、不使用重要性权重**,直接从 π 采样,估计量为简单样本均值

$$
\hat I_N=\frac1N\sum_{i=1}^N f(x_i),\qquad x_i=\mu_{k_i}+\sqrt{\sigma^2}\,z_i .
$$

区别仅在于离散桶 $k_i$ 与高斯 $z_i$ 的生成方式。

### Direct-MCbk-RQMC — MC 抽桶 + RQMC 高斯

$$
k_i\overset{\text{iid}}{\sim}\mathrm{Uniform}\{1,\dots,K\},\qquad
z_i=\Phi^{-1}(b_i),\quad \{b_i\}_{i=1}^N\subset(0,1)^d\ \text{scrambled Sobol'}.
$$

桶分配是普通蒙特卡洛,只有连续高斯部分用 RQMC。离散 MC 噪声主导
$\Rightarrow$ 收敛阶约 $O(N^{-1/2})$。

### Direct-Joint-RQMC — M1(联合 $(d{+}1)$ 维 RQMC)

取**一份** $(d{+}1)$ 维 scrambled Sobol' 点列 $\{u_i\}_{i=1}^N\subset(0,1)^{d+1}$,

$$
k_i=\big\lfloor K\,u_{i,0}\big\rfloor+1,\qquad
z_i=\Phi^{-1}\!\big(u_{i,1},\dots,u_{i,d}\big).
$$

等价地,把整条采样写成 $(0,1)^{d+1}$ 上的单一积分
$\mathbb E_\pi[f]=\int_{(0,1)^{d+1}} f(\phi(u))\,du$,其中
$\phi(u)=\mu_{\lfloor Ku_0\rfloor+1}+\sqrt{\sigma^2}\,\Phi^{-1}(u_{1:d})$,
再用一份 $(d{+}1)$ 维低差异点列估计。第一维负责分桶,base-2 Sobol' 在
$N=2^p$ 时把点严格 $N/K$ 分到各分量;其余 $d$ 维生成高斯。
**单一全局低差异点列**同时驱动桶与高斯,联合 cube 被低差异填充
$\Rightarrow$ 对光滑被积逼近 $O(N^{-1})$。

### Direct-MC — 纯 MC baseline

$$
k_i\overset{\text{iid}}{\sim}\mathrm{Uniform}\{1,\dots,K\},\qquad
z_i\overset{\text{iid}}{\sim}\mathcal N(0, I_d).
$$

桶和高斯都使用普通蒙特卡洛采样，是直接从 $\pi$ 采样的最朴素 baseline，
收敛阶为 $O(N^{-1/2})$。

> 对比:论文里 FM-ISQMC 用学到的 flow $q_\theta$ 作提议 + scrambled-Sobol' 基点 +
> 自归一化重要性采样 (SNIS) $w_i\propto\pi(x_i)/q_\theta(x_i)$;FM-MC/QMC/ISMC 是其
> MC / 无纠偏 / MC-基 的变体。这四种在 `evaluate.py` 中一并计算。

---

## 4. 复现

依赖:`torch numpy scipy matplotlib tqdm`。

### 训练(可选,已提供 ckpt)

```bash
python train.py --steps 800000 --ckpt results/fm_model.pt --gpu 0
```

网络:`dim=30, hidden_dim=512, num_blocks=8`,Logistic 基分布,OT 路径
$x_t=(1-t)x_0+tz$。RTX 4090 上 ~80 steps/s(~2.8h)。

### 评估 + 出图

```bash
python evaluate.py --ckpt results/fm_model.pt
```

产出 `results/final_complex.pdf`、`results/final_second.pdf`(各 7 条曲线：
4 种 FM 方法 + Direct-MCbk-RQMC + Direct-Joint-RQMC + Direct-MC，
以及参考斜率 -0.5 / -1 / FM-ISQMC 拟合),以及对应数据缓存 npz。
$N$:FM 方法 $2\to16384$,直接估计量 $4\to16384$,每个 $N$ 重复 10 次。

---

## 5. 结果摘要(本 ckpt,seed 0)

RMSE-vs-N 的 log-log 斜率:

| 被积 | FM-MC | FM-QMC | FM-ISMC | FM-ISQMC | Direct-MCbk-RQMC | Direct-Joint-RQMC (M1) | Direct-MC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| complex f | -0.36 | -0.26 | -0.51 | -0.55 | -0.54 | **-0.57** | -0.50 |
| 二阶矩 | -0.49 | -0.57 | -0.51 | -0.77 | -0.59 | **-0.99** | -0.50 |

- **Direct-MC** 作为纯 MC baseline，两条曲线斜率均约为 **-0.50**，验证了 MC 的 $O(N^{-1/2})$ 收敛。
- **二阶矩**(光滑无界):Direct-Joint-RQMC (M1) ≈ $-1$,比 Direct-MC 与 FM 方法低一个数量级。
- **complex f**(有界饱和,大方差推向准间断):M1 退化到 $\approx-0.57$,
  与 FM-ISQMC 打平，但仍优于 Direct-MC 的 -0.50。

详见 `results/final_{complex,second}.pdf`。
