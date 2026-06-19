"""
只单独跑新增的 Direct-MC（MC 抽桶 + MC 抽分量），
然后和 results/final_*_cache.npz 里已有的 6 条曲线（4 种 FM + 2 个 Direct）合并出 7 条曲线图。

  python evaluate_estimator3_only.py

输出：results/final_{complex,second}_with_est3.pdf
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from gmm import INTEGRANDS
from estimators import sample_estimator3

NUM_EXP = 10

SERIES_STYLE = [
    ("FM-MC", 'r', '-', 'o'), ("FM-QMC", 'b', '-', 's'),
    ("FM-ISMC", 'g', '-', '^'), ("FM-ISQMC", 'k', '-', 'D'),
    ("Direct-MCbk-RQMC", 'm', '-', 'P'), ("Direct-Joint-RQMC", '#d97a00', '-', 'v'),
    ("Direct-MC", 'c', '-', '*'),
]


def _slope(N, e):
    N = np.asarray(N, float); e = np.asarray(e, float)
    m = np.isfinite(N) & np.isfinite(e) & (N > 0) & (e > 0)
    return float(np.polyfit(np.log(N[m]), np.log(e[m]), 1)[0]) if m.sum() >= 2 else np.nan


def _rmse(ests, tv, vector):
    e = np.asarray(ests)
    if vector:
        mean = e.mean(axis=0)
        return float(np.sqrt(np.sum((mean - tv) ** 2) + np.mean(np.sum((e - mean) ** 2, axis=1))))
    return float(np.sqrt((e.mean() - tv) ** 2 + np.var(e)))


def _agg(x, key):
    spec = INTEGRANDS[key]
    fx = spec["fn"](torch.as_tensor(x, dtype=torch.float32))
    return fx.mean(0).cpu().numpy() if spec["is_vector"] else float(fx.mean())


def run_estimator3(keys):
    series = {k: {"Direct-MC": {"N": [], "rmse": []}} for k in keys}
    for p in tqdm(range(2, 15), desc="Direct-MC"):
        N = 2 ** p
        acc = {k: [] for k in keys}
        for r in range(NUM_EXP):
            x3 = sample_estimator3(N, r)
            for k in keys:
                acc[k].append(_agg(x3, k))
        for k in keys:
            vec = INTEGRANDS[k]["is_vector"]
            series[k]["Direct-MC"]["N"].append(N)
            series[k]["Direct-MC"]["rmse"].append(_rmse(acc[k], INTEGRANDS[k]["true"](), vec))
    return series


def load_cached(keys, cache_dir="results"):
    cached = {k: {} for k in keys}
    for k in keys:
        path = os.path.join(cache_dir, f"final_{k}_cache.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到缓存：{path}")
        z = np.load(path)
        # 旧缓存用 Estimator_1 / Estimator_2 命名，新缓存用 Direct_*；这里做兼容。
        key_map = {
            "FM-MC": "FM_MC",
            "FM-QMC": "FM_QMC",
            "FM-ISMC": "FM_ISMC",
            "FM-ISQMC": "FM_ISQMC",
            "Direct-MCbk-RQMC": ["Direct_MCbk_RQMC", "Estimator_1"],
            "Direct-Joint-RQMC": ["Direct_Joint_RQMC", "Estimator_2"],
        }
        for m, slugs in key_map.items():
            if isinstance(slugs, str):
                slugs = [slugs]
            for slug in slugs:
                if f"{slug}_N" in z:
                    cached[k][m] = {"N": z[f"{slug}_N"], "rmse": z[f"{slug}_rmse"]}
                    break
            else:
                raise KeyError(f"缓存 {path} 中找不到 {m} 的数据")
    return cached


def plot(series_k, save_path):
    plt.figure(figsize=(8, 6)); plt.rcParams.update({'font.size': 14})
    slopes = {}
    for name, color, ls, marker in SERIES_STYLE:
        if name not in series_k:
            continue
        N = np.array(series_k[name]["N"], float); e = np.array(series_k[name]["rmse"])
        plt.loglog(N, e, color=color, linestyle=ls, marker=marker, markersize=6,
                   linewidth=1.6, label=name)
        slopes[name] = _slope(N, e)
    Nref = np.array(series_k["FM-ISQMC"]["N"], float)
    e0, N0 = series_k["FM-ISQMC"]["rmse"][0], Nref[0]
    s_fq = slopes["FM-ISQMC"]
    plt.loglog(Nref, e0 * (Nref / N0) ** -0.5, '--', color='gray', linewidth=1.2, alpha=0.7, label='Slope = -0.5')
    plt.loglog(Nref, e0 * (Nref / N0) ** -1.0, '--', color='gray', linewidth=1.2, alpha=0.9, label='Slope = -1')
    plt.loglog(Nref, e0 * (Nref / N0) ** s_fq, ':', color='dimgray', linewidth=1.5, alpha=0.9,
               label=f'Slope = {s_fq:.2f}')
    plt.xlabel("Sample Size $N$"); plt.ylabel("RMSE")
    plt.legend(loc='lower left', fontsize=10, framealpha=0.95)
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()
    print(f"[{os.path.basename(save_path)}] slopes:", {k: round(v, 3) for k, v in slopes.items()})


def main():
    keys = list(INTEGRANDS.keys())
    est3_series = run_estimator3(keys)
    cached = load_cached(keys)

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    for k in keys:
        merged = {**cached[k], **est3_series[k]}
        save = os.path.join(out_dir, f"final_{k}_with_est3.pdf")
        plot(merged, save)
        print(f"saved {save}")


if __name__ == "__main__":
    main()
