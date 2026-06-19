"""
Evaluate all six estimators on the 30D GMM (cov = VAR*I_30) and reproduce the
two final figures:

    results/final_complex.pdf     (integrand: complex f)
    results/final_second.pdf      (integrand: per-coordinate second moment)

Six curves per figure:
  FM-MC      : flow proposal, i.i.d. Gaussian base, plain mean
  FM-QMC     : flow proposal, scrambled-Sobol' base, plain mean
  FM-ISMC    : flow proposal + SNIS (i.i.d. base)
  FM-ISQMC   : flow proposal + SNIS (scrambled-Sobol' base)
  Direct-MCbk-RQMC: MC bucket + RQMC Gaussian (no flow)
  Direct-Joint-RQMC: M1, joint (d+1)-dim RQMC  (no flow)
  Direct-MC: MC bucket + MC Gaussian (pure MC baseline, no flow)

FM methods use N = 2..16384 (p=1..14); the direct estimators use N = 4..16384
(p=2..14). 10 independent repetitions per N. RMSE is computed against the
closed-form true value.

  python evaluate.py --ckpt results/fm_model.pt
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import FlowMatchingOT, set_seed
from gmm import INTEGRANDS, log_prob_torch
from estimators import sample_estimator1, sample_estimator2, sample_estimator3

NUM_EXP = 10
SAMPLING_STEPS = 64
INTEGRATOR = "heun"
LOGPROB_STEPS = 64
LOGPROB_BS = 256

SERIES_STYLE = [
    ("FM-MC", 'r', '-', 'o'), ("FM-QMC", 'b', '-', 's'),
    ("FM-ISMC", 'g', '-', '^'), ("FM-ISQMC", 'k', '-', 'D'),
    ("Direct-MCbk-RQMC", 'm', '-', 'P'), ("Direct-Joint-RQMC", '#d97a00', '-', 'v'),
    ("Direct-MC", 'c', '-', '*'),
]


def load_model(device, ckpt):
    m = FlowMatchingOT(dim=30, hidden_dim=512, num_blocks=8, sigma=0.0, lr=1e-3,
                       device=device, base_dist="logistic", base_loc=0.0, base_scale=1.0)
    m.load(ckpt); m.eval(); return m


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


def _agg(x, key, w=None):
    spec = INTEGRANDS[key]
    fx = spec["fn"](x if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32))
    vec = spec["is_vector"]
    if w is not None:
        return (w.unsqueeze(1) * fx).sum(0).cpu().numpy() if vec else float((w * fx).sum())
    return fx.mean(0).cpu().numpy() if vec else float(fx.mean())


def run(ckpt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(0)
    model = load_model(device, ckpt)
    keys = list(INTEGRANDS.keys())
    tvs = {k: INTEGRANDS[k]["true"]() for k in keys}
    fm_methods = ["FM-MC", "FM-QMC", "FM-ISMC", "FM-ISQMC"]
    rq_methods = ["Direct-MCbk-RQMC", "Direct-Joint-RQMC", "Direct-MC"]
    series = {k: {m: {"N": [], "rmse": []} for m in fm_methods + rq_methods} for k in keys}

    # ---- flow-based methods: N = 2..16384 ----
    for p in tqdm(range(1, 15), desc="FM"):
        N = 2 ** p
        acc = {k: {m: [] for m in fm_methods} for k in keys}
        for r in range(NUM_EXP):
            x_mc = model.sample(N, sampling_steps=SAMPLING_STEPS, integrator=INTEGRATOR).to(device)
            x_q = model.sample_qmc(N, sampling_steps=SAMPLING_STEPS, exp=r, integrator=INTEGRATOR).to(device)
            lq = model.batched_log_prob(x_q, steps=LOGPROB_STEPS, batch_size=LOGPROB_BS, integrator=INTEGRATOR)
            lq_mc = model.batched_log_prob(x_mc, steps=LOGPROB_STEPS, batch_size=LOGPROB_BS, integrator=INTEGRATOR)
            with torch.no_grad():
                lw_q = log_prob_torch(x_q) - lq.detach()
                w_q = torch.exp(lw_q - torch.logsumexp(lw_q, 0))
                lw_mc = log_prob_torch(x_mc) - lq_mc.detach()
                w_mc = torch.exp(lw_mc - torch.logsumexp(lw_mc, 0))
                for k in keys:
                    acc[k]["FM-MC"].append(_agg(x_mc, k))
                    acc[k]["FM-QMC"].append(_agg(x_q, k))
                    acc[k]["FM-ISMC"].append(_agg(x_mc, k, w_mc))
                    acc[k]["FM-ISQMC"].append(_agg(x_q, k, w_q))
        for k in keys:
            vec = INTEGRANDS[k]["is_vector"]
            for m in fm_methods:
                series[k][m]["N"].append(N); series[k][m]["rmse"].append(_rmse(acc[k][m], tvs[k], vec))

    # ---- direct estimators: N = 4..16384 ----
    for p in tqdm(range(2, 15), desc="RQMC"):
        N = 2 ** p
        acc = {k: {"Direct-MCbk-RQMC": [], "Direct-Joint-RQMC": [], "Direct-MC": []} for k in keys}
        for r in range(NUM_EXP):
            x1 = sample_estimator1(N, r)
            x2 = sample_estimator2(N, r)
            x3 = sample_estimator3(N, r)
            for k in keys:
                acc[k]["Direct-MCbk-RQMC"].append(_agg(x1, k))
                acc[k]["Direct-Joint-RQMC"].append(_agg(x2, k))
                acc[k]["Direct-MC"].append(_agg(x3, k))
        for k in keys:
            vec = INTEGRANDS[k]["is_vector"]
            for m in rq_methods:
                series[k][m]["N"].append(N); series[k][m]["rmse"].append(_rmse(acc[k][m], tvs[k], vec))
    return series, tvs


def plot(series_k, save_path):
    plt.figure(figsize=(8, 6)); plt.rcParams.update({'font.size': 14})
    slopes = {}
    for name, color, ls, marker in SERIES_STYLE:
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
    print(f"[{os.path.basename(save_path)}] slopes:",
          {k: round(v, 3) for k, v in slopes.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="results/fm_model.pt")
    ap.add_argument("--out_dir", type=str, default="results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    series, tvs = run(args.ckpt)
    for k in INTEGRANDS:
        tv = tvs[k]
        print(f"true {k} = {tv if np.isscalar(tv) else tv[:2]}")
        save = os.path.join(args.out_dir, f"final_{k}.pdf")
        plot(series[k], save)
        cache = {}
        for name in series[k]:
            slug = name.replace(" ", "_").replace("-", "_")
            cache[f"{slug}_N"] = np.array(series[k][name]["N"])
            cache[f"{slug}_rmse"] = np.array(series[k][name]["rmse"])
        np.savez(os.path.join(args.out_dir, f"final_{k}_cache.npz"), **cache)
        print(f"saved {save}")


if __name__ == "__main__":
    main()
