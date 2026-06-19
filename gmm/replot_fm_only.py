"""
Standalone replotter: read the saved all_methods_2d.npz and produce a figure
with only the four FM estimators for the 'complex' integrand.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results_logi")


def _fit_loglog_slope(N, err, tail_k=None):
    x = np.asarray(N, dtype=float)
    y = np.asarray(err, dtype=float)
    if tail_k is not None and 2 <= tail_k <= len(x):
        x = x[-tail_k:]
        y = y[-tail_k:]
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[m], y[m]
    if x.size < 2:
        return np.nan
    s, _ = np.polyfit(np.log(x), np.log(y), 1)
    return float(s)


def plot_one_integrand(res, integrand_key, save_path, title, baseline_curves=None):
    keys = ["MC", "QMC", "ISMC", "ISQMC"]
    styles = {
        "MC": ('r-o', 'FM-MC'),
        "QMC": ('b-s', 'FM-QMC'),
        "ISMC": ('g-^', 'FM-ISMC'),
        "ISQMC": ('k-^', 'FM-ISQMC'),
    }

    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 14})

    slopes = {}
    N_arr = np.array(res["MC"][integrand_key]["sample_sizes"], dtype=float)

    for k in keys:
        N = res[k][integrand_key]["sample_sizes"]
        err = res[k][integrand_key]["mse_errors"]
        style, label = styles[k]
        plt.loglog(N, err, style, label=label)
        slopes[k] = _fit_loglog_slope(N, err)

    if baseline_curves:
        baseline_styles = {
            "Direct-MCbk-RQMC": ('m-D', 'Direct-MCbk-RQMC'),
            "Direct-RQMCbk-RQMC": ('y-v', 'Direct-RQMCbk-RQMC'),
        }
        for name, (Nb, errb) in baseline_curves.items():
            style, label = baseline_styles.get(name, ('c-x', name))
            plt.loglog(Nb, errb, style, label=label)
            slopes[name] = _fit_loglog_slope(Nb, errb)

    # reference / fit dashes anchored at FM-ISQMC's first point
    isqmc_err0 = res["ISQMC"][integrand_key]["mse_errors"][0]
    N0 = N_arr[0]
    plt.loglog(N_arr, isqmc_err0 * (N_arr / N0) ** (-0.5), '--',
               linewidth=1, alpha=0.7, label='Slope = -0.5')
    plt.loglog(N_arr, isqmc_err0 * (N_arr / N0) ** (-1.0), '--',
               linewidth=1, alpha=0.7, label='Slope = -1')
    if np.isfinite(slopes["ISQMC"]):
        plt.loglog(N_arr, isqmc_err0 * (N_arr / N0) ** slopes["ISQMC"], '--',
                   linewidth=1, alpha=0.7,
                   label=f'Slope = {slopes["ISQMC"]:.2f}')

    plt.xlabel("Sample Size")
    plt.ylabel("RMSE")
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved {save_path}")
    print(f"[{title}] slopes: " +
          ", ".join(f"{k}={slopes[k]:.3f}" for k in slopes))
    return slopes


def main():
    npz_path = os.path.join(OUT_DIR, "all_methods_2d.npz")
    npz = np.load(npz_path)

    sample_sizes = npz["sample_sizes"]
    keys = ["MC", "QMC", "ISMC", "ISQMC"]
    integrands = ["first", "second", "complex"]

    res = {m: {k: {"sample_sizes": sample_sizes,
                   "mse_errors": npz[f"rmse_{m}_{k}"]}
               for k in integrands}
           for m in keys}

    integrand_key = "complex"
    title = "2D GMM, Complex f: 4 FM estimators"
    save_path = os.path.join(OUT_DIR, f"fm_only_2d_{integrand_key}.pdf")
    plot_one_integrand(res, integrand_key, save_path, title,
                       baseline_curves=None)


if __name__ == "__main__":
    main()
