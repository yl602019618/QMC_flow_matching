"""
Final paper-quality plots: 4 figures (2D / 30D x second / complex).

Each figure shows:
    FM-MC, FM-QMC, FM-ISMC, FM-ISQMC                (4 FM curves, fixed legend order)
    Estimator 1  (= Direct-MCbk-RQMC,  ~ -0.5 slope)
    Estimator 2  (= M1 Joint (d+1)-Sobol', ~ -1 slope)
    Reference dashes: slope = -0.5, slope = -1, slope = FM-ISQMC fit  (3 dashes)
No title. Legend ordered: FM-MC, FM-QMC, FM-ISMC, FM-ISQMC, Estimator 1, Estimator 2.

Data sources:
    gmm/results_logi/all_methods_2d.npz        -> FM 4 methods + Estimator 1 (2D)
    gmm/results_logi/rqmc_advanced_2d.npz      -> Estimator 2 (M1) (2D)
    gmm30d/results30d/all_methods_30d_5x.npz   -> FM 4 methods + Estimator 1 (30D, 5x ckpt)
    gmm30d/results30d/rqmc_advanced_30d.npz    -> Estimator 2 (M1) (30D)

Outputs:
    figures/final_2d_second.pdf
    figures/final_2d_complex.pdf
    figures/final_30d_second.pdf
    figures/final_30d_complex.pdf
    figures/plot_data_cache.npz   -- consolidated data used by these plots
"""

import os
import numpy as np
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# Curve drawing specs (color, linestyle, marker) in fixed legend order
SERIES = [
    ("FM-MC",       'r', '-',  'o'),
    ("FM-QMC",      'b', '-',  's'),
    ("FM-ISMC",     'g', '-',  '^'),
    ("FM-ISQMC",    'k', '-',  'D'),
    ("Estimator 1", 'm', '-',  'P'),
    ("Estimator 2", '#d97a00', '-', 'v'),   # dark orange
]


def _slope(N, err):
    N = np.asarray(N, dtype=float); err = np.asarray(err, dtype=float)
    m = np.isfinite(N) & np.isfinite(err) & (N > 0) & (err > 0)
    if m.sum() < 2:
        return float('nan')
    s, _ = np.polyfit(np.log(N[m]), np.log(err[m]), 1)
    return float(s)


def load_cache():
    """Load all 4 figures' data into a single dict.

    Returns dict keyed by (setting, integrand) -> dict of series_name -> (N, err).
    setting in {"2d", "30d"}, integrand in {"second", "complex"}.
    """
    n2d = np.load(os.path.join(HERE, "gmm", "results_logi", "all_methods_2d.npz"))
    n2m = np.load(os.path.join(HERE, "gmm", "results_logi", "rqmc_advanced_2d.npz"))
    n30 = np.load(os.path.join(HERE, "gmm30d", "results30d", "all_methods_30d_5x.npz"))
    n3m = np.load(os.path.join(HERE, "gmm30d", "results30d", "rqmc_advanced_30d.npz"))

    cache = {}
    for setting, src_main, src_m1 in [("2d", n2d, n2m), ("30d", n30, n3m)]:
        N_main = src_main["sample_sizes"]
        N_m1   = src_m1["sample_sizes"]
        for key in ("second", "complex"):
            series = {
                "FM-MC":       (N_main, src_main[f"rmse_MC_{key}"]),
                "FM-QMC":      (N_main, src_main[f"rmse_QMC_{key}"]),
                "FM-ISMC":     (N_main, src_main[f"rmse_ISMC_{key}"]),
                "FM-ISQMC":    (N_main, src_main[f"rmse_ISQMC_{key}"]),
                "Estimator 1": (N_main, src_main[f"rmse_direct_mc_{key}"]),
                "Estimator 2": (N_m1,   src_m1[f"rmse_M1_Joint_d+1_{key}"]),
            }
            cache[(setting, key)] = series
    return cache


def save_cache(cache):
    """Flatten cache into a single npz with structured keys."""
    out = {}
    for (setting, key), series in cache.items():
        for sname, (N, err) in series.items():
            slug = sname.replace(" ", "_").replace("-", "_")
            out[f"{setting}_{key}_{slug}_N"] = np.asarray(N)
            out[f"{setting}_{key}_{slug}_rmse"] = np.asarray(err)
    path = os.path.join(FIG_DIR, "plot_data_cache.npz")
    np.savez(path, **out)
    print(f"[cache] saved {path}  ({len(out)} arrays)")
    return path


def plot_one(series, save_path):
    """series: dict series_name -> (N, err). Saves a PDF, no title."""
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.rcParams.update({'font.size': 14})

    # main curves in fixed order
    for name, color, ls, marker in SERIES:
        N, err = series[name]
        ax.loglog(N, err, color=color, linestyle=ls, marker=marker,
                  markersize=6, linewidth=1.6, label=name)

    # reference dashes anchored at FM-ISQMC's first point
    N_ref, err_ref = series["FM-ISQMC"]
    N_ref = np.asarray(N_ref, dtype=float); err_ref = np.asarray(err_ref, dtype=float)
    N0, e0 = N_ref[0], err_ref[0]
    s_isqmc = _slope(N_ref, err_ref)

    ax.loglog(N_ref, e0 * (N_ref / N0) ** (-0.5), '--',
              color='gray', linewidth=1.2, alpha=0.7,
              label='Slope = -0.5')
    ax.loglog(N_ref, e0 * (N_ref / N0) ** (-1.0), '--',
              color='gray', linewidth=1.2, alpha=0.9,
              label='Slope = -1')
    ax.loglog(N_ref, e0 * (N_ref / N0) ** s_isqmc, ':',
              color='dimgray', linewidth=1.5, alpha=0.9,
              label=f'Slope = {s_isqmc:.2f}')

    ax.set_xlabel("Sample Size $N$")
    ax.set_ylabel("RMSE")
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

    # legend in the order we constructed plots: FM-MC, FM-QMC, FM-ISMC,
    # FM-ISQMC, Estimator 1, Estimator 2, then the three reference dashes.
    ax.legend(loc='lower left', fontsize=10, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[plot] {save_path}  (FM-ISQMC slope = {s_isqmc:.3f})")


def print_slope_table(cache):
    print("\n=== slope summary ===")
    for (setting, key), series in sorted(cache.items()):
        print(f"-- {setting} {key} --")
        for name in ["FM-MC", "FM-QMC", "FM-ISMC", "FM-ISQMC", "Estimator 1", "Estimator 2"]:
            N, err = series[name]
            print(f"  {name:14s}: slope = {_slope(N, err):.3f}")


def main():
    cache = load_cache()
    save_cache(cache)
    out_map = {
        ("2d",  "second"):  "final_2d_second.pdf",
        ("2d",  "complex"): "final_2d_complex.pdf",
        ("30d", "second"):  "final_30d_second.pdf",
        ("30d", "complex"): "final_30d_complex.pdf",
    }
    for k, name in out_map.items():
        plot_one(cache[k], os.path.join(FIG_DIR, name))
    print_slope_table(cache)


if __name__ == "__main__":
    main()
