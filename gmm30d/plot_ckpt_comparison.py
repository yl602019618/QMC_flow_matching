"""
Three-ckpt comparison plots for the 30D GMM experiment.

For each integrand (first, second, complex), overlay the four FM methods
across the three training-length ckpts (1x, 2x, 5x). Outputs:
    results30d/ckpt_comparison_{first,second,complex}.pdf
    results30d/ckpt_summary.txt    -- slope table across all (ckpt, method, key)
"""

import os
import numpy as np
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results30d")

CKPTS = [
    ("1x", "all_methods_30d.npz",     'solid'),
    ("2x", "all_methods_30d_2x.npz",  'dashed'),
    ("5x", "all_methods_30d_5x.npz",  'dotted'),
]
METHODS = [
    ("MC",    'r', 'o'),
    ("QMC",   'b', 's'),
    ("ISMC",  'g', '^'),
    ("ISQMC", 'k', 'D'),
]
INTEGRANDS = ["first", "second", "complex"]
LABELS = {"first": "First moment", "second": "Second moment", "complex": "Complex f"}


def _slope(N, err):
    N = np.asarray(N, dtype=float); err = np.asarray(err, dtype=float)
    m = np.isfinite(N) & np.isfinite(err) & (N > 0) & (err > 0)
    if m.sum() < 2:
        return np.nan
    s, _ = np.polyfit(np.log(N[m]), np.log(err[m]), 1)
    return float(s)


def main():
    data = {}
    for tag, fname, _ in CKPTS:
        path = os.path.join(OUT_DIR, fname)
        if not os.path.exists(path):
            print(f"Missing: {path}; skip {tag}")
            continue
        data[tag] = np.load(path)

    slopes = {}
    summary_lines = ["# 30D GMM three-ckpt slope comparison", ""]
    for key in INTEGRANDS:
        fig, ax = plt.subplots(figsize=(9, 6))
        plt.rcParams.update({'font.size': 13})
        for m_name, color, marker in METHODS:
            for tag, _, ls in CKPTS:
                if tag not in data:
                    continue
                N = data[tag]["sample_sizes"]
                err = data[tag][f"rmse_{m_name}_{key}"]
                s = _slope(N, err)
                slopes[(tag, m_name, key)] = s
                ax.loglog(N, err, color=color, marker=marker, linestyle=ls,
                          markersize=5, linewidth=1.4,
                          label=f"FM-{m_name} {tag} (slope={s:.2f})")
        ax.set_xlabel("Sample Size $N$"); ax.set_ylabel("RMSE")
        ax.set_title(f"30D GMM, {LABELS[key]}: 1x / 2x / 5x ckpt comparison")
        ax.grid(True, which='both', ls='--', alpha=0.4)
        # arrange legend in 4 columns (one per method), 3 rows (per ckpt)
        ax.legend(loc='lower left', fontsize=8, ncol=4, columnspacing=0.8)
        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"ckpt_comparison_{key}.pdf")
        plt.savefig(out, dpi=200); plt.close()
        print(f"Saved {out}")

    # ------ summary table ------
    summary_lines.append(f"{'integrand':10s} {'method':7s} " +
                         " ".join(f"{t:>8s}" for t, *_ in CKPTS))
    summary_lines.append("-" * 60)
    for key in INTEGRANDS:
        for m_name, *_ in METHODS:
            row = [f"{key:10s}", f"{m_name:7s}"]
            for tag, *_ in CKPTS:
                s = slopes.get((tag, m_name, key), np.nan)
                row.append(f"{s:>8.3f}" if np.isfinite(s) else f"{'-':>8s}")
            summary_lines.append(" ".join(row))
        summary_lines.append("")

    summary_path = os.path.join(OUT_DIR, "ckpt_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nSaved {summary_path}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
