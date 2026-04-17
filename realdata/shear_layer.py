"""
Section 6 - Application to Shear Layer Experiment (Figure 5).

Reproduces the compressible shear layer analysis:
  - KRR fits for y_s (simulation, n=11) and y_i (physical, n=32)
  - Global FMMT test over [0, 1.5]
  - Subdomain FMMT tests over six equal intervals of length 0.25
  - ND2003 K1/K2 comparison tests

Data: embedded from Wang & Ding (2009), doi:10.1198/TECH.2009.07011
Output:
    results/figures/shear_layer_results.png   (Figure 5 in paper)
    results/tables/shear_layer_pvalues.csv

Usage:
    python -m realdata.shear_layer
"""
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.kde import KDE
from code.kernel_ridge import kernel_ridge_regression
from code.fourier_basis import create_decay_weight_function
from code.statistical_tests import (
    compute_1d_test_statistic_kde_fft,
    compute_1d_max_statistic,
    compute_1d_max_statistic_cdf,
)
from code.nd2003 import (
    get_rule_of_thumb_bandwidths,
    calculate_nd2003_statistics,
    wild_bootstrap_nd2003,
)
from code.utils import set_random_seed

NUM_BOOTSTRAPS = 200


# ---------------------------------------------------------------------------
# Embedded dataset (Wang & Ding 2009, doi:10.1198/TECH.2009.07011)
# ---------------------------------------------------------------------------

def load_shear_layer_data():
    """Load the compressible shear layer dataset."""
    ys_raw = [
        (0.100, 1.0000), (0.240, 1.0014), (0.380, 0.9596), (0.520, 0.8828),
        (0.660, 0.7977), (0.800, 0.7527), (0.940, 0.6346), (1.080, 0.5657),
        (1.220, 0.5112), (1.360, 0.4716), (1.500, 0.4531),
    ]
    yi_raw = [
        (0.992, 0.4640), (0.945, 0.4890), (0.059, 1.0000), (0.510, 0.9710),
        (0.342, 0.9780), (0.640, 0.7620), (0.428, 1.0000), (0.860, 0.5750),
        (0.476, 0.9810), (0.206, 0.9850), (0.636, 0.7520), (0.455, 0.8170),
        (0.821, 0.6010), (0.691, 0.5650), (0.928, 0.4600), (0.720, 0.6330),
        (1.119, 0.4530), (0.795, 0.5020), (1.309, 0.4220), (0.862, 0.4570),
        (1.440, 0.4400), (0.985, 0.4000), (0.270, 1.3500), (0.525, 1.0580),
        (0.519, 0.9570), (0.535, 0.8100), (0.589, 0.8120), (0.580, 0.9270),
        (0.668, 0.7330), (0.640, 0.8410), (0.825, 0.5350), (1.040, 0.5180),
    ]
    X_ys = np.array([r[0] for r in ys_raw]).reshape(-1, 1)
    y_ys = np.array([r[1] for r in ys_raw])
    X_yi = np.array([r[0] for r in yi_raw]).reshape(-1, 1)
    y_yi = np.array([r[1] for r in yi_raw])
    return X_ys, y_ys, X_yi, y_yi


# ---------------------------------------------------------------------------
# Subdomain FMMT
# ---------------------------------------------------------------------------

def _subdomain_fmmt(model_yi, model_ys, density_estimator, n, sigma_hat,
                    start, end, max_freq):
    """Compute FMMT p-value for subdomain [start, end]."""
    subdomain_length = end - start
    x_grid = np.linspace(start, end, 512).reshape(-1, 1)
    f_diff = model_yi.predict(x_grid).ravel() - model_ys.predict(x_grid).ravel()
    density = density_estimator(x_grid).ravel()
    scaled = f_diff * np.sqrt(n) * np.sqrt(density) / sigma_hat

    x_flat = x_grid.ravel()
    coeffs, decays = [], []

    # Frequency 0
    coeffs.append(np.trapezoid(scaled * np.ones_like(x_flat) * 2, x_flat) / subdomain_length)
    decays.append(1.0 / np.log(2))

    for freq in range(1, max_freq + 1):
        rho = 1.0 / np.log(freq + 2)
        coeffs.append(rho * np.trapezoid(
            scaled * 2 * np.sqrt(2) * np.cos(2 * np.pi * freq / subdomain_length * (x_flat - start)),
            x_flat) / subdomain_length)
        coeffs.append(rho * np.trapezoid(
            scaled * 2 * np.sqrt(2) * np.sin(2 * np.pi * freq / subdomain_length * (x_flat - start)),
            x_flat) / subdomain_length)
        decays.extend([rho, rho])

    T = float(np.max(np.abs(coeffs)))
    F_t = float(np.prod([2 * norm.cdf(T / rho) - 1 for rho in decays]))
    return T, 1.0 - F_t


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_shear_layer(seed=42, output_dir="results"):
    set_random_seed(seed)
    Path(f"{output_dir}/figures").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/tables").mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X_ys, y_ys, X_yi, y_yi = load_shear_layer_data()
    n_yi = len(X_yi)

    print("Fitting KRR models...")
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.gaussian_process.kernels import Matern

    matern_k = Matern(length_scale=1, nu=3.5)
    model_ys = KernelRidge(kernel=matern_k, alpha=0.0)
    model_ys.fit(X_ys, y_ys)

    model_yi, sigma_hat = kernel_ridge_regression(X_yi, y_yi)

    print("Estimating density (KDE)...")
    density_est = KDE(X_yi)

    print("Running FMMT tests...")
    weight_fn = create_decay_weight_function(1, "log")
    max_freq  = int(np.sqrt(n_yi))

    global_domain = np.array([[0.0, 1.5]])
    T_fn   = compute_1d_test_statistic_kde_fft(
        n_samples=n_yi, weight_function=weight_fn, sigma_hat=sigma_hat,
        domain_bounds=global_domain, fitted_function=model_yi.predict,
        true_function=model_ys.predict, density_estimate=density_est,
        grid_points=1024,
    )
    T_glob = compute_1d_max_statistic(max_freq, T_fn)
    p_glob = 1.0 - compute_1d_max_statistic_cdf(T_glob, max_freq, weight_fn)

    subdomain_edges = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75),
                       (0.75, 1.0), (1.0, 1.25), (1.25, 1.5)]
    sub_T, sub_p = [], []
    for (s, e) in subdomain_edges:
        T_s, p_s = _subdomain_fmmt(model_yi, model_ys, density_est,
                                    n_yi, sigma_hat, s, e, max_freq)
        sub_T.append(T_s)
        sub_p.append(p_s)

    print("Running ND2003 comparison tests...")
    h, _, _, g = get_rule_of_thumb_bandwidths(y_ys, X_ys.ravel(), y_yi, X_yi.ravel())
    K1, K2 = calculate_nd2003_statistics(X_ys.ravel(), y_ys, X_yi.ravel(), y_yi, h)
    bs = wild_bootstrap_nd2003(X_ys.ravel(), y_ys, X_yi.ravel(), y_yi, h, g, NUM_BOOTSTRAPS)
    p_K1 = float(np.mean(np.array(bs["K1"]) >= K1))
    p_K2 = float(np.mean(np.array(bs["K2"]) >= K2))

    print(f"\nResults:")
    print(f"  FMMT global:  T={T_glob:.4f}  p={p_glob:.4e}")
    for i, (T_s, p_s) in enumerate(zip(sub_T, sub_p), 1):
        star = " *" if p_s < 0.05 else ""
        print(f"  Subdomain {i} [{subdomain_edges[i-1][0]:.2f},{subdomain_edges[i-1][1]:.2f}]:  "
              f"T={T_s:.4f}  p={p_s:.4e}{star}")
    print(f"  ND2003 K1:    T={K1:.4f}  p={p_K1:.3f}")
    print(f"  ND2003 K2:    T={K2:.4f}  p={p_K2:.3f}")

    # -----------------------------------------------------------------------
    # Save results table
    # -----------------------------------------------------------------------
    rows = [{"Test": "Global FMMT", "Domain": "[0.00, 1.50]",
              "T": T_glob, "p_value": p_glob, "Reject_0.05": p_glob < 0.05}]
    for i, ((s, e), T_s, p_s) in enumerate(zip(subdomain_edges, sub_T, sub_p), 1):
        rows.append({"Test": f"Subdomain {i}", "Domain": f"[{s:.2f}, {e:.2f}]",
                     "T": T_s, "p_value": p_s, "Reject_0.05": p_s < 0.05})
    rows.append({"Test": "ND2003 K1", "Domain": "[0.00, 1.50]",
                 "T": K1, "p_value": p_K1, "Reject_0.05": p_K1 < 0.05})
    rows.append({"Test": "ND2003 K2", "Domain": "[0.00, 1.50]",
                 "T": K2, "p_value": p_K2, "Reject_0.05": p_K2 < 0.05})

    df = pd.DataFrame(rows)
    csv_path = Path(f"{output_dir}/tables/shear_layer_pvalues.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Table saved: {csv_path}")

    # -----------------------------------------------------------------------
    # Figure 5
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    x_plot = np.linspace(0.05, 1.5, 300).reshape(-1, 1)

    # Top-left: KRR fits
    ax = axes[0, 0]
    ax.scatter(X_yi, y_yi, color="steelblue", alpha=0.7, s=50, label=r"$y_i$ data")
    ax.scatter(X_ys, y_ys, facecolors="none", edgecolors="black",
               s=80, linewidth=1.5, label=r"$y_s$ data")
    ax.plot(x_plot, model_yi.predict(x_plot), "b-", lw=2, label=r"$\hat{f}_i$ (KRR)")
    ax.plot(x_plot, model_ys.predict(x_plot), "k-", lw=2, label=r"$\hat{f}_s$ (KRR)")
    ax.set_xlabel(r"Convective Mach Number $M_c$", fontsize=10)
    ax.set_ylabel(r"Compressibility Factor $\Phi$", fontsize=10)
    ax.set_title("KRR Fits", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: density
    ax = axes[0, 1]
    x_dense = np.linspace(0, 1.5, 300).reshape(-1, 1)
    ax.plot(x_dense, density_est(x_dense), "g-", lw=2, label="KDE")
    ax.hist(X_yi.ravel(), bins=12, density=True, alpha=0.4,
            color="lightblue", label=r"$y_i$ histogram")
    ax.set_xlabel(r"$M_c$", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(r"Estimated Density of $y_i$", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-left: test statistics
    ax = axes[1, 0]
    labels = ["Global"] + [f"Sub{i}" for i in range(1, 7)]
    T_vals = [T_glob] + sub_T
    colors = ["red"] + ["steelblue"] * 6
    bars = ax.bar(labels, T_vals, color=colors, alpha=0.75, width=0.6)
    ax.set_ylabel("Test Statistic $T$", fontsize=10)
    ax.set_title("FMMT Test Statistics", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, T_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    # Bottom-right: p-values (log scale)
    ax = axes[1, 1]
    p_vals = [p_glob] + sub_p
    p_plot = np.maximum(p_vals, 1e-16)
    bars = ax.bar(labels, p_plot, color=colors, alpha=0.75, width=0.6)
    ax.axhline(0.05, color="red", linestyle="--", lw=1.5, label=r"$\alpha=0.05$")
    ax.set_yscale("log")
    ax.set_ylabel("p-value", fontsize=10)
    ax.set_title("FMMT p-values", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, p in zip(bars, p_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                max(bar.get_height(), 1e-16) * 2.5,
                f"{p:.1e}", ha="center", va="bottom", fontsize=7, rotation=45)

    fig.suptitle("Compressible Shear Layer Application (Section 6)", fontsize=12)
    fig.tight_layout()
    fig_path = Path(f"{output_dir}/figures/shear_layer_results.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    return {
        "global_T": T_glob, "global_p": p_glob,
        "subdomain_T": sub_T, "subdomain_p": sub_p,
        "nd2003_K1_p": p_K1, "nd2003_K2_p": p_K2,
    }


if __name__ == "__main__":
    run_shear_layer()
