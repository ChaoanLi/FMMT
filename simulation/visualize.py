"""
Visualization for simulation study results (Figures 1-4 in paper).

Reads JSON outputs from sim_1d_global.py / sim_1d_subdomain.py /
sim_2d_global.py / sim_2d_subdomain.py and generates publication figures.

Usage:
    python -m simulation.visualize --step 1   # Figure 1 + Table 2
    python -m simulation.visualize --step 2   # Figure 2
    python -m simulation.visualize --step 3   # Figure 3 + Table 3
    python -m simulation.visualize --step 4   # Figure 4
    python -m simulation.visualize            # all figures
"""
import sys
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


STEP_CONFIG = {
    1: {
        "json":    "results/step1_1d_global.json",
        "fig":     "results/figures/step1_power_curves.png",
        "csv":     "results/tables/step1_type_i_error.csv",
        "n_sub":   3,
        "title":   "1D Global Alternatives (n=50)",
        "ncols":   3,
    },
    2: {
        "json":    "results/step2_1d_subdomain.json",
        "fig":     "results/figures/step2_power_curves.png",
        "csv":     "results/tables/step2_type_i_error.csv",
        "n_sub":   3,
        "title":   "1D Subdomain Alternatives (n=50)",
        "ncols":   3,
    },
    3: {
        "json":    "results/step3_2d_global.json",
        "fig":     "results/figures/step3_power_curves.png",
        "csv":     "results/tables/step3_type_i_error.csv",
        "n_sub":   4,
        "title":   "2D Global Alternatives (n=200)",
        "ncols":   2,
    },
    4: {
        "json":    "results/step4_2d_subdomain.json",
        "fig":     "results/figures/step4_power_curves.png",
        "csv":     "results/tables/step4_type_i_error.csv",
        "n_sub":   4,
        "title":   "2D Subdomain Alternatives (n=200)",
        "ncols":   3,
    },
}

COLORS     = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
LINESTYLES = ["-", "--", "-.", ":", "-"]
MARKERS    = ["o", "s", "^", "D", "v"]


def _load_json(path):
    with open(path) as fh:
        return json.load(fh)


def plot_power_curves(step, output_dir="."):
    """Generate power-curve figure for a given step."""
    cfg   = STEP_CONFIG[step]
    data  = _load_json(cfg["json"])
    n_sub = cfg["n_sub"]

    curve_labels = [f"subdomain_{i+1}" for i in range(n_sub)] + ["global", "bonferroni"]
    scenarios    = list(data.keys())
    n_sc         = len(scenarios)
    ncols        = cfg["ncols"]
    nrows        = (n_sc + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), dpi=150)
    axes = np.array(axes).flatten()

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        sc = data[scenario]
        c_vals = sorted(sc.keys(), key=lambda k: float(k.split("_", 1)[1]))
        xs = [sc[k]["c"] for k in c_vals]
        for li, label in enumerate(curve_labels):
            ys = [sc[k]["rejection_rates"].get(label, np.nan) for k in c_vals]
            ax.plot(xs, ys, label=label,
                    color=COLORS[li % len(COLORS)],
                    linestyle=LINESTYLES[li % len(LINESTYLES)],
                    marker=MARKERS[li % len(MARKERS)],
                    linewidth=2, markersize=5, markevery=2, alpha=0.85)
        ax.axhline(0.05, color="black", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.set_title(scenario.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("c", fontsize=10)
        ax.set_ylabel("Rejection Rate", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for i in range(n_sc, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(cfg["title"], fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = Path(cfg["fig"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out}")


def export_type_i_error_csv(step):
    """Export Type I error (c=0 rejection rates) to CSV."""
    cfg  = STEP_CONFIG[step]
    data = _load_json(cfg["json"])

    n_sub = cfg["n_sub"]
    sub_labels  = [f"subdomain_{i+1}" for i in range(n_sub)]
    all_labels  = sub_labels + ["global", "bonferroni"]
    header      = ["Scenario"] + all_labels

    rows = [header]
    for scenario, sc in data.items():
        c0_key = next(k for k in sc if abs(float(k.split("_", 1)[1])) < 1e-8)
        rr = sc[c0_key]["rejection_rates"]
        rows.append([scenario] + [f"{rr.get(lbl, np.nan):.4f}" for lbl in all_labels])

    out = Path(cfg["csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for row in rows:
            fh.write(",".join(str(x) for x in row) + "\n")
    print(f"  CSV saved: {out}")


def visualize_step(step):
    cfg = STEP_CONFIG[step]
    if not Path(cfg["json"]).exists():
        print(f"  JSON not found: {cfg['json']}  (run the simulation first)")
        return
    plot_power_curves(step)
    export_type_i_error_csv(step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figures 1-4")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4],
                        help="Which step to visualize (default: all)")
    args = parser.parse_args()
    steps = [args.step] if args.step else [1, 2, 3, 4]
    for s in steps:
        print(f"\nVisualizing Step {s}...")
        visualize_step(s)
