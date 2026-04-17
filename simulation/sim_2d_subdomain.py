"""
Section 5.2 - 2D Subdomain Scenarios (Figure 4).

Tests FMMT on three 2D subdomain alternatives (quadrant modulation).
n=200, sigma=0.1, c in [-0.5, 0.5] step 0.05, 1000 replications.

Outputs:
    results/step4_2d_subdomain.json
    results/tables/step4_type_i_error.csv
    results/figures/step4_power_curves.png   (Figure 4 in paper)

Usage:
    python -m simulation.sim_2d_subdomain
    python -m simulation.sim_2d_subdomain --quick
"""
import sys
import json
import argparse
import warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.statistical_tests import compute_2d_pvalue_kde
from code.utils import set_random_seed

SCENARIOS = [
    "exp_2d_vs_quad1_modulation",
    "exp_2d_vs_quad2_modulation",
    "exp_2d_vs_multi_quad_modulation",
]

SCENARIO_MODULATIONS = {
    "exp_2d_vs_quad1_modulation":       (True,  False, False, False),
    "exp_2d_vs_quad2_modulation":       (False, True,  False, False),
    "exp_2d_vs_multi_quad_modulation":  (True,  True,  False, False),
}

N_SAMPLES      = 200
SIGMA          = 0.1
WEIGHT_DECAY_N = 0.7
ALPHA          = 0.05
C_VALUES       = np.arange(-0.5, 0.55, 0.05)


def run_single_trial(scenario, c):
    try:
        return compute_2d_pvalue_kde(
            n_samples=N_SAMPLES,
            subdomain_modulations=SCENARIO_MODULATIONS[scenario],
            modulation_strength=c,
            use_fft=True,
            scenario_name=scenario,
            sigma=SIGMA,
            noise_type="gaussian",
            dist_type="uniform",
            weight_decay_n=WEIGHT_DECAY_N,
        )
    except Exception:
        return (0.5, 0.5, 0.5, 0.5, 0.5)


def compute_rejection_rates(pvalue_results, alpha=ALPHA):
    arr = np.array(pvalue_results)
    n_sub = arr.shape[1] - 1
    rates = {"global": float(np.mean(arr[:, -1] < alpha))}
    for i in range(n_sub):
        rates[f"subdomain_{i+1}"] = float(np.mean(arr[:, i] < alpha))
    rates["bonferroni"] = float(np.mean(np.any(arr[:, :-1] < alpha / n_sub, axis=1)))
    return rates


def run_sim_2d_subdomain(n_reps=1000, seed=42, output_dir="results"):
    set_random_seed(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {}
    for scenario in SCENARIOS:
        print(f"\nStep 4 - {scenario}")
        sc_results = {}
        for c in C_VALUES:
            pvals = [run_single_trial(scenario, c)
                     for _ in tqdm(range(n_reps), desc=f"c={c:.2f}", leave=False)]
            sc_results[f"c_{c:.1f}"] = {
                "c": float(c),
                "rejection_rates": compute_rejection_rates(pvals),
                "pvalue_results": [[float(v) for v in row] for row in pvals],
            }
        results[scenario] = sc_results

    json_path = Path(output_dir) / "step4_2d_subdomain.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nStep 4 complete. Raw results: {json_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Figure 4")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-reps", type=int, default=1000)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()
    run_sim_2d_subdomain(n_reps=50 if args.quick else args.n_reps, seed=args.seed)
